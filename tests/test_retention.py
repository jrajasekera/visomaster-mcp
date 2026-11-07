"""Retention: deletion and pruning against concurrent submission, reuse, upload and aliases."""

import fcntl
import hashlib
import io
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import common
import control
import pytest
import test_jobs
from conftest import apply_config, write_config

backend, wait_for = test_jobs.backend, test_jobs.wait_for  # The shared job backend fixture.
OLD = time.time() - 10 * 86400


def completed_job(job_id, finished, size=1000, name="normalized.mp4"):
    job = common.job_path(job_id)
    (job / "artifacts").mkdir(parents=True)
    common.atomic_json(job / "request.json", {"kind": "normalize"})
    state = {"job_id": job_id, "status": "completed", "created_at": 0}
    if finished is not None:
        state["finished_at"] = finished
    common.atomic_json(job / "state.json", state)
    artifact = job / "artifacts" / name
    artifact.write_bytes(b"r" * size)
    return artifact


def media_file(name, size=100):
    path = common.BASE / "media" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"m" * size)
    return path


def aged(path, mtime=OLD):
    os.utime(path, (mtime, mtime))
    return path


def blocked(target):
    """Start target in a thread and confirm it waits (on the retention lock)."""
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(0.5)
    assert thread.is_alive(), "expected the operation to wait for the retention lock"
    return thread


def names(entries):
    return sorted(Path(entry["path"]).name for entry in entries)


def job_ids(result):
    return sorted(job["job_id"] for job in result["jobs"])


def finish(job_id):
    wait_for(job_id, {"running"})
    (common.job_path(job_id) / "release").touch()
    wait_for(job_id, {"completed"})


def stdin(monkeypatch, data):
    monkeypatch.setattr(control.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(data)))


# Replacement jobs and the deleted generation (qpn.12)


def test_replacement_cannot_publish_inside_a_prune_decision(backend, monkeypatch):
    completed_job("reused", OLD, size=1000)
    expected = control.tree_bytes(common.job_path("reused"))
    original, threads, started = control.current, [], []

    def replace():
        try:
            control.delete_job("reused")
        except FileNotFoundError:
            pass  # Prune removed the old generation first.
        control.submit({"job_id": "reused", "kind": "detect"})

    def selecting(directory):
        # The earliest point after prune selects the old job: a replacement must wait.
        if directory.name == "reused" and not started:
            started.append(directory)
            threads.append(blocked(replace))
        return original(directory)

    monkeypatch.setattr(control, "current", selecting)
    pruned = control.prune(7, dry_run=False)
    monkeypatch.setattr(control, "current", original)
    threads[0].join(10)
    assert pruned["jobs"] == [
        {
            "job_id": "reused",
            "status": "completed",
            "bytes": expected,
            "finished_at": OLD,
            "deleted": True,
        }
    ]
    finish("reused")
    fresh = backend / "jobs/reused/artifacts/result.mp4"
    fresh.write_bytes(b"n" * 10)
    assert control.prune(7, dry_run=False)["jobs"] == []
    assert fresh.exists() and control.status("reused")["status"] == "completed"


def test_prune_uses_each_generation_and_skips_vanished_candidates(backend, monkeypatch):
    completed_job("old", OLD, size=1000)
    completed_job("fresh", time.time())
    for job_id, mtime in (("legacy-old", OLD), ("legacy-fresh", time.time())):
        completed_job(job_id, None)
        aged(common.job_path(job_id) / "state.json", mtime)
    completed_job("vanish", OLD)
    completed_job("changed", OLD)
    referenced = completed_job("referenced", OLD)
    control.submit({"job_id": "consumer", "kind": "render", "media_path": str(referenced)})
    wait_for("consumer", {"running"})
    sizes = {
        job_id: control.tree_bytes(common.job_path(job_id)) for job_id in ("old", "legacy-old")
    }
    original, calls = control.listing, []

    def racing(directory):
        entries = original(directory)
        if directory.name == "jobs":
            calls.append(directory)
            if len(calls) == 2:  # The candidate scan, after the reference snapshot.
                entries.insert(0, directory / "ghost")
                shutil.rmtree(directory / "vanish")
                state = common.read_json(directory / "changed/state.json")
                state["finished_at"] = time.time()
                common.atomic_json(directory / "changed/state.json", state)
        return entries

    monkeypatch.setattr(control, "listing", racing)
    pruned = control.prune(7, dry_run=False)
    monkeypatch.setattr(control, "listing", original)
    assert len(calls) == 2
    reported = {job["job_id"]: (job["bytes"], job["finished_at"]) for job in pruned["jobs"]}
    assert reported == {
        "old": (sizes["old"], OLD),
        "legacy-old": (sizes["legacy-old"], pytest.approx(OLD)),
    }
    remaining = sorted(path.name for path in (backend / "jobs").iterdir())
    assert remaining == ["changed", "consumer", "fresh", "legacy-fresh", "referenced"]
    assert not list(backend.glob("trash-*"))


# Resolved inputs of active jobs (qpn.16)


@pytest.mark.parametrize("field", ["media_path", "target_face", "source_faces"])
def test_redirected_alias_keeps_the_input_resolved_at_submission(backend, field):
    artifact = completed_job("producer", OLD, name="face-0.png")
    plain, other = media_file("clip.mp4"), media_file("other.png")
    alias = common.BASE / "media/current.png"
    alias.symlink_to(artifact)
    mapping = {"target_face": str(plain), "source_faces": [str(plain)]}
    request = {"job_id": "consumer", "kind": "render", "media_path": str(plain)}
    if field == "media_path":
        request["media_path"] = str(alias)
    else:
        mapping[field] = str(alias) if field == "target_face" else [str(alias)]
    request["mappings"] = [mapping]
    control.submit(request)
    wait_for("consumer", {"running"})
    resolved = common.read_json(backend / "jobs/consumer/resolved.json")
    assert artifact.resolve() in control.request_paths(resolved)
    # The raw request stays as submitted, so an identical retry remains idempotent.
    assert str(alias) in str(common.read_json(backend / "jobs/consumer/request.json"))
    alias.unlink()
    alias.symlink_to(other)
    assert control.submit(request)["job_id"] == "consumer"
    with pytest.raises(ValueError, match="artifacts as input"):
        control.delete_job("producer")
    for dry_run in (True, False):
        assert control.prune(0, dry_run=dry_run)["jobs"] == []
    assert artifact.exists()
    (backend / "jobs/consumer/release").touch()
    wait_for("consumer", {"completed"})
    assert control.delete_job("producer")["deleted"]


def test_worker_opens_the_inputs_resolved_at_submission(tmp_path, monkeypatch):
    import engine

    config = write_config(tmp_path / "remote-config.json", tmp_path)
    settings = apply_config(monkeypatch, config)
    monkeypatch.setattr(engine, "CONFIG_ERROR", None)
    monkeypatch.setattr(engine, "SETTINGS", settings)
    monkeypatch.setattr(engine.sys, "platform", "test")  # No parent-death signal for pytest.
    monkeypatch.setenv("VM_SUPERVISOR_PID", str(os.getppid()))
    monkeypatch.setenv("VM_CONFIG_SHA256", settings.sha256)
    opened = []
    monkeypatch.setattr(
        engine, "normalize_video", lambda request, directory: opened.append(request) or {}
    )
    first, second = media_file("first.mp4"), media_file("second.mp4")
    alias = common.BASE / "media/alias.mp4"
    alias.symlink_to(first)
    job = common.job_path("normalize")
    job.mkdir(parents=True)
    request = {"kind": "normalize", "media_path": str(alias)}
    common.atomic_json(job / "request.json", request)
    common.atomic_json(job / "resolved.json", control.resolve_inputs(request))
    alias.unlink()
    alias.symlink_to(second)
    engine.run(job)
    assert opened[-1]["media_path"] == str(first.resolve())
    # A job persisted by an earlier release has no resolved.json and keeps its raw request.
    (job / "resolved.json").unlink()
    engine.run(job)
    assert opened[-1]["media_path"] == str(alias)
    # Resolved paths are revalidated, so a tampered record cannot escape the media root.
    outside = tmp_path.parent / f"{tmp_path.name}-outside.mp4"
    outside.write_bytes(b"x")
    common.atomic_json(job / "resolved.json", {**request, "media_path": str(outside)})
    with pytest.raises(ValueError, match="inside"):
        engine.run(job)
    outside.unlink()


def test_submission_still_rejects_out_of_base_symlinks(backend, monkeypatch, tmp_path_factory):
    monkeypatch.setattr(control, "validate_request", common.validate_request)
    outside = tmp_path_factory.mktemp("outside") / "clip.mp4"
    outside.write_bytes(b"x")
    escape = common.BASE / "media/escape.mp4"
    escape.parent.mkdir(parents=True)
    escape.symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        control.submit({"job_id": "escape", "kind": "detect", "media_path": str(escape)})
    assert not common.job_path("escape").exists()
    request = {"job_id": "valid", "kind": "detect", "media_path": str(media_file("in.mp4"))}
    control.submit(request)
    assert control.submit(request)["job_id"] == "valid"


def test_nested_upload_tree_with_an_active_input_is_kept(backend):
    uploads = backend / "uploads"
    trees = {}
    for tree in ("session", "sessionx", "other"):
        trees[tree] = uploads / tree / "clip.mp4"
        trees[tree].parent.mkdir(parents=True)
        trees[tree].write_bytes(b"u" * 100)
        aged(trees[tree])
        aged(trees[tree].parent)
    flat = uploads / ("a" * 64 + ".mp4")
    flat.write_bytes(b"u" * 100)
    aged(flat)
    control.submit({"job_id": "consumer", "kind": "render", "media_path": str(trees["session"])})
    wait_for("consumer", {"running"})
    # "sessionx" shares the text prefix of the referenced tree but not a path component.
    eligible = sorted(["sessionx", "other", flat.name])
    preview = control.prune(7)
    assert names(preview["uploads"]) == eligible
    assert all(path.exists() for path in trees.values()) and flat.exists()
    pruned = control.prune(7, dry_run=False)
    assert names(pruned["uploads"]) == eligible
    assert pruned["total_bytes"] == preview["total_bytes"]
    assert sorted(path.name for path in uploads.iterdir()) == ["session"]
    assert trees["session"].exists()
    finish("consumer")
    assert names(control.prune(7, dry_run=False)["uploads"]) == ["session"]
    assert not trees["session"].exists()


def test_legacy_active_job_keeps_every_possible_input(backend):
    upload = backend / "uploads" / ("a" * 64 + ".mp4")
    upload.parent.mkdir(parents=True)
    upload.write_bytes(b"u" * 100)
    aged(upload)
    partial = backend / "uploads" / ("b" * 64 + ".mp4.1234abcd.partial")
    partial.write_bytes(b"p")
    aged(partial)
    completed_job("producer", OLD)
    # An active record from an earlier release: no resolved.json, so its inputs are unknown.
    legacy = common.job_path("legacy")
    legacy.mkdir(parents=True)
    common.atomic_json(legacy / "request.json", {"kind": "render", "media_path": str(upload)})
    common.atomic_json(
        legacy / "state.json", {"job_id": "legacy", "status": "running", "created_at": 0}
    )
    with (legacy / "lease").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        for dry_run in (True, False):
            result = control.prune(7, dry_run=dry_run)
            assert result["jobs"] == [] and result["uploads"] == []
            assert names(result["stale"]) == [partial.name]
        with pytest.raises(ValueError, match="artifacts as input"):
            control.delete_job("producer")
        assert upload.exists() and common.job_path("producer").exists()
    # The supervisor exited: status reconciliation makes the job terminal and cleanup resumes.
    assert control.status("legacy")["status"] == "failed"
    result = control.prune(7, dry_run=False)
    assert job_ids(result) == ["producer"] and names(result["uploads"]) == [upload.name]


# Upload reuse against prune (qpn.17)


def old_upload(data):
    digest = hashlib.sha256(data).hexdigest()
    path = control.STATE / "uploads" / f"{digest}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return digest, aged(path)


def test_prune_that_wins_turns_a_concurrent_reuse_into_a_miss(backend, monkeypatch):
    data = b"face" * 100
    digest, path = old_upload(data)
    original, found, threads = control.active_references, [], []

    def pruning():
        references = original()

        def reuse():
            found.append(control.find_upload(digest, ".png", len(data)))

        threads.append(blocked(reuse))
        return references

    monkeypatch.setattr(control, "active_references", pruning)
    pruned = control.prune(7, dry_run=False)
    monkeypatch.setattr(control, "active_references", original)
    threads[0].join(10)
    assert names(pruned["uploads"]) == [path.name] and found == [None]
    stdin(monkeypatch, data)
    stored = control.upload(digest, ".png", len(data))
    assert stored["path"] == str(path) and not stored["reused"] and path.read_bytes() == data
    assert control.find_upload(digest, ".png", len(data))["reused"]


def test_reuse_that_wins_keeps_the_upload_through_a_concurrent_prune(backend, monkeypatch):
    data = b"face" * 100
    digest, path = old_upload(data)
    original, pruned, threads = os.utime, [], []

    def refreshing(*args, **kwargs):
        threads.append(blocked(lambda: pruned.append(control.prune(7, dry_run=False))))
        return original(*args, **kwargs)

    monkeypatch.setattr(control.os, "utime", refreshing)
    found = control.find_upload(digest, ".png", len(data))
    monkeypatch.setattr(control.os, "utime", original)
    threads[0].join(10)
    assert found == {"path": str(path), "bytes": len(data), "sha256": digest, "reused": True}
    assert pruned[0]["uploads"] == [] and path.read_bytes() == data


def test_reuse_refreshes_age_and_misses_a_file_removed_underneath(backend, monkeypatch):
    used, used_path = old_upload(b"used" * 10)
    untouched, untouched_path = old_upload(b"untouched" * 10)
    assert control.find_upload(used, ".png", 40)["reused"]
    assert names(control.prune(7, dry_run=False)["uploads"]) == [untouched_path.name]
    assert used_path.exists() and not untouched_path.exists()
    assert control.find_upload(untouched, ".png", 90) is None
    original = os.utime

    def removed(path, *args, **kwargs):
        Path(path).unlink()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(control.os, "utime", removed)
    assert control.find_upload(used, ".png", 40) is None


# Entries that disappear during the scan (qpn.18)


@pytest.mark.parametrize("dry_run", [True, False])
def test_prune_continues_when_an_upload_publishes_during_the_scan(backend, monkeypatch, dry_run):
    completed_job("ancient", OLD)
    uploads = backend / "uploads"
    uploads.mkdir(parents=True)
    final = uploads / ("c" * 64 + ".mp4")
    transient = uploads / f"{final.name}.0badf00d.partial"
    transient.write_bytes(b"t" * 50)
    stale = uploads / ("d" * 64 + ".mp4.1234abcd.partial")
    stale.write_bytes(b"s" * 10)
    aged(stale)
    inflight = uploads / ("e" * 64 + ".mp4.5678abcd.partial")
    inflight.write_bytes(b"i" * 10)
    aged(inflight, time.time() - 60)
    unrelated = uploads / ("f" * 64 + ".png")
    unrelated.write_bytes(b"u" * 100)
    aged(unrelated)
    original = control.listing

    def publishing(directory):
        entries = original(directory)
        if directory.name == "uploads" and transient.exists():
            os.replace(transient, final)  # What upload publication does with its temporary.
        return entries

    monkeypatch.setattr(control, "listing", publishing)
    result = control.prune(7, dry_run=dry_run)
    assert job_ids(result) == ["ancient"]
    assert names(result["uploads"]) == [unrelated.name]
    assert names(result["stale"]) == [stale.name]
    assert result["total_bytes"] == result["jobs"][0]["bytes"] + 100 + 10
    assert final.read_bytes() == b"t" * 50 and inflight.exists()
    assert common.job_path("ancient").exists() is dry_run
    assert unrelated.exists() is dry_run and stale.exists() is dry_run


# Previews match removal (qpn.19)


def test_prune_preview_matches_removal_for_referenced_producers(backend):
    normalized = completed_job("media-producer", OLD)
    crop = completed_job("face-producer", OLD, size=500, name="face-0.png")
    completed_job("unused", OLD, size=700)
    completed_job("fresh", time.time())
    clip, face = media_file("clip.mp4"), media_file("face.png")
    plain = {"target_face": str(face), "source_faces": [str(face)]}
    control.submit(
        {"job_id": "running", "kind": "render", "media_path": str(normalized), "mappings": [plain]}
    )
    wait_for("running", {"running"})
    # The GPU lock keeps this consumer queued; it uses the crop as target and source.
    crops = {"target_face": str(crop), "source_faces": [str(crop)]}
    control.submit(
        {"job_id": "queued", "kind": "preview", "media_path": str(clip), "mappings": [crops]}
    )
    assert control.status("queued")["status"] == "queued"
    jobs = sorted(path.name for path in (backend / "jobs").iterdir())

    def compare(expected):
        preview = control.prune(7)
        assert sorted(path.name for path in (backend / "jobs").iterdir()) == jobs
        removal = control.prune(7, dry_run=False)
        assert job_ids(preview) == job_ids(removal) == expected
        assert [job["bytes"] for job in preview["jobs"]] == [
            job["bytes"] for job in removal["jobs"]
        ]
        assert preview["total_bytes"] == removal["total_bytes"]
        for job_id in expected:
            jobs.remove(job_id)
        assert sorted(path.name for path in (backend / "jobs").iterdir()) == jobs

    compare(["unused"])
    assert normalized.exists() and crop.exists()
    finish("running")
    assert control.status("queued")["status"] in {"queued", "running"}
    compare(["media-producer"])
    finish("queued")
    compare(["face-producer"])
    assert common.job_path("fresh").exists()

"""Seams between the vmcp-harden-pqj features: hashing before GPU admission, segment plans that
miss the video, retention of resumable inputs and shared chunks, weight-0 sources and pinned
comparison references."""

import os
import threading

import common
import control
import engine
import pytest
import test_cancel
import test_jobs
import test_preflight
import test_recipes
from test_cancel import until, wait_for
from test_retention import OLD, completed_job
from test_scheduling import gpu_held

library, site, worker = test_preflight.library, test_preflight.site, test_recipes.worker
backend, jobs_backend = test_cancel.backend, test_jobs.backend  # Stub runners.
swap, preflight, by_name = test_preflight.swap, test_preflight.preflight, test_preflight.by_name


@pytest.fixture(autouse=True)
def no_digests(monkeypatch):
    """Supervisors and workers set common.DIGESTS for their process; undo it after each test."""
    monkeypatch.setattr(common, "DIGESTS", None)


# Hashing before GPU admission (finding 1)


def tick(path):
    """Wait until the filesystem clock, which may be coarse, has passed path's change time."""
    probe = path.with_name(path.name + ".tick")
    probe.touch()
    while probe.stat().st_ctime_ns <= path.stat().st_ctime_ns:
        os.utime(probe)
    probe.unlink()


def test_render_inputs_are_hashed_before_the_job_joins_the_queue(site, monkeypatch):
    request = swap(site)
    with gpu_held(common.LOCK_DIR):
        control.submit({"job_id": "render", **request})
        job = common.job_path("render")
        # The supervisor hashes while the GPU is held elsewhere, then joins at its sequence.
        until(lambda: (job / "digests.json").exists())
        assert control.status("render")["status"] == "queued"
        until(lambda: control.status("render")["queue"]["reason"] == "gpu_busy")
        assert control.status("render")["sequence"] == 1
    digests = set(common.read_json(job / "digests.json").values())
    resolved = common.read_json(job / "resolved.json")
    record = common.provenance(resolved)
    hashed = [entry["sha256"] for entry in record["inputs"].values()]
    hashed += [model["sha256"] for model in record["models"].values()]
    hashed += list(record["adapter"]["code"].values())
    assert set(hashed) <= digests and None not in hashed

    # The worker's own provenance then reads no file: every hash comes from the supervisor's.
    common.DIGESTS = common.read_json(job / "digests.json")
    streamed, original = [], common.hashlib.sha256

    def sha256(*data):
        streamed.extend(() if data else [1])  # file_sha256 streams into an empty hash.
        return original(*data)

    with monkeypatch.context() as patch:
        patch.setattr(common.hashlib, "sha256", sha256)
        assert common.provenance(resolved) == record
    assert streamed == []


def test_a_cancel_while_hashing_stops_the_supervisor_and_others_use_the_gpu(
    backend, tmp_path, monkeypatch
):
    face = tmp_path / "face.png"
    face.write_bytes(b"face")
    mapping = {"target_face": str(face), "source_faces": [str(face)]}
    request = {"kind": "render", "media_path": str(face), "mappings": [mapping]}
    with monkeypatch.context() as patch:
        patch.setattr(control.subprocess, "Popen", lambda *a, **kw: None)
        control.submit({"job_id": "render", **request})
    job = common.job_path("render")
    hashing, resume = threading.Event(), threading.Event()

    def blocked(request, models=()):
        hashing.set()
        resume.wait(30)
        return {}

    monkeypatch.setattr(control, "provenance", blocked)
    supervisor = threading.Thread(
        target=control.supervise, args=(str(job), control.SETTINGS.sha256)
    )
    supervisor.start()
    try:
        assert hashing.wait(10)
        state = control.status("render")
        assert state["status"] == "queued" and state["progress"]["stage"] == "fingerprinting"
        assert state["queue"]["reason"] == "starting"  # Not eligible while it hashes.
        # A job submitted later takes the idle GPU meanwhile.
        control.submit({"job_id": "other", "kind": "detect"})
        until(lambda: (common.job_path("other") / "entered").exists())
        assert control.cancel("render")["cancel_requested"]
        supervisor.join(5)
        assert not supervisor.is_alive()
        assert control.status("render")["status"] == "cancelled"
        assert not (job / "worker").exists() and not (job / "digests.json").exists()
    finally:
        resume.set()
        supervisor.join(5)
    (common.job_path("other") / "release").touch()
    assert wait_for("other", {"completed"})["status"] == "completed"


def test_cached_digests_detect_rewritten_bytes(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"one")
    common.DIGESTS = {}
    digest = common.file_sha256(path)
    assert list(common.DIGESTS.values()) == [digest]
    # The cache answers for the unchanged file without reading it.
    (identity,) = common.DIGESTS
    common.DIGESTS[identity] = "f" * 64
    assert common.file_sha256(path) == "f" * 64
    # Same size and modification time, other bytes: the change time still differs.
    tick(path)
    info = path.stat()
    path.write_bytes(b"two")
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert common.file_sha256(path) == common.hashlib.sha256(b"two").hexdigest()


def test_worker_reuses_the_supervisors_media_hash_and_sees_a_change(worker):
    request = {**worker.request, "media_sha256": "f" * 64}
    common.atomic_json(worker.job / "resolved.json", control.fingerprint(request))
    with worker.clip.open("rb") as stream:
        identity = common.file_identity(stream)
    common.atomic_json(worker.job / "digests.json", {identity: "f" * 64})
    engine.run(worker.job)  # The recorded digest stands for the unchanged media.
    assert worker.loaded == [1]
    tick(worker.clip)
    worker.clip.write_bytes(b"edit")  # Rewritten: the worker hashes it and refuses.
    with pytest.raises(ValueError, match="changed after submission"):
        engine.run(worker.job)
    assert worker.loaded == [1]


# Segment plans past the end of the video (finding 2)


def test_preflight_blocks_segments_past_the_end_of_the_video(site):
    late = [{"id": "late", "start_frame": 40, "end_frame": 50, "passthrough": True}]
    report = preflight(swap(site, segments=late))
    checks = by_name(report)
    assert checks["request"]["status"] == "passed"  # Validation cannot know the length.
    assert checks["media"]["status"] == "blocked"
    assert checks["media"]["code"] == "segment_outside_range"
    assert report["readiness"] == "blocked"
    # Within the 25-frame clip the same plan is ready, and so is one clipped at its end.
    near = [{"id": "near", "start_frame": 20, "end_frame": 50, "passthrough": True}]
    assert by_name(preflight(swap(site, segments=near)))["media"]["status"] == "passed"
    every = [{"id": "all", "start_frame": 0, "end_frame": 50, "passthrough": True}]
    assert by_name(preflight(swap(site, segments=every)))["media"]["code"] == "all_passthrough"


def test_worker_refuses_segments_past_the_end_before_models_load(worker, library):
    worker.clip.write_bytes((library / "clip.mp4").read_bytes())
    late = [{"id": "late", "start_frame": 40, "end_frame": 50, "passthrough": True}]
    request = {**worker.request, "kind": "render", "max_frames": 60, "segments": late}
    common.atomic_json(worker.job / "resolved.json", control.fingerprint(request))
    with pytest.raises(ValueError, match="outside the rendered frames"):
        engine.run(worker.job)
    assert worker.loaded == []


# Retention of resumable renders and hard-linked chunks (findings 3 and 4)


def stopped_render(job_id, media, status="failed", checkpoint=True):
    job = common.job_path(job_id)
    (job / "artifacts").mkdir(parents=True)
    request = {"kind": "render", "media_path": str(media)}
    for name in ("request.json", "resolved.json"):
        common.atomic_json(job / name, request)
    state = {"job_id": job_id, "status": status, "created_at": 0, "finished_at": OLD}
    common.atomic_json(job / "state.json", state)
    if checkpoint:
        (job / "checkpoint/chunks").mkdir(parents=True)
        common.atomic_json(job / "checkpoint/manifest.json", {})
    return job


def test_inputs_of_resumable_renders_are_kept(jobs_backend):
    media = completed_job("producer", OLD)
    upload = common.STATE / "uploads" / ("a" * 64 + ".mp4")
    upload.parent.mkdir(parents=True)
    upload.write_bytes(b"u")
    os.utime(upload, (OLD, OLD))
    stopped_render("failed", media)
    stopped_render("cancelled", upload, status="cancelled")
    stopped_render("plain", completed_job("unprotected", OLD), checkpoint=False)
    with pytest.raises(ValueError, match="resumable failed or cancelled render"):
        control.delete_job("producer")
    preview = control.prune(0, dry_run=True)
    assert "producer" not in [entry["job_id"] for entry in preview["jobs"]]
    assert str(upload) not in [entry["path"] for entry in preview["uploads"]]
    # The renders themselves stay prunable, and a render without a checkpoint protects nothing.
    assert {"failed", "cancelled", "plain", "unprotected"} <= {
        entry["job_id"] for entry in preview["jobs"]
    }
    control.delete_job("failed")
    control.delete_job("cancelled")
    assert control.delete_job("producer")["deleted"]
    assert str(upload) in [entry["path"] for entry in control.prune(0, dry_run=True)["uploads"]]


def json_bytes(job):
    return sum(path.stat().st_size for path in job.rglob("*.json"))


def test_hard_linked_chunks_count_once_and_only_when_freed(jobs_backend):
    first, second = (stopped_render(name, common.BASE / "clip.mp4") for name in ("a", "b"))
    chunk = first / "checkpoint/chunks/00000.mp4"
    chunk.write_bytes(b"c" * 1000)
    os.link(chunk, second / "checkpoint/chunks/00000.mp4")  # As resume_render links it.
    os.link(chunk, first / "artifacts/copy.mp4")  # A second link inside the same job.
    own = {"a": json_bytes(first), "b": json_bytes(second)}
    # The other job's link keeps the chunk, so removing one job alone frees none of it.
    assert control.tree_bytes(first) == own["a"] and control.tree_bytes(second) == own["b"]
    preview = control.prune(0, dry_run=True)
    # Removed together they free it, counted once, for the first job listed.
    assert {entry["job_id"]: entry["bytes"] for entry in preview["jobs"]} == {
        "a": own["a"] + 1000,
        "b": own["b"],
    }
    assert preview["total_bytes"] == own["a"] + own["b"] + 1000
    assert control.delete_job("a")["bytes"] == own["a"]
    assert control.delete_job("b")["bytes"] == own["b"] + 1000  # Now its last link.

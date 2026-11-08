"""Submission: persisted retries, Fusion-free normalization, managed inputs, job IDs and health."""

import asyncio
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import common
import control
import engine
import pytest
import test_server
from conftest import ROOT, apply_config, write_config
from mcp import ClientSession
from mcp.client.stdio import stdio_client

server, call = test_server.server, test_server.call  # The stdio server over the local SSH shim.
WIDGETS = "app/ui/widgets"
# Stub worker. Normalization execs the real engine (FFmpeg only) once STATE/hold is absent; other
# kinds revalidate their resolved request with common.validate_request and wait for release.
RUNNER = (
    "#!/usr/bin/env python3\n"
    + """
import json, os, sys, time
from pathlib import Path
engine, job = sys.argv[1], Path(sys.argv[2])
request = json.loads((job / "resolved.json").read_text())
with open(job / "runs", "a") as runs:
    runs.write("run\\n")
(job / "entered").touch()
if request["kind"] == "normalize":
    while (job.parents[1] / "hold").exists():
        time.sleep(0.05)
    os.execv(sys.executable, [sys.executable, engine, str(job)])
sys.path.insert(0, os.path.dirname(engine))
import common
try:
    common.validate_request(request)
except Exception as error:
    (job / "error.json").write_text(json.dumps({"type": "ValueError", "message": str(error)}))
    sys.exit(1)
while not (job / "release").exists():
    time.sleep(0.05)
(job / "result.json").write_text(json.dumps({"validated": True}))
"""
)
SETTINGS_LAYOUT = (
    "LAYOUT = {'Detection': {'DetectorScoreSlider': "
    "{'default': '50', 'min_value': '1', 'max_value': '100'}}}\n"
)


def ffmpeg(*arguments):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *map(str, arguments)], check=True)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    directory = tmp_path_factory.mktemp("submit-media")
    clip, face = directory / "clip.mp4", directory / "face.png"
    lavfi = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=0.4"]
    ffmpeg(*lavfi, "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p", clip)
    ffmpeg("-i", clip, "-frames:v", "1", face)
    return SimpleNamespace(clip=clip, face=face)


def wait_for(job_id, statuses, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = control.status(job_id)
        if result["status"] in statuses:
            return result
        time.sleep(0.05)
    raise AssertionError(control.status(job_id))


def wait_file(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.05)


def settle(state):
    """Cancel jobs still active so no supervisor outlives the test."""
    for job in (state / "jobs").glob("*"):
        (job / "cancel").touch()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        statuses = []
        for path in (state / "jobs").glob("*/state.json"):
            try:
                statuses.append(json.loads(path.read_text())["status"])
            except (ValueError, KeyError):
                pass  # Fixture files that are not job state.
        if all(status in common.TERMINAL for status in statuses):
            return
        time.sleep(0.1)


@pytest.fixture
def make_site(tmp_path, monkeypatch, library):
    """A real-validation backend: media root, state nested (default) or external, stub runner."""
    sites = []

    def make(external=False):
        media = tmp_path / "media"
        media.mkdir()
        shutil.copy(library.clip, media / "clip.mp4")
        shutil.copy(library.face, media / "face.png")
        shutil.copy(library.face, media / "source.png")
        state = tmp_path / "runtime" if external else media / "state"
        runner = tmp_path / "bin/runner"
        runner.parent.mkdir()
        runner.write_text(RUNNER)
        runner.chmod(0o755)
        config = write_config(
            tmp_path / "remote-config.json",
            media,
            runner=runner,
            fusion_dir=str(tmp_path / "fusion"),
            state_dir=str(state),
        )
        settings = apply_config(monkeypatch, config)
        for name in ("SETTINGS", "APP", "CONFIG_ERROR"):
            monkeypatch.setattr(engine, name, getattr(common, name))
        site = SimpleNamespace(
            media=settings.base,
            state=settings.state,
            fusion=settings.app,
            config=config,
            clip=str(settings.base / "clip.mp4"),
            face=str(settings.base / "face.png"),
            source=str(settings.base / "source.png"),
        )
        sites.append(site)
        return site

    yield make
    for site in sites:
        (site.state / "hold").unlink(missing_ok=True)
        settle(site.state)


def remove_layouts(site):
    shutil.rmtree(site.fusion / "app")


def preview(site, job_id="shot-preview", **changes):
    mapping = {"target_face": site.face, "source_faces": [site.source], "parameters": {}}
    request = {
        "job_id": job_id,
        "kind": "preview",
        "media_path": site.clip,
        "mappings": [mapping],
        "frame": 0,
        "start_frame": 0,
        "max_frames": None,
        "controls": {},
    }
    return {**request, **changes}


def complete(job_id):
    wait_file(common.job_path(job_id) / "entered")
    (common.job_path(job_id) / "release").touch()
    return wait_for(job_id, {"completed"})


def control_command(site, payload):
    """A fresh control process, which loads the configuration from disk as SSH commands do."""
    environment = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG") if k in os.environ}
    environment["VM_CONFIG"] = str(site.config)
    result = subprocess.run(
        [sys.executable, str(ROOT / "remote/control.py")],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=environment,
    )
    return json.loads(result.stdout)


def counting_popen(monkeypatch):
    calls, real = [], subprocess.Popen

    def spawn(arguments, *args, **kwargs):
        calls.append(arguments)
        return real(arguments, *args, **kwargs)

    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    return calls


# Identical retries return the persisted job (qpn.20)


def test_identical_retry_returns_completed_job_after_inputs_and_layouts_vanish(
    make_site, monkeypatch
):
    site = make_site()
    request = preview(site)
    control.submit(dict(request))
    original = complete("shot-preview")
    assert original["result"] == {"validated": True}
    for path in (site.clip, site.face, site.source):
        os.unlink(path)
    remove_layouts(site)
    calls = counting_popen(monkeypatch)
    retried = control.submit(dict(request))
    assert retried["status"] == "completed" and retried["result"] == original["result"]
    assert retried["created_at"] == original["created_at"] and calls == []
    # A new control process loads the configuration without the layout files and agrees.
    answer = control_command(site, {"action": "submit", **request})
    assert answer["result"]["status"] == "completed"
    assert answer["result"]["created_at"] == original["created_at"]
    assert (common.job_path("shot-preview") / "runs").read_text() == "run\n"


@pytest.mark.parametrize(
    "changes",
    [{"frame": 1}, {"frame": False}, {"max_frames": 0}, {"kind": "render"}, {"extra": None}],
)
def test_changed_payload_is_a_conflict_even_with_missing_inputs(make_site, monkeypatch, changes):
    site = make_site()
    request = preview(site)
    control.submit(dict(request))
    complete("shot-preview")
    os.unlink(site.clip)
    calls = counting_popen(monkeypatch)
    with pytest.raises(ValueError, match="different request"):
        control.submit({**request, **changes})
    assert calls == []


def test_new_job_with_missing_input_publishes_nothing(make_site, monkeypatch):
    site = make_site()
    os.unlink(site.face)
    calls = counting_popen(monkeypatch)
    with pytest.raises(FileNotFoundError):
        control.submit(preview(site, job_id="fresh"))
    assert not (site.state / "jobs/fresh").exists() and calls == []
    assert list(site.state.glob("prepare-*")) == []


def test_concurrent_identical_submissions_start_one_supervisor(make_site, monkeypatch):
    site = make_site()
    calls = counting_popen(monkeypatch)
    barrier, results, errors = threading.Barrier(8), [], []

    def submit():
        barrier.wait()
        try:
            results.append(control.submit(preview(site, job_id="burst")))
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert errors == [] and len(results) == 8 and len(calls) == 1
    assert len({result["created_at"] for result in results}) == 1
    complete("burst")
    assert (common.job_path("burst") / "runs").read_text() == "run\n"


# Normalization without Fusion layouts or models (qpn.24)


def forbid_fusion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("normalization must not touch Fusion")

    for module in (common, engine, control):
        if hasattr(module, "layouts"):
            monkeypatch.setattr(module, "layouts", forbidden)
    monkeypatch.setattr(engine, "initialize", forbidden)


def test_normalization_without_layout_files_submits_runs_and_completes(make_site, monkeypatch):
    site = make_site()
    remove_layouts(site)
    forbid_fusion(monkeypatch)  # In this process; the worker has no layout files to read.
    control.submit({"job_id": "norm", "kind": "normalize", "media_path": site.clip, "fps": None})
    result = wait_for("norm", {"completed", "failed"})
    assert result["status"] == "completed", result
    assert result["result"]["media_path"] == str(site.state / "jobs/norm/artifacts/normalized.mp4")
    assert result["result"]["frames"] == 10


def test_queued_normalization_proceeds_after_layouts_vanish(make_site):
    site = make_site()
    (site.state / "hold").touch()
    control.submit({"job_id": "held", "kind": "normalize", "media_path": site.clip, "fps": "25"})
    wait_file(common.job_path("held") / "entered")
    remove_layouts(site)
    (site.state / "hold").unlink()
    result = wait_for("held", {"completed", "failed"})
    assert result["status"] == "completed", result


def test_worker_revalidates_normalization_without_fusion(make_site, monkeypatch, tmp_path):
    site = make_site()
    remove_layouts(site)
    forbid_fusion(monkeypatch)
    job = common.job_path("inline")
    (job / "artifacts").mkdir(parents=True)
    request = {"kind": "normalize", "media_path": site.clip, "fps": None}
    common.atomic_json(job / "resolved.json", request)
    monkeypatch.setenv("VM_SUPERVISOR_PID", str(os.getppid()))
    monkeypatch.setenv("VM_CONFIG_SHA256", common.SETTINGS.sha256)
    engine.run(job)
    assert common.read_json(job / "result.json")["frames"] == 10


@pytest.mark.parametrize(
    "changes, error, message",
    [
        ({"fps": "0"}, ValueError, "fps must be"),
        ({"fps": "1/0"}, ValueError, "fps must be"),
        ({"fps": True}, ValueError, "fps must be"),
        ({"media_path": "missing.mp4"}, ValueError, "absolute"),
        ({"media_path": "/nonexistent/clip.mp4"}, FileNotFoundError, "No such file"),
        ({"media_path": "outside"}, ValueError, "Media must be"),
        ({"controls": {}}, ValueError, "Normalization accepts only"),
        ({"mappings": []}, ValueError, "Normalization accepts only"),
        ({"frame": 0}, ValueError, "Normalization accepts only"),
    ],
)
def test_normalization_validation_errors_need_no_fusion(
    make_site, monkeypatch, tmp_path, changes, error, message
):
    site = make_site()
    remove_layouts(site)
    forbid_fusion(monkeypatch)
    outside = tmp_path / "outside.mp4"
    shutil.copy(site.clip, outside)
    (site.media / "escape.mp4").symlink_to(outside)
    if changes.get("media_path") == "outside":
        changes = {"media_path": str(site.media / "escape.mp4")}
    request = {"kind": "normalize", "media_path": site.clip, "fps": None, **changes}
    with pytest.raises(error, match=message):
        common.validate_request(request)
    with pytest.raises(error, match=message):
        control.submit({"job_id": "invalid", **request})
    assert not (site.state / "jobs/invalid").exists()


def test_gpu_kinds_keep_fusion_settings_validation(make_site):
    site = make_site()
    (site.fusion / WIDGETS / "settings_layout_data.py").write_text(SETTINGS_LAYOUT)
    detect = {"kind": "detect", "media_path": site.clip, "frame": 0}
    common.validate_request({**detect, "controls": {"DetectorScoreSlider": 60}})
    with pytest.raises(ValueError, match="Invalid value for DetectorScoreSlider"):
        common.validate_request({**detect, "controls": {"DetectorScoreSlider": 500}})
    with pytest.raises(ValueError, match="Unsupported setting Bogus"):
        common.validate_request({**detect, "controls": {"Bogus": 1}})
    request = preview(site)
    del request["job_id"]
    request["mappings"][0]["parameters"] = {"Bogus": 1}
    with pytest.raises(ValueError, match="Unsupported setting Bogus"):
        common.validate_request(request)
    for kind in ("preview", "render"):
        with pytest.raises(ValueError, match="At least one"):
            common.validate_request({"kind": kind, "media_path": site.clip})
    remove_layouts(site)
    with pytest.raises(ValueError, match="Fusion layout .*settings_layout_data.py"):
        common.validate_request({**detect, "controls": {}})


# Uploads and job artifacts as inputs with state outside media_root (qpn.25)


def upload(monkeypatch, path):
    data = Path(path).read_bytes()
    monkeypatch.setattr(control.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(data)))
    digest = hashlib.sha256(data).hexdigest()
    return control.upload(digest, Path(path).suffix, len(data))["path"]


def test_external_state_config_is_accepted_but_must_not_contain_media(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    config = write_config(tmp_path / "c.json", media, state_dir=str(tmp_path / "runtime"))
    assert common.load_config(config).state == (tmp_path / "runtime").resolve()
    for state in (media, tmp_path):
        data = json.loads(config.read_text())
        config.write_text(json.dumps({**data, "state_dir": str(state)}))
        with pytest.raises(common.ConfigError, match="state_dir must not be media_root"):
            common.load_config(config)


def test_upload_and_normalized_artifact_feed_later_jobs_with_external_state(make_site, monkeypatch):
    site = make_site(external=True)
    assert not site.state.is_relative_to(site.media)
    face = upload(monkeypatch, site.face)
    assert Path(face).parent == site.state / "uploads"
    assert control.dispatch("inspect", {"media_path": face})["streams"]
    # Producer: normalization from the media root publishes an artifact under external state.
    control.submit({"job_id": "producer", "kind": "normalize", "media_path": site.clip})
    produced = wait_for("producer", {"completed", "failed"})
    assert produced["status"] == "completed", produced
    normalized = produced["result"]["media_path"]
    assert Path(normalized).is_relative_to(site.state / "jobs/producer/artifacts")
    assert control.dispatch("inspect", {"media_path": normalized})["streams"]
    # Consumers: the worker revalidates the same paths, as media and as face references.
    mapping = {"target_face": face, "source_faces": [face], "parameters": {}}
    render = {"kind": "render", "media_path": normalized, "mappings": [mapping]}
    control.submit({"job_id": "consumer", **render, "start_frame": 0, "controls": {}})
    # Normal class, so it queues behind the render in FIFO order instead of passing it.
    crops = {"kind": "detect", "media_path": face, "frame": 0, "priority": "normal"}
    control.submit({"job_id": "crops", **crops})
    wait_file(common.job_path("consumer") / "entered")  # Detection waits for the GPU lock.
    # Retention compares resolved paths: the producer and the upload stay while consumers run.
    with pytest.raises(ValueError, match="uses this job's artifacts"):
        control.delete_job("producer")
    preview_prune = control.prune(0, dry_run=True)
    assert face not in [entry["path"] for entry in preview_prune["uploads"]]
    assert "producer" not in [entry["job_id"] for entry in preview_prune["jobs"]]
    for job_id in ("consumer", "crops"):
        assert complete(job_id)["result"] == {"validated": True}
    assert control.delete_job("producer")["deleted"]
    assert face in [entry["path"] for entry in control.prune(0, dry_run=True)["uploads"]]


@pytest.mark.parametrize("external", [True, False])
def test_only_managed_state_files_are_inputs(make_site, monkeypatch, tmp_path, external):
    site = make_site(external=external)
    state = site.state
    face = upload(monkeypatch, site.face)
    artifacts = state / "jobs/done/artifacts"
    (artifacts / "nested").mkdir(parents=True)
    for name in ("face-0.png", "nested/frame.png"):
        shutil.copy(site.face, artifacts / name)
    for name in ("request.json", "state.json", "worker.log", "result.json", "lease"):
        shutil.copy(site.face, state / "jobs/done" / name)
    common.atomic_json(state / "jobs/done/state.json", {"job_id": "done", "status": "completed"})
    shutil.copy(site.face, state / "uploads" / (Path(face).name + ".1234abcd.partial"))
    shutil.copy(site.face, state / "loose.png")
    (state / "trash-old-1234/artifacts").mkdir(parents=True)
    shutil.copy(site.face, state / "trash-old-1234/artifacts/face-0.png")
    (state / "jobs/bad.name/artifacts").mkdir(parents=True)
    shutil.copy(site.face, state / "jobs/bad.name/artifacts/face-0.png")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.copy(site.face, elsewhere / "face.png")
    # Symlinks resolve before the check, so they cannot extend or redirect the managed roots.
    (state / "uploads" / ("e" * 64 + ".png")).symlink_to(elsewhere / "face.png")
    (state / "jobs/linked").mkdir()
    (state / "jobs/linked/artifacts").symlink_to(elsewhere)
    (site.media / "log.png").symlink_to(state / "jobs/done/worker.log")
    (site.media / "alias.png").symlink_to(artifacts / "face-0.png")
    accepted = [
        site.clip,
        face,
        artifacts / "face-0.png",
        artifacts / "nested/frame.png",
        site.media / "alias.png",
    ]
    for path in accepted:
        assert common.input_path(str(path)) == Path(os.path.realpath(path))
    rejected = [
        *(state / "jobs/done" / name for name in ("request.json", "state.json", "worker.log")),
        state / "jobs/done/result.json",
        state / "jobs/done/lease",
        f"{artifacts}/../request.json",
        f"{state}/uploads/../loose.png",
        state / "uploads" / (Path(face).name + ".1234abcd.partial"),
        state / "loose.png",
        state / "trash-old-1234/artifacts/face-0.png",
        state / "jobs/bad.name/artifacts/face-0.png",
        state / "uploads" / ("e" * 64 + ".png"),
        state / "jobs/linked/artifacts/face.png",
        site.media / "log.png",
        elsewhere / "face.png",
        artifacts,
    ]
    for path in rejected:
        with pytest.raises(ValueError, match="Media must be"):
            common.input_path(str(path))


# Explicit job IDs (qpn.27)


def media(tmp_path):
    for name in ("clip.mp4", "face.png", "source.png"):
        (tmp_path / name).write_bytes(b"x")
    mapping = {
        "target_face": str(tmp_path / "face.png"),
        "source_faces": [str(tmp_path / "source.png")],
    }
    clip = str(tmp_path / "clip.mp4")
    return {
        "detect_faces": {"media_path": clip},
        "preview": {"media_path": clip, "mappings": [mapping], "mode": "preview"},
        "render": {"media_path": clip, "mappings": [mapping], "mode": "render"},
        "normalize_media": {"media_path": clip},
    }


def tool(name):
    return "submit_swap" if name in ("preview", "render") else name


def jobs(tmp_path):
    return sorted(p.name for p in (tmp_path / "state/jobs").iterdir())


@pytest.fixture
def stdio(server, tmp_path):
    yield server
    settle(tmp_path / "state")


def test_empty_job_id_is_rejected_by_every_submission_tool(stdio, tmp_path):
    requests = media(tmp_path)
    results = call(stdio, *((tool(n), {**a, "job_id": ""}) for n, a in requests.items()))
    for result in results:
        assert result.isError and "job_id must be 1" in result.content[0].text
    assert jobs(tmp_path) == ["done"]
    assert list((tmp_path / "state").glob("prepare-*")) == []


def test_omitted_null_and_explicit_job_ids(stdio, tmp_path):
    requests = media(tmp_path)
    calls = []
    for name, arguments in requests.items():
        calls += [(tool(name), arguments), (tool(name), {**arguments, "job_id": None})]
        explicit = {**arguments, "job_id": f"keep-{name.replace('_', '-')}"}
        calls += [(tool(name), explicit), (tool(name), explicit)]
        changed = {**explicit, "media_path": str(tmp_path / "face.png")}
        calls.append((tool(name), changed))
    results = iter(call(stdio, *calls))
    for name in requests:
        omitted, null, first, retry, conflict = (next(results) for _ in range(5))
        for generated in (omitted, null):
            assert not generated.isError, generated.content
            assert re.fullmatch(r"[0-9a-f]{32}", generated.structuredContent["job_id"])
        assert omitted.structuredContent["job_id"] != null.structuredContent["job_id"]
        expected = f"keep-{name.replace('_', '-')}"
        assert first.structuredContent["job_id"] == retry.structuredContent["job_id"] == expected
        assert first.structuredContent["created_at"] == retry.structuredContent["created_at"]
        assert conflict.isError and "different request" in conflict.content[0].text
    assert len(jobs(tmp_path)) == 1 + 3 * len(requests)


def test_submission_schemas_describe_the_job_id_format():
    async def schemas():
        params = test_server.StdioServerParameters(
            command=sys.executable, args=["-m", "visomaster_mcp.server"]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return {tool.name: tool.inputSchema for tool in (await session.list_tools()).tools}

    found = asyncio.run(schemas())
    for name in ("detect_faces", "submit_swap", "normalize_media", "analyze_references"):
        description = found[name]["properties"]["job_id"]["description"]
        assert "1–64 letters, digits, underscores or hyphens" in description
        assert "omit" in description.lower()


# Health compatibility shares the worker's guard (qpn.28)


def git(repository, *arguments):
    environment = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG") if k in os.environ}
    environment.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="test",
        GIT_AUTHOR_EMAIL="test@example.com",
        GIT_COMMITTER_NAME="test",
        GIT_COMMITTER_EMAIL="test@example.com",
    )
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A real Git Fusion stand-in at the supported revision, applied to control and engine."""
    media = tmp_path / "media"
    media.mkdir()
    config = write_config(tmp_path / "c.json", media, fusion_dir=str(tmp_path / "fusion"))
    fusion = tmp_path / "fusion"
    (fusion / ".git").rmdir()
    (fusion / "README.md").write_text("outside app\n")
    git(fusion, "init", "-q")
    git(fusion, "add", "-A")
    git(fusion, "commit", "-q", "-m", "fusion")
    revision = git(fusion, "rev-parse", "HEAD")
    apply_config(monkeypatch, config)
    for module in (common, control, engine):
        monkeypatch.setattr(module, "REVISION", revision)
    monkeypatch.setattr(engine, "APP", common.APP)
    return SimpleNamespace(
        path=fusion, revision=revision, layout=fusion / WIDGETS / "common_layout_data.py"
    )


def guard(monkeypatch):
    """engine.initialize's outcome: the guard's error, or None once it would import Fusion."""

    class Passed(Exception):
        pass

    def stop(path):
        raise Passed

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(engine.os, "chdir", stop)
    try:
        engine.initialize({})
    except Passed:
        return None
    except RuntimeError as error:
        return str(error)
    raise AssertionError("initialize returned without reaching the Fusion import")


def test_clean_supported_checkout_is_compatible(checkout, monkeypatch):
    result = control.health()
    assert result["compatible"] is True and result["compatibility_reason"] is None
    assert result["revision"] == result["supported_revision"] == checkout.revision
    for key in ("code", "gpu", "gpus", "app", "config_sha256", "checks", "release"):
        assert key in result
    assert guard(monkeypatch) is None


@pytest.mark.parametrize("staged", [False, True])
def test_tracked_app_change_is_incompatible_in_health_and_worker(checkout, monkeypatch, staged):
    checkout.layout.write_text("{'changed': {}}\n")
    if staged:
        git(checkout.path, "add", "-A")
    result = control.health()
    assert result["compatible"] is False and result["revision"] == checkout.revision
    assert "tracked local changes" in result["compatibility_reason"]
    assert guard(monkeypatch) == result["compatibility_reason"]


def test_deleted_layout_file_is_a_tracked_change(checkout, monkeypatch):
    checkout.layout.unlink()
    result = control.health()
    assert (
        result["compatible"] is False and "tracked local changes" in result["compatibility_reason"]
    )
    assert guard(monkeypatch) == result["compatibility_reason"]


def test_unsupported_revision_reports_revision_reason(checkout, monkeypatch):
    (checkout.path / "README.md").write_text("next\n")
    git(checkout.path, "commit", "-q", "-am", "next")
    result = control.health()
    assert result["compatible"] is False and result["revision"] != checkout.revision
    assert result["revision"] in result["compatibility_reason"]
    assert checkout.revision in result["compatibility_reason"]
    assert guard(monkeypatch) == result["compatibility_reason"]


def test_changes_outside_app_and_untracked_files_keep_startup_policy(checkout, monkeypatch):
    (checkout.path / "README.md").write_text("edited outside app\n")
    (checkout.path / WIDGETS / "untracked.py").write_text("x = 1\n")
    result = control.health()
    assert result["compatible"] is True and result["compatibility_reason"] is None
    assert guard(monkeypatch) is None


def test_unreadable_checkout_reports_reason(checkout, monkeypatch):
    shutil.rmtree(checkout.path / ".git")
    (checkout.path / ".git").mkdir()
    result = control.health()
    assert result["compatible"] is False and result["revision"] is None
    assert "Cannot read the Fusion revision" in result["compatibility_reason"]
    assert guard(monkeypatch) == result["compatibility_reason"]

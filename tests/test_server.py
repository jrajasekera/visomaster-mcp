"""Drive the real stdio server end to end, with an SSH shim that runs commands locally."""

import asyncio
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest
from conftest import write_config
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).parents[1]
# Ignore ssh options and the host; run the remote command (always the last argument) here.
SHIM = '#!/bin/sh\nfor command; do :; done\nexec /bin/sh -c "$command"\n'


@pytest.fixture
def server(tmp_path):
    shim = tmp_path / "ssh"
    shim.write_text(SHIM)
    shim.chmod(0o755)
    state = tmp_path / "state"
    job = state / "jobs/done"
    (job / "artifacts").mkdir(parents=True)
    (job / "state.json").write_text(json.dumps({"job_id": "done", "status": "completed"}))
    (job / "result.json").write_text("{}")
    (job / "artifacts/preview.jpg").write_bytes(b"\xff\xd8jpeg")
    (job / "artifacts/result.mp4").write_bytes(os.urandom(3 * 1024 * 1024))
    config = write_config(tmp_path / "remote-config.json", tmp_path, state_dir=str(state))
    # Only what the client needs, so failure output never shows unrelated environment values.
    environment = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG") if k in os.environ}
    environment.update(
        VISOMASTER_SSH=str(shim),
        VISOMASTER_HOST="gpu-host",
        VISOMASTER_REMOTE=str(ROOT),
        VISOMASTER_DOWNLOADS=str(tmp_path / "downloads"),
        VM_CONFIG=str(config),
    )
    return StdioServerParameters(
        command=sys.executable, args=["-m", "visomaster_mcp.server"], env=environment
    )


def call(params, *calls):
    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return [await session.call_tool(name, arguments) for name, arguments in calls]

    return asyncio.run(run())


def test_upload_round_trip_and_extension_rejection(server, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(os.urandom(100_000))
    (tmp_path / "notes.txt").write_text("text")
    uploaded, rejected = call(
        server,
        ("upload_media", {"local_path": str(media)}),
        ("upload_media", {"local_path": str(tmp_path / "notes.txt")}),
    )
    assert not uploaded.isError
    result = uploaded.structuredContent
    remote = Path(result["path"])
    assert remote.parent == tmp_path / "state/uploads" and remote.suffix == ".mp4"
    assert remote.read_bytes() == media.read_bytes() and result["bytes"] == 100_000
    assert remote.stem == hashlib.sha256(media.read_bytes()).hexdigest() and not result["reused"]
    assert rejected.isError and "supported media" in rejected.content[0].text


def test_reupload_reuses_existing_content(server, tmp_path):
    media = tmp_path / "face.png"
    media.write_bytes(os.urandom(50_000))
    copy = tmp_path / "copy.png"
    copy.write_bytes(media.read_bytes())
    first, second = call(
        server,
        ("upload_media", {"local_path": str(media)}),
        ("upload_media", {"local_path": str(copy)}),
    )
    assert not first.structuredContent["reused"] and second.structuredContent["reused"]
    assert first.structuredContent["path"] == second.structuredContent["path"]
    assert list((tmp_path / "state/uploads").iterdir()) == [Path(first.structuredContent["path"])]


def test_delete_and_prune_through_tools(server, tmp_path):
    old = time.time() - 40 * 86400
    stale = tmp_path / "state/uploads" / ("f" * 32 + ".mp4")
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"v" * 4096)
    os.utime(stale, (old, old))
    preview, kept, deleted, gone, invalid = call(
        server,
        ("prune_storage", {}),
        ("job_status", {"job_id": "done"}),
        ("delete_job", {"job_id": "done"}),
        ("delete_job", {"job_id": "done"}),
        ("prune_storage", {"older_than_days": -1}),
    )
    assert preview.structuredContent["dry_run"] and stale.exists()
    assert preview.structuredContent["total_bytes"] == 4096
    assert not kept.isError and deleted.structuredContent["deleted"]
    assert not (tmp_path / "state/jobs/done").exists()
    assert gone.isError and invalid.isError
    (pruned,) = call(server, ("prune_storage", {"dry_run": False}))
    assert pruned.structuredContent["uploads"][0]["path"] == str(stale) and not stale.exists()


def test_download_and_view_artifacts(server, tmp_path):
    downloaded, image, video, missing = call(
        server,
        ("download_artifact", {"job_id": "done"}),
        ("view_artifact", {"job_id": "done"}),
        ("view_artifact", {"job_id": "done", "name": "result.mp4"}),
        ("job_status", {"job_id": "absent"}),
    )
    assert not downloaded.isError
    local = Path(downloaded.structuredContent["path"])
    assert local == tmp_path / "downloads/done/result.mp4"
    assert local.read_bytes() == (tmp_path / "state/jobs/done/artifacts/result.mp4").read_bytes()
    assert list(local.parent.iterdir()) == [local]
    assert not image.isError and image.content[0].type == "image"
    assert image.content[0].mimeType == "image/jpeg"
    assert base64.b64decode(image.content[0].data) == b"\xff\xd8jpeg"
    assert video.isError and "PNG/JPEG" in video.content[0].text
    # Remote control.py failures surface as tool errors with the remote message.
    assert missing.isError and "No such file" in missing.content[0].text


def test_job_status_wait_and_log_opt_in(server, tmp_path):
    (tmp_path / "state/jobs/done/worker.log").write_text("worker output\n")
    started = time.monotonic()
    quick, logged, invalid = call(
        server,
        ("job_status", {"job_id": "done", "wait_seconds": 50}),
        ("job_status", {"job_id": "done", "include_log": True}),
        ("job_status", {"job_id": "done", "wait_seconds": 51}),
    )
    # A terminal job ends the wait immediately.
    assert time.monotonic() - started < 20
    assert not quick.isError and quick.structuredContent["status"] == "completed"
    assert "log_tail" not in quick.structuredContent
    assert logged.structuredContent["log_tail"] == "worker output\n"
    assert invalid.isError


def test_list_jobs_filters(server, tmp_path):
    (tmp_path / "state/jobs/done/request.json").write_text(
        json.dumps({"kind": "detect", "media_path": "/m/a.mp4", "frame": 0})
    )
    everything, detects, renders, invalid = call(
        server,
        ("list_jobs", {}),
        ("list_jobs", {"kind": "detect", "status": "completed", "limit": 1}),
        ("list_jobs", {"kind": "render"}),
        ("list_jobs", {"limit": 201}),
    )
    assert [job["job_id"] for job in everything.structuredContent["result"]] == ["done"]
    assert detects.structuredContent["result"][0]["media_path"] == "/m/a.mp4"
    assert renders.structuredContent["result"] == []
    assert invalid.isError


def test_normalize_media_submits_without_waiting_for_the_gpu(server, tmp_path):
    import fcntl

    # The configured runner (from write_config) stands in for the normalization worker.
    runner = tmp_path / "bin/runner"
    runner.write_text('#!/bin/sh\necho \'{"kind": "normalize"}\' > "$2/result.json"\n')
    runner.chmod(0o755)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    arguments = {"media_path": str(media), "fps": "30000/1001", "job_id": "norm"}
    with (tmp_path / "state/gpu-0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # A GPU job is running.
        submitted, repeated, waited, invalid = call(
            server,
            ("normalize_media", arguments),
            ("normalize_media", arguments),
            ("job_status", {"job_id": "norm", "wait_seconds": 10}),
            ("normalize_media", {"media_path": str(media), "fps": "0", "job_id": "bad"}),
        )
        status = waited.structuredContent
        if status["status"] != "completed":
            (again,) = call(server, ("job_status", {"job_id": "norm", "wait_seconds": 10}))
            status = again.structuredContent
    assert submitted.structuredContent["job_id"] == repeated.structuredContent["job_id"] == "norm"
    request = json.loads((tmp_path / "state/jobs/norm/request.json").read_text())
    assert request["kind"] == "normalize" and request["fps"] == "30000/1001"
    assert status["status"] == "completed" and status["result"] == {"kind": "normalize"}
    assert invalid.isError and "fps" in invalid.content[0].text

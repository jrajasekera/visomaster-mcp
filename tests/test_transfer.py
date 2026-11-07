"""Artifact transfers end to end: generation-bound streaming and confined local downloads.

The remote interpreter is a wrapper that runs control.py unchanged but can run a shell hook
once, right after the first bytes it writes to stdout (the artifact header) reach the client.
Hooks swap job directories at exactly that point, so each race is deterministic.
"""

import asyncio
import base64
import json
import os
import shlex
import sys
from pathlib import Path

import pytest
from conftest import ROOT, write_config
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from visomaster_mcp import server as client

# Ignore ssh options and the host; run the remote command (always the last argument) here.
SHIM = '#!/bin/sh\nfor command; do :; done\nexec /bin/sh -c "$command"\n'
HOOKED_PYTHON = """
import io, os, runpy, subprocess, sys

class Hooked(io.RawIOBase):
    written = 0

    def writable(self):
        return True

    def write(self, data):
        data = bytes(data)
        view = memoryview(data)
        while view:
            view = view[os.write(1, view):]
        self.written += len(data)
        if self.written > int(os.environ.get("HOOK_AFTER") or 0):
            fire()
        return len(data)

def fire():
    command = os.environ.get("HOOK_COMMAND")
    if not command:
        return
    try:  # Once per test, whichever remote process gets there first.
        os.close(os.open(os.environ["HOOK_MARKER"], os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return
    subprocess.run(["/bin/sh", "-c", command], check=True)

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(script))
sys.stdout = io.TextIOWrapper(io.BufferedWriter(Hooked()), encoding="utf-8")
try:
    runpy.run_path(script, run_name="__main__")
finally:
    sys.stdout.flush()
"""
ORIGINAL = os.urandom(3 * 1024 * 1024)
JPEG = b"\xff\xd8original-jpeg"


def make_job(directory, status="completed", **artifacts):
    (directory / "artifacts").mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"job_id": "done", "status": status}))
    (directory / "result.json").write_text("{}")
    for name, data in artifacts.items():
        (directory / "artifacts" / name.replace("_", ".")).write_bytes(data)
    return directory


class Site:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.state = tmp_path / "state"
        self.downloads = tmp_path / "downloads"
        self.marker = tmp_path / "hook-fired"
        shim = tmp_path / "ssh"
        shim.write_text(SHIM)
        python = tmp_path / "python"
        (tmp_path / "hooked.py").write_text(HOOKED_PYTHON)
        python.write_text(
            f'#!/bin/sh\nexec {shlex.join([sys.executable, str(tmp_path / "hooked.py")])} "$@"\n'
        )
        for program in (shim, python):
            program.chmod(0o755)
        make_job(self.state / "jobs/done", result_mp4=ORIGINAL, preview_jpg=JPEG)
        config = write_config(tmp_path / "remote-config.json", tmp_path, state_dir=str(self.state))
        # Only what the client needs, so failure output never shows unrelated environment values.
        self.environment = {
            k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG") if k in os.environ
        }
        self.environment.update(
            VISOMASTER_SSH=str(shim),
            VISOMASTER_HOST="gpu-host",
            VISOMASTER_REMOTE=str(ROOT),
            VISOMASTER_REMOTE_PYTHON=str(python),
            VISOMASTER_DOWNLOADS=str(self.downloads),
            VM_CONFIG=str(config),
            HOOK_MARKER=str(self.marker),
        )

    def call(self, *calls, together=False, cwd=None, **environment):
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "visomaster_mcp.server"],
            env={**self.environment, **environment},
            cwd=cwd,
        )

        async def run():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    requests = [session.call_tool(name, arguments) for name, arguments in calls]
                    if together:
                        return await asyncio.gather(*requests)
                    return [await request for request in requests]

        return asyncio.run(run())

    def leftovers(self):
        return sorted(p.name for p in self.downloads.rglob(".download-*"))


@pytest.fixture
def site(tmp_path):
    return Site(tmp_path)


def text(result):
    return result.content[0].text


def swap_job(site, replacement):
    """Hook: delete the job (rename to trash) and publish a different one under the same ID."""
    jobs = site.state / "jobs"
    return (
        f"mv {shlex.quote(str(jobs / 'done'))} {shlex.quote(str(site.state / 'trash-old'))} && "
        f"mv {shlex.quote(str(replacement))} {shlex.quote(str(jobs / 'done'))}"
    )


# vmcp-review-remediation-qpn.14


def test_download_keeps_validated_bytes_when_job_is_replaced(site):
    replacement = make_job(site.state / "replacement", result_mp4=os.urandom(len(ORIGINAL)))
    swapped = (replacement / "artifacts/result.mp4").read_bytes()
    (result,) = site.call(
        ("download_artifact", {"job_id": "done"}), HOOK_COMMAND=swap_job(site, replacement)
    )
    assert site.marker.exists()  # The swap happened between the header and the bytes.
    assert (site.state / "jobs/done/artifacts/result.mp4").read_bytes() == swapped
    assert not result.isError, text(result)
    local = Path(result.structuredContent["path"])
    assert local.read_bytes() == ORIGINAL and result.structuredContent["bytes"] == len(ORIGINAL)
    assert site.leftovers() == []


def test_inline_view_of_replaced_artifact_stays_bounded(site):
    oversized = b"\xff\xd8" + os.urandom(12_000_000)
    replacement = make_job(site.state / "replacement", preview_jpg=oversized)
    (viewed,) = site.call(
        ("view_artifact", {"job_id": "done"}), HOOK_COMMAND=swap_job(site, replacement)
    )
    assert site.marker.exists()
    assert not viewed.isError, text(viewed)
    assert viewed.content[0].mimeType == "image/jpeg"
    assert base64.b64decode(viewed.content[0].data) == JPEG
    # Without a race, the oversized replacement is refused before any byte is sent.
    (refused,) = site.call(("view_artifact", {"job_id": "done"}))
    assert refused.isError and "10000000-byte limit" in text(refused)


def test_png_view_and_normal_download_replace_existing_copy(site):
    png = b"\x89PNG\r\n\x1a\n" + os.urandom(1000)
    (site.state / "jobs/done/artifacts/result.png").write_bytes(png)
    existing = site.downloads / "done/result.mp4"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"older copy")
    viewed, downloaded = site.call(
        ("view_artifact", {"job_id": "done", "name": "result.png"}),
        ("download_artifact", {"job_id": "done"}),
    )
    assert viewed.content[0].mimeType == "image/png"
    assert base64.b64decode(viewed.content[0].data) == png
    assert downloaded.structuredContent == {"path": str(existing), "bytes": len(ORIGINAL)}
    assert existing.read_bytes() == ORIGINAL
    assert list(existing.parent.iterdir()) == [existing] and site.leftovers() == []


def test_large_download_streams_to_a_temporary_file(site):
    seen = site.tmp / "seen"
    # Runs once over 1.5 MB are written: wait until the partial file exists and holds bytes.
    command = (
        f"for i in $(seq 200); do for f in {shlex.quote(str(site.downloads))}/.download-*; do "
        f'if [ -s "$f" ]; then basename "$f" > {shlex.quote(str(seen))}; exit 0; fi; done; '
        "sleep 0.05; done; exit 1"
    )
    (result,) = site.call(
        ("download_artifact", {"job_id": "done"}), HOOK_COMMAND=command, HOOK_AFTER="1500000"
    )
    assert not result.isError, text(result)
    assert seen.read_text().startswith(".download-")
    assert Path(result.structuredContent["path"]).read_bytes() == ORIGINAL
    assert site.leftovers() == []


def test_interrupted_download_preserves_prior_copy(site):
    existing = site.downloads / "done/result.mp4"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"prior copy")
    # Kill the remote process after the header, mid-transfer.
    (result,) = site.call(("download_artifact", {"job_id": "done"}), HOOK_COMMAND="kill -9 $PPID")
    assert site.marker.exists()
    assert result.isError and "interrupted" in text(result)
    assert existing.read_bytes() == b"prior copy"
    assert list(existing.parent.iterdir()) == [existing] and site.leftovers() == []


def lying_remote(tmp_path, monkeypatch, size, sent):
    """An SSH program that announces `size` bytes, then sends `sent` zero bytes.

    It is one process, like ssh, so killing it closes the client's pipes.
    """
    fake = tmp_path / "liar"
    fake.write_text(
        "#!/bin/sh\n"
        f"""printf '{{"result": {{"path": "/x/preview.jpg", "size": {size}}}}}\\n'\n"""
        f"exec head -c {sent} /dev/zero\n"
    )
    fake.chmod(0o755)
    for key, value in {
        "VISOMASTER_SSH": str(fake),
        "VISOMASTER_HOST": "gpu-host",
        "VISOMASTER_REMOTE": "/srv/unused",
        "VISOMASTER_DOWNLOADS": str(tmp_path / "downloads"),
    }.items():
        monkeypatch.setenv(key, value)


def test_client_reads_at_most_one_byte_past_the_announced_size(tmp_path, monkeypatch):
    lying_remote(tmp_path, monkeypatch, 4, 20_000_000)
    written = []
    with pytest.raises(RuntimeError, match="exceeded"):
        asyncio.run(client.stream_artifact("done", "preview.jpg", lambda info: written.append))
    assert sum(map(len, written)) <= 4
    lying_remote(tmp_path, monkeypatch, 20_000_000, 20_000_000)
    with pytest.raises(ValueError, match="10000000-byte limit"):
        asyncio.run(client.view_artifact("done"))


def test_short_transfer_publishes_nothing(tmp_path, monkeypatch):
    lying_remote(tmp_path, monkeypatch, 100, 10)
    existing = tmp_path / "downloads/done/preview.jpg"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"prior copy")
    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(client.download_artifact("done", "preview.jpg"))
    assert existing.read_bytes() == b"prior copy"
    assert list((tmp_path / "downloads").rglob(".download-*")) == []


def test_unfinished_jobs_and_escapes_are_refused(site):
    for status in ("queued", "running", "failed", "cancelled"):
        make_job(site.state / "jobs" / status, status, preview_jpg=JPEG)
    outside = site.tmp / "outside"
    make_job(outside, preview_jpg=JPEG)
    (outside / "secret.jpg").write_bytes(b"\xff\xd8secret")
    artifacts = site.state / "jobs/done/artifacts"
    (artifacts / "link.jpg").symlink_to(outside / "secret.jpg")
    (site.state / "jobs/linked").symlink_to(outside)
    make_job(site.state / "jobs/linkdir")
    (site.state / "jobs/linkdir/artifacts").rmdir()
    (site.state / "jobs/linkdir/artifacts").symlink_to(outside / "artifacts")
    calls = []
    for tool in ("view_artifact", "download_artifact"):
        for status in ("queued", "running", "failed", "cancelled"):
            calls.append((tool, {"job_id": status, "name": "preview.jpg"}, "successful completion"))
        calls += [
            (tool, {"job_id": "done", "name": "../artifacts/preview.jpg"}, "escapes"),
            (tool, {"job_id": "done", "name": "link.jpg"}, "escapes"),
            (tool, {"job_id": "linked", "name": "preview.jpg"}, "escapes"),
            (tool, {"job_id": "linkdir", "name": "preview.jpg"}, "escapes"),
            (tool, {"job_id": "../outside", "name": "preview.jpg"}, "job_id"),
        ]
    results = site.call(*[(tool, arguments) for tool, arguments, _ in calls])
    for (tool, arguments, message), result in zip(calls, results, strict=True):
        assert result.isError and message in text(result), (tool, arguments, text(result))
    assert not site.downloads.exists() or list(site.downloads.iterdir()) == []


def test_concurrent_downloads_of_distinct_jobs(site):
    other = os.urandom(2 * 1024 * 1024)
    make_job(site.state / "jobs/other", result_mp4=other)
    first, second = site.call(
        ("download_artifact", {"job_id": "done"}),
        ("download_artifact", {"job_id": "other"}),
        together=True,
    )
    assert Path(first.structuredContent["path"]).read_bytes() == ORIGINAL
    assert Path(second.structuredContent["path"]).read_bytes() == other


# vmcp-review-remediation-qpn.26


def external_directory(site):
    outside = site.tmp / "elsewhere"
    outside.mkdir()
    (outside / "result.mp4").write_bytes(b"sentinel")
    return outside


def test_symlinked_job_directory_is_refused(site):
    outside = external_directory(site)
    site.downloads.mkdir()
    (site.downloads / "done").symlink_to(outside)
    (result,) = site.call(("download_artifact", {"job_id": "done"}))
    assert result.isError and "real directory" in text(result)
    assert list(outside.iterdir()) == [outside / "result.mp4"]
    assert (outside / "result.mp4").read_bytes() == b"sentinel" and site.leftovers() == []


def test_job_directory_swapped_for_symlink_during_transfer(site):
    outside = external_directory(site)
    (site.downloads / "done").mkdir(parents=True)
    job = shlex.quote(str(site.downloads / "done"))
    command = f"mv {job} {shlex.quote(str(site.downloads / 'moved'))} && ln -s {outside} {job}"
    (result,) = site.call(("download_artifact", {"job_id": "done"}), HOOK_COMMAND=command)
    assert site.marker.exists()
    assert result.isError and "real directory" in text(result)
    assert list(outside.iterdir()) == [outside / "result.mp4"]
    assert (outside / "result.mp4").read_bytes() == b"sentinel"
    assert list((site.downloads / "moved").iterdir()) == [] and site.leftovers() == []


def test_job_directory_renamed_out_of_root_during_transfer(site):
    outside = external_directory(site)
    (site.downloads / "done").mkdir(parents=True)
    command = f"mv {shlex.quote(str(site.downloads / 'done'))} {shlex.quote(str(outside))}/done"
    (result,) = site.call(("download_artifact", {"job_id": "done"}), HOOK_COMMAND=command)
    assert site.marker.exists()
    assert not result.isError, text(result)
    local = Path(result.structuredContent["path"])
    assert local == site.downloads / "done/result.mp4" and local.read_bytes() == ORIGINAL
    assert list((outside / "done").iterdir()) == [] and site.leftovers() == []


def test_downloads_root_expands_tilde_and_rejects_relative_paths(site):
    home = site.tmp / "home"
    home.mkdir()
    (expanded,) = site.call(
        ("download_artifact", {"job_id": "done"}), HOME=str(home), VISOMASTER_DOWNLOADS="~/dl"
    )
    assert not expanded.isError, text(expanded)
    path = expanded.structuredContent["path"]
    assert os.path.isabs(path) and Path(path) == home / "dl/done/result.mp4"
    assert Path(path).read_bytes() == ORIGINAL
    workdir = site.tmp / "cwd"
    workdir.mkdir()
    (relative,) = site.call(
        ("download_artifact", {"job_id": "done"}), cwd=workdir, VISOMASTER_DOWNLOADS="dl"
    )
    assert relative.isError and "VISOMASTER_DOWNLOADS must be an absolute path" in text(relative)
    assert list(workdir.iterdir()) == []

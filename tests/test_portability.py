"""Non-default configuration end to end through an SSH shim that scrubs the environment.

Like real SSH, the shim passes no VISOMASTER_* or VM_* variables to the remote command, so the
remote side must find everything through its deployment's remote-config.json.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time

import pytest
from conftest import ROOT, write_config
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ALIAS = "vm-portability-test"
SHIM = """#!/bin/sh
printf '%s\\036' "$*" >> "$SHIM_LOG"
for command; do :; done
exec env -i PATH=/usr/bin:/bin HOME="$HOME" /bin/sh -c "$command"
"""
RECORDING_RUNNER = """#!{python}
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import common
job = Path(sys.argv[2])
keys = ("CUDA_VISIBLE_DEVICES", "VM_CONFIG", "VM_CONFIG_SHA256", "VM_SUPERVISOR_PID")
(job / "result.json").write_text(json.dumps({{
    "argv": sys.argv,
    "env": {{k: os.environ.get(k) for k in keys}},
    "leaked": sorted(k for k in os.environ if k.startswith("VISOMASTER_")),
    "parent_is_supervisor": os.getppid() == int(os.environ["VM_SUPERVISOR_PID"]),
    "config": common.load_config().effective(),
    "artifacts": [],
}}))
"""


@pytest.fixture
def site(tmp_path):
    """A non-default media root, deployment root, state directory, runner and interpreter."""
    media = tmp_path / "media root"
    media.mkdir()
    deployment = tmp_path / "deploy-root"
    deployment.mkdir()
    runner = media / "launchers/recording-runner"
    runner.parent.mkdir()
    runner.write_text(RECORDING_RUNNER.format(python=sys.executable))
    runner.chmod(0o755)
    config = write_config(
        deployment / "remote-config.json",
        media,
        runner=runner,
        state_dir=str(media / "jobs-state"),
        gpu="2",
    )
    shim = tmp_path / "custom-ssh"
    shim.write_text(SHIM)
    shim.chmod(0o755)
    # Only what the client needs, so failure output never shows unrelated environment values.
    environment = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG") if k in os.environ}
    environment.update(
        VISOMASTER_HOST=ALIAS,
        VISOMASTER_REMOTE=str(deployment),
        VISOMASTER_SSH=str(shim),
        VISOMASTER_REMOTE_PYTHON=sys.executable,
        VISOMASTER_DOWNLOADS=str(tmp_path / "my downloads"),
        SHIM_LOG=str(tmp_path / "ssh.log"),
    )
    return {
        "media": media,
        "deployment": deployment,
        "config": config,
        "env": environment,
        "log": tmp_path / "ssh.log",
        "tmp": tmp_path,
    }


def install_release(deployment, name="r1"):
    release = deployment / "releases" / name
    release.mkdir(parents=True)
    for file in ("common.py", "control.py", "engine.py"):
        shutil.copy(ROOT / "remote" / file, release)
    link = deployment / "remote"
    link.unlink(missing_ok=True)
    link.symlink_to(f"releases/{name}")
    return release


def deploy(site, *arguments, **overrides):
    environment = {**site["env"], **overrides}
    return subprocess.run(
        ["bash", str(ROOT / "scripts/deploy.sh"), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )


def ssh_log(site):
    """One entry per SSH invocation (records are separated by ASCII RS)."""
    return site["log"].read_text().split("\036")[:-1] if site["log"].exists() else []


@pytest.mark.parametrize(
    "variable, value",
    [
        ("VISOMASTER_HOST", "-oProxyCommand=x"),
        ("VISOMASTER_HOST", "two words"),
        ("VISOMASTER_REMOTE", "relative/path"),
        ("VISOMASTER_REMOTE", "/srv/../etc"),
        ("VISOMASTER_REMOTE_PYTHON", "python3;id"),
        ("VISOMASTER_HOST", ""),
    ],
)
def test_deploy_and_server_refuse_the_same_settings_before_ssh(site, variable, value):
    result = deploy(site, **{variable: value})
    assert result.returncode == 2 and variable in result.stderr
    assert ssh_log(site) == []  # No SSH call was made.
    from visomaster_mcp import transport

    with pytest.raises(transport.TransportError, match=variable):
        transport.load({**site["env"], variable: value})


def test_deploy_uploads_preflights_and_switches(site):
    result = deploy(site)
    assert result.returncode == 0, result.stderr
    releases = list((site["deployment"] / "releases").iterdir())
    assert len(releases) == 1
    for file in ("common.py", "control.py", "engine.py"):
        assert (releases[0] / file).read_bytes() == (ROOT / "remote" / file).read_bytes()
    log = ssh_log(site)
    assert log and all(f" -- {ALIAS} " in f" {line} " for line in log)
    assert any("control.py" in line and sys.executable in line for line in log)
    assert (site["deployment"] / "remote").resolve() == releases[0].resolve()


def test_deploy_preflight_refuses_invalid_config_without_switching(site):
    install_release(site["deployment"])
    site["config"].write_text('{"version": 1, "media_root": "relative"}')
    # The current release cannot list jobs with this file either, so a plain deploy refuses.
    refused = deploy(site)
    assert refused.returncode == 1 and "Cannot list active jobs" in refused.stderr
    result = deploy(site, "--force")
    assert result.returncode == 1
    assert "remote configuration rejected" in result.stderr and "unchanged" in result.stderr
    assert os.readlink(site["deployment"] / "remote") == "releases/r1"
    assert [p.name for p in (site["deployment"] / "releases").iterdir()] == ["r1"]


def test_deploy_refuses_while_jobs_are_active(site):
    install_release(site["deployment"])
    job = site["media"] / "jobs-state/jobs/busy"
    (job / "artifacts").mkdir(parents=True)
    (job / "request.json").write_text(json.dumps({"kind": "detect"}))
    (job / "state.json").write_text(
        json.dumps({"job_id": "busy", "status": "queued", "created_at": time.time()})
    )
    result = deploy(site)
    assert result.returncode == 1 and "jobs are active: busy" in result.stderr
    assert [p.name for p in (site["deployment"] / "releases").iterdir()] == ["r1"]


def session(site, *calls):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "visomaster_mcp.server"],
        env=site["env"],
        cwd=str(site["tmp"]),
    )

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                results = []
                for name, arguments in calls:
                    result = await client.call_tool(name, arguments)
                    assert not result.isError, result.content
                    results.append(result.structuredContent)
                return results

    return asyncio.run(run())


def test_custom_settings_reach_control_supervisor_and_worker_across_reconnects(site):
    install_release(site["deployment"])
    # Quotes, command substitution and spaces travel only inside JSON.
    media = site["media"] / "it's $(touch pwned) x.png"
    media.write_bytes(b"not decoded by the recording runner")
    health, submitted, waited = session(
        site,
        ("health_check", {}),
        ("detect_faces", {"media_path": str(media), "job_id": "portable"}),
        ("job_status", {"job_id": "portable", "wait_seconds": 20}),
    )
    assert health["config_error"] is None
    assert health["config_path"] == str(site["config"].resolve())
    assert health["input_root"] == str(site["media"].resolve()) and health["gpu"] == "2"
    assert health["release"] == "r1" and health["code_matches"] is True
    assert submitted["job_id"] == "portable"
    status = waited
    if status["status"] != "completed":
        (status,) = session(site, ("job_status", {"job_id": "portable", "wait_seconds": 20}))
    assert status["status"] == "completed", status
    seen = status["result"]
    release = (site["deployment"] / "releases/r1").resolve()
    assert seen["argv"][1:] == [
        str(release / "engine.py"),
        str(site["media"].resolve() / "jobs-state/jobs/portable"),
    ]
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert seen["env"]["VM_CONFIG"] == str(site["config"].resolve())
    assert seen["env"]["VM_CONFIG_SHA256"] == health["config_sha256"]
    assert seen["parent_is_supervisor"] and seen["leaked"] == []
    assert seen["config"]["state_dir"] == str(site["media"].resolve() / "jobs-state")
    assert not (site["tmp"] / "pwned").exists() and not (site["media"] / "pwned").exists()
    # A new session (reconnect) sees the same job and settings.
    listed, again, settings = session(
        site,
        ("list_jobs", {}),
        ("job_status", {"job_id": "portable"}),
        ("health_check", {}),
    )
    assert [job["job_id"] for job in listed["result"]] == ["portable"]
    assert listed["result"][0]["media_path"] == str(media)
    assert again["result"]["env"] == seen["env"]
    assert settings["config_sha256"] == health["config_sha256"]
    for line in ssh_log(site):
        assert f" -- {ALIAS} " in f" {line} "

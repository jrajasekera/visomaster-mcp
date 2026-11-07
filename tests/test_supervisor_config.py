"""Configuration identity across submit, supervisor and worker; GPU lock sharing; health."""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import common
import control
import pytest
from conftest import ROOT, apply_config, write_config


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    for variable in common.LEGACY:
        monkeypatch.delenv(variable, raising=False)
    config = write_config(tmp_path / "remote-config.json", tmp_path, state_dir=str(tmp_path / "a"))
    settings = apply_config(monkeypatch, config)
    monkeypatch.setattr(control, "validate_request", lambda request: request)
    yield config, settings
    for state in tmp_path.glob("*/jobs/*/state.json"):
        (state.parent / "cancel").touch()
        (state.parent / "release").touch()
    time.sleep(0.5)


def state_of(directory):
    return json.loads((Path(directory) / "state.json").read_text())


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def edit(path, **changes):
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


class Recorder:
    """Stands in for Popen during submit, so the supervisor can be started later."""

    calls = []

    def __init__(self, arguments, **options):
        Recorder.calls.append(arguments)


def submit_deferred(monkeypatch, job_id):
    Recorder.calls = []
    with monkeypatch.context() as patch:
        patch.setattr(control.subprocess, "Popen", Recorder)
        control.submit({"job_id": job_id, "kind": "detect"})
    return Recorder.calls[0]


def test_submit_records_identity_and_supervisor_argv(deployment, monkeypatch):
    config, settings = deployment
    argv = submit_deferred(monkeypatch, "ident")
    directory = settings.state / "jobs/ident"
    assert argv[-3:] == ["supervise", str(directory), settings.sha256]
    assert state_of(directory)["config_sha256"] == settings.sha256


@pytest.mark.parametrize(
    "change, message",
    [
        ("moved", "state_dir differs"),
        ("malformed", "became invalid"),
        ("deleted", "not found"),
        ("edited", "changed after submission"),
    ],
)
def test_config_change_before_supervisor_start_fails_job(deployment, monkeypatch, change, message):
    config, settings = deployment
    argv = submit_deferred(monkeypatch, "changed")
    if change == "moved":
        edit(config, state_dir=str(settings.base / "b"))
    elif change == "malformed":
        config.write_text("{")
    elif change == "deleted":
        config.unlink()
    else:
        edit(config, gpu="1")
    subprocess.run(argv, check=True, timeout=30)
    state = state_of(settings.state / "jobs/changed")
    assert state["status"] == "failed" and message in state["error"]
    assert "finished_at" in state


def test_config_change_while_waiting_for_gpu_fails_job(deployment):
    config, settings = deployment
    with (settings.lock_dir / "gpu-0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # Another job holds the GPU.
        control.submit({"job_id": "waiting", "kind": "detect"})
        time.sleep(0.6)
        assert control.status("waiting")["status"] == "queued"
        edit(config, runner=[*settings.runner, "--changed"])
        state = wait_until(lambda: (s := control.status("waiting"))["status"] != "queued" and s)
    assert state["status"] == "failed" and "changed after submission" in state["error"]


def test_supervisor_rejects_non_job_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, control.__file__, "supervise", str(tmp_path), "0" * 64],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "not a job directory" in result.stderr


def test_worker_refuses_mismatched_identity(deployment, monkeypatch, tmp_path):
    import engine

    config, settings = deployment
    monkeypatch.setattr(engine, "SETTINGS", settings)
    monkeypatch.setattr(engine, "CONFIG_ERROR", None)
    monkeypatch.setenv("VM_SUPERVISOR_PID", str(os.getppid()))
    monkeypatch.setenv("VM_CONFIG_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="changed after submission"):
        engine.run(tmp_path)
    monkeypatch.setattr(engine, "CONFIG_ERROR", "broken config")
    with pytest.raises(RuntimeError, match="broken config"):
        engine.run(tmp_path)


def test_runner_without_exec_gets_actionable_error(deployment):
    config, settings = deployment
    # Runs Python as a child instead of exec-ing it, so the worker is not the supervisor's child.
    Path(settings.runner[0]).write_text(f'#!/bin/sh\n"{sys.executable}" "$@"\n')
    control.submit({"job_id": "noexec", "kind": "detect"})
    state = wait_until(lambda: (s := control.status("noexec"))["status"] == "failed" and s)
    assert "must exec the Fusion Python interpreter" in state["error"]


def test_supervisor_and_worker_see_configured_gpu_and_lock_dir(deployment, tmp_path, monkeypatch):
    config, settings = deployment
    locks = tmp_path / "locks"
    locks.mkdir()
    edit(config, gpu="3", lock_dir=str(locks))
    settings = apply_config(monkeypatch, config)
    Path(settings.runner[0]).write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys, time\nfrom pathlib import Path\n"
        "job = Path(sys.argv[2])\n"
        "keys = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'VM_CONFIG', 'VM_CONFIG_SHA256')\n"
        "(job / 'env.json').write_text(json.dumps({k: os.environ.get(k) for k in keys}))\n"
        "while not (job / 'release').exists(): time.sleep(.05)\n"
        "(job / 'result.json').write_text('{}')\n"
    )
    control.submit({"job_id": "gpu3", "kind": "detect"})
    job = settings.state / "jobs/gpu3"
    seen = json.loads(
        wait_until(lambda: (job / "env.json").exists() and job / "env.json").read_text()
    )
    assert seen == {
        "CUDA_VISIBLE_DEVICES": "3",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "VM_CONFIG": str(config),
        "VM_CONFIG_SHA256": settings.sha256,
    }
    with (locks / "gpu-3.lock").open("a") as lock:
        with pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # The worker holds it.
    assert not (settings.state / "gpu-3.lock").exists()
    (job / "release").touch()
    assert wait_until(lambda: control.status("gpu3")["status"] == "completed")


def second_deployment(tmp_path, name, lock_dir):
    config = write_config(
        tmp_path / f"{name}.json",
        tmp_path,
        runner=tmp_path / "bin/runner",
        state_dir=str(tmp_path / name),
        lock_dir=str(lock_dir),
    )
    return config


@pytest.mark.parametrize("shared", [True, False])
def test_two_deployments_serialize_only_on_a_shared_lock_dir(
    deployment, tmp_path, monkeypatch, shared
):
    config, settings = deployment
    edit(config, lock_dir=str(settings.state))
    apply_config(monkeypatch, config)
    control.submit({"job_id": "first", "kind": "detect"})
    first = settings.state / "jobs/first"
    wait_until(lambda: (first / "entered").exists())
    other = tmp_path / "other-locks"
    other.mkdir()
    second = apply_config(
        monkeypatch, second_deployment(tmp_path, "b", settings.state if shared else other)
    )
    control.submit({"job_id": "second", "kind": "detect"})
    job = second.state / "jobs/second"
    if shared:
        time.sleep(0.8)
        assert control.status("second")["status"] == "queued"
        (first / "release").touch()
        wait_until(lambda: (job / "entered").exists())
    else:
        wait_until(lambda: (job / "entered").exists())
        assert not (first / "result.json").exists()  # Both ran at once.
        (first / "release").touch()
    (job / "release").touch()
    assert wait_until(lambda: control.status("second")["status"] == "completed")


def test_health_reports_config_and_survives_missing_tools(deployment, tmp_path, monkeypatch):
    config, settings = deployment
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    result = control.dispatch("health", {})
    assert result["config_path"] and result["config_error"] is None
    assert result["config_sha256"] == settings.sha256
    assert result["checks"]["git"] is False and result["checks"]["nvidia-smi"] is False
    assert result["revision"] is None and result["compatible"] is False and result["gpus"] is None
    assert result["checks"]["state_writable"] and result["runner"]["executable"]
    assert result["state"] == str(settings.state) and result["lock_dir"] == str(settings.lock_dir)


def test_config_error_blocks_actions_but_not_health(deployment, monkeypatch):
    monkeypatch.setattr(control, "CONFIG_ERROR", "Remote configuration not found at /x")
    health = control.dispatch("health", {})
    assert health["config_error"].startswith("Remote configuration not found")
    assert health["compatible"] is False and "code" in health
    with pytest.raises(ValueError, match="not found"):
        control.dispatch("list", {})
    result = subprocess.run(
        [sys.executable, str(ROOT / "remote/control.py"), "upload", "a" * 64, ".png", "1"],
        env=dict(os.environ, VM_CONFIG="/nonexistent/remote-config.json"),
        capture_output=True,
        text=True,
        input="x",
    )
    assert result.returncode == 1 and "not found" in json.loads(result.stdout)["error"]

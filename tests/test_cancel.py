"""Cancellation of owned process groups, cancel/replacement races and stable finish times."""

import json
import os
import signal
import subprocess
import sys
import threading
import time

import common
import control
import pytest
from conftest import apply_config, write_config

# Stub worker. With "descendant" it forks a helper into its process group that appends heartbeats
# and either ignores SIGTERM ("ignore") or exits on it ("exit"). The helper stops after 30 s so a
# failing run cannot leave it behind.
RUNNER = (
    "#!/usr/bin/env python3\n"
    + """
import json, os, signal, sys, time
from pathlib import Path
job = Path(sys.argv[2])
request = json.loads((job / "request.json").read_text())
(job / "worker").write_text(str(os.getpid()))
(job / "supervisor").write_text(os.environ["VM_SUPERVISOR_PID"])
behavior = request.get("descendant")
if behavior:
    pid = os.fork()
    if pid == 0:
        if behavior == "ignore":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        else:
            signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
        end = time.monotonic() + 30
        with open(job / "heartbeat", "ab", buffering=0) as beat:
            while time.monotonic() < end:
                beat.write(b".")
                time.sleep(0.01)
        os._exit(0)
    (job / "descendant").write_text(str(pid))
(job / "entered").touch()
while not (job / "release").exists():
    time.sleep(0.05)
(job / "result.json").write_text('{"ok": true}')
"""
)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    state = tmp_path / "runtime"
    runner = tmp_path / "bin/runner"
    runner.parent.mkdir()
    runner.write_text(RUNNER)
    runner.chmod(0o755)
    config = write_config(
        tmp_path / "remote-config.json", tmp_path, runner=runner, state_dir=str(state)
    )
    apply_config(monkeypatch, config)
    monkeypatch.setattr(control, "validate_request", lambda request: request)
    yield state
    for job in (state / "jobs").glob("*"):
        (job / "cancel").touch()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        jobs = list((state / "jobs").glob("*/state.json"))
        if all(json.loads(p.read_text())["status"] in common.TERMINAL for p in jobs):
            break
        time.sleep(0.1)


def until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def wait_for(job_id, statuses):
    return until(lambda: (s := control.status(job_id))["status"] in statuses and s)


def started(job, heartbeat=True):
    until(lambda: (job / "entered").exists())
    if heartbeat:
        until(lambda: (job / "heartbeat").exists() and (job / "heartbeat").stat().st_size)


def beats(job):
    return (job / "heartbeat").stat().st_size


def assert_stopped(job):
    """The descendant writes a heartbeat every 10 ms while it can run."""
    size = beats(job)
    time.sleep(0.3)
    assert beats(job) == size


def seed(job_id, **request):
    job = common.job_path(job_id)
    (job / "artifacts").mkdir(parents=True)
    common.atomic_json(job / "request.json", request)
    # As submit stores it; an active record without it makes retention protect every input.
    common.atomic_json(job / "resolved.json", control.resolve_inputs(request))
    common.atomic_json(
        job / "state.json", {"job_id": job_id, "status": "queued", "created_at": time.time()}
    )
    return job


def run_and_release(job_id):
    control.submit({"job_id": job_id, "kind": "detect"})
    wait_for(job_id, {"running"})
    job = common.job_path(job_id)
    until(lambda: (job / "entered").exists())
    (job / "release").touch()
    assert wait_for(job_id, {"completed"})["result"] == {"ok": True}


@pytest.mark.parametrize("kind", ["detect", "normalize"])
def test_cancel_stops_term_ignoring_descendant_before_publishing(backend, kind):
    job = seed("stubborn", kind=kind, descendant="ignore")
    supervisor = threading.Thread(
        target=control.supervise, args=(str(job), control.SETTINGS.sha256)
    )
    supervisor.start()
    started(job)
    (job / "cancel").touch()
    # The first status that reports cancellation already finds the descendant stopped.
    wait_for("stubborn", {"cancelled"})
    assert_stopped(job)
    supervisor.join(10)
    assert not supervisor.is_alive()
    # The supervisor reaped its direct child.
    with pytest.raises(ChildProcessError):
        os.waitpid(int((job / "worker").read_text()), os.WNOHANG)
    # The GPU lock is free for the next GPU job.
    run_and_release("next")


def test_term_responsive_group_exits_in_grace_and_unrelated_group_survives(backend):
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        control.submit({"job_id": "polite", "kind": "detect", "descendant": "exit"})
        job = common.job_path("polite")
        started(job)
        began = time.monotonic()
        assert control.dispatch("cancel", {"job_id": "polite"})["cancel_requested"]
        wait_for("polite", {"cancelled"})
        assert time.monotonic() - began < control.GRACE
        assert_stopped(job)
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait()
    run_and_release("after")


def test_queued_cancel_starts_no_worker_and_repeats_safely(backend):
    control.submit({"job_id": "holder", "kind": "detect"})
    wait_for("holder", {"running"})
    control.submit({"job_id": "waiting", "kind": "detect"})
    assert control.dispatch("cancel", {"job_id": "waiting"})["cancel_requested"]
    assert wait_for("waiting", {"cancelled"})
    assert not (backend / "jobs/waiting/worker").exists()
    # Repeated cancellation of a running worker stays safe.
    until(lambda: (backend / "jobs/holder/entered").exists())
    for _ in range(3):
        control.dispatch("cancel", {"job_id": "holder"})
    wait_for("holder", {"cancelled"})
    again = control.dispatch("cancel", {"job_id": "holder"})
    assert again == {"job_id": "holder", "cancel_requested": False, "status": "cancelled"}
    run_and_release("third")


def test_cancel_of_terminal_job_adds_no_marker(backend):
    run_and_release("done")
    result = control.dispatch("cancel", {"job_id": "done"})
    assert result == {"job_id": "done", "cancel_requested": False, "status": "completed"}
    assert not (backend / "jobs/done/cancel").exists()
    assert control.status("done")["result"] == {"ok": True}


def test_supervisor_loss_stops_worker_tree_and_reconciles(backend, monkeypatch):
    control.submit({"job_id": "orphaned", "kind": "detect", "descendant": "ignore"})
    job = common.job_path("orphaned")
    started(job)
    supervisor = int((job / "supervisor").read_text())
    worker = int((job / "worker").read_text())
    os.kill(supervisor, signal.SIGKILL)

    def gone(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False

    until(lambda: gone(worker))
    assert_stopped(job)
    # Past the startup grace period, status reconciles the job to a stable terminal state.
    real = time.time
    monkeypatch.setattr(control.time, "time", lambda: real() + 60)
    first = control.status("orphaned")
    assert first["status"] == "failed"
    assert first["error"] == "Supervisor exited or host restarted"
    assert common.read_json(job / "state.json")["finished_at"] == first["finished_at"]
    monkeypatch.setattr(control.time, "time", lambda: real() + 120)
    assert control.status("orphaned")["finished_at"] == first["finished_at"]
    monkeypatch.setattr(control.time, "time", real)
    run_and_release("successor")


def test_cancel_holds_off_replacement_of_the_observed_job(backend, monkeypatch):
    control.submit({"job_id": "shot", "kind": "detect"})
    wait_for("shot", {"running"})
    job = common.job_path("shot")
    until(lambda: (job / "entered").exists())
    selected, resume = threading.Event(), threading.Event()
    real_status = control.status

    def paused(job_id, *args, **kwargs):
        state = real_status(job_id, *args, **kwargs)
        selected.set()
        assert resume.wait(10)
        return state

    monkeypatch.setattr(control, "status", paused)
    results = []
    canceller = threading.Thread(
        target=lambda: results.append(control.dispatch("cancel", {"job_id": "shot"}))
    )
    canceller.start()
    assert selected.wait(10)
    monkeypatch.setattr(control, "status", real_status)
    # Meanwhile the job completes. Delete and resubmission need the submit lock, which the paused
    # cancellation holds between selecting the job and publishing its marker.
    (job / "release").touch()
    wait_for("shot", {"completed"})
    with (backend / "submit.lock").open("a") as probe:
        with pytest.raises(BlockingIOError):
            control.fcntl.flock(probe, control.fcntl.LOCK_EX | control.fcntl.LOCK_NB)
    resume.set()
    canceller.join(10)
    assert results == [{"job_id": "shot", "cancel_requested": True, "status": "running"}]
    # The late marker landed in the completed generation, which keeps its result.
    assert control.status("shot")["result"] == {"ok": True}
    assert control.dispatch("delete", {"job_id": "shot"})["deleted"]
    control.submit({"job_id": "shot", "kind": "detect"})
    wait_for("shot", {"running"})
    assert not (job / "cancel").exists()
    until(lambda: (job / "entered").exists())
    (job / "release").touch()
    assert wait_for("shot", {"completed"})["result"] == {"ok": True}


def elapsed(job_id):
    return next(j for j in control.list_jobs() if j["job_id"] == job_id)["elapsed_seconds"]


def test_orphan_reconciliation_records_finish_time(backend, monkeypatch):
    job = seed("lost", kind="detect")
    state = common.read_json(job / "state.json")
    common.atomic_json(job / "state.json", {**state, "status": "running", "created_at": 1000})
    monkeypatch.setattr(control.time, "time", lambda: 5000.0)
    reconciled = control.status("lost")
    assert reconciled["status"] == "failed"
    assert common.read_json(job / "state.json")["finished_at"] == 5000.0
    assert elapsed("lost") == 4000.0
    monkeypatch.setattr(control.time, "time", lambda: 9000.0)
    assert elapsed("lost") == 4000.0
    assert control.status("lost")["finished_at"] == 5000.0


def test_launch_failure_records_finish_time_and_keeps_error(backend, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("cannot fork supervisor")

    monkeypatch.setattr(control.subprocess, "Popen", refuse)
    with pytest.raises(OSError, match="cannot fork supervisor"):
        control.submit({"job_id": "unlaunched", "kind": "detect"})
    state = control.status("unlaunched")
    assert state["status"] == "failed"
    assert state["error"] == "cannot fork supervisor"
    assert state["finished_at"] >= state["created_at"]
    first = elapsed("unlaunched")
    real = time.time
    monkeypatch.setattr(control.time, "time", lambda: real() + 3600)
    assert elapsed("unlaunched") == first


def test_legacy_terminal_record_has_stable_duration_and_ages(backend, monkeypatch):
    now = time.time()
    old = seed("legacy", kind="detect")
    common.atomic_json(
        old / "state.json", {"job_id": "legacy", "status": "failed", "created_at": now - 3 * 86400}
    )
    os.utime(old / "state.json", (now - 2 * 86400, now - 2 * 86400))
    recent = seed("recent", kind="detect")
    common.atomic_json(
        recent / "state.json", {"job_id": "recent", "status": "cancelled", "created_at": now - 60}
    )
    active = seed("active", kind="detect")
    common.atomic_json(
        active / "state.json", {"job_id": "active", "status": "running", "created_at": now}
    )
    with (active / "lease").open("a") as lease:
        # Holding the lease stands in for a live supervisor.
        control.fcntl.flock(lease, control.fcntl.LOCK_EX)
        assert control.status("legacy")["finished_at"] == pytest.approx(now - 2 * 86400)
        first = {job_id: elapsed(job_id) for job_id in ("legacy", "recent", "active")}
        assert first["legacy"] == pytest.approx(86400, abs=1)
        monkeypatch.setattr(control.time, "time", lambda: now + 600)
        later = {job_id: elapsed(job_id) for job_id in ("legacy", "recent", "active")}
        assert later["legacy"] == first["legacy"]
        assert later["recent"] == first["recent"]
        assert later["active"] > first["active"] + 500
        monkeypatch.setattr(control.time, "time", lambda: now)
        pruned = {entry["job_id"] for entry in control.prune(1)["jobs"]}
        common.atomic_json(active / "state.json", {"job_id": "active", "status": "cancelled"})
    assert pruned == {"legacy"}

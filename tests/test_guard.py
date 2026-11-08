"""Loss of the process group guard: the supervisor stops the job, and reconciliation reaps the
group's survivors if the supervisor dies too, without signalling a reused group or process ID."""

import contextlib
import os
import secrets
import signal
import subprocess
import sys

import common
import control
import pytest
import test_cancel
from test_cancel import assert_stopped, beats, run_and_release, seed, started, until, wait_for

backend = test_cancel.backend
LINUX = sys.platform.startswith("linux")
linux_only = pytest.mark.skipif(not LINUX, reason="survivor reconciliation needs /proc and pidfds")


def gone(pid):
    """Whether `pid` has exited (a zombie counts as exited)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    if LINUX:
        with open(f"/proc/{pid}/stat", "rb") as stat:
            return stat.read().rsplit(b")", 1)[1].split()[0] in (b"Z", b"X")
    return False


def pids(job):
    group = os.getpgid(int((job / "worker").read_text()))
    worker, descendant = (int((job / name).read_text()) for name in ("worker", "descendant"))
    assert os.getpgid(worker) == os.getpgid(descendant) == group
    return group, worker, descendant


def sleeper(**environment):
    """An unrelated process leading its own new process group."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        env={"PATH": os.environ["PATH"], **environment},
    )


def test_guard_loss_stops_worker_tree_and_fails_job(backend):
    control.submit({"job_id": "unguarded", "kind": "detect", "descendant": "ignore"})
    job = common.job_path("unguarded")
    started(job)
    guard, worker, descendant = pids(job)
    os.kill(guard, signal.SIGKILL)
    state = wait_for("unguarded", {"failed", "completed", "cancelled"})
    assert state["status"] == "failed"
    assert state["error"] == (
        "Process group guard exited unexpectedly (code -9); the job's processes were stopped"
    )
    # The first status that reports the failure already finds the TERM-ignoring descendant stopped.
    assert_stopped(job)
    until(lambda: gone(worker) and gone(descendant))
    # The terminal state is stable and the GPU lock is free for the next job.
    assert control.status("unguarded")["finished_at"] == state["finished_at"]
    run_and_release("after")


@linux_only
def test_guard_then_supervisor_loss_strands_no_descendants(backend):
    control.submit({"job_id": "stranded", "kind": "detect", "descendant": "ignore"})
    job = common.job_path("stranded")
    started(job)
    guard, worker, descendant = pids(job)
    supervisor = int((job / "supervisor").read_text())
    # Stopped first, the supervisor cannot react to the guard's death before it dies too.
    os.kill(supervisor, signal.SIGSTOP)
    try:
        os.kill(guard, signal.SIGKILL)
        os.kill(supervisor, signal.SIGKILL)
        until(lambda: gone(supervisor) and gone(guard))
        # Nothing is left to stop the group: the stub worker, unlike the engine, sets no parent
        # death signal, and the descendant keeps running.
        size = beats(job)
        until(lambda: beats(job) > size)
        state = control.status("stranded")
        assert state["status"] == "failed"
        assert state["error"] == "Supervisor exited or host restarted"
        assert_stopped(job)
        until(lambda: gone(worker) and gone(descendant))
        assert control.status("stranded")["finished_at"] == state["finished_at"]
    finally:
        (job / "release").touch()
        with contextlib.suppress(ProcessLookupError):
            os.kill(supervisor, signal.SIGCONT)
    # The worker tree held the GPU lock; it is free again.
    run_and_release("successor")


def test_reused_group_id_is_never_signalled(backend):
    token = secrets.token_hex(16)
    recycled = sleeper()  # Now leads the group ID the dead job recorded.
    carrier = sleeper(VM_JOB_TOKEN=token)  # Carries the token, but in another group.
    member = sleeper(VM_JOB_TOKEN=(other := secrets.token_hex(16)))  # A genuine survivor.
    try:
        job = seed("recycled", kind="detect")
        common.atomic_json(job / "group.json", {"pgid": recycled.pid, "token": token})
        survivor = seed("survivor", kind="detect")
        common.atomic_json(survivor / "group.json", {"pgid": member.pid, "token": other})
        # The record shows the supervisor had started, so reconciliation needs no startup grace.
        for job_id in ("recycled", "survivor"):
            state = control.status(job_id)
            assert state["status"] == "failed"
            assert state["error"] == "Supervisor exited or host restarted"
        assert recycled.poll() is None
        assert carrier.poll() is None
        # Elsewhere reconciliation is a no-op; on Linux it waits for the member to exit.
        assert member.poll() == (-signal.SIGKILL if LINUX else None)
    finally:
        for process in (recycled, carrier, member):
            process.kill()
            process.wait()


@linux_only
def test_recycled_process_id_is_rechecked_before_signalling(backend, monkeypatch):
    token = secrets.token_hex(16)
    process = sleeper(VM_JOB_TOKEN=token)
    checks = []
    real = control.owned

    def recycled_after_first_check(pid, group, token):
        # The first check passes; by the pidfd recheck the ID names another process.
        checks.append(pid)
        return real(pid, group, token) and checks.count(pid) == 1

    monkeypatch.setattr(control, "owned", recycled_after_first_check)
    try:
        control.stop_survivors({"pgid": process.pid, "token": token}, timeout=1)
        assert checks.count(process.pid) == 2
        assert process.poll() is None
    finally:
        process.kill()
        process.wait()

"""GPU scheduling: durable sequences, interactive priority, the fairness bound, queue standing,
long polls, cancellation and crash recovery, across supervisors and shared lock directories."""

import contextlib
import fcntl
import os
import signal
import subprocess
import threading
import time

import common
import control
import pytest
import test_cancel
from conftest import apply_config, write_config
from test_cancel import until, wait_for

backend = test_cancel.backend


@contextlib.contextmanager
def gpu_held(lock_dir):
    """Hold the GPU lock as a running job would, so jobs queue in a known state."""
    with (lock_dir / "gpu-0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def submit(job_id, kind, **extra):
    return control.submit({"job_id": job_id, "kind": kind, **extra})


def deferred(monkeypatch, job_id, kind):
    """Submit without starting the supervisor: a job whose supervisor has not started yet."""
    with monkeypatch.context() as patch:
        patch.setattr(control.subprocess, "Popen", lambda *a, **kw: None)
        return submit(job_id, kind)


def abandon(job_id):
    """Move a job without a live supervisor past the startup grace period, so it reconciles."""
    job = common.job_path(job_id)
    state = common.read_json(job / "state.json")
    common.atomic_json(job / "state.json", state | {"created_at": state["created_at"] - 60})
    result = control.status(job_id)
    assert result["status"] == "failed"
    return result


def run_next(pending):
    """Wait for the one running job among pending, let it finish and return it."""
    running = until(lambda: [j for j in pending if control.status(j)["status"] == "running"])
    assert len(running) == 1
    job = common.job_path(running[0])
    until(lambda: (job / "entered").exists())
    (job / "release").touch()
    wait_for(running[0], {"completed"})
    pending.remove(running[0])
    return running[0]


def admission_order(job_ids):
    pending = list(job_ids)
    return [run_next(pending) for _ in job_ids]


def standing(job_id):
    queue = control.status(job_id)["queue"]
    return queue["position"], queue["reason"]


# The policy itself, on synthetic entries.


def entry(name, priority, sequence):
    return {
        "job": name,
        "job_id": name,
        "priority": priority,
        "sequence": sequence,
        "enqueued_at": 0,
    }


def plan(entries, now=1000.0, **fairness):
    found = {name: [] for name in common.CLASSES}
    for item in sorted(entries, key=control.fifo):
        found[item["priority"]].append(item)
    queue = {"next_sequence": 99, "streak": 0, "passed_since": None, "waiting": entries}
    return [item["job"] for item in control.predicted(found, queue | fairness, now)]


def test_policy_orders_fifo_per_class_and_bounds_interactive_streaks():
    previews = [entry(f"p{n}", "interactive", n) for n in range(1, 6)]
    renders = [entry("r1", "normal", 0), entry("r2", "normal", 6)]
    assert control.MAX_STREAK == 3
    assert plan(previews + renders) == ["p1", "p2", "p3", "r1", "p4", "p5", "r2"]
    # Earlier interactive admissions ahead of waiting normal work count toward the bound.
    assert plan(previews + renders, streak=2) == ["p1", "r1", "p2", "p3", "p4", "r2", "p5"]
    assert plan(previews) == ["p1", "p2", "p3", "p4", "p5"]
    assert plan(renders) == ["r1", "r2"]
    # Equal sequences (a rebuilt queue) still order deterministically.
    twins = [entry("b", "normal", 4), entry("a", "normal", 4)]
    assert plan(twins) == ["a", "b"]


def test_policy_ages_waiting_normal_work():
    work = [entry("p1", "interactive", 2), entry("r1", "normal", 1)]
    now = 5000.0
    assert plan(work, now, streak=1, passed_since=now - control.AGE_LIMIT + 1) == ["p1", "r1"]
    assert plan(work, now, streak=1, passed_since=now - control.AGE_LIMIT) == ["r1", "p1"]


def test_sustained_interactive_stream_cannot_starve_normal_work():
    queue = {"next_sequence": 1, "streak": 0, "passed_since": None, "waiting": []}
    renders = [entry(f"r{n}", "normal", n) for n in range(3)]
    previews, started, now = [], [], 0.0
    for step in range(40):
        previews.append(entry(f"p{step}", "interactive", 100 + step))  # One more every turn.
        found = {"interactive": list(previews), "normal": list(renders)}
        chosen = control.pick(found, queue, now)
        control.record(queue, chosen, found, now)
        (previews if chosen["priority"] == "interactive" else renders).remove(chosen)
        started.append(chosen["job"])
        now += 1
    positions = [started.index(name) for name in ("r0", "r1", "r2")]
    assert positions == [3, 7, 11]  # Every MAX_STREAK interactive admissions, one render.
    # Interactive admissions while no normal work waits do not count toward the bound.
    queue.update(streak=2, passed_since=now)
    found = {"interactive": [entry("p", "interactive", 1)], "normal": []}
    control.record(queue, found["interactive"][0], found, now)
    assert (queue["streak"], queue["passed_since"]) == (0, None)


def test_queue_file_damage_is_contained(tmp_path):
    path = tmp_path / "gpu-0.queue.json"
    for text in ("{", "[]", '{"waiting": 3}'):
        path.write_text(text)
        assert control.load_queue(path)["waiting"] == []
    good, bad = entry(str(tmp_path / "jobs/a"), "normal", 7), entry("x", "urgent", 9)
    common.atomic_json(path, {"next_sequence": 3, "streak": True, "waiting": [good, bad, 5]})
    queue = control.load_queue(path)
    assert queue == {"next_sequence": 8, "streak": 0, "passed_since": None, "waiting": [good]}


def test_priority_classes_are_validated(monkeypatch):
    assert common.job_priority({"kind": "detect"}) == "interactive"
    assert common.job_priority({"kind": "preview"}) == "interactive"
    assert common.job_priority({"kind": "analyze"}) == "interactive"
    assert common.job_priority({"kind": "render"}) == "normal"
    assert common.job_priority({"kind": "future_gpu_kind"}) == common.DEFAULT_PRIORITY
    assert common.job_priority({"kind": "normalize"}) is None
    assert common.job_priority({"kind": "render", "priority": "interactive"}) == "interactive"
    for value in ("urgent", None, 1):
        with pytest.raises(ValueError, match="priority must be"):
            common.job_priority({"kind": "preview", "priority": value})
    monkeypatch.setattr(common, "layouts", lambda: ({}, {}))
    monkeypatch.setattr(common, "input_path", lambda path: path)
    request = {"kind": "detect", "media_path": "/m/clip.mp4"}
    with pytest.raises(ValueError, match="priority must be"):
        common.validate_request({**request, "priority": "high"})
    common.validate_request({**request, "priority": "normal"})
    with pytest.raises(ValueError, match="only media_path and fps"):
        common.validate_request({**request, "kind": "normalize", "priority": "normal"})
    analysis = {"kind": "analyze", "references": ["/m/face.png"]}
    with pytest.raises(ValueError, match="priority must be"):
        common.validate_request({**analysis, "priority": "high"})


# Real supervisors and stub workers.


def test_interactive_jobs_start_first_and_fifo_holds_within_each_class(backend):
    with gpu_held(backend):
        for job_id, kind in [("r1", "render"), ("p1", "preview"), ("r2", "render")]:
            submit(job_id, kind)
        submit("d1", "detect")
        submit("p2", "preview")
        assert [control.status(j)["sequence"] for j in ("r1", "p1", "r2", "d1", "p2")] == [
            1,
            2,
            3,
            4,
            5,
        ]
        assert control.status("p1")["priority"] == "interactive"
        assert control.status("r1")["priority"] == "normal"
        assert standing("p1") == (1, "gpu_busy")
        assert standing("d1") == (2, "behind_earlier")
        assert standing("r1") == (4, "behind_interactive")
        assert standing("r2") == (5, "behind_interactive")
        time.sleep(0.5)  # Supervisors poll meanwhile; nothing passes the held GPU.
        assert all(control.status(j)["status"] == "queued" for j in ("r1", "p1", "d1"))
    assert admission_order(["r1", "p1", "r2", "d1", "p2"]) == ["p1", "d1", "p2", "r1", "r2"]
    # The sequence is part of the durable record after the job ran.
    assert control.status("r1")["sequence"] == 1 and "queue" not in control.status("r1")


def test_sustained_preview_stream_admits_the_waiting_render(backend):
    with gpu_held(backend):
        submit("render", "render")
        submit("p0", "preview")
    started, pending, count = [], ["render", "p0"], 0
    while "render" in pending:
        running = until(lambda: [j for j in pending if control.status(j)["status"] == "running"])
        # Another preview arrives while each job runs, so interactive work never runs out.
        count += 1
        submit(f"p{count}", "preview")
        pending.append(f"p{count}")
        started.append(run_next(pending))
        assert running == started[-1:]
    assert started == ["p0", "p1", "p2", "render"]
    assert run_next(pending) == "p3"


def test_simultaneous_submissions_and_retries_keep_their_sequence(backend):
    names = [f"job{n}" for n in range(8)]
    barrier = threading.Barrier(len(names))
    errors = []

    def send(name):
        barrier.wait()
        try:
            submit(name, "preview")
        except Exception as error:  # Reported below rather than lost in the thread.
            errors.append(error)

    with gpu_held(backend):
        threads = [threading.Thread(target=send, args=(name,)) for name in names]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        sequences = {name: control.status(name)["sequence"] for name in names}
        assert sorted(sequences.values()) == list(range(1, 9))
        # Exact retries return the persisted job; a changed one is refused.
        for name in names:
            assert submit(name, "preview")["sequence"] == sequences[name]
        with pytest.raises(ValueError, match="different request"):
            submit(names[0], "preview", priority="normal")
        expected = sorted(names, key=sequences.get)
        assert [standing(name)[0] for name in expected] == list(range(1, 9))
    assert admission_order(names) == expected
    assert submit(expected[0], "preview")["sequence"] == sequences[expected[0]]


def test_cancelling_a_queued_job_removes_its_eligibility(backend, monkeypatch):
    with gpu_held(backend):
        deferred(monkeypatch, "parked", "preview")  # Head of the queue; its supervisor never runs.
        submit("render", "render")
        assert standing("parked") == (1, "gpu_busy")
        assert standing("render") == (2, "behind_interactive")
        assert control.cancel("parked")["cancel_requested"]
        assert standing("parked") == (None, "cancel_requested")
        assert standing("render") == (1, "gpu_busy")
    assert admission_order(["render"]) == ["render"]
    assert not (backend / "jobs/parked/entered").exists()
    # A queued job with a live supervisor ends cancelled without ever running.
    with gpu_held(backend):
        submit("waiting", "preview")
        control.cancel("waiting")
        assert wait_for("waiting", {"cancelled"})["status"] == "cancelled"
    assert not (backend / "jobs/waiting/entered").exists()
    waiting = control.load_queue(backend / "gpu-0.queue.json")["waiting"]
    assert [item["job_id"] for item in waiting] == ["parked"]  # The other left on exit.
    abandon("parked")
    assert control.queue_view() == {}  # Ended entries are dropped.


def test_next_reason_when_the_gpu_is_free(backend, monkeypatch):
    deferred(monkeypatch, "soon", "detect")
    assert standing("soon") == (1, "next")
    listed = control.list_jobs()[0]
    assert listed["priority"] == "interactive" and listed["sequence"] == 1
    assert listed["queue"]["position"] == 1 and listed["queue"]["reason"] == "next"
    assert listed["queue"]["waiting_seconds"] >= 0 and "interactive_streak" in listed["queue"]
    abandon("soon")


def test_running_job_is_not_preempted_and_list_reports_standing(backend):
    submit("render", "render")
    wait_for("render", {"running"})
    until(lambda: (backend / "jobs/render/entered").exists())
    submit("preview", "preview")
    submit("normalize", "normalize")
    time.sleep(0.6)
    assert control.status("render")["status"] == "running"
    assert standing("preview") == (1, "gpu_busy")
    jobs = {job["job_id"]: job for job in control.list_jobs()}
    assert jobs["preview"]["queue"]["reason"] == "gpu_busy"
    assert "queue" not in jobs["render"] and jobs["render"]["priority"] == "normal"
    assert "priority" not in jobs["normalize"] and "sequence" not in jobs["normalize"]
    assert set(jobs["preview"]["queue"]) == {
        "position",
        "reason",
        "interactive_streak",
        "waiting_seconds",
    }
    (backend / "jobs/normalize/release").touch()
    assert admission_order(["render", "preview"]) == ["render", "preview"]


def test_long_poll_returns_on_queue_changes_and_stays_bounded(backend):
    with gpu_held(backend):
        submit("render", "render")
        started = time.monotonic()
        quiet = control.status("render", wait=0.6)
        assert 0.5 < time.monotonic() - started < 2
        assert quiet["queue"]["position"] == 1 and "log_tail" not in quiet
        results = []
        poller = threading.Thread(target=lambda: results.append(control.status("render", wait=10)))
        poller.start()
        time.sleep(0.4)
        started = time.monotonic()
        submit("preview", "preview")  # Passes the render: its position changes.
        poller.join(5)
        assert not poller.is_alive() and time.monotonic() - started < 2
        assert (results[0]["queue"]["position"], results[0]["queue"]["reason"]) == (
            2,
            "behind_interactive",
        )
    assert admission_order(["render", "preview"]) == ["preview", "render"]


def supervisor_of(job):
    found = subprocess.run(
        ["pgrep", "-f", f"control.py supervise {job} "], capture_output=True, text=True
    )
    pids = [int(pid) for pid in found.stdout.split()]
    assert len(pids) == 1, found
    return pids[0]


def test_dead_waiting_supervisor_is_reconciled_and_never_runs(backend, monkeypatch):
    with gpu_held(backend):
        submit("lost", "preview")
        submit("render", "render")
        job = common.job_path("lost")
        os.kill(supervisor_of(job), signal.SIGKILL)
        time.sleep(0.5)
        # Within the startup grace period the job still holds its place.
        assert standing("lost") == (1, "gpu_busy")
        assert abandon("lost")["error"] == "Supervisor exited or host restarted"
        assert standing("render") == (1, "gpu_busy")
    assert admission_order(["render"]) == ["render"]
    assert not (job / "entered").exists()


def test_lost_queue_file_is_rebuilt_at_recorded_sequences(backend):
    queue_file = backend / "gpu-0.queue.json"
    with gpu_held(backend):
        for job_id, kind in [("r1", "render"), ("r2", "render"), ("p1", "preview")]:
            submit(job_id, kind)
        queue_file.write_text("not json")
        # Each waiting supervisor rejoins at the sequence its job recorded at submission.
        until(lambda: len(control.load_queue(queue_file)["waiting"]) == 3)
        assert {e["job_id"]: e["sequence"] for e in control.load_queue(queue_file)["waiting"]} == {
            "r1": 1,
            "r2": 2,
            "p1": 3,
        }
        submit("p2", "preview")
        assert control.status("p2")["sequence"] == 4
    assert admission_order(["r1", "r2", "p1", "p2"]) == ["p1", "p2", "r1", "r2"]
    assert all(control.status(j)["status"] == "completed" for j in ("r1", "r2", "p1", "p2"))


def test_supervisor_loss_after_admission_keeps_the_gpu_until_the_worker_exits(backend):
    submit("render", "render", descendant="ignore")
    job = common.job_path("render")
    test_cancel.started(job)
    submit("preview", "preview")
    time.sleep(0.5)
    assert standing("preview") == (1, "gpu_busy")  # The worker and its helper hold the lock.
    os.kill(int((job / "supervisor").read_text()), signal.SIGKILL)
    entered = backend / "jobs/preview/entered"
    until(lambda: entered.exists())
    # The preview started only after the render's lock-holding helper stopped beating.
    test_cancel.assert_stopped(job)
    assert (job / "heartbeat").stat().st_mtime_ns <= entered.stat().st_mtime_ns
    assert admission_order(["preview"]) == ["preview"]
    # Reconciliation fails the render once; it is never admitted again.
    assert control.status("render")["status"] == "failed"
    assert control.queue_view() == {}


def test_deployments_sharing_a_lock_directory_share_one_queue(backend, tmp_path, monkeypatch):
    first = control.SETTINGS
    runner = tmp_path / "bin/runner"
    other = write_config(
        tmp_path / "other.json",
        tmp_path,
        runner=runner,
        state_dir=str(tmp_path / "other"),
        lock_dir=str(first.lock_dir),
    )
    with gpu_held(first.lock_dir):
        submit("render", "render")
        apply_config(monkeypatch, other)
        submit("preview", "preview")
        assert control.status("preview")["sequence"] == 2
        assert standing("preview") == (1, "gpu_busy")
        apply_config(monkeypatch, first.path)
        assert standing("render") == (2, "behind_interactive")
    apply_config(monkeypatch, other)
    until(lambda: control.status("preview")["status"] == "running")
    apply_config(monkeypatch, first.path)
    assert control.status("render")["status"] == "queued"
    apply_config(monkeypatch, other)
    assert admission_order(["preview"]) == ["preview"]
    apply_config(monkeypatch, first.path)
    assert admission_order(["render"]) == ["render"]


def test_client_sends_priority_only_when_chosen(monkeypatch):
    import asyncio

    from visomaster_mcp import server

    sent = []

    async def command(action, **values):
        sent.append(values)
        return {}

    monkeypatch.setattr(server, "command", command)
    mapping = server.FaceMapping(target_face="/m/t.png", source_faces=["/m/s.png"])

    async def calls():
        await server.detect_faces("/m/clip.mp4", job_id="d")
        await server.detect_faces("/m/clip.mp4", job_id="d", priority="normal")
        await server.submit_swap("/m/clip.mp4", [mapping], mode="render", job_id="r")
        await server.submit_swap("/m/clip.mp4", [mapping], job_id="p", priority="normal")

    asyncio.run(calls())
    assert ["priority" in values for values in sent] == [False, True, False, True]
    assert sent[1]["priority"] == sent[3]["priority"] == "normal"

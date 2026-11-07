"""The GPU smoke script's gates and evidence, driven by a fake MCP session (no GPU)."""

import argparse
import asyncio
import contextlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import smoke_gpu  # noqa: E402

GOOD = {
    "compatible": True,
    "code_matches": True,
    "release": "release-1",
    "revision": "abc",
    "supported_revision": "abc",
}
TERMINAL = {"completed", "failed", "cancelled"}


class Workstation:
    """Backend shared by every fake session; records submissions and cancellations.

    `disconnect` scripts the reconnect job: `initial` is what the reconnected session first
    sees, `at_cancel` the state when cancel_job arrives (default: unchanged) and `after` the
    state once a non-terminal job has been cancelled (default: cancelled). `fail_render` and
    `stuck` script the ordinary render; `lose_job` hides the job from a reconnected session.
    """

    def __init__(
        self, health=GOOD, disconnect=None, stuck=False, fail_render=False, lose_job=False
    ):
        self.health = health
        self.disconnect = {"initial": "running", **(disconnect or {})}
        self.stuck = stuck
        self.fail_render = fail_render
        self.lose_job = lose_job
        self.submitted = []
        self.cancelled = []
        self.sessions = 0
        self.jobs = {}

    @contextlib.asynccontextmanager
    async def connect(self):
        self.sessions += 1
        yield Session(self, reconnected=self.sessions > 1)


def reply(value):
    structured = value if isinstance(value, dict) else None
    return SimpleNamespace(isError=False, structuredContent=structured, content=value)


class Session:
    def __init__(self, workstation, reconnected):
        self.ws = workstation
        self.reconnected = reconnected

    async def call_tool(self, name, arguments):
        ws = self.ws
        job_id = arguments.get("job_id")
        if name == "health_check":
            return reply(ws.health)
        if name in ("detect_faces", "submit_swap"):
            ws.submitted.append((name, job_id))
            if job_id.endswith("-disconnect"):
                ws.jobs[job_id] = ws.disconnect["initial"]
            elif job_id.endswith("-render") and ws.fail_render:
                ws.jobs[job_id] = "failed"
            else:
                ws.jobs.setdefault(job_id, "completed")
            return reply({"job_id": job_id, "status": "queued"})
        if name == "job_status":
            if ws.lose_job and self.reconnected:
                return reply({"job_id": "other", "status": "running"})
            status = ws.jobs[job_id]
            if ws.stuck and job_id.endswith("-render"):
                status = "running"
            job = {"job_id": job_id, "status": status}
            if status == "completed" and job_id.endswith("-detect"):
                job["result"] = {"faces": [{"artifact": "crop.jpg", "reference_path": "/ref.jpg"}]}
            if status == "failed":
                job["log_tail"] = "CUDA out of memory"
            return reply(job)
        if name == "cancel_job":
            ws.cancelled.append(job_id)
            if job_id.endswith("-disconnect"):
                ws.jobs[job_id] = ws.disconnect.get("at_cancel", ws.jobs[job_id])
            at_cancel = ws.jobs[job_id]
            requested = at_cancel not in TERMINAL
            if requested:
                ws.jobs[job_id] = ws.disconnect.get("after", "cancelled")
            return reply({"job_id": job_id, "cancel_requested": requested, "status": at_cancel})
        if name == "view_artifact":
            return reply([SimpleNamespace(type="image")])
        return reply({})


def run(ws):
    arguments = argparse.Namespace(media="/m.mp4", source="/s.png", frames=4, parameters="{}")
    return asyncio.run(smoke_gpu.run(arguments, connect=ws.connect, poll=0, timeout=0.2))


@pytest.mark.parametrize(
    "health, text, hash_status",
    [
        ({**GOOD, "code_matches": False}, "differs", "mismatch"),
        ({**GOOD, "code_matches": None}, "unavailable", "unknown"),
        ({k: v for k, v in GOOD.items() if k != "code_matches"}, "unavailable", "unknown"),
        ({**GOOD, "compatible": False}, "incompatible", "match"),
    ],
)
def test_preflight_fails_before_any_submission(health, text, hash_status):
    ws = Workstation(health=health)
    with pytest.raises(smoke_gpu.SmokeFailure, match=text) as caught:
        run(ws)
    assert ws.submitted == [] and ws.cancelled == []
    assert caught.value.evidence["preflight"]["hash_status"] == hash_status


def test_hash_mismatch_message_names_release_and_recovery():
    ws = Workstation(health={**GOOD, "code_matches": False})
    with pytest.raises(smoke_gpu.SmokeFailure, match=r"release-1.*deploy\.sh.*No job was queued"):
        run(ws)


def test_hash_match_proceeds_and_is_recorded():
    ws = Workstation()
    evidence = run(ws)
    assert evidence["preflight"] == {
        "release": "release-1",
        "code_matches": True,
        "hash_status": "match",
        "revision": "abc",
        "supported_revision": "abc",
    }
    assert evidence["health"]["release"] == "release-1"
    assert ("detect_faces", evidence["prefix"] + "-detect") in ws.submitted
    assert [kind for kind, _ in ws.submitted].count("submit_swap") == 5  # 2 x 2 + disconnect


def test_running_then_cancelled_records_a_pass():
    evidence = run(Workstation())
    assert evidence["reconnect"] == {"found_job": True, "state": "running"}
    assert evidence["cancellation"] == {
        "state_before_cancel": "running",
        "state_at_cancel": "running",
        "cancel_requested": True,
        "final_state": "cancelled",
        "result": "passed",
        "path": "running",
    }


def test_queued_cancellation_is_recorded_separately():
    cancellation = run(Workstation(disconnect={"initial": "queued"}))["cancellation"]
    assert cancellation["path"] == "queued"
    assert cancellation["result"] == "passed_queued"
    assert cancellation["state_at_cancel"] == "queued"


def test_queued_job_that_starts_before_the_cancel_counts_as_running():
    cancellation = run(Workstation(disconnect={"initial": "queued", "at_cancel": "running"}))[
        "cancellation"
    ]
    assert cancellation["state_before_cancel"] == "queued"
    assert cancellation["result"] == "passed" and cancellation["path"] == "running"


def test_render_completing_first_is_unexercised_not_a_failure_or_pass():
    evidence = run(Workstation(disconnect={"initial": "completed"}))
    assert evidence["reconnect"] == {"found_job": True, "state": "completed"}
    cancellation = evidence["cancellation"]
    assert cancellation["result"] == "unexercised"
    assert cancellation["final_state"] == "completed"
    assert "already completed" in cancellation["reason"]
    assert "path" not in cancellation


def test_finish_between_status_and_cancel_is_unexercised():
    evidence = run(Workstation(disconnect={"initial": "running", "at_cancel": "completed"}))
    assert evidence["cancellation"]["result"] == "unexercised"
    assert evidence["cancellation"]["state_before_cancel"] == "running"


def test_worker_that_finishes_despite_cancel_is_unexercised():
    evidence = run(Workstation(disconnect={"after": "completed"}))
    assert evidence["cancellation"]["result"] == "unexercised"
    assert "finished first" in evidence["cancellation"]["reason"]


def test_reconnect_that_cannot_find_the_job_fails():
    with pytest.raises(smoke_gpu.SmokeFailure, match="did not find job"):
        run(Workstation(lose_job=True))


def test_failed_disconnect_job_is_an_actionable_failure():
    with pytest.raises(smoke_gpu.SmokeFailure, match=r"failed instead of being cancelled.*CUDA"):
        run(Workstation(disconnect={"initial": "failed"}))


def test_failed_render_is_an_actionable_failure_before_the_disconnect_job():
    ws = Workstation(fail_render=True)
    with pytest.raises(smoke_gpu.SmokeFailure, match=r"ended failed.*CUDA out of memory"):
        run(ws)
    assert not any(job.endswith("-disconnect") for _, job in ws.submitted)


def test_timeout_cancels_and_reports():
    ws = Workstation(stuck=True)
    with pytest.raises(smoke_gpu.SmokeFailure, match=r"Timed out.*-render.*running"):
        run(ws)
    assert any(job.endswith("-render") for job in ws.cancelled)


def test_gates_survive_optimized_python():
    code = (
        "import sys; sys.path.insert(0, 'scripts'); import smoke_gpu\n"
        "try:\n smoke_gpu.require(False, 'x')\nexcept smoke_gpu.SmokeFailure: print('gate held')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "gate held"
    assert "assert " not in (ROOT / "scripts/smoke_gpu.py").read_text()


def test_cli_exits_nonzero_with_evidence_on_preflight_failure(monkeypatch, capsys):
    ws = Workstation(health={**GOOD, "code_matches": False})
    monkeypatch.setattr(smoke_gpu, "open_session", ws.connect)
    status = smoke_gpu.main(["--media", "/m.mp4", "--source", "/s.png"])
    captured = capsys.readouterr()
    assert status == 1
    assert "SMOKE FAILED" in captured.err and "No job was queued" in captured.err
    assert json.loads(captured.out)["preflight"]["hash_status"] == "mismatch"
    assert ws.submitted == []


def test_cli_prints_evidence_and_exits_zero(monkeypatch, capsys):
    ws = Workstation()
    monkeypatch.setattr(smoke_gpu, "open_session", ws.connect)
    monkeypatch.setattr(smoke_gpu.asyncio, "sleep", lambda _: asyncio.sleep(0))
    assert smoke_gpu.main(["--media", "/m.mp4", "--source", "/s.png", "--frames", "4"]) == 0
    output = capsys.readouterr().out
    evidence = json.loads(output[output.rindex("\n{\n") :])  # job lines precede the evidence
    assert evidence["cancellation"]["result"] == "passed"

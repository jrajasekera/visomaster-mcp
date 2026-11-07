"""Opt-in integration test against the installed worker using supplied face-swap fixtures.

Usage: uv run python scripts/smoke_gpu.py --media REMOTE_CLIP --source REMOTE_FACE_IMAGE
Creates persistent test jobs and artifacts. Uses the first detected face as the test target.

Refuses to queue any job unless health_check confirms the deployed worker code matches this
checkout. The final JSON evidence records the release and hash status, the reconnect result and
the cancellation outcome: "running" (passed), "queued" (a different, weaker path) or
"unexercised" (the render finished before the cancel arrived, so cancellation proves nothing).
"""

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
import uuid

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

TERMINAL = {"completed", "cancelled", "failed"}


class SmokeFailure(Exception):
    """A qualification gate failed; the message says what to check."""

    evidence = None


def require(condition, message):
    # An explicit check, not assert: `python -O` must not strip qualification gates.
    if not condition:
        raise SmokeFailure(message)


@contextlib.asynccontextmanager
async def open_session():
    # Pass the full environment so VISOMASTER_* overrides reach the server; the SDK default drops them.
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "visomaster_mcp.server"], env=dict(os.environ)
    )
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        yield session


def check_deployed_code(health, evidence):
    """Record release and hash status, and refuse unless the deployed code is this checkout's."""
    matches = health.get("code_matches")
    status = "match" if matches is True else "mismatch" if matches is False else "unknown"
    record = {
        "release": health.get("release"),
        "code_matches": matches,
        "hash_status": status,
        "revision": health.get("revision"),
        "supported_revision": health.get("supported_revision"),
    }
    evidence["preflight"] = record
    require(
        health.get("compatible"),
        "health_check reports an incompatible or misconfigured installation "
        f"(revision {health.get('revision')!r}, supported {health.get('supported_revision')!r}, "
        f"config_error {health.get('config_error')!r}); no job was queued.",
    )
    reason = {
        "mismatch": "the deployed worker code differs from this checkout. Deploy with "
        "scripts/deploy.sh and confirm code_matches is true",
        "unknown": "the deployed worker code hashes are unavailable (no local remote/ sources or no "
        "'code' in the health response), so the release cannot be verified",
    }
    require(
        status == "match",
        f"Refusing to qualify release {record['release']!r}: {reason.get(status)}. "
        "No job was queued.",
    )


def classify_cancellation(before, response, final):
    """Describe which cancellation path a reconnect-then-cancel actually exercised.

    `before` is job_status from the reconnected session, `response` the cancel_job reply (its
    status is the job state at the moment of the request) and `final` the terminal state.
    Cancellation counts as passed only when the cancel hit a queued or running job and the job
    ended `cancelled`; a job that was already terminal, or finished despite the request, leaves
    cancellation unexercised rather than passed or failed.
    """
    at_cancel = response.get("status")
    outcome = {
        "state_before_cancel": before.get("status"),
        "state_at_cancel": at_cancel,
        "cancel_requested": response.get("cancel_requested"),
        "final_state": final.get("status"),
    }
    if final["status"] == "failed":
        raise SmokeFailure(
            f"Job {final['job_id']} failed instead of being cancelled; check its log_tail: "
            + json.dumps(final)[:1000]
        )
    if at_cancel in TERMINAL or not response.get("cancel_requested"):
        outcome.update(
            result="unexercised",
            reason=f"job was already {at_cancel} when cancellation was requested, so no "
            "running or queued job was cancelled; rerun with a longer clip or more frames",
        )
    elif final["status"] == "completed":
        outcome.update(
            result="unexercised",
            reason=f"job was {at_cancel} when cancellation was requested but the worker "
            "finished first and kept its result; rerun with a longer clip or more frames",
        )
    elif final["status"] != "cancelled":
        raise SmokeFailure(f"Unexpected final state after cancel: {json.dumps(final)[:1000]}")
    elif at_cancel == "running":
        outcome.update(result="passed", path="running")
    else:
        outcome.update(
            result="passed_queued",
            path="queued",
            reason="job had not started, so this does not show a running worker being stopped",
        )
    return outcome


async def run(args, connect=None, poll=2.0, timeout=600.0):
    connect = connect or open_session
    prefix = "verify-" + uuid.uuid4().hex[:12]
    evidence = {"prefix": prefix, "tools": [], "jobs": []}
    try:
        await qualify(args, connect, poll, timeout, prefix, evidence)
    except SmokeFailure as failure:
        failure.evidence = evidence
        raise
    return evidence


async def qualify(args, connect, poll, timeout, prefix, evidence):
    async def call(session, tool_name, **arguments):
        result = await session.call_tool(tool_name, arguments)
        require(not result.isError, f"{tool_name} returned an error: {result.content}")
        evidence["tools"].append(tool_name)
        return result.structuredContent or result.content

    async def wait(session, job_id):
        deadline = time.monotonic() + timeout
        job = None
        while time.monotonic() < deadline:
            job = await call(session, "job_status", job_id=job_id)
            if job["status"] in TERMINAL:
                print(json.dumps(job), flush=True)
                evidence["jobs"].append(job)
                return job
            await asyncio.sleep(poll)
        await call(session, "cancel_job", job_id=job_id)
        raise SmokeFailure(
            f"Timed out after {timeout:g}s waiting for job {job_id} (last status "
            f"{job and job['status']!r}); cancellation was requested. Check GPU use, the "
            "worker log and whether the job is stuck queued behind another GPU job."
        )

    async def completed(session, job_id):
        job = await wait(session, job_id)
        require(
            job["status"] == "completed",
            f"Job {job_id} ended {job['status']}, expected completed: " + json.dumps(job)[:1000],
        )
        return job

    async with connect() as session:
        health = await call(session, "health_check")
        evidence["health"] = health
        check_deployed_code(health, evidence)
        await call(session, "get_settings")
        await call(session, "inspect_media", media_path=args.media)
        detection = await call(
            session, "detect_faces", media_path=args.media, job_id=prefix + "-detect"
        )
        detection = await completed(session, detection["job_id"])
        require(detection["result"]["faces"], "Detection found no face in the supplied media.")
        face = detection["result"]["faces"][0]
        images = await call(
            session, "view_artifact", job_id=detection["job_id"], name=face["artifact"]
        )
        require(images[0].type == "image", "view_artifact did not return an inline image.")
        mapping = {
            "target_face": face["reference_path"],
            "source_faces": [args.source],
            "parameters": json.loads(args.parameters),
        }
        for mode in ("preview", "render"):
            submission = dict(
                media_path=args.media,
                mappings=[mapping],
                mode=mode,
                max_frames=args.frames,
                job_id=prefix + "-" + mode,
            )
            job = await call(session, "submit_swap", **submission)
            duplicate = await call(session, "submit_swap", **submission)
            require(
                duplicate["job_id"] == job["job_id"],
                f"Duplicate {mode} submission created a different job.",
            )
            job = await completed(session, job["job_id"])
            await call(session, "view_artifact", job_id=job["job_id"])
        # Leave a submitted job running when this MCP session exits.
        await call(
            session,
            "submit_swap",
            media_path=args.media,
            mappings=[mapping],
            mode="render",
            job_id=prefix + "-disconnect",
        )

    disconnect = prefix + "-disconnect"
    async with connect() as session:
        # Reconnection: the new session must find the job the old one left behind.
        before = await call(session, "job_status", job_id=disconnect)
        require(
            before.get("job_id") == disconnect,
            f"The reconnected session did not find job {disconnect}: {json.dumps(before)[:500]}",
        )
        evidence["reconnect"] = {"found_job": True, "state": before["status"]}
        # Cancellation is judged separately, from the state observed at the request.
        response = await call(session, "cancel_job", job_id=disconnect)
        final = await wait(session, disconnect)
        evidence["cancellation"] = classify_cancellation(before, response, final)
        await call(session, "list_jobs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--parameters", default="{}")
    try:
        evidence = asyncio.run(run(parser.parse_args(argv)))
    except SmokeFailure as failure:
        print(json.dumps(failure.evidence, indent=2))
        print(f"SMOKE FAILED: {failure}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

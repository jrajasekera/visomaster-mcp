"""Opt-in integration test against the installed worker using supplied face-swap fixtures.

Usage: uv run python scripts/smoke_gpu.py --media REMOTE_CLIP --source REMOTE_FACE_IMAGE
Creates persistent test jobs and artifacts. Uses the first detected face as the test target.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run(args):
    prefix = "verify-" + uuid.uuid4().hex[:12]
    # Pass the full environment so VISOMASTER_* overrides reach the server; the SDK default drops them.
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "visomaster_mcp.server"], env=dict(os.environ)
    )
    evidence = {"prefix": prefix, "tools": [], "jobs": []}

    async def call(session, tool_name, **arguments):
        result = await session.call_tool(tool_name, arguments)
        if result.isError:
            raise RuntimeError(result.content)
        evidence["tools"].append(tool_name)
        return result.structuredContent or result.content

    async def wait(session, job_id):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            job = await call(session, "job_status", job_id=job_id)
            if job["status"] in {"completed", "cancelled", "failed"}:
                print(json.dumps(job), flush=True)
                evidence["jobs"].append(job)
                return job
            await asyncio.sleep(2)
        await call(session, "cancel_job", job_id=job_id)
        raise TimeoutError(job_id)

    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        evidence["health"] = await call(session, "health_check")
        assert evidence["health"]["compatible"]
        await call(session, "get_settings")
        await call(session, "inspect_media", media_path=args.media)
        detection = await call(
            session, "detect_faces", media_path=args.media, job_id=prefix + "-detect"
        )
        detection = await wait(session, detection["job_id"])
        assert detection["status"] == "completed"
        face = detection["result"]["faces"][0]
        images = await call(
            session, "view_artifact", job_id=detection["job_id"], name=face["artifact"]
        )
        assert images[0].type == "image"
        mapping = {
            "target_face": face["reference_path"],
            "source_faces": [args.source],
            "parameters": json.loads(args.parameters),
        }
        for mode in ("preview", "render"):
            job = await call(
                session,
                "submit_swap",
                media_path=args.media,
                mappings=[mapping],
                mode=mode,
                max_frames=args.frames,
                job_id=prefix + "-" + mode,
            )
            duplicate = await call(
                session,
                "submit_swap",
                media_path=args.media,
                mappings=[mapping],
                mode=mode,
                max_frames=args.frames,
                job_id=prefix + "-" + mode,
            )
            assert duplicate["job_id"] == job["job_id"]
            job = await wait(session, job["job_id"])
            assert job["status"] == "completed"
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

    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        await call(session, "job_status", job_id=prefix + "-disconnect")
        await call(session, "cancel_job", job_id=prefix + "-disconnect")
        assert (await wait(session, prefix + "-disconnect"))["status"] == "cancelled"
        await call(session, "list_jobs")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--parameters", default="{}")
    asyncio.run(run(parser.parse_args()))

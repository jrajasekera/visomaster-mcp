"""MCP transport on the Mac; authenticated SSH controls the GPU workstation."""

import asyncio
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from visomaster_mcp.transport import load as transport

# The client runs from a source checkout (editable install), whose remote/ holds the code
# deploy.sh publishes; code_matches is null when these files are absent.
LOCAL_CODE = Path(__file__).resolve().parents[2] / "remote"
mcp = FastMCP(
    "VisoMaster Fusion",
    instructions=(
        "Run health_check and get_settings first. Paths are absolute workstation paths. "
        "Detect faces and inspect their crops before assigning identities; view_artifact "
        "contact.jpg shows every labelled crop of a detection job at once. Use explicit target/source "
        "references, preview before long renders, and follow jobs with job_status(wait_seconds=50), "
        "which returns early on any state change, instead of polling rapidly. "
        "Use a stable job_id to make retries idempotent. Jobs survive client disconnection."
    ),
)
READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
DESTROY = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)


class FaceMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_face: str = Field(
        description="Absolute workstation image path containing one target face"
    )
    source_faces: list[str] = Field(
        min_length=1, max_length=20, description="Workstation images of the replacement identity"
    )
    parameters: dict = Field(
        default_factory=dict, description="Validated Fusion keys from get_settings"
    )


async def ssh(arguments, data=None, timeout=45, stdin=None, stdout=asyncio.subprocess.PIPE):
    # Settings are validated per call, so discovery works before the client is configured.
    settings = transport()
    process = await asyncio.create_subprocess_exec(
        *settings.ssh_prefix(),
        settings.command(arguments),
        stdin=stdin
        or (asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL),
        stdout=stdout,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        output, errors = await asyncio.wait_for(process.communicate(data), timeout)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        message = ((output or b"") + errors).decode(errors="replace")[-6000:]
        raise RuntimeError(settings.hint(message) if process.returncode == 255 else message)
    return output


async def command(action, timeout=45, **values):
    payload = json.dumps({"action": action, **values}, allow_nan=False).encode()
    raw = await ssh(transport().control(), payload, timeout)
    response = json.loads(raw)
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["result"]


@mcp.tool(annotations=READ)
async def health_check() -> dict[str, Any]:
    """Check installed revision, deployed release, GPU, memory use and media/state paths.

    code_matches compares deployed worker hashes with this checkout, or is null if unknown.
    """
    result = await command("health")
    try:
        local = {
            name: hashlib.sha256((LOCAL_CODE / name).read_bytes()).hexdigest()
            for name in ("common.py", "control.py", "engine.py")
        }
    except OSError:
        local = None
    result["code_matches"] = (
        None if local is None or "code" not in result else result["code"] == local
    )
    return result


@mcp.tool(annotations=READ)
async def get_settings() -> dict[str, Any]:
    """Return supported Fusion parameter names, defaults, choices and numeric limits."""
    return await command("settings")


@mcp.tool(annotations=READ)
async def inspect_media(media_path: str) -> dict[str, Any]:
    """Read video/audio codec, frame rate, dimensions and duration from a workstation file."""
    return await command("inspect", media_path=media_path)


@mcp.tool(annotations=WRITE)
async def detect_faces(
    media_path: str,
    frame: int = 0,
    sample_frames: Annotated[int | None, Field(ge=1, le=64)] = None,
    frames: Annotated[list[int] | None, Field(min_length=1, max_length=64)] = None,
    controls: dict | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Queue face detection. Poll job_status, then inspect crops with view_artifact.

    Single-frame detection (`frame`, zero based) returns `faces` with crop reference paths
    usable as FaceMapping.target_face. To find who appears across a video, pass
    `sample_frames` (N evenly spaced frames) or explicit `frames` (at most 64) instead: faces
    are grouped into `identities` by recognition-embedding similarity at Fusion's default
    similarity threshold. Each identity lists its appearances, frames, sample boxes and a
    representative crop (`reference_path`, verified to re-detect as one face) usable as
    target_face. Both modes write contact.jpg, one grid of every crop labelled "face N" or
    "id N: <appearances>x" (* marks an unverified reference); view it first.
    """
    return await command(
        "submit",
        job_id=job_id or uuid.uuid4().hex,
        kind="detect",
        media_path=media_path,
        frame=frame,
        controls=controls or {},
        # Omitted when unset so single-frame requests keep their earlier, idempotent form.
        **{
            k: v for k, v in (("sample_frames", sample_frames), ("frames", frames)) if v is not None
        },
    )


@mcp.tool(annotations=WRITE)
async def submit_swap(
    media_path: str,
    mappings: list[FaceMapping],
    mode: Literal["preview", "render"] = "preview",
    frame: int = 0,
    start_frame: int = 0,
    max_frames: int | None = None,
    controls: dict | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Queue an image/frame preview or CFR video render using explicit identity mappings.

    Preview processes `frame`. Render processes `start_frame` and at most `max_frames`
    (None means through the end). Outputs are new job artifacts. Video output is H.264,
    with the selected interval's first audio stream encoded to AAC. Preview defaults to
    Inswapper128. Consult get_settings before changing parameters. GPU jobs run one at a time.
    Renders reject VFR, nonzero start times, odd dimensions and rotation; use normalize_media.
    """
    return await command(
        "submit",
        job_id=job_id or uuid.uuid4().hex,
        kind=mode,
        media_path=media_path,
        mappings=[m.model_dump() for m in mappings],
        frame=frame,
        start_frame=start_frame,
        max_frames=max_frames,
        controls=controls or {},
    )


@mcp.tool(annotations=WRITE)
async def normalize_media(
    media_path: str,
    fps: Annotated[
        str | float | None,
        Field(description="Output rate such as 25, 29.97 or '30000/1001'; default from the source"),
    ] = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Queue conversion of a video into a render-ready copy. Runs on CPU without the GPU lock.

    Output is CFR H.264 (CRF 18, yuv420p) starting at zero, with rotation applied, metadata
    removed and odd dimensions padded by one pixel. The first audio stream becomes AAC aligned
    to the first video frame. Without fps, the source average rate is used, snapped to a
    standard rate within 2%. The completed result's media_path is usable in submit_swap.
    """
    return await command(
        "submit",
        job_id=job_id or uuid.uuid4().hex,
        kind="normalize",
        media_path=media_path,
        fps=None if fps is None else str(fps),
    )


@mcp.tool(annotations=READ)
async def job_status(
    job_id: str,
    wait_seconds: Annotated[int, Field(ge=0, le=50)] = 0,
    include_log: bool = False,
) -> dict[str, Any]:
    """Read durable queued/running/completed/failed/cancelled state, progress and results.

    With wait_seconds > 0 the call blocks on the workstation until the job's status or
    progress stage differs from its value at call time, the job is terminal, or the wait
    elapses, then returns the current state. Frame counts advancing within a stage do not
    end the wait. The worker log tail is returned when include_log is true or the job failed.
    """
    return await command(
        "status",
        timeout=45 + wait_seconds,
        job_id=job_id,
        wait_seconds=wait_seconds,
        include_log=include_log,
    )


@mcp.tool(annotations=READ)
async def list_jobs(
    status: Literal["queued", "running", "completed", "failed", "cancelled"] | None = None,
    kind: Literal["detect", "preview", "render", "normalize"] | None = None,
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
) -> list[dict[str, Any]]:
    """List the most recently updated jobs, including jobs submitted by another client.

    Each summary has kind, status, media_path, frame (start_frame/max_frames for renders, fps
    for normalization), created/started/finished timestamps, elapsed_seconds since submission
    (until finish), error and progress. Use job_status for results and logs.
    """
    return await command("list", status=status, kind=kind, limit=limit)


@mcp.tool(annotations=WRITE)
async def cancel_job(job_id: str) -> dict[str, Any]:
    """Request cancellation of a queued/running job. Poll until its state is terminal."""
    return await command("cancel", job_id=job_id)


@mcp.tool(annotations=DESTROY)
async def delete_job(job_id: str) -> dict[str, Any]:
    """Permanently delete a completed, failed or cancelled job with its logs and artifacts.

    Queued and running jobs are refused; cancel them and wait for a terminal state first.
    """
    return await command("delete", job_id=job_id)


@mcp.tool(annotations=DESTROY)
async def prune_storage(
    older_than_days: Annotated[float, Field(ge=0, le=36500)] = 30, dry_run: bool = True
) -> dict[str, Any]:
    """Report or delete old terminal jobs, unused uploads and files left by crashed transfers.

    Removes terminal jobs finished more than `older_than_days` ago and uploads untouched for
    that long that no queued/running job references. Dry run by default; returns the paths,
    jobs and byte totals that were (or would be) deleted.
    """
    return await command("prune", older_than_days=older_than_days, dry_run=dry_run)


@mcp.tool(annotations=READ)
async def view_artifact(job_id: str, name: str = "preview.jpg") -> Image:
    """Show a completed job's PNG/JPEG crop or preview inline in Claude or Codex."""
    info = await command("artifact", job_id=job_id, name=name)
    suffix = Path(info["path"]).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"} or info["size"] > 10_000_000:
        raise ValueError("Inline artifacts must be PNG/JPEG images smaller than 10 MB")
    data = await ssh(["cat", "--", info["path"]])
    return Image(data=data, format="png" if suffix == ".png" else "jpeg")


@mcp.tool(annotations=WRITE)
async def download_artifact(job_id: str, name: str = "result.mp4") -> dict[str, Any]:
    """Download a completed artifact into the configured local downloads directory.

    Returns its absolute Mac path. Large videos stream directly to disk.
    """
    info = await command("artifact", job_id=job_id, name=name)
    directory = transport().downloads / job_id
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / Path(info["path"]).name
    temporary = directory / f".download-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            await ssh(["cat", "--", info["path"]], timeout=3600, stdout=stream)
        if temporary.stat().st_size != info["size"]:
            raise RuntimeError("Download size does not match the artifact")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), "bytes": info["size"]}


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@mcp.tool(annotations=WRITE)
async def upload_media(local_path: str) -> dict[str, Any]:
    """Stream a Mac image/video to private workstation storage and return its remote path.

    Accepts PNG, JPEG, WebP, MP4, MKV and MOV up to 20 GiB. Use this path in other tools.
    Uploads are content-addressed: re-uploading identical media returns the existing path.
    """
    path = Path(local_path).expanduser().resolve(strict=True)
    suffix = path.suffix.lower()
    if not path.is_file() or suffix not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".mp4",
        ".mkv",
        ".mov",
    }:
        raise ValueError("Expected a supported media file")
    size = path.stat().st_size
    if not 0 < size <= 20 * 1024**3:
        raise ValueError("Media must be between 1 byte and 20 GiB")
    digest = await asyncio.to_thread(sha256_file, path)
    existing = await command("find_upload", sha256=digest, suffix=suffix, size=size)
    if existing:
        return existing
    arguments = transport().control("upload", digest, suffix, str(size))
    with path.open("rb") as stream:
        stdout = await ssh(arguments, timeout=3600, stdin=stream)
    return json.loads(stdout)["result"]


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

"""MCP transport on the Mac; authenticated SSH controls the GPU workstation."""

import asyncio
import contextlib
import errno
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
# Only an omitted or null ID is generated; any other value goes to the workstation's validation.
JobId = Annotated[
    str | None,
    Field(
        description="Stable ID that makes retries idempotent: 1–64 letters, digits, underscores "
        "or hyphens, starting with a letter or digit. Omit it (or pass null) to generate one."
    ),
]


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
    similarity_threshold: Annotated[
        int | None,
        Field(
            ge=1,
            le=100,
            description="Multi-frame only: score (Fusion's 1-100 similarity scale) at which faces "
            "join an identity; default is Fusion's SimilarityThresholdSlider default (60). Lower "
            "it (e.g. 50) when one person's profile and frontal views split into identities.",
        ),
    ] = None,
    controls: dict | None = None,
    job_id: JobId = None,
) -> dict[str, Any]:
    """Queue face detection. Poll job_status, then inspect crops with view_artifact.

    Single-frame detection (`frame`, zero based) returns `faces`. To find who appears across a
    video, pass `sample_frames` (N evenly spaced frames) or explicit `frames` (at most 64)
    instead: faces are grouped into `identities` by recognition-embedding similarity at
    `similarity_threshold`, each with its appearances, frames and sample boxes. Every face or
    identity has a crop `artifact`. Its `reference_path` is usable as FaceMapping.target_face
    only when `reference_verified` is true: the crop re-detects as exactly one face, the same
    face, as submit_swap requires (a tighter crop is tried when a neighbour intrudes).
    Otherwise `reference_path` is null and the crop is for inspection only; a result without
    any verified reference also carries a `warning`. Both modes write contact.jpg, one grid of
    every crop labelled "face N" or "id N: <appearances>x"; unverified crops are marked * and
    framed in red. View it first.
    """
    optional = (
        ("sample_frames", sample_frames),
        ("frames", frames),
        ("similarity_threshold", similarity_threshold),
    )
    return await command(
        "submit",
        job_id=uuid.uuid4().hex if job_id is None else job_id,
        kind="detect",
        media_path=media_path,
        frame=frame,
        controls=controls or {},
        # Omitted when unset so single-frame requests keep their earlier, idempotent form.
        **{k: v for k, v in optional if v is not None},
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
    job_id: JobId = None,
) -> dict[str, Any]:
    """Queue an image/frame preview or CFR video render using explicit identity mappings.

    Preview processes `frame`. Render processes `start_frame` and at most `max_frames`
    (None means through the end). Outputs are new job artifacts. Video output is H.264,
    with the selected interval's first audio stream encoded to AAC, or no audio if that stream
    has no samples there. Preview defaults to Inswapper128. Consult get_settings before changing parameters. GPU jobs run one at a time.
    Renders reject VFR, nonzero start times, odd dimensions, rotation and display flips; use
    normalize_media. Each target swaps only its best-matching face per frame by default
    (control SwapOnlyBestMatchEnableToggle), so a look-alike that clears the mapping's
    SimilarityThresholdSlider is left alone while the target is in frame. Turn it off only to
    swap several faces of one identity in a frame. When the target leaves the frame, another
    face can still match; raise SimilarityThresholdSlider (e.g. 75) if the preview shows that.
    """
    return await command(
        "submit",
        job_id=uuid.uuid4().hex if job_id is None else job_id,
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
    job_id: JobId = None,
) -> dict[str, Any]:
    """Queue conversion of a video into a render-ready copy. Runs on CPU without the GPU lock.

    Output is CFR H.264 (CRF 18, 4:2:0, 10-bit for deeper sources) starting at zero, with
    rotation and flips applied, metadata removed and odd dimensions padded by one pixel. The
    first audio stream becomes AAC aligned to the first video frame and ending with the video.
    A clip shorter than half a frame period keeps one frame. Without fps, the source average
    rate is used, snapped to a
    standard rate within 2%. The completed result's media_path is usable in submit_swap.
    """
    return await command(
        "submit",
        job_id=uuid.uuid4().hex if job_id is None else job_id,
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

    Each summary has kind, status, media_path, frame (sample_frames or frames, and any
    similarity_threshold, for multi-frame detection; start_frame/max_frames for renders; fps
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
    that long, except files a queued/running job uses as input. Dry run by default and with
    the same checks; returns the paths, jobs and byte totals that were (or would be) deleted.
    """
    return await command("prune", older_than_days=older_than_days, dry_run=dry_run)


INLINE_LIMIT = 10_000_000


async def stream_artifact(job_id, name, receive, limit=None, timeout=45):
    """Stream a completed artifact's bytes to the writer that receive(header) returns.

    The workstation opens the file once and sends a JSON header with its size, then exactly that
    many bytes from the same descriptor, so metadata and content come from one job generation.
    At most size + 1 bytes are read before a mismatch fails the transfer.
    """
    settings = transport()
    arguments = settings.control("artifact", job_id, name, *([] if limit is None else [str(limit)]))
    process = await asyncio.create_subprocess_exec(
        *settings.ssh_prefix(),
        settings.command(arguments),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    errors = asyncio.ensure_future(process.stderr.read())

    async def transfer():
        header = await process.stdout.readline()
        try:
            response = json.loads(header)
        except ValueError:
            await process.wait()
            message = (header + await errors).decode(errors="replace")[-6000:]
            raise RuntimeError(
                settings.hint(message) if process.returncode == 255 else message
            ) from None
        if "error" in response:
            raise RuntimeError(response["error"])
        info = response["result"]
        size = info["size"]
        if limit is not None and size > limit:
            raise ValueError(f"Artifact is larger than the {limit}-byte limit")
        write, received = receive(info), 0
        while chunk := await process.stdout.read(min(1024 * 1024, size - received + 1)):
            received += len(chunk)
            if received > size:
                raise RuntimeError("Artifact transfer exceeded the announced size")
            write(chunk)
        if await process.wait() or received != size:
            raise RuntimeError("Artifact transfer was interrupted or its size changed")
        return info

    try:
        return await asyncio.wait_for(transfer(), timeout)
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            # Reading resumes the paused pipe, so its end of file lets wait() finish.
            while await process.stdout.read(1024 * 1024):
                pass
            await process.wait()
        errors.cancel()


@mcp.tool(annotations=READ)
async def view_artifact(job_id: str, name: str = "preview.jpg") -> Image:
    """Show a completed job's PNG/JPEG crop or preview inline in Claude or Codex."""
    suffix = Path(name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Inline artifacts must be PNG/JPEG images smaller than 10 MB")
    data = bytearray()
    await stream_artifact(job_id, name, lambda info: data.extend, limit=INLINE_LIMIT)
    return Image(data=bytes(data), format="png" if suffix == ".png" else "jpeg")


def plain_name(value):
    if value in ("", ".", "..") or "/" in value or "\0" in value:
        raise ValueError(f"{value!r} is not a plain file name")
    return value


def download_directory(root, job_id):
    """Open the job's download directory below the root descriptor, refusing symlinks."""
    try:
        os.mkdir(plain_name(job_id), dir_fd=root)
    except FileExistsError:
        pass
    try:
        return os.open(job_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise RuntimeError(
                f"Download directory {job_id} must be a real directory inside the downloads root"
            ) from None
        raise


@mcp.tool(annotations=WRITE)
async def download_artifact(job_id: str, name: str = "result.mp4") -> dict[str, Any]:
    """Download a completed artifact into the configured local downloads directory.

    Returns its absolute Mac path. Large videos stream directly to disk.
    """
    downloads = transport().downloads
    downloads.mkdir(parents=True, exist_ok=True)
    # The configured root is the trust boundary: the partial file stays in it, and the job
    # directory is reopened without following symlinks when the complete file is published.
    root = os.open(downloads, os.O_RDONLY | os.O_DIRECTORY)
    temporary = f".download-{uuid.uuid4().hex}"
    output = None

    def receive(info):
        nonlocal output
        plain_name(name)
        os.close(download_directory(root, job_id))  # Refuse a bad directory before streaming.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        output = open(os.open(temporary, flags, 0o666, dir_fd=root), "wb")
        return output.write

    try:
        try:
            info = await stream_artifact(job_id, name, receive, timeout=3600)
        finally:
            if output is not None:
                output.close()
        directory = download_directory(root, job_id)
        try:
            os.replace(temporary, name, src_dir_fd=root, dst_dir_fd=directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=root)
        os.close(root)
    return {"path": str(downloads / job_id / name), "bytes": info["size"]}


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

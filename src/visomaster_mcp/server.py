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
        "contact.jpg shows every labelled crop of a detection job at once. Before swapping with "
        "several source images, run analyze_references to check their quality, duplicates and "
        "consistency. Use explicit target/source references, check a request with "
        "preflight_swap (same arguments as submit_swap, nothing starts), preview before long "
        "renders (compare_variants previews several settings on the same frames or short clip "
        "in one job), and follow jobs with job_status(wait_seconds=50), "
        "which returns early on any state change, instead of polling rapidly. "
        "Use a stable job_id to make retries idempotent. Jobs survive client disconnection; "
        "resume_render continues a failed or cancelled render from its checkpointed chunks."
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
FAIRNESS = (
    "Interactive jobs start first; a waiting normal job goes next after 3 interactive jobs in a "
    "row or 300 s of them. Never interrupts a running job."
)


def priority_field(default):
    return Annotated[
        Literal["interactive", "normal"] | None,
        Field(description=f"GPU queue class. Default: {default}. {FAIRNESS}"),
    ]


Priority = priority_field(
    "interactive for detection, previews, reference analysis and still comparisons, normal "
    "for renders and clip comparisons. Pass normal for batch work that should not delay "
    "others' previews"
)
ResumePriority = priority_field(
    "the earlier render's priority if it was given explicitly, otherwise normal"
)
RecipePriority = priority_field(
    "the priority the recipe stores, otherwise interactive for a preview and normal for a render"
)
StoredPriority = Annotated[
    Literal["interactive", "normal"] | None,
    Field(
        description="GPU queue class stored with the recipe for the jobs run_recipe queues "
        "from it (run_recipe can override it). Default: none stored, so a preview queues as "
        f"interactive and a render as normal. {FAIRNESS}"
    ),
]


Weight = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class FaceMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_face: str = Field(
        description="Absolute workstation image path containing one target face"
    )
    source_faces: list[str] = Field(
        min_length=1, max_length=20, description="Workstation images of the replacement identity"
    )
    source_weights: list[Weight] | None = Field(
        None,
        description="Optional weight per source image, in source_faces order: finite, "
        "nonnegative, with a positive sum. 0 excludes an image (it is not read). Embeddings are "
        "averaged with these weights, then L2-normalized as Fusion's merge does, so only their "
        "ratios matter. Omit for equal weights (the plain average).",
    )
    source_sha256: list[Digest | None] | None = Field(
        None,
        description="Optional SHA-256 per source image, in source_faces order, as "
        "analyze_references reported them. Submission fails if any file's bytes changed since. "
        "null pins nothing for that image; an image with weight 0 is never read or checked.",
    )
    parameters: dict = Field(
        default_factory=dict, description="Validated Fusion keys from get_settings"
    )


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,19}$",
        description="Label of this variant: 1–20 letters, digits, underscores or hyphens; "
        "unique, ignoring case. Names the variant's artifacts and grid tiles.",
    )
    parameters: dict = Field(
        default_factory=dict,
        description="Fusion parameters from get_settings, merged over every mapping's parameters",
    )
    controls: dict = Field(
        default_factory=dict, description="Controls from get_settings, merged over `controls`"
    )


class ReferenceThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_face_pixels: int | None = Field(
        None, ge=1, le=4096, description="Shorter face-box side in pixels; default 112"
    )
    min_sharpness: float | None = Field(
        None, ge=0, le=1_000_000, description="Laplacian variance at 112x112; default 40"
    )
    duplicate_similarity: float | None = Field(
        None,
        ge=50,
        le=100,
        description="Fusion similarity marking near duplicates; default 95 (96 for SimSwap, 95.5 "
        "for Ghost and 98 for CSCS recognizers)",
    )
    outlier_similarity: float | None = Field(
        None,
        ge=0,
        le=100,
        description="Leave-one-out similarity below which a reference is an outlier; default "
        "70 (85 for the CSCS recognizer)",
    )


class VisibleRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mapping: int = Field(ge=0, le=19, description="Zero-based index into mappings")
    start_frame: int = Field(ge=0, description="First zero-based source frame")
    end_frame: int = Field(ge=1, description="First source frame after the range (exclusive)")


class SwapReportOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_images: int = Field(
        default=6, ge=0, le=24, description="Most labelled before/after review images to write"
    )
    expected_visible: list[VisibleRange] = Field(
        default_factory=list,
        max_length=100,
        description="Ranges where a mapping's person is known to be visible; unswapped frames "
        "there are reported, and with min_coverage they are the coverage denominator",
    )
    min_coverage: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description="Strict: fail the render if any mapping's swapped share of its expected-"
        "visible frames (or, without ranges, of all rendered frames) is below this",
    )


class Segment(BaseModel):
    """Frames [start_frame, end_frame) of a render with their own intent: passthrough, or any of
    mappings, parameters and controls replacing the base request's within them."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
        description="Stable segment ID, unique in the request, reported in plan.json and results",
    )
    start_frame: int = Field(ge=0, description="First zero-based source frame")
    end_frame: int = Field(ge=1, description="First source frame after the segment (exclusive)")
    passthrough: Literal[True] | None = Field(
        None,
        description="Leave these frames exactly as decoded (no detection or swap); excludes the "
        "other changes",
    )
    mappings: list[int] | None = Field(
        None,
        min_length=1,
        max_length=20,
        description="Zero-based indices into mappings that swap here; the others are not "
        "requested in these frames. Default: all",
    )
    parameters: dict | None = Field(
        None,
        description="Fusion parameters (see get_settings) applied over every active mapping's "
        "own here; SwapModelSelection holds for the whole render",
    )
    controls: dict | None = Field(
        None,
        description="Controls over the request's here; DetectorModelSelection and the tracking "
        "toggles hold for the whole render",
    )


Segments = Annotated[
    list[Segment] | None,
    Field(
        max_length=100,
        description="Renders only: ordered, non-overlapping source-frame segments. Frames outside "
        "every segment (gaps) use the base mappings, parameters and controls. Segments are "
        "clipped to the rendered frames and must intersect them. Omit for a uniform render.",
    ),
]


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


def local_code():
    """SHA-256 of this checkout's worker files, or None when they are absent."""
    try:
        return {
            name: hashlib.sha256((LOCAL_CODE / name).read_bytes()).hexdigest()
            for name in ("common.py", "control.py", "engine.py")
        }
    except OSError:
        return None


@mcp.tool(annotations=READ)
async def health_check() -> dict[str, Any]:
    """Check installed revision, deployed release, GPU, memory use and media/state paths.

    code_matches compares deployed worker hashes with this checkout, or is null if unknown.
    """
    result = await command("health")
    local = local_code()
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
    priority: Priority = None,
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
        ("priority", priority),
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
    report: SwapReportOptions | None = None,
    segments: Segments = None,
    job_id: JobId = None,
    priority: Priority = None,
) -> dict[str, Any]:
    """Queue an image/frame preview or CFR video render using explicit identity mappings.

    Preview processes `frame`. Render processes `start_frame` and at most `max_frames`
    (None means through the end). Outputs are new job artifacts. Video output is H.264,
    with the selected interval's first audio stream encoded to AAC, or no audio if that stream
    has no samples there. Preview defaults to Inswapper128. Consult get_settings before changing parameters. GPU jobs run one at a time:
    previews queue as interactive and renders as normal (see `priority`).
    Renders reject VFR, nonzero start times, odd dimensions, rotation and display flips; use
    normalize_media. Each target swaps only its best-matching face per frame by default
    (control SwapOnlyBestMatchEnableToggle), so a look-alike that clears the mapping's
    SimilarityThresholdSlider is left alone while the target is in frame. Turn it off only to
    swap several faces of one identity in a frame. When the target leaves the frame, another
    face can still match; raise SimilarityThresholdSlider (e.g. 75) if the preview shows that.

    Several source images are averaged (source_weights sets explicit weights, 0 excludes one)
    and the result's `references` lists, per mapping, each source's SHA-256, weight and share.
    Every reference's bytes are fingerprinted at submission; a file changed before the worker
    reads it fails the job instead of being used.

    Results carry `swap_report`, a per-mapping summary (mapping = index into mappings) of
    swapped frames, coverage, ambiguous frames, suspicious intervals and warnings such as a
    mapping that never swapped; warnings never fail the job. The full report is the
    swap-report.json artifact, with half-open [start_frame, end_frame) source-frame intervals
    and times. Renders also write up to `report.review_images` labelled before/after images
    (review-NN.jpg). `report` applies to renders only; omit it for the defaults.

    preflight_swap takes these same arguments and reports, without starting anything, whether
    this request can start and what to fix.

    Renders are encoded in validated, checkpointed chunks of 250 frames and joined without
    re-encoding; the result's `checkpoint` describes them. If a render fails or is cancelled,
    resume_render queues a new job that reuses the chunks when nothing it depends on changed.

    `segments` (renders only) change intent by source-frame range [start_frame, end_frame)
    without changing the timeline: a segment may restrict which `mappings` swap, override
    `parameters` (for every active mapping) or `controls`, or pass frames through untouched.
    Gaps use the base settings. Overlaps, unknown or render-wide settings, and contradictory
    or empty segments are refused at submission; a render whose frames are all passthrough is
    refused. Frame count, order, timing and audio are unchanged. Temporal tracking state is
    reset at every segment boundary. The result's `plan` lists each run (segment or gap) with
    its frames and swaps, and plan.json records the resolved settings of every run; the render
    still fails when no face swaps outside passthrough frames, and a segment that swapped
    nothing is a warning.
    """
    request = swap_request(
        media_path,
        mappings,
        mode,
        frame,
        start_frame,
        max_frames,
        controls,
        report,
        priority,
        segments,
    )
    return await command("submit", job_id=uuid.uuid4().hex if job_id is None else job_id, **request)


def swap_request(
    media_path, mappings, mode, frame, start_frame, max_frames, controls, report, priority, segments
):
    """The submit_swap request without its job_id; preflight_swap checks exactly this."""
    return {
        "kind": mode,
        "media_path": media_path,
        # Unset optional fields are omitted, so earlier requests keep their idempotent form.
        "mappings": [m.model_dump(exclude_none=True) for m in mappings],
        "frame": frame,
        "start_frame": start_frame,
        "max_frames": max_frames,
        "controls": controls or {},
        **({} if report is None else {"report": report.model_dump(exclude_none=True)}),
        **({} if segments is None else {"segments": dump_segments(segments)}),
        **({} if priority is None else {"priority": priority}),
    }


@mcp.tool(annotations=WRITE)
async def preflight_swap(
    media_path: str,
    mappings: list[FaceMapping],
    mode: Literal["preview", "render"] = "preview",
    frame: int = 0,
    start_frame: int = 0,
    max_frames: int | None = None,
    controls: dict | None = None,
    report: SwapReportOptions | None = None,
    segments: Segments = None,
    job_id: JobId = None,
    priority: Priority = None,
    analyze_references: Annotated[
        bool,
        Field(
            description="Also queue GPU reference analysis of each mapping's sources (an "
            "ordinary analyze job, shared GPU lock, interactive class). Repeating the preflight "
            "finds the same jobs and reports their results."
        ),
    ] = False,
) -> dict[str, Any]:
    """Check whether a submit_swap request can start, without starting it or using the GPU.

    Takes submit_swap's arguments (validated by the same models and, on the workstation, by
    the same request validation); job_id, if given, is checked for an existing job. Cheap
    checks read files and run FFprobe, git and nvidia-smi: Fusion and its models are not
    loaded and no job starts. They cover the configuration, Fusion revision, deployed code,
    job_id, media and reference access (with each reference's SHA-256), settings, the model
    files those settings load, the worker's own media checks (timestamps read only through the
    requested frames), free space against an estimate, the GPU's class and memory use,
    recorded GPU qualification and submit_swap's validation itself.

    `readiness` is blocked (a check would make submission or the job fail), pending
    (analysis still running), warnings, or ready. Each of `checks` has `check`, `status`
    (passed, warning, blocked, pending or not_checked), `severity`, a stable `code`, `message`,
    `evidence` and `actions` (tool calls with arguments, or operator steps); `not_checked`
    lists what was skipped. `qualification` reports combinations no recorded evidence covers
    as `unqualified`, a warning, never a refusal. Media that normalize_media can fix suggests
    it; nothing is converted or changed. `fingerprint` hashes the request, the inputs (media
    identity, reference SHA-256) and the configuration (settings, revision, worker code, model
    files), with `checked_at`. A preflight reserves nothing: submit_swap checks again.
    """
    request = swap_request(
        media_path,
        mappings,
        mode,
        frame,
        start_frame,
        max_frames,
        controls,
        report,
        priority,
        segments,
    )
    return await command(
        "preflight",
        timeout=300,
        request={**({} if job_id is None else {"job_id": job_id}), **request},
        analyze=analyze_references,
        client_code=local_code(),
    )


def dump_segments(segments):
    return [segment.model_dump(exclude_none=True) for segment in segments]


@mcp.tool(annotations=WRITE)
async def compare_variants(
    media_path: str,
    mappings: list[FaceMapping],
    variants: Annotated[list[Variant], Field(min_length=2, max_length=6)],
    frames: Annotated[
        list[int] | None,
        Field(
            min_length=1,
            max_length=4,
            description="Still comparison: 1–4 distinct zero-based frames (images: [0])",
        ),
    ] = None,
    start_frame: Annotated[
        int | None, Field(ge=0, description="Clip comparison: first frame (default 0)")
    ] = None,
    max_frames: Annotated[
        int | None,
        Field(ge=1, le=120, description="Clip comparison: frames to render per variant"),
    ] = None,
    controls: dict | None = None,
    job_id: JobId = None,
    priority: Priority = None,
) -> dict[str, Any]:
    """Queue one job that runs 2–6 named settings variants over the same frames or clip.

    Give `frames` for stills, or `max_frames` (and `start_frame`) for a contiguous CFR clip.
    Every variant uses the same `mappings` and `controls`; its own `parameters` are merged
    over each mapping's and its `controls` over the common ones, and each merged request is
    validated like submit_swap before anything is queued. Variants run in order in one worker
    under one GPU admission, on identical decoded frames; cancel_job stops them all. Stills
    queue as interactive, clips as normal (see `priority`).

    The result (also compare.json) lists per variant its status, error, overrides, effective
    `settings`, `references`, swap report summaries, artifacts and `submit_swap`: arguments
    that repeat that variant alone (add a job_id; change frame or the range as needed).
    compare-f<N>.jpg grids show the original and each variant labelled "<index> <name>", at
    display aspect; view them first. Full-resolution PNGs are original-f<N>.png and
    v<index>-<name>-f<N>.png; clips are v<index>-<name>.mp4 with the same range, frame count,
    duration and source audio. A failed variant keeps its reason, appears as a red FAILED
    tile and has no artifacts; the job then completes with `complete: false` and a `warning`,
    and fails only when every variant fails. Nothing chooses a best variant.
    """
    selection = (("frames", frames), ("start_frame", start_frame), ("max_frames", max_frames))
    return await command(
        "submit",
        job_id=uuid.uuid4().hex if job_id is None else job_id,
        kind="compare",
        media_path=media_path,
        mappings=[m.model_dump(exclude_none=True) for m in mappings],
        variants=[v.model_dump() for v in variants],
        controls=controls or {},
        **{k: v for k, v in selection if v is not None},
        **({} if priority is None else {"priority": priority}),
    )


@mcp.tool(annotations=WRITE)
async def analyze_references(
    references: Annotated[
        list[str],
        Field(min_length=1, max_length=20, description="Absolute workstation image paths"),
    ],
    swap_model: Annotated[
        str | None,
        Field(
            description="SwapModelSelection whose ArcFace embedding is compared, as the swap "
            "averages it; default Fusion's default swap model"
        ),
    ] = None,
    thresholds: ReferenceThresholds | None = None,
    controls: dict | None = None,
    job_id: JobId = None,
    priority: Priority = None,
) -> dict[str, Any]:
    """Queue a GPU check of candidate source images. Poll job_status, then view contact.jpg.

    Each reference gets an id (r0, r1, ... in request order), its SHA-256, face count, face
    box, `face_pixels` (shorter box side, source pixels), `sharpness` (Laplacian variance of
    the box's luma at 112x112), `consistency_score`, a crop artifact when usable and
    `findings` (code, severity, message). `usable` means submit_swap accepts it: exactly one
    detected face with an embedding; unreadable, no_face, multiple_faces and unrecognized are
    errors. Warnings (small_face, blurry, exact_duplicate, near_duplicate, outlier,
    inconsistent_pair) report concerns for review; nothing is removed or changed. Outlier
    scores compare each embedding with the mean of the others on Fusion's similarity scale,
    need 3 distinct usable references (2 give only `pair_similarity`), and suggest a different
    person, view or lighting rather than establishing identity. Detection uses the same
    `controls` as submit_swap. Pass the reported sha256 values as FaceMapping.source_sha256 so
    a later swap fails if the files change, and source_weights to exclude or down-weight.
    Analysis queues as interactive unless `priority` says otherwise.
    """
    optional = (("swap_model", swap_model), ("controls", controls), ("priority", priority))
    limits = thresholds.model_dump(exclude_none=True) if thresholds else {}
    return await command(
        "submit",
        job_id=uuid.uuid4().hex if job_id is None else job_id,
        kind="analyze",
        references=references,
        **({"thresholds": limits} if limits else {}),
        **{k: v for k, v in optional if v is not None},
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

    With wait_seconds > 0 the call blocks on the workstation until the job's status, progress
    stage (or a comparison's running variant) or queue position/reason differs from its value
    at call time, the job is terminal, or the wait elapses, then returns the current state.
    Frame counts advancing within a stage do not end the wait. The worker log tail is returned
    when include_log is true or the job failed. GPU jobs carry `priority` and `sequence` (FIFO
    order within a class); while queued, `queue` gives an advisory `position` (1 = next to
    start), `reason` (next, gpu_busy, behind_earlier, behind_interactive, behind_due_normal,
    cancel_requested, or starting: no queue entry yet, while being admitted or, for a render or
    recipe job, while its inputs are hashed before it joins, with progress stage
    fingerprinting), `interactive_streak` and `waiting_seconds`. Positions can change as jobs
    arrive, end or become due under the fairness rule; there is no start-time estimate.
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
    kind: Literal["detect", "preview", "render", "normalize", "analyze", "compare"] | None = None,
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
) -> list[dict[str, Any]]:
    """List the most recently updated jobs, including jobs submitted by another client.

    Each summary has kind, status, media_path, frame (sample_frames or frames, and any
    similarity_threshold, for multi-frame detection; start_frame/max_frames for renders; fps
    for normalization; references for reference analysis; frames or start_frame/max_frames
    and variant names for comparisons), created/started/finished
    timestamps, elapsed_seconds since submission (until finish), error and progress; GPU
    jobs add priority and sequence, queued ones the advisory queue standing that job_status
    describes, and a resumed render resume_from. Use job_status for results and logs.
    """
    return await command("list", status=status, kind=kind, limit=limit)


@mcp.tool(annotations=WRITE)
async def cancel_job(job_id: str) -> dict[str, Any]:
    """Request cancellation of a queued/running job. Poll until its state is terminal."""
    return await command("cancel", job_id=job_id)


@mcp.tool(annotations=WRITE)
async def resume_render(
    from_job_id: Annotated[
        str, Field(description="The failed or cancelled render to continue; it is not changed")
    ],
    job_id: JobId = None,
    priority: ResumePriority = None,
) -> dict[str, Any]:
    """Queue a new render that reuses the verified chunks of a failed or cancelled render.

    Renders are encoded in chunks of 250 source frames, each published only after its frame
    count and timing are checked and committed with its SHA-256 in a checkpoint manifest. The
    new job, with its own job_id, repeats the earlier job's request and links to it through
    `resume_from`; the earlier job keeps its terminal state. Before anything is reused, the
    media, references, effective settings, model files, Fusion revision, adapter code,
    configuration and FFmpeg build are hashed and compared with the earlier attempt's; any
    difference refuses the resume with a list of the changes (submit a fresh render instead).
    Renders with FaceTrackingEnableToggle or KPSSmoothingEnableToggle are refused, because
    their tracking state is not checkpointed. The worker reuses the longest run of chunks that
    still verify, renders the remaining frames and joins all chunks by stream copy, with the
    audio cut once over the whole range; the result and swap report cover every frame. A retry
    with the same job_id and arguments returns the same job. Cancelling a job never resumes it.
    """
    return await command(
        "resume",
        timeout=HASHING_TIMEOUT,
        job_id=uuid.uuid4().hex if job_id is None else job_id,
        from_job_id=from_job_id,
        **({} if priority is None else {"priority": priority}),
    )


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
    that long, except files a saved recipe or a queued/running job uses as input. Dry run by default and with
    the same checks; returns the paths, jobs and byte totals that were (or would be) deleted.
    """
    return await command("prune", older_than_days=older_than_days, dry_run=dry_run)


RecipeName = Annotated[
    str,
    Field(
        description="Recipe name: 1–64 letters, digits, underscores or hyphens, starting with a "
        "letter or digit"
    ),
]
RecipeRevision = Annotated[
    int | None, Field(ge=1, description="Revision number; default the latest")
]
HASHING_TIMEOUT = 900  # Recipe calls hash media and model files on the workstation.
RECIPE_LIMIT = 4 * 1024**2


class RecipeRange(BaseModel):
    """Frames to process: `frame` for a preview; `start_frame` and `max_frames` (null: through
    the end) for a render. Zero-based, in decoded order."""

    model_config = ConfigDict(extra="forbid")
    frame: int | None = Field(None, ge=0)
    start_frame: int | None = Field(None, ge=0)
    max_frames: int | None = Field(None, ge=1)


@mcp.tool(annotations=WRITE)
async def save_recipe(
    name: RecipeName,
    job_id: Annotated[
        str | None, Field(description="A completed preview or render job to save")
    ] = None,
    media_path: str | None = None,
    mappings: list[FaceMapping] | None = None,
    mode: Literal["preview", "render"] = "preview",
    frame: int = 0,
    start_frame: int = 0,
    max_frames: int | None = None,
    controls: dict | None = None,
    report: SwapReportOptions | None = None,
    segments: Segments = None,
    priority: StoredPriority = None,
    note: Annotated[str | None, Field(max_length=2000)] = None,
) -> dict[str, Any]:
    """Save a new immutable revision of a named, reusable swap configuration (a recipe).

    Give either `job_id` (a completed preview or render, recorded with the release, Fusion
    revision and models it ran with) or an explicit request (media_path, mappings, mode and its
    range, controls) as submit_swap takes it. The recipe stores the resolved request with every
    adjustable setting pinned at its effective value, the full effective settings, the SHA-256
    of the media and of every target and source image, source weights, the frame convention,
    the Fusion revision, the adapter release and its code hashes, the configuration identity and
    the names, paths and SHA-256 of the selected and loaded model files (never their weights).
    Face references are copied into the content-addressed upload store and kept while a recipe
    names them; `retention` says which inputs are retained. Media and models are hashed now,
    so a large video takes a while. Saving again under a name adds a revision; earlier
    revisions, and jobs that name them, never change. A render's `segments` are kept as given.
    """
    if job_id is not None:
        return await command(
            "recipe_save", timeout=HASHING_TIMEOUT, name=name, job_id=job_id, note=note
        )
    if media_path is None or not mappings:
        raise ValueError("Give job_id, or media_path and mappings")
    frames = {"frame": frame} if mode == "preview" else {"start_frame": start_frame}
    if mode == "render":
        frames["max_frames"] = max_frames
    optional = (
        ("report", None if report is None else report.model_dump(exclude_none=True)),
        ("segments", None if segments is None else dump_segments(segments)),
        ("priority", priority),
        ("note", note),
    )
    return await command(
        "recipe_save",
        timeout=HASHING_TIMEOUT,
        name=name,
        kind=mode,
        media_path=media_path,
        mappings=[m.model_dump(exclude_none=True) for m in mappings],
        controls=controls or {},
        **frames,
        **{key: value for key, value in optional if value is not None},
    )


@mcp.tool(annotations=READ)
async def get_recipe(
    name: RecipeName | None = None,
    revision: RecipeRevision = None,
    verify: Annotated[
        bool,
        Field(
            description="Hash the recipe's inputs and models now and report what changed since "
            "it was saved"
        ),
    ] = False,
) -> dict[str, Any]:
    """Without a name, list recipes and their revisions. With one, return a revision's document.

    The document is what export_recipe writes. `retention` marks each input retained (kept by
    retention while the recipe names it) or unretained (outside state storage), and missing.
    With `verify`, `verification.changes` lists, readably, every input whose bytes changed or
    that is unavailable, changed settings or defaults, a different Fusion revision, adapter
    code, configuration, and changed or missing model files.
    """
    return await command(
        "recipe_get",
        timeout=HASHING_TIMEOUT if verify else 45,
        name=name,
        revision=revision,
        verify=verify,
    )


@mcp.tool(annotations=WRITE)
async def run_recipe(
    name: RecipeName,
    revision: RecipeRevision = None,
    mode: Annotated[
        Literal["preview", "render"] | None, Field(description="Default: the recipe's mode")
    ] = None,
    media_path: Annotated[
        str | None,
        Field(description="Substitute media; content that differs needs targets and range"),
    ] = None,
    targets: Annotated[
        list[str] | None,
        Field(description="Substitute target_face paths, one per recipe mapping, in order"),
    ] = None,
    frame_range: Annotated[
        RecipeRange | None,
        Field(description="Substitute frames; required with a new clip or a changed mode"),
    ] = None,
    report: SwapReportOptions | None = None,
    priority: RecipePriority = None,
    accept_changes: Annotated[
        bool,
        Field(
            description="Run even though inputs, defaults, models, adapter code or the Fusion "
            "revision changed since the recipe was saved; the changes are recorded in the job"
        ),
    ] = False,
    job_id: JobId = None,
) -> dict[str, Any]:
    """Queue a new preview or render job from a recipe revision, with explicit substitutions.

    The job runs the recipe's explicit request with its pinned settings, sources and weights.
    Before submitting, every input and model file is hashed and compared with the recipe; if
    anything changed (input bytes at the same path, defaults, model files, adapter code, Fusion
    revision or configuration) the call fails with a readable list of the changes unless
    `accept_changes` is true. Unavailable inputs always fail with the inputs to restore. Media
    with different content is a new clip: it needs explicit `targets` from a detection on that
    clip and an explicit `frame_range` (and `report` if the recipe has expected_visible
    ranges). The job's request and result record `recipe`: name, revision, the document's
    sha256, the invocation, substitutions and accepted changes. The worker hashes the media
    again before decoding it. Outputs are expected to be equivalent, not bit-identical: GPU
    kernels and drivers are not pinned. A retry with the same job_id and arguments returns the
    existing job.
    """
    values = {
        "mode": mode,
        "media_path": media_path,
        "targets": targets,
        "range": None if frame_range is None else frame_range.model_dump(exclude_unset=True),
        "report": None if report is None else report.model_dump(exclude_none=True),
        "priority": priority,
    }
    return await command(
        "recipe_run",
        timeout=HASHING_TIMEOUT,
        job_id=uuid.uuid4().hex if job_id is None else job_id,
        name=name,
        revision=revision,
        accept_changes=accept_changes,
        **{key: value for key, value in values.items() if value is not None},
    )


@mcp.tool(annotations=WRITE)
async def export_recipe(name: RecipeName, revision: RecipeRevision = None) -> dict[str, Any]:
    """Write a recipe revision as portable JSON in the local downloads directory.

    The file holds configuration and provenance (paths, fingerprints, settings, versions and
    model checksums), never media or model weights, and carries its own sha256, which
    import_recipe checks. Returns its local path.
    """
    document = (await command("recipe_get", name=name, revision=revision))["recipe"]
    data = json.dumps(document, indent=2, sort_keys=True).encode() + b"\n"
    downloads = transport().downloads
    downloads.mkdir(parents=True, exist_ok=True)
    root = os.open(downloads, os.O_RDONLY | os.O_DIRECTORY)
    file_name = plain_name(f"{document['name']}-r{document['revision']}.json")
    temporary = f".recipe-{uuid.uuid4().hex}"
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        with open(os.open(temporary, flags, 0o666, dir_fd=root), "wb") as output:
            output.write(data)
        directory = download_directory(root, "recipes")
        try:
            os.replace(temporary, file_name, src_dir_fd=root, dst_dir_fd=directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=root)
        os.close(root)
    return {
        "path": str(downloads / "recipes" / file_name),
        "name": document["name"],
        "revision": document["revision"],
        "sha256": document["sha256"],
    }


@mcp.tool(annotations=WRITE)
async def import_recipe(
    local_path: str,
    name: Annotated[
        RecipeName | None, Field(description="Store under this name; default the recipe's own")
    ] = None,
) -> dict[str, Any]:
    """Import an exported recipe file as a new revision on this workstation.

    Refuses unsupported format versions, documents whose sha256 does not match their content,
    malformed fields, inputs outside the allowed media and state roots, and settings the
    installed Fusion does not accept. Every input must exist with its recorded bytes, at its
    recorded path or as an upload with those bytes (upload_media the file first); otherwise the
    error lists what to provide. Recorded provenance is kept, so run_recipe reports version
    differences here.
    """
    path = Path(local_path).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size > RECIPE_LIMIT:
        raise ValueError("Expected a recipe JSON file of at most 4 MiB")
    try:
        document = json.loads(path.read_bytes())
    except ValueError as error:
        raise ValueError(f"{path} is not JSON: {error}") from None
    return await command("recipe_import", timeout=HASHING_TIMEOUT, document=document, name=name)


@mcp.tool(annotations=DESTROY)
async def delete_recipe(name: RecipeName, revision: RecipeRevision = None) -> dict[str, Any]:
    """Delete one revision of a recipe, or all of them. Its number is never reused.

    Deletes no media or uploads: inputs another recipe or a queued or running job uses stay
    protected, and prune_storage may remove the rest once they are old enough.
    """
    return await command("recipe_delete", name=name, revision=revision)


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

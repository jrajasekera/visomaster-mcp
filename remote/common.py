"""Shared configuration, validation and atomic state for the workstation (stdlib only)."""

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

REVISION = "f2d6f5ebe190b631ee5c8972332083cf7f8a3e72"
CODE = ("common.py", "control.py", "engine.py")
CONFIG_NAME = "remote-config.json"
CONFIG_KEYS = {"version", "media_root", "fusion_dir", "state_dir", "gpu", "runner", "lock_dir"}
LAYOUT_NAMES = ("settings", "common", "swapper", "face_editor", "denoiser")
# Environment variables read by earlier releases; a differing value must not be ignored silently.
LEGACY = {"VM_BASE": "base", "VM_APP": "app", "VM_STATE": "state", "VM_GPU": "gpu"}
JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
TERMINAL = {"completed", "failed", "cancelled"}
# GPU admission class of each job kind (docs/explanation/gpu-scheduling.md); a request's `priority`
# overrides it. A GPU kind missing here is scheduled as DEFAULT_PRIORITY. Normalization never
# waits for the GPU and has no class.
PRIORITIES = {
    "detect": "interactive",
    "preview": "interactive",
    "analyze": "interactive",
    "compare": "interactive",  # Still comparisons; clip comparisons are normal (job_priority).
    "render": "normal",
}
DEFAULT_PRIORITY = "normal"
CLASSES = ("interactive", "normal")
PARAMETERS = {
    "SwapModelSelection",
    "SwapperResSelection",
    "SimilarityThresholdSlider",
    "FaceRestorerEnableToggle",
    "FaceRestorerTypeSelection",
    "FaceRestorerBlendSlider",
    "FaceRestorerDetTypeSelection",
    "FaceFidelityWeightDecimalSlider",
    "FaceExpressionEnableBothToggle",
    "BorderTopSlider",
    "BorderBottomSlider",
    "BorderLeftSlider",
    "BorderRightSlider",
    "OverallMaskBlendAmountSlider",
    "OccluderEnableToggle",
    "DFLXSegEnableToggle",
    "StrengthEnableToggle",
    "StrengthAmountSlider",
}
# Demuxers that read only the file they are given (mov is MP4/MOV, matroska is MKV/WebM).
DEMUXERS = "mov,matroska,avi,mpegts,png_pipe,jpeg_pipe,webp_pipe,bmp_pipe"
MAX_DETECT_FRAMES = 64
REPORT_KEYS = {"review_images", "expected_visible", "min_coverage"}
MAX_REVIEW_IMAGES = 24
MAX_EXPECTED_RANGES = 100
NORMALIZE_KEYS = {"kind", "media_path", "fps"}
ANALYZE_KEYS = {
    "kind",
    "references",
    "reference_sha256",
    "swap_model",
    "thresholds",
    "controls",
    "priority",
}
# Reference analysis thresholds: (type, minimum, maximum). The engine documents their defaults.
THRESHOLDS = {
    "min_face_pixels": (int, 1, 4096),
    "min_sharpness": (float, 0, 1_000_000),
    "duplicate_similarity": (float, 50, 100),
    "outlier_similarity": (float, 0, 100),
}
MAX_REFERENCES = 20
# Variant comparisons (docs/explanation/video-processing.md, Variant comparisons): bounded so a
# request cannot schedule a sweep.
COMPARE_KEYS = {
    "kind",
    "media_path",
    "mappings",
    "controls",
    "frames",
    "start_frame",
    "max_frames",
    "variants",
    "priority",
}
VARIANT_KEYS = {"name", "parameters", "controls"}
VARIANT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,19}")
MIN_VARIANTS, MAX_VARIANTS = 2, 6
MAX_COMPARE_FRAMES = 4  # Explicit still frames.
MAX_COMPARE_CLIP = 120  # Frames of a contiguous clip.
MAX_COMPARE_PIXELS = 3840 * 2160  # Source frame area; outputs keep the source dimensions.
MAX_REFERENCE_BYTES = 256 * 1024**2
SHA256 = re.compile(r"[0-9a-f]{64}")
CONTROLS = {
    "DetectorModelSelection",
    "DetectorScoreSlider",
    "MaxFacesToDetectSlider",
    "FaceTrackingEnableToggle",
    "ByteTrackTrackThreshSlider",
    "ByteTrackMatchThreshSlider",
    "ByteTrackTrackBufferSlider",
    "KPSSmoothingEnableToggle",
    "KPSEmaAlphaSlider",
    "SwapOnlyBestMatchEnableToggle",
}
# Defaults the adapter applies over Fusion's; get_settings reports them.
ADAPTER_CONTROLS = {
    "DetectorModelSelection": "Yolov8",
    "MaxFacesToDetectSlider": 20,
    "SwapOnlyBestMatchEnableToggle": True,
}
# Fusion model names each selection loads at REVISION (face_detectors.py detector_map,
# models_data.arcface_mapping_model_dict, face_swappers.py, face_restorers.py, face_masks.py and
# face_editors.py); models_data.models_list names their files. Preflight checks those files.
DETECTOR_MODELS = {
    "RetinaFace": "RetinaFace",
    "SCRFD": "SCRFD2.5g",
    "Yolov8": "YoloFace8n",
    "Yunet": "YunetN",
    "Yolov11 VR180": "YoloFace11nVR180",
    "Yolov12 VR180": "YoloFace12nVR180",
}
SWAPPER_MODELS = {
    "Inswapper128": ("Inswapper128", "Inswapper128ArcFace"),
    "InStyleSwapper256 Version A": ("InStyleSwapper256 Version A", "Inswapper128ArcFace"),
    "InStyleSwapper256 Version B": ("InStyleSwapper256 Version B", "Inswapper128ArcFace"),
    "InStyleSwapper256 Version C": ("InStyleSwapper256 Version C", "Inswapper128ArcFace"),
    "SimSwap512": ("SimSwap512", "SimSwapArcFace"),
    "GhostFace-v1": ("GhostFacev1", "GhostArcFace"),
    "GhostFace-v2": ("GhostFacev2", "GhostArcFace"),
    "GhostFace-v3": ("GhostFacev3", "GhostArcFace"),
    "CSCS": ("CSCS", "CSCSArcFace", "CSCSIDArcFace"),
}
RESTORER_MODELS = {
    "GFPGAN-v1.4": "GFPGANv1.4",
    "GFPGAN-1024": "GFPGAN1024",
    "CodeFormer": "CodeFormer",
    "GPEN-256": "GPENBFR256",
    "GPEN-512": "GPENBFR512",
    "GPEN-1024": "GPENBFR1024",
    "GPEN-2048": "GPENBFR2048",
    "RestoreFormer++": "RestoreFormerPlusPlus",
    "VQFR-v2": "VQFRv2",
}
EXPRESSION_MODELS = (
    "LivePortraitMotionExtractor",
    "LivePortraitAppearanceFeatureExtractor",
    "LivePortraitStitchingEye",
    "LivePortraitStitchingLip",
    "LivePortraitStitching",
    "LivePortraitWarpingSpade",
)
# GPU qualification evidence (VERIFICATION.md): what was exercised, where, when. Preflight
# reports a request no record covers as unqualified; that alone never refuses it. Hardware is a
# class (architecture, compute capability, VRAM class), never a model name. Each listed field
# holds the values exercised; max_mappings and max_sources bound the counts per request and
# per mapping (sources with a positive weight).
QUALIFICATIONS = (
    {
        "id": "baseline-2026-11-07",
        "date": "2026-11-07",
        "source": "VERIFICATION.md: Verification on 2026-11-07; Detection reference and "
        "best-match qualification on 2026-11-07",
        "fusion_revision": REVISION,
        "hardware": {"architecture": "Maxwell", "compute_capability": "5.2", "vram_class": "24 GB"},
        "modes": ["preview", "render"],
        "detector": ["Yolov8"],
        "swap_model": ["Inswapper128"],
        "restorer": [None, "GFPGAN-v1.4"],
        "expression": [False, True],
        "occluder": [False],
        "xseg": [False],
        "max_mappings": 1,
        "max_sources": 1,
    },
    {
        "id": "mappings-2026-11-08",
        "date": "2026-11-08",
        "source": "VERIFICATION.md: Feature qualification on 2026-11-08",
        "fusion_revision": REVISION,
        "hardware": {"architecture": "Maxwell", "compute_capability": "5.2", "vram_class": "24 GB"},
        "modes": ["preview", "render"],
        "detector": ["Yolov8"],
        "swap_model": ["Inswapper128"],
        "restorer": [None, "GFPGAN-v1.4"],
        "expression": [False],
        "occluder": [False],
        "xseg": [False],
        "max_mappings": 3,
        "max_sources": 2,
    },
    {
        "id": "sources-2026-11-08",
        "date": "2026-11-08",
        "source": "VERIFICATION.md: Feature qualification on 2026-11-08",
        "fusion_revision": REVISION,
        "hardware": {"architecture": "Maxwell", "compute_capability": "5.2", "vram_class": "24 GB"},
        "modes": ["preview"],
        "detector": ["Yolov8"],
        "swap_model": ["Inswapper128"],
        "restorer": [None],
        "expression": [False],
        "occluder": [False],
        "xseg": [False],
        "max_mappings": 1,
        "max_sources": 4,
    },
    {
        "id": "ghostface-2026-11-08",
        "date": "2026-11-08",
        "source": "VERIFICATION.md: Feature qualification on 2026-11-08",
        "fusion_revision": REVISION,
        "hardware": {"architecture": "Maxwell", "compute_capability": "5.2", "vram_class": "24 GB"},
        "modes": ["preview"],
        "detector": ["Yolov8"],
        "swap_model": ["GhostFace-v1"],
        "restorer": [None],
        "expression": [False],
        "occluder": [False],
        "xseg": [False],
        "max_mappings": 1,
        "max_sources": 2,
    },
)
# Render segments (docs/explanation/video-processing.md, Segments). Selections that load models or
# choose the detection path hold for the whole render, so a segment cannot set them.
MAX_SEGMENTS = 100
SEGMENT_CHANGES = ("mappings", "parameters", "controls")
SEGMENT_KEYS = {"id", "start_frame", "end_frame", "passthrough", *SEGMENT_CHANGES}
RENDER_WIDE = {
    "SwapModelSelection",
    "DetectorModelSelection",
    "FaceTrackingEnableToggle",
    "KPSSmoothingEnableToggle",
}


class ConfigError(Exception):
    """The workstation configuration file is missing, malformed or inconsistent."""


class Refusal(ValueError):
    """A refused input with a stable `code` that preflight reports. `normalize` marks media that
    normalize_media can make acceptable."""

    def __init__(self, code, message, normalize=False):
        super().__init__(message)
        self.code, self.normalize = code, normalize


@dataclass(frozen=True)
class Settings:
    path: Path
    base: Path
    app: Path
    state: Path
    gpu: str
    runner: tuple
    lock_dir: Path

    def effective(self):
        return {
            "version": 1,
            "media_root": str(self.base),
            "fusion_dir": str(self.app),
            "state_dir": str(self.state),
            "gpu": self.gpu,
            "runner": list(self.runner),
            "lock_dir": str(self.lock_dir),
        }

    @property
    def sha256(self):
        """Identity of the effective, resolved settings, recorded with each submitted job."""
        canonical = json.dumps(self.effective(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def config_path():
    """VM_CONFIG, else remote-config.json in the deployment root that holds this code.

    The root is two levels above a release (<root>/releases/<release>/common.py) and one
    level above a legacy flat <root>/remote/ directory or a source checkout's remote/.
    """
    override = os.environ.get("VM_CONFIG")
    if override is not None:
        if not os.path.isabs(override):
            raise ConfigError(f"VM_CONFIG must be an absolute path, not {override!r}")
        return Path(override)
    here = Path(__file__).resolve().parent
    root = here.parent.parent if here.parent.name == "releases" else here.parent
    return root / CONFIG_NAME


def _directory(path, key, value, must_exist=True):
    if not isinstance(value, str) or not os.path.isabs(value):
        raise ConfigError(f"{path}: {key} must be an absolute path string, not {value!r}")
    resolved = Path(value).resolve()
    if must_exist and not resolved.is_dir():
        raise ConfigError(f"{path}: {key} {value} is not an existing directory")
    return resolved


def load_config(path=None):
    """Read and validate the remote configuration; raise ConfigError naming file, key and fix."""
    path = Path(path) if path is not None else config_path()
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise ConfigError(
            f"Remote configuration not found at {path}; create it from "
            "config/remote-config.example.json (see docs/reference/configuration.md)"
        ) from None
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path}: cannot read JSON ({error})") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the configuration must be a JSON object")
    unknown = sorted(set(raw) - CONFIG_KEYS)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {', '.join(unknown)}; check for typos")
    if raw.get("version") != 1:
        raise ConfigError(f"{path}: version must be 1")
    if "media_root" not in raw:
        raise ConfigError(f"{path}: media_root is required (absolute workstation media directory)")
    base = _directory(path, "media_root", raw["media_root"])
    app = raw.get("fusion_dir", str(base / "tools/visomaster-fusion"))
    app = _directory(path, "fusion_dir", app)
    # Layout files are read only by jobs that need Fusion settings (see layouts), so a missing
    # one cannot block normalization or status; health reports it as a tracked app change.
    if not (app / ".git").exists():
        raise ConfigError(f"{path}: fusion_dir {app} is not a Fusion checkout (missing .git)")
    state = raw.get("state_dir", str(base / "runtime/visomaster-mcp"))
    state = _directory(path, "state_dir", state, must_exist=False)
    # Inside media_root or elsewhere; only its uploads and job artifacts are inputs (input_path).
    if base.is_relative_to(state):
        raise ConfigError(f"{path}: state_dir must not be media_root or contain it ({base})")
    if state.is_relative_to(app):
        raise ConfigError(f"{path}: state_dir must not lie inside fusion_dir {app}")
    gpu = raw.get("gpu", "0")
    # Index only: "0" and a UUID naming the same device would otherwise use different locks.
    if not isinstance(gpu, str) or not re.fullmatch(r"0|[1-9][0-9]*", gpu):
        raise ConfigError(
            f'{path}: gpu must be a device index string such as "0" '
            "(no leading zeros, UUIDs or lists)"
        )
    runner = raw.get("runner")
    if isinstance(runner, str):
        runner = [runner]
    if not isinstance(runner, list) or not runner or not all(isinstance(a, str) for a in runner):
        raise ConfigError(
            f"{path}: runner is required: a path, or a list of arguments whose first element "
            "is an absolute executable (see docs/reference/configuration.md, Runner)"
        )
    program = Path(runner[0])
    if not program.is_absolute() or not program.is_file() or not os.access(program, os.X_OK):
        raise ConfigError(f"{path}: runner {runner[0]!r} must be an absolute path to an executable")
    lock = raw.get("lock_dir")
    try:
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as error:
        raise ConfigError(f"{path}: cannot create state_dir {state} ({error})") from None
    lock_dir = state if lock is None else _directory(path, "lock_dir", lock)
    if not os.access(lock_dir, os.W_OK | os.X_OK):
        raise ConfigError(f"{path}: lock_dir {lock_dir} must be writable")
    settings = Settings(path, base, app, state, gpu, tuple(runner), lock_dir)
    for variable, field in LEGACY.items():
        value = os.environ.get(variable)
        if value is None:
            continue
        effective = getattr(settings, field)
        if (value != effective) if field == "gpu" else (Path(value).resolve() != effective):
            raise ConfigError(
                f"{variable}={value} is no longer read; remove it from the workstation "
                f"environment and set the value in {path}"
            )
    return settings


# Import never raises: control.py reports CONFIG_ERROR from health and refuses other actions.
try:
    SETTINGS, CONFIG_ERROR = load_config(), None
except ConfigError as error:
    SETTINGS, CONFIG_ERROR = None, str(error)
BASE = SETTINGS.base if SETTINGS else None
APP = SETTINGS.app if SETTINGS else None
STATE = SETTINGS.state if SETTINGS else None
GPU = SETTINGS.gpu if SETTINGS else None
RUNNER = list(SETTINGS.runner) if SETTINGS else None
LOCK_DIR = SETTINGS.lock_dir if SETTINGS else None


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def job_path(job_id):
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise ValueError("job_id must be 1–64 letters, digits, underscores or hyphens")
    return STATE / "jobs" / job_id


def managed(path):
    """Whether a resolved path inside STATE is an upload or a job artifact.

    The rest of the state tree (requests, logs, locks, partial uploads, staging and trash) is
    never an input, wherever state_dir lies.
    """
    parts = path.relative_to(STATE).parts
    if len(parts) == 2 and parts[0] == "uploads":
        return not parts[1].endswith(".partial")
    return (
        len(parts) >= 4
        and parts[0] == "jobs"
        and JOB_ID.fullmatch(parts[1]) is not None
        and parts[2] == "artifacts"
    )


def require_completed(job_id):
    """Refuse the artifacts of a job that has not completed: a queued or running job is still
    writing them, and a failed or cancelled one may have left them partial.

    Submission checks under the submit lock. A completed state is never rewritten, and
    retention neither deletes nor (by deletion and resubmission) replaces a job whose artifacts
    a queued or running job uses, so the answer still holds when that job's worker revalidates.
    """
    try:
        status = read_json(STATE / "jobs" / job_id / "state.json")["status"]
    except (OSError, ValueError, KeyError, TypeError):
        status = None
    if status == "completed":
        return
    if status in TERMINAL:
        raise Refusal(
            "artifact_job_incomplete",
            f"Job {job_id} did not complete ({status}); only completed jobs' artifacts are "
            "inputs, so rerun it under a new job_id and use that job's artifacts",
        )
    if status is None:
        raise Refusal(
            "artifact_job_incomplete",
            f"Job {job_id} has no readable state; its artifacts are not inputs",
        )
    raise Refusal(
        "artifact_job_incomplete",
        f"Job {job_id} is {status}; wait for it to complete before using its artifacts",
    )


def input_path(value):
    """A media input: a regular file in media_root or a managed file of state_dir.

    The path is resolved first, so traversal and symlinks (in the file name or any ancestor)
    are judged by their target and cannot extend either root. A job artifact is an input only
    once the job holding it has completed.
    """
    path = Path(value)
    if not path.is_absolute():
        raise Refusal("path_not_absolute", "Media paths must be absolute workstation paths")
    path = path.resolve(strict=True)
    parts = ()
    if STATE is not None and path.is_relative_to(STATE):
        inside = managed(path)
        parts = path.relative_to(STATE).parts
    else:
        inside = path.is_relative_to(BASE)
    if not inside or not path.is_file():
        raise Refusal(
            "path_outside_roots",
            f"Media must be a regular file inside {BASE}, an upload or a job artifact",
        )
    if parts[0:1] == ("jobs",):
        require_completed(parts[1])
    return path


# Digests a job's supervisor took before GPU admission, keyed by the file's identity and times,
# so its worker reuses them for unchanged files (docs/explanation/video-processing.md,
# Checkpoints). None elsewhere.
DIGESTS = None


def file_identity(stream):
    """Device, inode, size and modification and change times of an open file. Rewriting the
    bytes changes the change time, which, unlike the modification time, cannot be set back."""
    info = os.fstat(stream.fileno())
    return f"{info.st_dev}:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}:{info.st_ctime_ns}"


def file_sha256(path):
    with open(path, "rb") as stream:
        identity = file_identity(stream)
        if DIGESTS is not None and identity in DIGESTS:
            return DIGESTS[identity]
        digest = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        if DIGESTS is not None and file_identity(stream) == identity:
            DIGESTS[identity] = digest.hexdigest()  # Not when it changed while being read.
    return digest.hexdigest()


REANALYZE = "Analyze it again and resubmit with the new fingerprint."
# A resume repeats the fingerprints of the render it continues, so a new one cannot help.
RESUBMIT = (
    "The checkpointed chunks cannot be reused; submit a fresh render with submit_swap instead."
)


def reference_path(value, expected=None, remedy=REANALYZE):
    """A face reference input; with `expected`, its bytes must still have that SHA-256.

    Submission checks fingerprints a caller took from analyze_references, and the worker checks
    those recorded at submission, so a changed file is reported, with `remedy`, rather than
    silently used.
    """
    path = input_path(value)
    if path.stat().st_size > MAX_REFERENCE_BYTES:
        raise Refusal(
            "reference_too_large", f"Face references must be images up to 256 MiB: {value}"
        )
    if expected is not None and file_sha256(path) != expected:
        raise Refusal(
            "reference_changed",
            f"Reference {value} no longer has SHA-256 {expected}: its content changed after it "
            f"was fingerprinted. {remedy}",
        )
    return path


def digests(values, count, key):
    """Expected SHA-256 values, one per reference, or Nones when omitted. A null entry pins
    nothing; submission records the file's SHA-256 unless its weight is 0."""
    if values is None:
        return [None] * count
    if (
        not isinstance(values, list)
        or len(values) != count
        or not all(v is None or isinstance(v, str) and SHA256.fullmatch(v) for v in values)
    ):
        raise ValueError(f"{key} must list one 64-hex-digit SHA-256 per reference, or null")
    return values


def positive(mapping):
    """Indexes of the sources a mapping reads: those with a positive (or omitted) weight. A
    source of weight 0 is excluded without being read, hashed or required to exist."""
    count = len(mapping["source_faces"])
    weights = mapping.get("source_weights")
    weights = weights if isinstance(weights, list) and len(weights) == count else [1] * count
    return [i for i, w in enumerate(weights) if type(w) in (int, float) and w > 0]


def validate_weights(weights, count):
    """Explicit source weights: finite, nonnegative, one per source, with a positive sum."""
    if weights is None:
        return
    if not isinstance(weights, list) or len(weights) != count:
        raise ValueError("source_weights must list one weight per source image")
    if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("source_weights must be finite nonnegative numbers")
    total = sum(weights)
    if not (math.isfinite(total) and total > 0):
        raise ValueError("source_weights must have a positive, finite sum")


def artifact_path(job_id, name):
    directory = job_path(job_id) / "artifacts"
    path = (directory / name).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Artifact path escapes the job")
    return path


def ffmpeg_input(path, ffprobe=False):
    """Arguments that open `path` as one confined FFmpeg or FFprobe input.

    Libavformat refuses a detected format outside DEMUXERS before reading its header, so
    playlists and concat lists cannot open files input_path never checked. Only local files
    are readable. FFprobe also accepts image2, which some JPEGs are detected as, with pattern
    expansion off; the FFmpeg CLI rejects -pattern_type for other demuxers, so it never uses it.
    """
    formats = DEMUXERS + (",image2" if ffprobe else "")
    options = ["-protocol_whitelist", "file", "-format_whitelist", formats]
    if ffprobe:
        options += ["-pattern_type", "none"]
    return [*options, "-i", str(path)]


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json"]
        + ffmpeg_input(path, ffprobe=True),
        capture_output=True,
        timeout=30,
    )
    refused = re.search(rb"^\[(\w+) @ [^]]*\] Format not on whitelist", result.stderr, re.M)
    if refused:
        raise Refusal(
            "unsupported_container",
            f"Unsupported container '{refused[1].decode()}'; use a self-contained video (MP4, "
            "MOV, Matroska, WebM, AVI or MPEG-TS) or image (PNG, JPEG, WebP or BMP), not a "
            "playlist or concat list",
        )
    result.check_returncode()
    return json.loads(result.stdout)


def fusion_compatibility(app):
    """(revision, reason) for the Fusion checkout; reason is None when GPU jobs may load it.

    Health and worker startup share this check: the checkout must be at REVISION with no
    tracked changes (staged or not) under app/. Untracked files and other directories are
    not checked.
    """

    def git(*arguments):
        try:
            return subprocess.run(
                ["git", "-C", str(app), *arguments],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise OSError(error.stderr.strip() or str(error)) from None

    try:
        revision = git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"Cannot read the Fusion revision in {app} ({error})"
    if revision != REVISION:
        return revision, (
            f"Fusion revision changed to {revision}, but the adapter supports {REVISION}; "
            "qualify the adapter before rendering"
        )
    try:
        changed = git("diff", "--no-ext-diff", "HEAD", "--", "app")
    except (OSError, subprocess.SubprocessError) as error:
        return revision, f"Cannot check Fusion app for tracked changes ({error})"
    if changed:
        return revision, "Fusion app has tracked local changes; qualify them before rendering"
    return revision, None


def layouts():
    """Read literal metadata without importing Qt actions or constructing a GUI."""
    control, parameters = {}, {}
    for name in LAYOUT_NAMES:
        path = APP / f"app/ui/widgets/{name}_layout_data.py"
        try:
            tree = ast.parse(path.read_text())
        except OSError as error:
            raise ValueError(
                f"Cannot read Fusion layout {path} ({error.strerror}); fusion_dir must be a "
                "Fusion checkout (see health_check)"
            ) from None
        destination = control if name == "settings" else parameters
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or not isinstance(value, ast.Dict):
                    continue
                fields = {}
                for field, item in zip(value.keys, value.values):
                    if isinstance(field, ast.Constant) and field.value in {
                        "default",
                        "options",
                        "min_value",
                        "max_value",
                        "help",
                    }:
                        try:
                            fields[field.value] = ast.literal_eval(item)
                        except (ValueError, TypeError):
                            pass
                if "default" in fields:
                    default = fields["default"]
                    if "DecimalSlider" in key.value:
                        fields["default"] = float(default)
                    elif "Slider" in key.value:
                        fields["default"] = int(default)
                    destination[key.value] = fields
    control["ProvidersPrioritySelection"] = {"default": "CUDA"}
    parameters["DFMModelSelection"] = {"default": ""}
    return control, parameters


def validate_settings(values, definitions, allowed):
    if not isinstance(values, dict):
        raise ValueError("Settings must be an object")
    for key, value in values.items():
        if key not in allowed or key not in definitions:
            raise Refusal("setting_unsupported", f"Unsupported setting {key}; use get_settings")
        spec = definitions[key]
        default = spec["default"]
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, (float, int)):
            valid = type(value) in (float, int) and math.isfinite(value)
            if isinstance(default, int):
                valid = valid and isinstance(value, int)
            if valid:
                valid = (
                    float(spec.get("min_value", -math.inf))
                    <= value
                    <= float(spec.get("max_value", math.inf))
                )
        else:
            valid = isinstance(value, str)
        if not valid or ("options" in spec and value not in spec["options"]):
            raise Refusal("setting_invalid", f"Invalid value for {key}")
        if key == "SwapModelSelection" and value in {"DeepFaceLive (DFM)", "AlphaFace"}:
            raise Refusal(
                "swapper_unsupported", "This adapter supports embedding-based swappers only"
            )


def job_priority(request):
    """The scheduling class of a GPU job request, or None for normalization."""
    kind = request.get("kind")
    if kind == "normalize":
        return None
    default = PRIORITIES.get(kind, DEFAULT_PRIORITY)
    if kind == "compare" and request.get("max_frames") is not None:
        default = "normal"  # A clip comparison renders up to MAX_VARIANTS clips.
    priority = request.get("priority", default)
    if priority not in CLASSES:
        raise ValueError(f"priority must be one of {', '.join(CLASSES)}")
    return priority


def validate_request(request):
    """Validate a job request; normalization alone never reads Fusion's layout metadata."""
    if request.get("kind") not in {
        "detect",
        "preview",
        "render",
        "normalize",
        "analyze",
        "compare",
    }:
        raise ValueError("Unknown job kind")
    if request["kind"] == "analyze":
        job_priority(request)
        return validate_analysis(request)
    if request["kind"] == "compare":
        job_priority(request)
        return validate_comparison(request)
    input_path(request["media_path"])
    fps = request.get("fps")
    if fps is not None:
        try:
            rate = Fraction(str(fps)) if type(fps) in (int, float, str) else None
        except (ValueError, ZeroDivisionError):
            rate = None
        if rate is None or not 1 <= rate <= 240 or rate.denominator > 1001:
            raise ValueError("fps must be a rate from 1 to 240, such as 25, 29.97 or 30000/1001")
    if request["kind"] == "normalize":
        # FFmpeg-only: GPU fields would be ignored, so they are refused rather than validated.
        extra = sorted(set(request) - NORMALIZE_KEYS)
        if extra:
            raise ValueError(
                f"Normalization accepts only media_path and fps, not {', '.join(extra)}"
            )
        return request
    job_priority(request)
    return validate_fusion_request(request)


def validate_fusion_request(request):
    """Frame selection, settings and mappings of a detect, preview or render request."""
    for key in ("frame", "start_frame"):
        value = request.get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    frames, samples = request.get("frames"), request.get("sample_frames")
    if frames is not None or samples is not None:
        if request["kind"] != "detect":
            raise ValueError("frames and sample_frames apply only to detection")
        if frames is not None and samples is not None:
            raise ValueError("Use either frames or sample_frames")
        if request.get("frame", 0):
            raise ValueError("Use frame or multi-frame detection, not both")
    threshold = request.get("similarity_threshold")
    if threshold is not None and frames is None and samples is None:
        raise ValueError("similarity_threshold applies only to multi-frame detection")
    if samples is not None and (type(samples) is not int or not 1 <= samples <= MAX_DETECT_FRAMES):
        raise ValueError(f"sample_frames must be 1–{MAX_DETECT_FRAMES}")
    if frames is not None and (
        not isinstance(frames, list)
        or not 1 <= len(frames) <= MAX_DETECT_FRAMES
        or any(type(f) is not int or f < 0 for f in frames)
        or len(set(frames)) != len(frames)
    ):
        raise ValueError(f"frames must list 1–{MAX_DETECT_FRAMES} distinct nonnegative integers")
    count = request.get("max_frames")
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("max_frames must be positive")
    # Recipe instantiation records these; the worker hashes the media before decoding it.
    media = request.get("media_sha256")
    if media is not None and not (isinstance(media, str) and SHA256.fullmatch(media)):
        raise ValueError("media_sha256 must be a 64-hex-digit SHA-256")
    if not isinstance(request.get("recipe", {}), dict):
        raise ValueError("recipe must be an object")
    resume = request.get("resume")
    if resume is not None and (
        request["kind"] != "render"
        or not isinstance(resume, dict)
        or not isinstance(resume.get("from_job_id"), str)
    ):
        raise ValueError("resume links a render to the attempt it resumes; use resume_render")
    control, parameters = layouts()
    if threshold is not None:
        # Identities are grouped on SimilarityThresholdSlider's scale, within its limits.
        spec = parameters.get("SimilarityThresholdSlider", {})
        low, high = int(spec.get("min_value", 1)), int(spec.get("max_value", 100))
        if type(threshold) is not int or not low <= threshold <= high:
            raise ValueError(f"similarity_threshold must be an integer from {low} to {high}")
    validate_settings(request.get("controls", {}), control, CONTROLS)
    mappings = request.get("mappings", [])
    if request["kind"] not in {"detect", "normalize"} and not mappings:
        raise ValueError("At least one explicit target/source mapping is required")
    if len(mappings) > 20:
        raise ValueError("At most 20 mappings are supported")
    remedy = REANALYZE if request.get("resume") is None else RESUBMIT
    for mapping in mappings:
        target = mapping.get("target_sha256")
        if target is not None and not (isinstance(target, str) and SHA256.fullmatch(target)):
            raise ValueError("target_sha256 must be a 64-hex-digit SHA-256")
        reference_path(mapping["target_face"], target, remedy)
        sources = mapping["source_faces"]
        if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_REFERENCES:
            raise ValueError("Each mapping requires 1–20 source images")
        expected = digests(mapping.get("source_sha256"), len(sources), "source_sha256")
        validate_weights(mapping.get("source_weights"), len(sources))
        read = positive(mapping)
        for number, (source, digest) in enumerate(zip(sources, expected)):
            if number in read:
                reference_path(source, digest, remedy)
            elif not (isinstance(source, str) and os.path.isabs(source)):
                message = f"Source images must be absolute workstation paths: {source!r}"
                raise Refusal("path_not_absolute", message)
        validate_settings(mapping.get("parameters", {}), parameters, PARAMETERS)
    if "report" in request:
        if request["kind"] != "render":
            raise ValueError("report options apply only to renders")
        validate_report(request["report"], request, len(mappings))
    if "segments" in request:
        validate_segments(request, len(mappings), control, parameters)
    return request


def variant_request(request, variant, frame=None):
    """The single-variant preview or render request of one comparison variant.

    The variant's parameters are merged over every mapping's and its controls over the common
    controls; everything else is the comparison's, so variants differ only in their overrides.
    A still comparison gives a preview of `frame` (default its lowest frame, the first compared),
    a clip comparison a render of the same range.
    """
    overrides = variant.get("parameters", {})
    mappings = [
        {**mapping, "parameters": {**mapping.get("parameters", {}), **overrides}}
        for mapping in request["mappings"]
    ]
    single = {
        "kind": "preview" if "frames" in request else "render",
        "media_path": request["media_path"],
        "mappings": mappings,
        "controls": {**request.get("controls", {}), **variant.get("controls", {})},
    }
    if "frames" in request:
        single["frame"] = min(request["frames"]) if frame is None else frame
    else:
        single.update(start_frame=request.get("start_frame", 0), max_frames=request["max_frames"])
    return single


def validate_comparison(request):
    """Limits, distinct variants and every variant's merged request, validated in full before
    anything is queued."""
    extra = sorted(set(request) - COMPARE_KEYS)
    if extra:
        raise ValueError(f"Comparisons do not accept {', '.join(extra)}")
    frames, count = request.get("frames"), request.get("max_frames")
    if (frames is None) == (count is None):
        raise ValueError("Compare either explicit frames or a clip of max_frames frames")
    if frames is not None:
        if "start_frame" in request:
            raise ValueError("start_frame applies only to clip comparisons; list frames instead")
        if (
            not isinstance(frames, list)
            or not 1 <= len(frames) <= MAX_COMPARE_FRAMES
            or any(type(f) is not int or f < 0 for f in frames)
            or len(set(frames)) != len(frames)
        ):
            raise ValueError(
                f"frames must list 1–{MAX_COMPARE_FRAMES} distinct nonnegative integers"
            )
    elif type(count) is not int or not 1 <= count <= MAX_COMPARE_CLIP:
        raise ValueError(f"max_frames of a clip comparison must be 1–{MAX_COMPARE_CLIP}")
    mappings = request.get("mappings")
    if not isinstance(mappings, list) or not all(isinstance(m, dict) for m in mappings):
        raise ValueError("mappings must list target/source mappings")
    if not isinstance(request.get("controls", {}), dict):
        raise ValueError("controls must be an object")
    variants = request.get("variants")
    if not isinstance(variants, list) or not MIN_VARIANTS <= len(variants) <= MAX_VARIANTS:
        raise ValueError(f"variants must list {MIN_VARIANTS}–{MAX_VARIANTS} named variants")
    names, overrides = set(), set()
    for variant in variants:
        if not isinstance(variant, dict) or "name" not in variant or set(variant) - VARIANT_KEYS:
            raise ValueError("Each variant has a name and optional parameters and controls")
        name = variant["name"]
        if not isinstance(name, str) or not VARIANT_NAME.fullmatch(name):
            raise ValueError(
                "Variant names are 1–20 letters, digits, underscores or hyphens, starting with "
                "a letter or digit"
            )
        if name.lower() in names:
            raise ValueError(f"Duplicate variant name {name} (names ignore case)")
        names.add(name.lower())
        if not all(isinstance(variant.get(key, {}), dict) for key in ("parameters", "controls")):
            raise ValueError(f"Variant {name}: parameters and controls must be objects")
        settings = [variant.get("parameters", {}), variant.get("controls", {})]
        settings = json.dumps(settings, sort_keys=True)
        if settings in overrides:
            raise ValueError(f"Variant {name} repeats another variant's overrides")
        overrides.add(settings)
    for number, variant in enumerate(variants):
        single = variant_request(request, variant)
        if number:
            # Every variant reads the same references, whose pinned bytes the first one checked;
            # hashing them again per variant would hold the submit lock for nothing.
            for mapping in single["mappings"]:
                mapping.pop("target_sha256", None)
                mapping.pop("source_sha256", None)
        try:
            validate_fusion_request(single)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            message = str(error) if isinstance(error, ValueError) else f"invalid request ({error})"
            raise ValueError(f"Variant {variant['name']}: {message}") from None
    return request


def validate_segments(request, count, control, parameters):
    """A render's segments: half-open source-frame ranges [start_frame, end_frame), listed in
    ascending order without overlap, each with a unique id and either passthrough or changes
    (mappings, parameters, controls). A segment must intersect the rendered frames; the worker
    checks again once the video's length is known."""
    segments = request["segments"]
    if request["kind"] != "render":
        raise ValueError("segments apply only to renders")
    if not isinstance(segments, list) or not 1 <= len(segments) <= MAX_SEGMENTS:
        raise ValueError(
            f"segments must list 1–{MAX_SEGMENTS} segments; omit it for a uniform render"
        )
    first, limit = request.get("start_frame", 0), request.get("max_frames")
    names, previous = set(), None
    for segment in segments:
        if not isinstance(segment, dict) or not {"id", "start_frame", "end_frame"} <= set(segment):
            raise ValueError("Each segment needs id, start_frame and end_frame")
        unknown = sorted(set(segment) - SEGMENT_KEYS)
        if unknown:
            raise ValueError(f"Unknown segment fields {', '.join(unknown)}")
        name, low, high = segment["id"], segment["start_frame"], segment["end_frame"]
        if not isinstance(name, str) or not JOB_ID.fullmatch(name):
            raise ValueError(
                "Segment ids are 1–64 letters, digits, underscores or hyphens, starting with a "
                "letter or digit"
            )
        if name in names:
            raise ValueError(f"Segment id {name} is used twice")
        names.add(name)
        if type(low) is not int or type(high) is not int or not 0 <= low < high:
            raise ValueError(
                f"Segment {name} needs integer frames with 0 <= start_frame < end_frame "
                "(end_frame is the first frame after it)"
            )
        if previous is not None and low < previous["end_frame"]:
            if low < previous["start_frame"]:
                raise ValueError("Segments must be listed in ascending start_frame order")
            raise ValueError(
                f"Segments {previous['id']} [{previous['start_frame']}, {previous['end_frame']}) "
                f"and {name} [{low}, {high}) overlap; a frame belongs to at most one segment"
            )
        previous = segment
        if high <= first:
            raise ValueError(
                f"Segment {name} [{low}, {high}) ends before the rendered frames start at {first}"
            )
        changes = [key for key in SEGMENT_CHANGES if key in segment]
        if "passthrough" in segment:
            if segment["passthrough"] is not True:
                raise ValueError(f"Segment {name}: passthrough must be true when given")
            if changes:
                raise ValueError(
                    f"Segment {name} is passthrough, so it cannot also set {', '.join(changes)}"
                )
            continue
        if not any(segment[key] for key in changes):
            raise ValueError(
                f"Segment {name} changes nothing; give passthrough, mappings, parameters or controls"
            )
        chosen = segment.get("mappings")
        if chosen is not None and (
            not isinstance(chosen, list)
            or not chosen
            or any(type(i) is not int or not 0 <= i < count for i in chosen)
            or len(set(chosen)) != len(chosen)
        ):
            raise ValueError(
                f"Segment {name} mappings must list distinct indices into mappings (0–{count - 1})"
            )
        for key, specs, allowed in (
            ("parameters", parameters, PARAMETERS),
            ("controls", control, CONTROLS),
        ):
            values = segment.get(key, {})
            fixed = sorted(set(values) & RENDER_WIDE) if isinstance(values, dict) else []
            if fixed:
                raise ValueError(
                    f"Segment {name} cannot set {', '.join(fixed)}: model selections and tracking "
                    "toggles hold for the whole render"
                )
            validate_settings(values, specs, allowed)
    if limit is not None:
        segment_plan(segments, first, first + limit, count)


def segment_plan(segments, start, end, count):
    """Runs covering the rendered frames [start, end) in order: each segment clipped to them,
    and the gaps between, which use the base mappings, parameters and controls.

    A run has its segment's id (None for a gap), range, passthrough flag, active mapping indices
    and the segment's parameter and control overrides. Raises ValueError for a segment outside
    the frames, or when every frame is passthrough, which leaves nothing to swap.
    """
    runs, cursor = [], start

    def run(segment, low, high):
        segment = segment or {}
        passthrough = segment.get("passthrough", False)
        return {
            "segment": segment.get("id"),
            "start_frame": low,
            "end_frame": high,
            "passthrough": passthrough,
            "mappings": [] if passthrough else list(segment.get("mappings", range(count))),
            "parameters": segment.get("parameters", {}),
            "controls": segment.get("controls", {}),
        }

    for segment in segments:
        low, high = max(segment["start_frame"], start), min(segment["end_frame"], end)
        if low >= high:
            raise Refusal(
                "segment_outside_range",
                f"Segment {segment['id']} [{segment['start_frame']}, {segment['end_frame']}) lies "
                f"outside the rendered frames [{start}, {end})",
            )
        if low > cursor:
            runs.append(run(None, cursor, low))
        runs.append(run(segment, low, high))
        cursor = high
    if cursor < end:
        runs.append(run(None, cursor, end))
    if all(entry["passthrough"] for entry in runs):
        raise Refusal(
            "all_passthrough",
            f"Every rendered frame [{start}, {end}) is passthrough, so there is nothing to swap; "
            "use the source media, or render frames outside the passthrough segments",
        )
    return runs


def validate_analysis(request):
    """References, thresholds and settings of a reference-analysis request."""
    extra = sorted(set(request) - ANALYZE_KEYS)
    if extra:
        raise ValueError(f"Reference analysis does not accept {', '.join(extra)}")
    references = request.get("references")
    if (
        not isinstance(references, list)
        or not 1 <= len(references) <= MAX_REFERENCES
        or not all(isinstance(path, str) for path in references)
    ):
        raise ValueError(f"references must list 1–{MAX_REFERENCES} image paths")
    expected = digests(request.get("reference_sha256"), len(references), "reference_sha256")
    for path, digest in zip(references, expected):
        reference_path(path, digest)
    thresholds = request.get("thresholds")
    thresholds = {} if thresholds is None else thresholds
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be an object")
    for key, value in thresholds.items():
        if key not in THRESHOLDS:
            raise ValueError(f"Unknown threshold {key}; use {', '.join(THRESHOLDS)}")
        kind, low, high = THRESHOLDS[key]
        valid = type(value) in ((int,) if kind is int else (int, float))
        if not valid or not math.isfinite(value) or not low <= value <= high:
            number = "an integer" if kind is int else "a number"
            raise ValueError(f"Threshold {key} must be {number} from {low} to {high}")
    control, parameters = layouts()
    validate_settings(request.get("controls", {}), control, CONTROLS)
    if request.get("swap_model") is not None:
        validate_settings({"SwapModelSelection": request["swap_model"]}, parameters, PARAMETERS)
    return request


def validate_report(options, request, mappings):
    """Swap report options of a render. Expected-visible ranges are half-open source-frame
    ranges [start_frame, end_frame) inside the requested range; the worker also checks that
    they end within the video."""
    if not isinstance(options, dict):
        raise ValueError("report must be an object")
    unknown = sorted(set(options) - REPORT_KEYS)
    if unknown:
        raise ValueError(f"Unknown report options {', '.join(unknown)}")
    images = options.get("review_images", 0)
    if type(images) is not int or not 0 <= images <= MAX_REVIEW_IMAGES:
        raise ValueError(f"review_images must be an integer from 0 to {MAX_REVIEW_IMAGES}")
    minimum = options.get("min_coverage")
    if minimum is not None and (
        type(minimum) not in (int, float) or not math.isfinite(minimum) or not 0 < minimum <= 1
    ):
        raise ValueError("min_coverage must be a number greater than 0 and at most 1")
    ranges = options.get("expected_visible", [])
    if not isinstance(ranges, list) or len(ranges) > MAX_EXPECTED_RANGES:
        raise ValueError(f"expected_visible must list at most {MAX_EXPECTED_RANGES} ranges")
    first = request.get("start_frame", 0)
    count = request.get("max_frames")
    for item in ranges:
        if not isinstance(item, dict) or set(item) != {"mapping", "start_frame", "end_frame"}:
            raise ValueError("Each expected_visible range needs mapping, start_frame and end_frame")
        if any(type(value) is not int for value in item.values()):
            raise ValueError("expected_visible mapping and frames must be integers")
        if not 0 <= item["mapping"] < mappings:
            raise ValueError(f"expected_visible mapping {item['mapping']} does not exist")
        if not first <= item["start_frame"] < item["end_frame"] or (
            count is not None and item["end_frame"] > first + count
        ):
            raise ValueError(
                "expected_visible ranges must satisfy start_frame < end_frame (exclusive) "
                "within the requested frames"
            )


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def initial_controls(specs, requested):
    """Fusion's control defaults, then the adapter's, then the request's validated controls."""
    control = {k: v["default"] for k, v in specs.items()}
    control.update(ADAPTER_CONTROLS, LandmarkDetectToggle=False, ForceSwapToggle=False)
    control.update(requested)
    return control


def effective_settings(request):
    """Every control and per-mapping parameter a swap runs with, defaults included."""
    control, parameters = layouts()
    defaults = {k: v["default"] for k, v in parameters.items()}
    return {
        "controls": initial_controls(control, request.get("controls", {})),
        "parameters": [{**defaults, **m.get("parameters", {})} for m in request["mappings"]],
    }


def model_text(node):
    """A models_data string: a literal, or an f-string over models_dir (APP/model_assets)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if not isinstance(node, ast.JoinedStr):
        return None
    parts = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append(value.value)
        elif (
            isinstance(value, ast.FormattedValue)
            and isinstance(value.value, ast.Name)
            and value.value.id == "models_dir"
            and value.conversion == -1
            and value.format_spec is None
        ):
            parts.append(str(APP / "model_assets"))
        else:
            return None
    return "".join(parts)


def model_catalog():
    """Fusion's declared models (name: path and declared SHA-256) and its swap-to-ArcFace and
    detector maps, read from models_data.py without importing Fusion; empty when unreadable."""
    models, maps = {}, {}
    try:
        tree = ast.parse((APP / "app/processors/models_data.py").read_text())
    except (OSError, SyntaxError):
        return models, maps
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name in ("arcface_mapping_model_dict", "detection_model_mapping"):
            try:
                maps[name] = ast.literal_eval(node.value)
            except ValueError:
                pass
        elif name == "models_list" and isinstance(node.value, ast.List):
            for item in node.value.elts:
                if isinstance(item, ast.Dict):
                    fields = {
                        key.value: model_text(value)
                        for key, value in zip(item.keys, item.values)
                        if isinstance(key, ast.Constant)
                    }
                    if fields.get("model_name"):
                        models[fields["model_name"]] = (
                            fields.get("local_path"),
                            fields.get("hash"),
                        )
    return models, maps


def model_choices(settings):
    """Each mapping's model selections in a swap's effective settings (effective_settings):
    detector, recognizer, swap model, restorer (None when off) and the toggles that load a
    model. Preflight checks, and provenance hashes, the files these select."""
    controls = settings["controls"]
    return [
        {
            "detector": controls.get("DetectorModelSelection"),
            "recognition": controls.get("RecognitionModelSelection"),
            "swap_model": values.get("SwapModelSelection"),
            "restorer": values.get("FaceRestorerTypeSelection")
            if values.get("FaceRestorerEnableToggle") is True
            else None,
            "expression": values.get("FaceExpressionEnableBothToggle") is True,
            "occluder": values.get("OccluderEnableToggle") is True,
            "xseg": values.get("DFLXSegEnableToggle") is True,
        }
        for values in settings["parameters"]
    ]


def required_models(choice):
    """Fusion model names a mapping's choices load (DETECTOR_MODELS, SWAPPER_MODELS,
    RESTORER_MODELS, the mask and expression models), and selections the tables lack."""
    names, unknown = set(), []

    def need(table, key, value):
        found = table.get(value) if isinstance(value, str) else None
        if found is None:
            unknown.append(f"{key}={value!r}")
        else:
            names.update((found,) if isinstance(found, str) else found)

    need(DETECTOR_MODELS, "DetectorModelSelection", choice["detector"])
    if isinstance(choice["recognition"], str):
        names.add(choice["recognition"])
    need(SWAPPER_MODELS, "SwapModelSelection", choice["swap_model"])
    if choice["restorer"] is not None:
        need(RESTORER_MODELS, "FaceRestorerTypeSelection", choice["restorer"])
    if choice["occluder"]:
        names.add("Occluder")
    if choice["xseg"]:
        names.add("XSeg")
    if choice["expression"]:
        names.update(EXPRESSION_MODELS)
    return names, unknown


def setting_variants(settings, segments=None):
    """(segment id, mapping indexes, settings) of a render's base settings, then of each segment
    that overrides parameters or controls, merged as the engine's Segments.settings merges them
    (the base mapping's parameters, then the segment's; the base controls, then the segment's).
    A segment can enable a restorer or mask the base settings leave off."""
    count = len(settings["parameters"])
    variants = [(None, list(range(count)), settings)]
    for segment in segments if isinstance(segments, list) else []:
        if not isinstance(segment, dict) or segment.get("passthrough"):
            continue
        parameters, controls = segment.get("parameters"), segment.get("controls")
        parameters = parameters if isinstance(parameters, dict) else {}
        controls = controls if isinstance(controls, dict) else {}
        if not (parameters or controls):
            continue
        chosen = segment.get("mappings")
        chosen = chosen if isinstance(chosen, list) else range(count)
        chosen = [i for i in chosen if type(i) is int and 0 <= i < count]
        variant = {
            "controls": {**settings["controls"], **controls},
            "parameters": [{**settings["parameters"][i], **parameters} for i in chosen],
        }
        variants.append((segment.get("id"), chosen, variant))
    return variants


def model_names(settings, segments=None):
    """Models a swap's effective settings, and its segments' overrides, load, from the same
    tables preflight checks: the detector, recognizer, swapper with its ArcFace models, and an
    enabled restorer, occluder, XSeg or expression restorer. A selection the tables lack adds
    nothing; at another Fusion revision provenance reports the revision change instead."""
    names = set()
    for _, _, variant in setting_variants(settings, segments):
        for choice in model_choices(variant):
            names |= required_models(choice)[0]
    return names


def model_identity(names, catalog):
    """Per model: its file, the SHA-256 Fusion declares and the SHA-256 of the file's bytes."""
    models, _ = catalog
    identity = {}
    for name in sorted(names):
        path, declared = models.get(name, (None, None))
        try:
            digest = file_sha256(path) if path else None
        except OSError:
            digest = None
        identity[name] = {"path": path, "declared_sha256": declared, "sha256": digest}
    return identity


def code_identity(directory=None):
    """A release directory's name and the SHA-256 of each adapter file (None once removed)."""
    directory = Path(directory) if directory else Path(__file__).resolve().parent
    try:
        files = {name: file_sha256(directory / name) for name in CODE}
    except OSError:
        files = None
    return {"release": directory.name, "code": files}


def recorded_inputs(request):
    """Each input a swap request reads keyed by its place, with the SHA-256 the request
    records; sources of weight 0 are not read, so they are not inputs."""
    inputs = {"media_path": {"path": request["media_path"], "sha256": request.get("media_sha256")}}
    for index, mapping in enumerate(request["mappings"]):
        place = f"mappings[{index}]"
        inputs[f"{place}.target_face"] = {
            "path": mapping["target_face"],
            "sha256": mapping.get("target_sha256"),
        }
        digests = mapping.get("source_sha256") or [None] * len(mapping["source_faces"])
        read = positive(mapping)  # The output does not depend on a source of weight 0.
        for number, (path, digest) in enumerate(zip(mapping["source_faces"], digests)):
            if number in read:
                inputs[f"{place}.source_faces[{number}]"] = {"path": path, "sha256": digest}
    return inputs


def current_digest(path):
    """SHA-256 of an allowed input's bytes now, or None if it is missing or not an input."""
    try:
        return file_sha256(input_path(path))
    except (OSError, ValueError):
        return None


def provenance(request, models=()):
    """What a swap request's output depends on, apart from the GPU, its driver and CUDA stack.

    The bytes of every input (at its resolved path) and of the selected and `models` model
    files, the effective settings, the Fusion revision, the adapter release and its code, and
    the configuration identity. Each call hashes every file in a streamed pass, so callers use
    it at deliberate boundaries (recipe creation, verification and instantiation), never per
    status poll. provenance_changes compares two records, so recorded work can be checked
    before it is reused.
    """
    settings = effective_settings(request)
    catalog = model_catalog()
    inputs = recorded_inputs(request)
    hashes = {}
    for entry in inputs.values():
        if entry["path"] not in hashes:
            hashes[entry["path"]] = current_digest(entry["path"])
        entry["sha256"] = hashes[entry["path"]]
    return {
        "inputs": inputs,
        "settings": settings,
        "fusion_revision": fusion_compatibility(APP)[0],
        "adapter": code_identity(),
        "config_sha256": SETTINGS.sha256,
        "models": model_identity(
            model_names(settings, request.get("segments")) | set(models), catalog
        ),
    }


def short(value):
    return value[:12] + "…" if isinstance(value, str) and SHA256.fullmatch(value) else value


def provenance_changes(recorded, current):
    """Readable differences between two provenance records, as a list of entries.

    An input whose path differs is a substitution, which the caller records, not a change; one
    that became unavailable is a change whose current SHA-256 is None. Release names alone are
    not compared: a new release with identical code runs the same adapter.
    """
    found = []

    def add(kind, item, before, after, message):
        found.append(
            {"kind": kind, "item": item, "recorded": before, "current": after, "message": message}
        )

    for place, entry in recorded["inputs"].items():
        now = current["inputs"].get(place)
        if now is None or now["path"] != entry["path"] or now["sha256"] == entry["sha256"]:
            continue
        state = (
            "is missing or no longer an allowed input"
            if now["sha256"] is None
            else f"has different bytes (SHA-256 {short(entry['sha256'])} now {short(now['sha256'])})"
        )
        add("input", place, entry["sha256"], now["sha256"], f"{place} {entry['path']} {state}")
    groups = [("controls", recorded["settings"]["controls"], current["settings"]["controls"])]
    for index, pair in enumerate(
        zip(recorded["settings"]["parameters"], current["settings"]["parameters"])
    ):
        groups.append((f"mappings[{index}].parameters", *pair))
    for group, before, after in groups:
        for key in sorted(set(before) | set(after)):
            was, now = before.get(key, "(absent)"), after.get(key, "(absent)")
            if canonical(was) != canonical(now):
                add("setting", f"{group}.{key}", was, now, f"{group}.{key} {was!r} now {now!r}")
    was, now = recorded["fusion_revision"], current["fusion_revision"]
    if was != now:
        add("fusion", "revision", was, now, f"Fusion revision {was} now {now}")
    before, after = recorded["adapter"], current["adapter"]
    for name in CODE:
        was = (before["code"] or {}).get(name)
        now = (after["code"] or {}).get(name)
        if was is None or was != now:
            add(
                "adapter",
                name,
                was,
                now,
                f"Adapter {name} of release {before['release']} "
                + ("is unknown (release removed)" if was is None else f"was {short(was)}")
                + f"; release {after['release']} has {short(now)}",
            )
    if recorded["config_sha256"] != current["config_sha256"]:
        add(
            "config",
            "config_sha256",
            recorded["config_sha256"],
            current["config_sha256"],
            "The workstation configuration (paths, GPU or runner) changed",
        )
    for name in sorted(set(recorded["models"]) | set(current["models"])):
        was, now = recorded["models"].get(name, {}), current["models"].get(name, {})
        if canonical(was) != canonical(now):
            if now.get("sha256") is None:
                state = "is not in this Fusion's model list or its file is missing"
            elif name not in recorded["models"]:
                # Revisions saved before restorer, mask, expression and CSCS adapter models were
                # fingerprinted lack them, so their bytes then are unknown.
                state = (
                    "was not recorded when this revision was saved, so whether it changed is "
                    "unknown; save a new revision to record it"
                )
            elif was.get("sha256") is None:
                state = "was not recorded"
            else:
                state = f"has different bytes ({short(was['sha256'])} now {short(now['sha256'])})"
            add("model", name, was.get("sha256"), now.get("sha256"), f"Model {name} {state}")
    return found


# Render checkpoints (docs/explanation/video-processing.md, Checkpoints): validated chunks
# committed by a versioned manifest, which records the provenance a resumed attempt must match
# before it reuses them.
CHECKPOINT_FORMAT, CHECKPOINT_VERSION = "visomaster-checkpoint", 1
CHECKPOINT_KEYS = {
    "format",
    "version",
    "job_id",
    "created_at",
    "resumed_from",
    "intent_sha256",
    "range",
    "chunk_frames",
    "chunking",
    "temporal",
    "runtime",
    "codec",
    "provenance",
    "chunks",
}
# Fusion carries tracking and landmark-smoothing state from frame to frame, which a checkpoint
# does not hold, so renders with either toggle are not resumable.
TEMPORAL = ("FaceTrackingEnableToggle", "KPSSmoothingEnableToggle")


def intent_digest(request):
    """SHA-256 of what a render's frames depend on in its resolved request: everything except
    the link to a resumed attempt and the scheduling class."""
    intent = {key: value for key, value in request.items() if key not in ("resume", "priority")}
    return hashlib.sha256(canonical(intent).encode()).hexdigest()


def temporal(controls):
    """The temporal toggles a set of effective controls enables."""
    return [key for key in TEMPORAL if controls.get(key) is True]


def ffmpeg_version():
    """FFmpeg's version line: chunks encoded by another build may not join without re-encoding."""
    output = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=30)
    return output.stdout.partition("\n")[0].strip()


def concat_input(path):
    """Arguments that open a checkpoint's concat list as one confined FFmpeg input. The list
    names chunk files by safe relative paths, and the concat demuxer passes its whitelists on,
    so each chunk also opens with the file protocol only and as MP4."""
    options = ["-f", "concat", "-safe", "1", "-protocol_whitelist", "file"]
    return [*options, "-format_whitelist", "concat,mov", "-i", str(path)]


def read_checkpoint(path):
    """A checkpoint manifest, refused with a reason when it is missing, damaged or newer."""
    try:
        document = read_json(path)
    except FileNotFoundError:
        raise ValueError("it has no checkpoint manifest") from None
    except (OSError, ValueError):
        raise ValueError("its checkpoint manifest is not readable JSON") from None
    if not isinstance(document, dict) or document.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("its checkpoint manifest has an unknown format")
    if document.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"its checkpoint manifest has version {document.get('version')!r}; this release "
            f"reads version {CHECKPOINT_VERSION} only"
        )
    if set(document) != CHECKPOINT_KEYS or not isinstance(document["chunks"], list):
        raise ValueError("its checkpoint manifest has unexpected or missing fields")
    return document


def resume_problems(manifest, request, record):
    """Why `manifest`'s chunks cannot be reused for `request` now, given its current provenance
    `record`; empty when they can."""
    found = []
    if manifest["intent_sha256"] != intent_digest(request):
        found.append(
            "The request differs from the checkpointed render's, or its inputs now resolve to "
            "other files"
        )
    if record is None or manifest["provenance"] is None:
        return found + ["No provenance was recorded, so compatibility cannot be checked"]
    enabled = temporal(record["settings"]["controls"])
    if enabled:
        found.append(
            f"{' and '.join(enabled)} {'carries' if len(enabled) == 1 else 'carry'} tracking "
            "state from frame to frame, which "
            "checkpoints do not hold; resuming would restart it at the boundary"
        )
    now = ffmpeg_version()
    if manifest["runtime"] != now:
        found.append(f"FFmpeg changed ({manifest['runtime']!r} now {now!r})")
    found += [change["message"] for change in provenance_changes(manifest["provenance"], record)]
    return found

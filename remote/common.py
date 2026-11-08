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
CONFIG_NAME = "remote-config.json"
CONFIG_KEYS = {"version", "media_root", "fusion_dir", "state_dir", "gpu", "runner", "lock_dir"}
LAYOUT_NAMES = ("settings", "common", "swapper", "face_editor", "denoiser")
# Environment variables read by earlier releases; a differing value must not be ignored silently.
LEGACY = {"VM_BASE": "base", "VM_APP": "app", "VM_STATE": "state", "VM_GPU": "gpu"}
JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
TERMINAL = {"completed", "failed", "cancelled"}
# GPU admission class of each job kind (DESIGN.md, GPU scheduling); a request's `priority`
# overrides it. A GPU kind missing here is scheduled as DEFAULT_PRIORITY. Normalization never
# waits for the GPU and has no class.
PRIORITIES = {
    "detect": "interactive",
    "preview": "interactive",
    "analyze": "interactive",
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


class ConfigError(Exception):
    """The workstation configuration file is missing, malformed or inconsistent."""


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
            "config/remote-config.example.json (see README, Remote configuration)"
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
            "is an absolute executable (see README, Runner)"
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
        raise ValueError(
            f"Job {job_id} did not complete ({status}); only completed jobs' artifacts are "
            "inputs, so rerun it under a new job_id and use that job's artifacts"
        )
    if status is None:
        raise ValueError(f"Job {job_id} has no readable state; its artifacts are not inputs")
    raise ValueError(
        f"Job {job_id} is {status}; wait for it to complete before using its artifacts"
    )


def input_path(value):
    """A media input: a regular file in media_root or a managed file of state_dir.

    The path is resolved first, so traversal and symlinks (in the file name or any ancestor)
    are judged by their target and cannot extend either root. A job artifact is an input only
    once the job holding it has completed.
    """
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Media paths must be absolute workstation paths")
    path = path.resolve(strict=True)
    parts = ()
    if STATE is not None and path.is_relative_to(STATE):
        inside = managed(path)
        parts = path.relative_to(STATE).parts
    else:
        inside = path.is_relative_to(BASE)
    if not inside or not path.is_file():
        raise ValueError(f"Media must be a regular file inside {BASE}, an upload or a job artifact")
    if parts[0:1] == ("jobs",):
        require_completed(parts[1])
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def reference_path(value, expected=None):
    """A face reference input; with `expected`, its bytes must still have that SHA-256.

    Submission checks fingerprints a caller took from analyze_references, and the worker checks
    those recorded at submission, so a changed file is reported rather than silently used.
    """
    path = input_path(value)
    if path.stat().st_size > MAX_REFERENCE_BYTES:
        raise ValueError(f"Face references must be images up to 256 MiB: {value}")
    if expected is not None and file_sha256(path) != expected:
        raise ValueError(
            f"Reference {value} no longer has SHA-256 {expected}: its content changed after it "
            "was fingerprinted. Analyze it again and resubmit with the new fingerprint."
        )
    return path


def digests(values, count, key):
    """Expected SHA-256 values, one per reference, or Nones when omitted."""
    if values is None:
        return [None] * count
    if (
        not isinstance(values, list)
        or len(values) != count
        or not all(isinstance(v, str) and SHA256.fullmatch(v) for v in values)
    ):
        raise ValueError(f"{key} must list one 64-hex-digit SHA-256 per reference")
    return values


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
        raise ValueError(
            f"Unsupported container '{refused[1].decode()}'; use a self-contained video (MP4, "
            "MOV, Matroska, WebM, AVI or MPEG-TS) or image (PNG, JPEG, WebP or BMP), not a "
            "playlist or concat list"
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
            raise ValueError(f"Unsupported setting {key}; use get_settings")
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
            raise ValueError(f"Invalid value for {key}")
        if key == "SwapModelSelection" and value in {"DeepFaceLive (DFM)", "AlphaFace"}:
            raise ValueError("This adapter supports embedding-based swappers only")


def job_priority(request):
    """The scheduling class of a GPU job request, or None for normalization."""
    if request.get("kind") == "normalize":
        return None
    priority = request.get("priority", PRIORITIES.get(request.get("kind"), DEFAULT_PRIORITY))
    if priority not in CLASSES:
        raise ValueError(f"priority must be one of {', '.join(CLASSES)}")
    return priority


def validate_request(request):
    """Validate a job request; normalization alone never reads Fusion's layout metadata."""
    if request.get("kind") not in {"detect", "preview", "render", "normalize", "analyze"}:
        raise ValueError("Unknown job kind")
    if request["kind"] == "analyze":
        job_priority(request)
        return validate_analysis(request)
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
    for mapping in mappings:
        target = mapping.get("target_sha256")
        if target is not None and not (isinstance(target, str) and SHA256.fullmatch(target)):
            raise ValueError("target_sha256 must be a 64-hex-digit SHA-256")
        reference_path(mapping["target_face"], target)
        sources = mapping["source_faces"]
        if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_REFERENCES:
            raise ValueError("Each mapping requires 1–20 source images")
        expected = digests(mapping.get("source_sha256"), len(sources), "source_sha256")
        for source, digest in zip(sources, expected):
            reference_path(source, digest)
        validate_weights(mapping.get("source_weights"), len(sources))
        validate_settings(mapping.get("parameters", {}), parameters, PARAMETERS)
    if "report" in request:
        if request["kind"] != "render":
            raise ValueError("report options apply only to renders")
        validate_report(request["report"], request, len(mappings))
    return request


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

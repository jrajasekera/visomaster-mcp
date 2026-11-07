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
MAX_DETECT_FRAMES = 64
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
    required = (".git", *(f"app/ui/widgets/{name}_layout_data.py" for name in LAYOUT_NAMES))
    missing = [name for name in required if not (app / name).exists()]
    if missing:
        raise ConfigError(
            f"{path}: fusion_dir {app} is not a Fusion checkout (missing {', '.join(missing)})"
        )
    state = raw.get("state_dir", str(base / "runtime/visomaster-mcp"))
    state = _directory(path, "state_dir", state, must_exist=False)
    # Uploads and job artifacts are reused as media inputs, which must lie inside media_root.
    if state == base or not state.is_relative_to(base):
        raise ConfigError(f"{path}: state_dir must be a directory inside media_root {base}")
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


def input_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Media paths must be absolute workstation paths")
    path = path.resolve(strict=True)
    if not path.is_relative_to(BASE) or not path.is_file():
        raise ValueError(f"Media must be a regular file inside {BASE}")
    return path


def artifact_path(job_id, name):
    directory = job_path(job_id) / "artifacts"
    path = (directory / name).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Artifact path escapes the job")
    return path


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def layouts():
    """Read literal metadata without importing Qt actions or constructing a GUI."""
    control, parameters = {}, {}
    for name in LAYOUT_NAMES:
        tree = ast.parse((APP / f"app/ui/widgets/{name}_layout_data.py").read_text())
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


def validate_request(request):
    if request.get("kind") not in {"detect", "preview", "render", "normalize"}:
        raise ValueError("Unknown job kind")
    input_path(request["media_path"])
    fps = request.get("fps")
    if fps is not None:
        try:
            rate = Fraction(str(fps)) if type(fps) in (int, float, str) else None
        except (ValueError, ZeroDivisionError):
            rate = None
        if rate is None or not 1 <= rate <= 240 or rate.denominator > 1001:
            raise ValueError("fps must be a rate from 1 to 240, such as 25, 29.97 or 30000/1001")
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
    validate_settings(request.get("controls", {}), control, CONTROLS)
    mappings = request.get("mappings", [])
    if request["kind"] not in {"detect", "normalize"} and not mappings:
        raise ValueError("At least one explicit target/source mapping is required")
    if len(mappings) > 20:
        raise ValueError("At most 20 mappings are supported")
    for mapping in mappings:
        input_path(mapping["target_face"])
        sources = mapping["source_faces"]
        if not isinstance(sources, list) or not 1 <= len(sources) <= 20:
            raise ValueError("Each mapping requires 1–20 source images")
        for source in sources:
            input_path(source)
        validate_settings(mapping.get("parameters", {}), parameters, PARAMETERS)
    return request

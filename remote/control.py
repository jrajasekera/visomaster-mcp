"""Short-lived SSH commands and durable per-job supervisors. No GPU imports here."""

import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from common import (
    APP,
    BASE,
    CONFIG_ERROR,
    CONTROLS,
    GPU,
    JOB_ID,
    LOCK_DIR,
    PARAMETERS,
    REVISION,
    RUNNER,
    SETTINGS,
    STATE,
    TERMINAL,
    ConfigError,
    artifact_path,
    atomic_json,
    config_path,
    input_path,
    job_path,
    layouts,
    load_config,
    probe,
    read_json,
    validate_request,
)

MAX_WAIT = 50
GRACE = 10  # Seconds a cancelled worker gets to exit on SIGTERM before its group is killed.
# Leads the job's process group so the group ID stays reserved until the supervisor reaps it. It
# ignores the cancellation SIGTERM and kills the whole group when the supervisor's pipe closes,
# which also covers a supervisor that dies.
GUARD = """import os, signal
for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(number, signal.SIG_IGN)
os.write(1, b"+")
while os.read(0, 1):
    pass
os.killpg(0, signal.SIGKILL)
"""


def current(directory):
    state = read_json(directory / "state.json")
    if state["status"] not in TERMINAL:
        # The supervisor holds this lease from startup through final state publication.
        with (directory / "lease").open("a") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                state = read_json(directory / "state.json")
                if state["status"] not in TERMINAL and time.time() - state["created_at"] > 15:
                    state.update(
                        status="failed",
                        error="Supervisor exited or host restarted",
                        finished_at=time.time(),
                    )
                    atomic_json(directory / "state.json", state)
    if state["status"] in TERMINAL and not state.get("finished_at"):
        # Terminal records written before finish times were recorded: the state file is never
        # rewritten once terminal, so its modification time is the finish time.
        state["finished_at"] = modified(directory / "state.json")
    progress = directory / "progress.json"
    if progress.exists():
        state["progress"] = read_json(progress)
    return state


def status(job_id, wait=0, include_log=False):
    directory = job_path(job_id)
    state = current(directory)
    # Long poll: return once status or progress stage changes, or the job becomes terminal.
    initial = (state["status"], state.get("progress", {}).get("stage"))
    deadline = time.monotonic() + min(max(float(wait), 0.0), MAX_WAIT)
    while state["status"] not in TERMINAL and time.monotonic() < deadline:
        time.sleep(0.25)
        state = current(directory)
        if (state["status"], state.get("progress", {}).get("stage")) != initial:
            break
    result = directory / "result.json"
    if state["status"] == "completed" and result.exists():
        state["result"] = read_json(result)
    log = directory / "worker.log"
    if (include_log or state["status"] == "failed") and log.exists():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 6000))
            state["log_tail"] = stream.read().decode(errors="replace")
    return state


def modified(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def list_jobs(only_status=None, only_kind=None, limit=50):
    limit = min(max(int(limit), 1), 200)
    paths = sorted((STATE / "jobs").glob("*/state.json"), key=modified, reverse=True)
    jobs = []
    for path in paths:
        try:
            state = current(path.parent)
            request = read_json(path.parent / "request.json")
        except (OSError, ValueError):
            continue  # Removed or still being published.
        if (only_status and state["status"] != only_status) or (
            only_kind and request.get("kind") != only_kind
        ):
            continue
        frames = {"render": ("start_frame", "max_frames"), "normalize": ("fps",)}.get(
            request.get("kind"), ("frame",)
        )
        created, end = state.get("created_at"), state.get("finished_at") or time.time()
        jobs.append(
            {
                "job_id": state["job_id"],
                "kind": request.get("kind"),
                "status": state["status"],
                "media_path": request.get("media_path"),
                **{key: request.get(key) for key in frames},
                **{key: state.get(key) for key in ("created_at", "started_at", "finished_at")},
                "elapsed_seconds": round(end - created, 1) if created else None,
                "error": state.get("error"),
                "progress": state.get("progress"),
            }
        )
        if len(jobs) == limit:
            break
    return jobs


def submit(request):
    request = dict(request)
    job_id = request.pop("job_id")
    directory = job_path(job_id)
    (STATE / "jobs").mkdir(parents=True, exist_ok=True, mode=0o700)
    with (STATE / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Validate under the lock so pruning cannot remove an input before the job exists.
        validate_request(request)
        if directory.exists():
            if read_json(directory / "request.json") != request:
                raise ValueError("job_id already exists with a different request")
            return status(job_id)
        with tempfile.TemporaryDirectory(dir=STATE, prefix="prepare-") as temporary:
            staging = Path(temporary)
            (staging / "artifacts").mkdir()
            atomic_json(staging / "request.json", request)
            atomic_json(
                staging / "state.json",
                {
                    "job_id": job_id,
                    "status": "queued",
                    "created_at": time.time(),
                    "gpu": GPU,
                    "config_sha256": SETTINGS.sha256,
                },
            )
            os.replace(staging, directory)
        # The supervisor gets the job directory and configuration identity, so a later edit of
        # remote-config.json cannot make it look the job up elsewhere.
        supervisor = [
            sys.executable,
            str(Path(__file__).resolve()),
            "supervise",
            str(directory),
            SETTINGS.sha256,
        ]
        with (directory / "worker.log").open("ab", buffering=0) as log:
            try:
                subprocess.Popen(
                    supervisor,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            except Exception as error:
                state = read_json(directory / "state.json")
                state.update(status="failed", error=str(error), finished_at=time.time())
                atomic_json(directory / "state.json", state)
                raise
    return status(job_id)


def verify_config(identity, directory):
    """Reload the configuration and confirm it is the one the job was submitted under."""
    try:
        settings = load_config()
    except ConfigError as error:
        raise RuntimeError(
            f"Remote configuration became invalid after submission; fix it and resubmit: {error}"
        ) from None
    if (settings.state / "jobs" / directory.name).resolve() != directory.resolve():
        raise RuntimeError(
            "Remote configuration changed after submission (state_dir differs); resubmit"
        )
    if settings.sha256 != identity:
        raise RuntimeError("Remote configuration changed after submission; resubmit the job")
    return settings


class ConfigWatch:
    """Cheap repeated identity check: rehash only when the file's stat signature changes."""

    def __init__(self, settings, identity, directory):
        self.path, self.identity, self.directory = settings.path, identity, directory
        self.stamp = self.signature()

    def signature(self):
        try:
            info = os.stat(self.path)
        except OSError:
            return None
        return info.st_ino, info.st_size, info.st_mtime_ns

    def check(self, force=False):
        stamp = self.signature()
        if force or stamp != self.stamp:
            verify_config(self.identity, self.directory)
            self.stamp = stamp


def valid_job_directory(value):
    """Structural checks that need no configuration, so a broken file cannot strand the job."""
    directory = Path(value)
    state = directory / "state.json"
    return (
        directory.is_absolute()
        and JOB_ID.fullmatch(directory.name) is not None
        and directory.parent.name == "jobs"
        and not state.is_symlink()
        and state.is_file()
    )


def spawn(arguments, group, **options):
    """Start a process in process group `group`, or in a new group it leads when `group` is 0."""
    if sys.version_info >= (3, 11):
        return subprocess.Popen(arguments, process_group=group, **options)
    # Older system Pythons lack process_group; the supervisor is single-threaded.
    return subprocess.Popen(arguments, preexec_fn=lambda: os.setpgid(0, group), **options)


def signal_group(group, number):
    try:
        os.killpg(group, number)
    except (ProcessLookupError, PermissionError):
        pass  # Only exited members remain; macOS reports EPERM for those.


def stop(guard, worker):
    """Kill the job's process group, then reap the worker and the guard.

    The unreaped guard reserves the group ID, so the signal cannot reach another group.
    """
    if guard is None or guard.returncode is not None:
        return
    signal_group(guard.pid, signal.SIGKILL)
    if worker is not None:
        worker.wait()
    guard.wait()
    guard.stdin.close()
    guard.stdout.close()
    # Killed members finish exiting asynchronously. Signal 0 cannot affect a reused group ID.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.killpg(guard.pid, 0)
        except (ProcessLookupError, PermissionError):
            break
        time.sleep(0.02)


def supervise(job_directory, identity):
    directory = Path(job_directory)
    if not valid_job_directory(directory):
        print(f"Refusing to supervise {job_directory}: not a job directory", file=sys.stderr)
        sys.exit(2)
    with (directory / "lease").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        state = {"job_id": directory.name, "status": "queued"}
        guard = child = None
        try:
            # Terminal state always goes to this directory, whatever the configuration says now.
            state = read_json(directory / "state.json")
            settings = verify_config(identity, directory)
            watch = ConfigWatch(settings, identity, directory)
            # Normalization is FFmpeg-only, so it neither waits for nor holds the GPU lock.
            gpu = read_json(directory / "request.json").get("kind") != "normalize"
            lock_path = settings.lock_dir / f"gpu-{settings.gpu}.lock"
            with lock_path.open("a") as gpu_lock:
                while True:
                    if (directory / "cancel").exists():
                        state["status"] = "cancelled"
                        return
                    watch.check()
                    if not gpu:
                        break
                    try:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(0.2)
                watch.check(force=True)
                state.update(status="running", started_at=time.time())
                atomic_json(directory / "state.json", state)
                environment = dict(
                    os.environ,
                    CUDA_DEVICE_ORDER="PCI_BUS_ID",
                    CUDA_VISIBLE_DEVICES=settings.gpu,
                    QT_QPA_PLATFORM="offscreen",
                    VM_SUPERVISOR_PID=str(os.getpid()),
                    VM_CONFIG=str(settings.path),
                    VM_CONFIG_SHA256=identity,
                    PYTHONUNBUFFERED="1",
                    OMP_NUM_THREADS="2",
                )
                arguments = [
                    *settings.runner,
                    str(Path(__file__).with_name("engine.py")),
                    str(directory),
                ]
                guard = spawn(
                    [sys.executable, "-I", "-S", "-c", GUARD],
                    0,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                )
                if guard.stdout.read(1) != b"+":
                    raise RuntimeError("Cannot start the process group guard")
                # Inherit the GPU lease so a supervisor crash cannot release it while rendering.
                try:
                    child = spawn(
                        arguments,
                        guard.pid,
                        env=environment,
                        pass_fds=(gpu_lock.fileno(),) if gpu else (),
                    )
                except OSError as error:
                    raise RuntimeError(
                        f"Cannot start the configured runner {settings.runner[0]}: {error}"
                    ) from None
                while child.poll() is None:
                    if (directory / "cancel").exists():
                        signal_group(guard.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=GRACE)
                        except subprocess.TimeoutExpired:
                            pass
                        break
                    time.sleep(0.2)
                # Helpers left in the group, whatever the worker's exit, die before any result.
                stop(guard, child)
                # A cancel arriving after the worker finished must not discard its result.
                if child.returncode == 0 and (directory / "result.json").exists():
                    state["status"] = "completed"
                elif (directory / "cancel").exists():
                    state["status"] = "cancelled"
                else:
                    error = f"Worker exited with code {child.returncode}"
                    try:
                        error = read_json(directory / "error.json")["message"]
                    except (OSError, ValueError, KeyError):
                        pass
                    state.update(status="failed", error=error)
        except Exception as error:
            state.update(status="failed", error=str(error))
        finally:
            stop(guard, child)
            state["finished_at"] = time.time()
            atomic_json(directory / "state.json", state)


def cancel(job_id):
    directory = job_path(job_id)
    # Delete and resubmission replace job directories under the submit lock, so holding it from
    # the status check to the marker keeps a replacement job from receiving this cancellation.
    with (STATE / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = status(job_id)
        active = state["status"] not in TERMINAL
        if active:
            (directory / "cancel").touch()
    return {"job_id": job_id, "cancel_requested": active, "status": state["status"]}


def tree_bytes(path):
    try:
        if path.is_symlink() or not path.is_dir():
            return path.lstat().st_size
        total = 0
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).lstat().st_size
                except FileNotFoundError:
                    pass
        return total
    except FileNotFoundError:
        return 0


def delete_job(job_id):
    directory = job_path(job_id)
    status(job_id)  # Reconciles jobs whose supervisor has exited.
    with (STATE / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (directory / "lease").open("a") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Job is still active; cancel it and wait for a terminal state")
            state = read_json(directory / "state.json")
            if state["status"] not in TERMINAL:
                raise ValueError("Only completed, failed or cancelled jobs can be deleted")
            # Artifacts such as normalized media or face crops can be another job's input.
            inside = os.path.realpath(directory) + os.sep
            if any(path.startswith(inside) for path in active_references()):
                raise ValueError("A queued or running job uses this job's artifacts as input")
            size = tree_bytes(directory)
            # Rename first so the job disappears atomically, then remove it outside the locks.
            trash = STATE / f"trash-{job_id}-{os.urandom(4).hex()}"
            os.replace(directory, trash)
    shutil.rmtree(trash, ignore_errors=True)
    return {"job_id": job_id, "status": state["status"], "deleted": True, "bytes": size}


def request_paths(request):
    paths = [request.get("media_path")]
    for mapping in request.get("mappings") or []:
        if isinstance(mapping, dict):
            paths.append(mapping.get("target_face"))
            sources = mapping.get("source_faces")
            paths.extend(sources if isinstance(sources, list) else [])
    return {os.path.realpath(path) for path in paths if isinstance(path, str)}


def active_references():
    """Input paths of queued/running jobs; callers hold the submit lock."""
    referenced = set()
    for request_file in (STATE / "jobs").glob("*/request.json"):
        try:
            if read_json(request_file.with_name("state.json"))["status"] in TERMINAL:
                continue
        except (OSError, ValueError, KeyError):
            pass  # Unreadable state counts as active.
        try:
            referenced |= request_paths(read_json(request_file))
        except (OSError, ValueError, AttributeError):
            pass
    return referenced


def prune(older_than_days, dry_run=True):
    if type(older_than_days) not in (int, float) or not 0 <= older_than_days <= 36500:
        raise ValueError("older_than_days must be between 0 and 36500")
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a boolean")
    now = time.time()
    cutoff = now - older_than_days * 86400
    result = {"dry_run": dry_run, "jobs": [], "uploads": [], "stale": []}
    for state_file in sorted((STATE / "jobs").glob("*/state.json")):
        job_id = state_file.parent.name
        try:
            state = status(job_id)
            finished = state.get("finished_at") or state_file.stat().st_mtime
            if state["status"] not in TERMINAL or finished >= cutoff:
                continue
            if dry_run:
                entry = {
                    "job_id": job_id,
                    "status": state["status"],
                    "bytes": tree_bytes(state_file.parent),
                }
            else:
                entry = delete_job(job_id)
        except (OSError, ValueError):
            continue  # Invalid, concurrently deleted or newly active jobs are left alone.
        entry["finished_at"] = finished
        result["jobs"].append(entry)
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (STATE / "submit.lock").open("a") as lock:
        # Submissions validate inputs under this lock, so no new reference can appear mid-prune.
        fcntl.flock(lock, fcntl.LOCK_EX)
        referenced = active_references()
        candidates = []
        for path in sorted((STATE / "uploads").glob("*")):
            modified = path.lstat().st_mtime
            if path.name.endswith(".partial"):
                # Active transfers keep writing; an hour of silence means a dead upload.
                if modified < min(cutoff, now - 3600):
                    candidates.append(("stale", path))
            elif modified < cutoff and os.path.realpath(path) not in referenced:
                candidates.append(("uploads", path))
        # Staging exists only inside submit's lock; trash only until delete_job finishes.
        for pattern in ("prepare-*", "trash-*"):
            candidates += [("stale", path) for path in sorted(STATE.glob(pattern))]
        for category, path in candidates:
            result[category].append({"path": str(path), "bytes": tree_bytes(path)})
            if not dry_run:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
    result["total_bytes"] = sum(
        entry["bytes"] for key in ("jobs", "uploads", "stale") for entry in result[key]
    )
    return result


def output(arguments):
    """Command output, or None when the tool is missing or fails, so health still answers."""
    try:
        return subprocess.check_output(arguments, text=True, stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None


def writable(path):
    return path is not None and os.access(path, os.W_OK | os.X_OK)


def health():
    release = Path(__file__).resolve().parent
    try:
        attempted = str(config_path())
    except ConfigError:
        attempted = None
    checks = {
        name: shutil.which(name) is not None for name in ("git", "nvidia-smi", "ffmpeg", "ffprobe")
    }
    checks["python"] = platform.python_version()
    checks["state_writable"] = writable(STATE)
    checks["lock_dir_writable"] = writable(LOCK_DIR)
    result = {
        "release": release.name,
        "code": {
            name: hashlib.sha256((release / name).read_bytes()).hexdigest()
            for name in ("common.py", "control.py", "engine.py")
        },
        "supported_revision": REVISION,
        "config_path": attempted,
        "config_error": CONFIG_ERROR,
        "checks": checks,
    }
    if CONFIG_ERROR:
        return {**result, "revision": None, "compatible": False}
    revision = output(["git", "-C", str(APP), "rev-parse", "HEAD"])
    revision = revision.strip() if revision else None
    gpus = output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    program = Path(RUNNER[0])
    return {
        **result,
        "revision": revision,
        "compatible": revision == REVISION,
        "gpu": GPU,
        "gpus": gpus,
        "app": str(APP),
        "state": str(STATE),
        "input_root": str(BASE),
        "lock_dir": str(LOCK_DIR),
        "runner": {
            "argv": RUNNER,
            "executable": program.is_file() and os.access(program, os.X_OK),
        },
        "config_sha256": SETTINGS.sha256,
    }


def dispatch(action, request):
    if action == "health":
        return health()
    if CONFIG_ERROR:
        raise ValueError(CONFIG_ERROR)
    if action == "settings":
        controls, parameters = layouts()
        controls["DetectorModelSelection"]["default"] = "Yolov8"
        controls["MaxFacesToDetectSlider"]["default"] = 20
        parameters["SwapModelSelection"]["options"] = [
            value
            for value in parameters["SwapModelSelection"]["options"]
            if value not in {"AlphaFace", "DeepFaceLive (DFM)"}
        ]
        return {
            "controls": {k: v for k, v in controls.items() if k in CONTROLS},
            "parameters": {k: v for k, v in parameters.items() if k in PARAMETERS},
            "excluded_swappers": ["AlphaFace", "DeepFaceLive (DFM)"],
        }
    if action == "inspect":
        return probe(input_path(request["media_path"]))
    if action == "submit":
        return submit(request)
    if action == "status":
        return status(
            request["job_id"], request.get("wait_seconds", 0), request.get("include_log", False)
        )
    if action == "cancel":
        return cancel(request["job_id"])
    if action == "list":
        return list_jobs(request.get("status"), request.get("kind"), request.get("limit", 50))
    if action == "artifact":
        if status(request["job_id"])["status"] != "completed":
            raise ValueError("Artifacts become available after successful completion")
        path = artifact_path(request["job_id"], request["name"])
        return {"path": str(path), "size": path.stat().st_size}
    if action == "delete":
        return delete_job(request["job_id"])
    if action == "prune":
        return prune(request["older_than_days"], request.get("dry_run", True))
    if action == "find_upload":
        return find_upload(request["sha256"], request["suffix"], request["size"])
    raise ValueError(f"Unknown action {action}")


def upload_destination(upload_id, suffix, expected_size):
    # 64 hex digits name content-addressed uploads; 32 remain for older random-named clients.
    if not re.fullmatch(r"[a-f0-9]{32}|[a-f0-9]{64}", upload_id) or suffix not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".mp4",
        ".mkv",
        ".mov",
    }:
        raise ValueError("Invalid upload name or media extension")
    size = int(expected_size)
    if not 0 < size <= 20 * 1024**3:
        raise ValueError("Uploads must be between 1 byte and 20 GiB")
    return STATE / "uploads" / (upload_id + suffix), size


def find_upload(sha256, suffix, size):
    if not isinstance(sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("sha256 must be 64 lowercase hex digits")
    destination, size = upload_destination(sha256, suffix, size)
    try:
        info = destination.lstat()
    except FileNotFoundError:
        return None
    if destination.is_symlink() or not destination.is_file() or info.st_size != size:
        return None
    os.utime(destination)  # Reuse counts as recent use for retention pruning.
    return {"path": str(destination), "bytes": size, "sha256": sha256, "reused": True}


def upload(upload_id, suffix, expected_size):
    destination, size = upload_destination(upload_id, suffix, expected_size)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # A unique name lets identical concurrent uploads proceed; the last complete one wins.
    temporary = destination.with_name(f"{destination.name}.{os.urandom(4).hex()}.partial")
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as stream:
            remaining = size + 1
            while remaining:
                chunk = sys.stdin.buffer.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                stream.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
        if temporary.stat().st_size != size:
            raise ValueError("Upload was interrupted or its size changed")
        if len(upload_id) == 64 and digest.hexdigest() != upload_id:
            raise ValueError("Upload content does not match its SHA-256")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), "bytes": size, "sha256": digest.hexdigest(), "reused": False}


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "supervise":
        supervise(sys.argv[2], sys.argv[3])
    else:
        try:
            if len(sys.argv) == 5 and sys.argv[1] == "upload":
                if CONFIG_ERROR:
                    raise ValueError(CONFIG_ERROR)
                result = upload(*sys.argv[2:])
            else:
                payload = json.load(sys.stdin)
                result = dispatch(payload.pop("action"), payload)
            print(json.dumps({"result": result}))
        except Exception as error:
            print(json.dumps({"error": str(error)}))
            sys.exit(1)

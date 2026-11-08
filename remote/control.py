"""Short-lived SSH commands and durable per-job supervisors. No GPU imports here."""

import contextlib
import copy
import errno
import fcntl
import hashlib
import json
import os
import platform
import re
import secrets
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Pin each command to one release. One started through the deployment's `remote` symlink re-runs
# from the release directory it resolves to before any project module loads, so a concurrent
# switch cannot mix releases within the command, its supervisor or its engine.
if __name__ == "__main__" and os.path.realpath(__file__) != os.path.abspath(__file__):
    os.execv(sys.executable, [sys.executable, os.path.realpath(__file__), *sys.argv[1:]])

from common import (  # noqa: E402
    ADAPTER_CONTROLS,
    APP,
    BASE,
    CLASSES,
    CODE,
    CONFIG_ERROR,
    CONTROLS,
    GPU,
    JOB_ID,
    LOCK_DIR,
    PARAMETERS,
    QUALIFICATIONS,
    REVISION,
    RUNNER,
    SETTINGS,
    SHA256,
    STATE,
    TERMINAL,
    ConfigError,
    atomic_json,
    canonical,
    code_identity,
    config_path,
    current_digest,
    file_sha256,
    fusion_compatibility,
    initial_controls,
    input_path,
    intent_digest,
    job_path,
    job_priority,
    layouts,
    load_config,
    model_catalog,
    model_choices,
    probe,
    provenance,
    provenance_changes,
    read_checkpoint,
    read_json,
    recorded_inputs,
    reference_path,
    required_models,
    resume_problems,
    setting_variants,
    validate_request,
    validate_settings,
)

RELEASE = Path(os.path.realpath(__file__)).parent
KEEP_RELEASES = 5  # Newest release directories kept, besides any an active job still uses.
MAX_WAIT = 50
GRACE = 10  # Seconds a cancelled worker gets to exit on SIGTERM before its group is killed.
# Leads the job's process group so the group ID stays reserved until the supervisor reaps it. It
# ignores the cancellation SIGTERM and kills the whole group when the supervisor's pipe closes,
# which also covers a supervisor that dies. A supervisor that sees the guard die stops the job.
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
                # The supervisor records its group after taking the lease, so it has died rather
                # than not started yet, and so may the guard that would stop the group.
                group = directory / "group.json"
                if state["status"] not in TERMINAL and (
                    group.exists() or time.time() - state["created_at"] > 15
                ):
                    if group.exists():
                        stop_survivors(read_json(group))
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


def observed(state):
    """What ends a long poll when it changes: status, progress stage (and a comparison's
    variant) and queue standing."""
    queue, progress = state.get("queue") or {}, state.get("progress") or {}
    stage = progress.get("stage"), progress.get("variant")
    return state["status"], stage, queue.get("position"), queue.get("reason")


def status(job_id, wait=0, include_log=False):
    directory = job_path(job_id)
    state = queued(directory, current(directory))
    # Long poll: return once status, progress stage or queue standing changes, or the job ends.
    initial = observed(state)
    deadline = time.monotonic() + min(max(float(wait), 0.0), MAX_WAIT)
    while state["status"] not in TERMINAL and time.monotonic() < deadline:
        time.sleep(0.25)
        state = queued(directory, current(directory))
        if observed(state) != initial:
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
    jobs, view = [], None
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
        if state["status"] == "queued" and state.get("priority"):
            view = queue_view() if view is None else view  # One snapshot for the whole list.
            queued(path.parent, state, view)
        frames = {
            "render": ("start_frame", "max_frames"),
            "normalize": ("fps",),
            "analyze": ("references",),
            "compare": ("frames", "start_frame", "max_frames"),
        }.get(request.get("kind"), ("frame",))
        if request.get("kind") == "detect":
            # Multi-frame detection reports its sampling instead of the unused single frame.
            multi = ("sample_frames", "frames", "similarity_threshold")
            frames = tuple(k for k in multi if request.get(k) is not None) or frames
        selection = {key: request.get(key) for key in frames}
        if request.get("kind") == "compare":
            # Stills or a clip, and the variant names in order.
            selection = {key: value for key, value in selection.items() if key in request}
            variants = request.get("variants") or []
            selection["variants"] = [v.get("name") for v in variants if isinstance(v, dict)]
        created, end = state.get("created_at"), state.get("finished_at") or time.time()
        jobs.append(
            {
                "job_id": state["job_id"],
                "kind": request.get("kind"),
                "status": state["status"],
                "media_path": request.get("media_path"),
                **selection,
                **{key: state.get(key) for key in ("created_at", "started_at", "finished_at")},
                "elapsed_seconds": round(end - created, 1) if created else None,
                "error": state.get("error"),
                "progress": state.get("progress"),
                # GPU jobs only, and queue standing only while queued.
                **{
                    key: state[key]
                    for key in ("priority", "sequence", "queue", "resume_from")
                    if key in state
                },
            }
        )
        if len(jobs) == limit:
            break
    return jobs


# GPU scheduling (DESIGN.md, GPU scheduling). Each GPU's queue lives beside its lock in the lock
# directory, so every deployment sharing that directory shares one queue and one decision.
MAX_STREAK = 3  # Interactive admissions in a row ahead of waiting normal work before it is due.
AGE_LIMIT = 300  # Seconds after the first of them at which waiting normal work is due anyway.
ENTRY = {
    "job": (str,),
    "job_id": (str,),
    "priority": (str,),
    "sequence": (int,),
    "enqueued_at": (int, float),
}


def load_queue(path):
    """The queue file's contents; malformed parts are dropped and supervisors rejoin."""
    try:
        raw = read_json(path)
    except (OSError, ValueError):
        raw = {}
    raw = raw if isinstance(raw, dict) else {}
    entries = raw.get("waiting") if isinstance(raw.get("waiting"), list) else []
    waiting = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and all(type(entry.get(key)) in types for key, types in ENTRY.items())
        and entry["priority"] in CLASSES
    ]
    number = raw.get("next_sequence")
    since = raw.get("passed_since")
    return {
        "next_sequence": max(
            [number if type(number) is int else 1] + [entry["sequence"] + 1 for entry in waiting]
        ),
        "streak": raw["streak"] if type(raw.get("streak")) is int else 0,
        "passed_since": since if type(since) in (int, float) else None,
        "waiting": waiting,
    }


@contextlib.contextmanager
def scheduler(lock_dir, gpu):
    """This GPU's queue under its scheduling lock, saved on normal exit if changed.

    Submission, admission, status and listing read and change the queue only under this lock,
    so one decision at a time picks the next job and that job takes the GPU lock within it.
    """
    path = Path(lock_dir) / f"gpu-{gpu}.queue.json"
    with (Path(lock_dir) / f"gpu-{gpu}.queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = load_queue(path)
        before = canonical(queue)
        yield queue
        if canonical(queue) != before:
            atomic_json(path, queue)


def fifo(entry):
    return entry["sequence"], entry["enqueued_at"], entry["job"]


def enqueue(queue, directory, state):
    """Queue a job at its recorded sequence, or record the next sequence in its state."""
    if state.get("sequence") is None:
        state["sequence"] = queue["next_sequence"]
    queue["next_sequence"] = max(queue["next_sequence"], state["sequence"] + 1)
    queue["waiting"].append(
        {
            "job": str(directory),
            "job_id": state["job_id"],
            "priority": state["priority"],
            "sequence": state["sequence"],
            "enqueued_at": state.get("created_at") or time.time(),
        }
    )


def standing(entry):
    """ "waiting" for an eligible entry, "cancelled" while its cancelled job is still exiting,
    or "drop" when its job ended, vanished or was replaced under the same ID."""
    directory = Path(entry["job"])
    try:
        if not valid_job_directory(directory):
            return "drop"
        state = current(directory)  # Fails a job whose supervisor died.
    except (OSError, ValueError, KeyError, TypeError):
        return "drop"  # A live supervisor rejoins at its recorded sequence.
    if state["status"] != "queued" or state.get("sequence") != entry["sequence"]:
        return "drop"
    return "cancelled" if (directory / "cancel").exists() else "waiting"


def lanes(queue, heads=False):
    """Eligible entries of each class in FIFO order and cancelled ones; drops ended entries.
    With heads, only the first eligible entry of each class is looked for."""
    found, cancelled = {name: [] for name in CLASSES}, []
    for entry in sorted(queue["waiting"], key=fifo):
        if heads and found[entry["priority"]]:
            continue
        verdict = standing(entry)
        if verdict == "drop":
            queue["waiting"].remove(entry)
        else:
            (cancelled if verdict == "cancelled" else found[entry["priority"]]).append(entry)
    return found, cancelled


def pick(found, queue, now):
    """The next admission: the interactive head, unless waiting normal work is due."""
    interactive, normal = found["interactive"], found["normal"]
    since = queue["passed_since"]
    due = queue["streak"] >= MAX_STREAK or (since is not None and now - since >= AGE_LIMIT)
    if normal and (due or not interactive):
        return normal[0]
    return interactive[0] if interactive else None


def record(queue, entry, found, now):
    """Count an interactive admission that passes waiting normal work; anything else resets."""
    if entry["priority"] == "normal" or not found["normal"]:
        queue.update(streak=0, passed_since=None)
    else:
        since = queue["passed_since"]
        queue.update(streak=queue["streak"] + 1, passed_since=now if since is None else since)


def predicted(found, queue, now):
    """Admission order if nothing else arrives and no time passes; advisory."""
    found = {name: list(entries) for name, entries in found.items()}
    queue, order = dict(queue), []
    while (entry := pick(found, queue, now)) is not None:
        record(queue, entry, found, now)
        found[entry["priority"]].remove(entry)
        order.append(entry)
    return order


def admit(settings, directory, state, gpu_lock):
    """One scheduling decision: True once this job is next and holds the GPU lock. The choice,
    the lock and leaving the queue happen under the scheduling lock, so they are one step."""
    name = str(directory)
    with scheduler(settings.lock_dir, settings.gpu) as queue:
        if not any(entry["job"] == name for entry in queue["waiting"]):
            enqueue(queue, directory, state)  # The queue file was lost: rejoin in place.
        now = time.time()
        found, _ = lanes(queue, heads=True)
        chosen = pick(found, queue, now)
        if chosen is None or chosen["job"] != name or (directory / "cancel").exists():
            return False
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False  # A running worker or an unqueued holder has the GPU.
        record(queue, chosen, found, now)
        queue["waiting"].remove(chosen)
        return True


def leave(settings, directory):
    with scheduler(settings.lock_dir, settings.gpu) as queue:
        queue["waiting"] = [entry for entry in queue["waiting"] if entry["job"] != str(directory)]


def gpu_busy(lock_dir, gpu):
    """Whether something holds the GPU lock. Callers hold the scheduling lock, so the probe
    cannot delay an admission."""
    with (Path(lock_dir) / f"gpu-{gpu}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def queue_view():
    """Advisory standing of this GPU's waiting jobs, keyed by job directory."""
    now = time.time()
    with scheduler(LOCK_DIR, GPU) as queue:
        found, cancelled = lanes(queue)
        order = predicted(found, queue, now)
        busy = bool(order) and gpu_busy(LOCK_DIR, GPU)
        streak = queue["streak"]
    view = {entry["job"]: (None, "cancel_requested") for entry in cancelled}
    for index, entry in enumerate(order):
        passed = {other["priority"] for other in order[:index]} - {entry["priority"]}
        if index == 0:
            reason = "gpu_busy" if busy else "next"
        elif passed:
            reason = "behind_interactive" if entry["priority"] == "normal" else "behind_due_normal"
        else:
            reason = "behind_earlier"
        view[entry["job"]] = (index + 1, reason)
    return {
        job: {"position": position, "reason": reason, "interactive_streak": streak}
        for job, (position, reason) in view.items()
    }


def queued(directory, state, view=None):
    """Add the advisory queue standing of a queued GPU job to its state."""
    if state["status"] == "queued" and state.get("priority"):
        view = queue_view() if view is None else view
        info = view.get(str(directory)) or {"position": None, "reason": "starting"}
        waited = round(time.time() - state.get("created_at", time.time()), 1)
        state["queue"] = {**info, "waiting_seconds": waited}
    return state


class Superseded(Exception):
    """Another release became selected before this submission took the submit lock."""


def selected_release():
    """The release the deployment's `remote` link selects now; a checkout or flat directory
    outside a releases directory selects itself."""
    if RELEASE.parent.name != "releases":
        return RELEASE
    selected = Path(os.path.realpath(RELEASE.parent.parent / "remote"))
    return selected if (selected / "control.py").is_file() else RELEASE


def submit(request):
    request = dict(request)
    job_id = request.pop("job_id")
    directory = job_path(job_id)
    (STATE / "jobs").mkdir(parents=True, exist_ok=True, mode=0o700)
    with (STATE / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Promotion switches releases under this lock, so a deployment either sees this job or
        # the job is submitted, supervised and run by the release it selected.
        selected = selected_release()
        if selected != RELEASE:
            raise Superseded(selected)
        # An exact retry returns the persisted job before its inputs are consulted, so it still
        # answers after they are moved or pruned. Canonical JSON keeps 1, 1.0 and true distinct.
        if directory.exists():
            if canonical(read_json(directory / "request.json")) != canonical(request):
                raise ValueError("job_id already exists with a different request")
            return status(job_id)
        # Validate under the lock so pruning cannot remove an input before the job exists.
        validate_request(request)
        priority = job_priority(request)
        with tempfile.TemporaryDirectory(dir=STATE, prefix="prepare-") as temporary:
            staging = Path(temporary)
            (staging / "artifacts").mkdir()
            atomic_json(staging / "request.json", request)
            # The worker opens and retention protects these files, whatever aliases point to later.
            atomic_json(staging / "resolved.json", fingerprint(resolve_inputs(request)))
            if request.get("resume") is not None:
                inherit(request, staging)
            state = {
                "job_id": job_id,
                "status": "queued",
                "created_at": time.time(),
                "gpu": GPU,
                "config_sha256": SETTINGS.sha256,
                "release": str(RELEASE),  # Deployment cleanup keeps it while the job is active.
            }
            if request.get("resume") is not None:
                state["resume_from"] = request["resume"]["from_job_id"]
            # A GPU job takes its queue sequence and appears in the queue in one step with its
            # publication, so a retry finds the sequence and no scheduler sees a missing job.
            with scheduler(LOCK_DIR, GPU) if priority else contextlib.nullcontext() as queue:
                if priority:
                    state["priority"] = priority
                    enqueue(queue, directory, state)
                atomic_json(staging / "state.json", state)
                os.replace(staging, directory)
        # The supervisor gets the job directory and configuration identity, so a later edit of
        # remote-config.json cannot make it look the job up elsewhere.
        supervisor = [
            sys.executable,
            str(RELEASE / "control.py"),
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


# Resumed renders (DESIGN.md, Checkpoints).
RESUME_KEYS = {"job_id", "from_job_id", "priority"}
CHUNK_NAME = re.compile(r"[0-9]{5}\.(mp4|json)")


def resumable(origin):
    """The resolved request and checkpoint manifest of render `origin`, which must have failed
    or been cancelled; a refusal says why it cannot be resumed."""
    source = job_path(origin)
    try:
        status = current(source)["status"]
        recorded = read_json(source / "resolved.json")
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError(
            f"Job {origin} does not exist or has no readable state and request"
        ) from None
    if recorded.get("kind") != "render":
        raise ValueError(f"Job {origin} is a {recorded.get('kind')} job; only renders resume")
    if status == "completed":
        raise ValueError(f"Job {origin} completed; its result is final and nothing is left to do")
    if status not in TERMINAL:
        raise ValueError(f"Job {origin} is {status}; wait for it to end, or cancel it, first")
    try:
        manifest = read_checkpoint(source / "checkpoint" / "manifest.json")
    except ValueError as error:
        raise ValueError(
            f"Job {origin} cannot be resumed: {error} (it ended before its first chunk, or ran "
            "before checkpoints existed); submit a fresh render"
        ) from None
    return recorded, manifest


def inherit(request, staging):
    """Give a resumed render its own links to the chunks of the attempt it continues.

    The caller holds the submit lock, so that attempt is not deleted or replaced meanwhile.
    Chunk files are never rewritten, so a hard link shares them safely, and afterwards the
    links keep them whatever retention does to that attempt. The worker verifies them.
    """
    origin = request["resume"]["from_job_id"]
    recorded, manifest = resumable(origin)
    if intent_digest(recorded) != intent_digest(request):
        raise ValueError(
            f"The request differs from job {origin}'s; a resume repeats the attempt's own "
            "request (resume_render builds it)"
        )
    source, target = job_path(origin) / "checkpoint" / "chunks", staging / "checkpoint" / "chunks"
    target.mkdir(parents=True)
    for entry in manifest["chunks"]:
        for key in ("file", "state"):
            name = entry.get(key) if isinstance(entry, dict) else None
            path = source / name if isinstance(name, str) and CHUNK_NAME.fullmatch(name) else None
            if path is None or path.is_symlink() or not path.is_file():
                continue  # The worker finds it missing and renders those frames again.
            try:
                os.link(path, target / name)
            except OSError:
                shutil.copyfile(path, target / name)  # A filesystem without hard links.
    atomic_json(staging / "checkpoint" / "inherited.json", manifest)


def resume_render(request):
    """Submit a new render that reuses the verified chunks of a failed or cancelled render."""
    extra = sorted(set(request) - RESUME_KEYS)
    if extra:
        raise ValueError(f"resume_render does not accept {', '.join(extra)}")
    job_id, origin = request.get("job_id"), request.get("from_job_id")
    directory = job_path(job_id)
    job_path(origin)
    if job_id == origin:
        raise ValueError("A resume is a new job; give it a job_id of its own")
    invocation = {"from_job_id": origin, "priority": request.get("priority")}
    # A retry returns the job before the earlier attempt is consulted, so it still answers once
    # that attempt is deleted.
    if (directory / "request.json").exists():
        try:
            previous = read_json(directory / "request.json")["resume"]["invocation"]
        except (OSError, ValueError, KeyError, TypeError):
            previous = None
        if previous is None or canonical(previous) != canonical(invocation):
            raise ValueError("job_id already exists with a different request")
        return status(job_id)
    recorded, manifest = resumable(origin)
    new = {key: value for key, value in recorded.items() if key not in ("resume", "priority")}
    if (request.get("priority") or recorded.get("priority")) is not None:
        new["priority"] = request.get("priority") or recorded["priority"]
    attempt = (recorded.get("resume") or {}).get("attempt", 1) + 1
    new["resume"] = {"from_job_id": origin, "attempt": attempt, "invocation": invocation}
    # Inputs, settings, models, Fusion, adapter code and configuration are hashed and compared
    # here for an early answer; the worker compares them again before it reuses anything.
    problems = resume_problems(manifest, new, provenance(new))
    if problems:
        raise ValueError(
            f"The chunks of job {origin} cannot be reused:\n- "
            + "\n- ".join(problems)
            + "\nSubmit a fresh render with submit_swap instead"
        )
    return submit({"job_id": job_id, **new})


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


def owned(pid, group, token):
    """Whether live process `pid` is in process group `group` and carries the job's token."""
    proc = Path("/proc", str(pid))
    try:
        stat = (proc / "stat").read_bytes()
        fields = stat[stat.rindex(b")") + 2 :].split()
        return (
            fields[0] not in (b"Z", b"X")
            and int(fields[2]) == group
            and f"VM_JOB_TOKEN={token}".encode() in (proc / "environ").read_bytes().split(b"\0")
        )
    except (OSError, ValueError, IndexError):
        return False


def pidfd(number, *arguments):
    """Linux pidfd system call `number` (the same on every architecture) through libc.

    Some Python builds lack os.pidfd_open and signal.pidfd_send_signal, so call the kernel directly.
    """
    import ctypes

    result = ctypes.CDLL(None, use_errno=True).syscall(number, *arguments)
    if result < 0:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
    return result


PIDFD_OPEN, PIDFD_SEND_SIGNAL = 434, 424


def stop_survivors(record, timeout=5):
    """SIGKILL what remains of a dead supervisor's job group and wait for it to exit.

    The group ID alone may name a reused group, so only members whose environment carries the
    job's token count, and each is checked and signalled through a pidfd, which cannot follow a
    reused process ID. Linux only; elsewhere the guard is the only thing that stops the group.
    """
    if not sys.platform.startswith("linux"):
        return
    group, token = record["pgid"], record["token"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handles = []
        for pid in (int(name) for name in os.listdir("/proc") if name.isdigit()):
            if not owned(pid, group, token):
                continue
            try:
                handle = pidfd(PIDFD_OPEN, pid, 0)
            except OSError:
                continue
            try:
                # Rechecked now that the handle pins the process the signal will reach.
                if owned(pid, group, token):
                    pidfd(PIDFD_SEND_SIGNAL, handle, signal.SIGKILL, None, 0)
                    handles.append(handle)
                    continue
            except OSError:
                pass
            os.close(handle)
        if not handles:
            return
        for handle in handles:
            select.select([handle], [], [], max(0, deadline - time.monotonic()))
            os.close(handle)


def supervise(job_directory, identity):
    directory = Path(job_directory)
    if not valid_job_directory(directory):
        print(f"Refusing to supervise {job_directory}: not a job directory", file=sys.stderr)
        sys.exit(2)
    with (directory / "lease").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        state = {"job_id": directory.name, "status": "queued"}
        guard = child = settings = None
        gpu = admitted = False
        try:
            # Terminal state always goes to this directory, whatever the configuration says now.
            state = read_json(directory / "state.json")
            settings = verify_config(identity, directory)
            watch = ConfigWatch(settings, identity, directory)
            request = read_json(directory / "request.json")
            # Normalization is FFmpeg-only, so it neither waits for nor holds the GPU lock.
            gpu = request.get("kind") != "normalize"
            if gpu and state.get("sequence") is None:
                # A job record written without a queue position joins at the next one.
                state.setdefault("priority", job_priority(request))
                with scheduler(settings.lock_dir, settings.gpu) as queue:
                    enqueue(queue, directory, state)
                    atomic_json(directory / "state.json", state)
            lock_path = settings.lock_dir / f"gpu-{settings.gpu}.lock"
            with lock_path.open("a") as gpu_lock:
                while True:
                    if (directory / "cancel").exists():
                        state["status"] = "cancelled"
                        return
                    watch.check()
                    if not gpu:
                        break
                    if admit(settings, directory, state, gpu_lock):
                        admitted = True
                        break
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
                    VM_JOB_TOKEN=secrets.token_hex(16),
                )
                arguments = [
                    *settings.runner,
                    str(RELEASE / "engine.py"),
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
                # Lets status reconciliation find the group's survivors should the guard and then
                # the supervisor die.
                atomic_json(
                    directory / "group.json",
                    {"pgid": guard.pid, "token": environment["VM_JOB_TOKEN"]},
                )
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
                guard_lost = False
                while child.poll() is None:
                    if (directory / "cancel").exists():
                        signal_group(guard.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=GRACE)
                        except subprocess.TimeoutExpired:
                            pass
                        break
                    # The guard's output closes only when it exits. Without it a supervisor death
                    # would strand the group, so stop the job while the unreaped guard still
                    # reserves the group ID.
                    if select.select([guard.stdout], [], [], 0.2)[0]:
                        guard_lost = True
                        break
                # Helpers left in the group, whatever the worker's exit, die before any result.
                stop(guard, child)
                # A cancel arriving after the worker finished must not discard its result.
                if child.returncode == 0 and (directory / "result.json").exists():
                    state["status"] = "completed"
                elif (directory / "cancel").exists():
                    state["status"] = "cancelled"
                elif guard_lost:
                    state.update(
                        status="failed",
                        error=f"Process group guard exited unexpectedly (code {guard.returncode});"
                        " the job's processes were stopped",
                    )
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
            if gpu and not admitted and settings is not None:
                with contextlib.suppress(Exception):
                    leave(settings, directory)  # Others would drop the ended entry anyway.
            state["finished_at"] = time.time()
            atomic_json(directory / "state.json", state)
            if state["status"] == "completed":
                # The output is final, so a render's chunks are no longer needed. Should this
                # fail, they go with the job when it is deleted or pruned.
                shutil.rmtree(directory / "checkpoint", ignore_errors=True)


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


@contextlib.contextmanager
def retention_lock():
    """The submit lock: submission, deletion, pruning and upload reuse or publication decide
    under it, so a retention decision and its removal see one consistent set of jobs and files."""
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (STATE / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def listing(directory):
    return sorted(directory.glob("*"))


def resolve_inputs(request):
    """The request with each input path resolved; persisted at submission as resolved.json."""
    resolved = copy.deepcopy(request)
    if isinstance(resolved.get("media_path"), str):
        resolved["media_path"] = os.path.realpath(resolved["media_path"])
    if isinstance(resolved.get("references"), list):
        resolved["references"] = [
            os.path.realpath(path) if isinstance(path, str) else path
            for path in resolved["references"]
        ]
    for mapping in resolved.get("mappings") or []:
        if isinstance(mapping, dict):
            if isinstance(mapping.get("target_face"), str):
                mapping["target_face"] = os.path.realpath(mapping["target_face"])
            if isinstance(mapping.get("source_faces"), list):
                mapping["source_faces"] = [
                    os.path.realpath(path) if isinstance(path, str) else path
                    for path in mapping["source_faces"]
                ]
    return resolved


def fingerprint(resolved):
    """Record the SHA-256 of each face reference the worker will read, unless the request gave
    one (validation has compared those), so the worker uses exactly the bytes submitted."""
    if isinstance(resolved.get("references"), list) and resolved.get("reference_sha256") is None:
        resolved["reference_sha256"] = [file_sha256(path) for path in resolved["references"]]
    for mapping in resolved.get("mappings") or []:
        if mapping.get("target_sha256") is None:
            mapping["target_sha256"] = file_sha256(mapping["target_face"])
        if mapping.get("source_sha256") is None:
            mapping["source_sha256"] = [file_sha256(path) for path in mapping["source_faces"]]
    return resolved


def request_paths(request):
    paths = [request.get("media_path")]
    if isinstance(request.get("references"), list):
        paths.extend(request["references"])
    for mapping in request.get("mappings") or []:
        if isinstance(mapping, dict):
            paths.append(mapping.get("target_face"))
            sources = mapping.get("source_faces")
            paths.extend(sources if isinstance(sources, list) else [])
    return {Path(path) for path in paths if isinstance(path, str)}


def active_references():
    """Inputs of queued/running jobs as resolved at submission; callers hold the retention lock.

    None means unknown: an active job without resolved.json (submitted by an earlier release)
    may have resolved an alias differently, so nothing that could be an input is removable.
    """
    referenced = set()
    for directory in listing(STATE / "jobs"):
        if not (directory / "request.json").exists():
            continue  # Not a job, or deleted.
        try:
            if current(directory)["status"] in TERMINAL:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Unreadable state counts as active.
        try:
            referenced |= request_paths(read_json(directory / "resolved.json"))
        except (OSError, ValueError, AttributeError):
            return None
    return referenced


def recipe_references():
    """Inputs saved recipe revisions reference, which retention keeps; None when a revision is
    unreadable, so nothing that could be its input is removed. Callers hold the retention lock."""
    referenced = set()
    for path in sorted((STATE / "recipes").glob("*/*.json")):
        try:
            referenced |= request_paths(read_json(path)["request"])
        except FileNotFoundError:
            continue  # Deleted meanwhile.
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
    return referenced


def protected_inputs():
    """Inputs of queued or running jobs and of saved recipes, or None when unknown."""
    active, saved = active_references(), recipe_references()
    return None if active is None or saved is None else active | saved


def protected(path, referenced):
    """Whether removing path (a file, or a tree by component ancestry) removes an input."""
    if referenced is None:
        return True
    targets = {Path(os.path.abspath(path)), Path(os.path.realpath(path))}
    return any(item.is_relative_to(target) for item in referenced for target in targets)


def claim_job(directory, referenced, cutoff=None, remove=False):
    """Retention eligibility shared by delete, prune and its preview; the caller holds the lock.

    Returns None for a job newer than cutoff, raises ValueError for an ineligible one, and
    otherwise returns its report entry and, when removing, the trash path it was renamed to.
    The checks and the rename see one generation: replacing a job needs the retention lock,
    and the lease shows that no supervisor is still publishing its state.
    """
    current(directory)  # Reconciles jobs whose supervisor has exited.
    with (directory / "lease").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Job is still active; cancel it and wait for a terminal state")
        state = current(directory)  # The generation renamed below, with its finish time.
        if state["status"] not in TERMINAL:
            raise ValueError("Only completed, failed or cancelled jobs can be deleted")
        finished = state["finished_at"]
        if cutoff is not None and finished >= cutoff:
            return None
        # Artifacts such as normalized media or face crops can be another job's input.
        if protected(directory, referenced):
            raise ValueError(
                "A queued or running job or a saved recipe uses this job's artifacts as input"
            )
        entry = {
            "job_id": directory.name,
            "status": state["status"],
            "bytes": tree_bytes(directory),
            "finished_at": finished,
        }
        trash = None
        if remove:
            # Rename first so the job disappears atomically; callers remove it outside the lock.
            trash = STATE / f"trash-{directory.name}-{os.urandom(4).hex()}"
            os.replace(directory, trash)
        return entry, trash


def delete_job(job_id):
    directory = job_path(job_id)
    with retention_lock():
        entry, trash = claim_job(directory, protected_inputs(), remove=True)
    shutil.rmtree(trash, ignore_errors=True)
    return {"job_id": job_id, "status": entry["status"], "deleted": True, "bytes": entry["bytes"]}


def prune(older_than_days, dry_run=True):
    if type(older_than_days) not in (int, float) or not 0 <= older_than_days <= 36500:
        raise ValueError("older_than_days must be between 0 and 36500")
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a boolean")
    now = time.time()
    cutoff = now - older_than_days * 86400
    result = {"dry_run": dry_run, "jobs": [], "uploads": [], "stale": []}
    trash = []
    # Previews run the same checks as removal under the same lock, so both report alike.
    with retention_lock():
        referenced = protected_inputs()
        for directory in listing(STATE / "jobs"):
            try:
                claimed = claim_job(directory, referenced, cutoff, remove=not dry_run)
            except (OSError, ValueError, KeyError, TypeError):
                continue  # Invalid, vanished, active or referenced jobs are left alone.
            if claimed:
                entry, removed = claimed
                if removed:
                    entry["deleted"] = True
                    trash.append(removed)
                result["jobs"].append(entry)
        candidates = []
        for path in listing(STATE / "uploads"):
            try:
                modified = path.lstat().st_mtime
            except FileNotFoundError:
                continue  # A transfer published or discarded its temporary file.
            if path.name.endswith(".partial"):
                # Active transfers keep writing; an hour of silence means a dead upload.
                if modified < min(cutoff, now - 3600):
                    candidates.append(("stale", path))
            elif modified < cutoff and not protected(path, referenced):
                candidates.append(("uploads", path))
        # Staging exists only inside submit's lock; trash only until its remover finishes.
        for pattern in ("prepare-*", "trash-*"):
            candidates += [
                ("stale", path) for path in sorted(STATE.glob(pattern)) if path not in trash
            ]
        for category, path in candidates:
            result[category].append({"path": str(path), "bytes": tree_bytes(path)})
            if not dry_run:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
    for path in trash:
        shutil.rmtree(path, ignore_errors=True)
    result["total_bytes"] = sum(
        entry["bytes"] for key in ("jobs", "uploads", "stale") for entry in result[key]
    )
    return result


def active_jobs():
    """Queued and running job IDs with the release each records (None if unknown); callers hold
    the submit lock. Unreadable state counts as active."""
    jobs = {}
    for directory in listing(STATE / "jobs"):
        if not (directory / "request.json").exists():
            continue  # Not a job, or deleted.
        try:
            state = current(directory)
        except (OSError, ValueError, KeyError, TypeError):
            state = {}
        if state.get("status") not in TERMINAL:
            jobs[directory.name] = state.get("release")
    return jobs


def prune_releases(releases, jobs):
    """Remove release directories beyond the newest few, except this one and active jobs' ones."""
    if None in jobs.values():
        return []  # A job from an earlier release does not record which release it needs.
    needed = {RELEASE, *(Path(path) for path in jobs.values())}
    directories = [path for path in releases.iterdir() if path.is_dir() and not path.is_symlink()]
    removed = []
    for path in sorted(directories, key=modified, reverse=True)[KEEP_RELEASES:]:
        if path not in needed:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    return removed


def promote(force=False):
    """Select this release; deploy.sh runs it from the new candidate after its preflight.

    The final active-job check, the switch and release cleanup hold the submit lock, so a job is
    either seen here or submitted by the selected release. A unique staging link keeps
    concurrent deployments from replacing each other's candidate before it is selected.
    """
    if not isinstance(force, bool):
        raise ValueError("force must be a boolean")
    releases = RELEASE.parent
    if releases.name != "releases" or not all((RELEASE / name).is_file() for name in CODE):
        raise ValueError(f"{RELEASE} is not a complete release in a releases directory")
    root, target = releases.parent, f"releases/{RELEASE.name}"
    link = root / "remote"
    with retention_lock():
        jobs = active_jobs()
        if jobs and not force:
            names = " ".join(sorted(jobs))
            raise ValueError(
                f"Refusing to deploy while jobs are active: {names} (--force overrides)"
            )
        if link.is_dir() and not link.is_symlink():
            # The flat directory of an earlier deployment becomes an ordinary release.
            os.rename(link, releases / time.strftime("legacy-%Y%m%dT%H%M%SZ", time.gmtime()))
        staging = root / f"remote.next-{os.urandom(4).hex()}"
        staging.symlink_to(target)
        try:
            os.replace(staging, link)
        finally:
            staging.unlink(missing_ok=True)
        if os.readlink(link) != target:
            raise RuntimeError(f"{link} does not select {target} after promotion")
        removed = prune_releases(releases, jobs)
    return {"release": RELEASE.name, "active_jobs": sorted(jobs), "removed": removed}


def output(arguments):
    """Command output, or None when the tool is missing or fails, so health still answers."""
    try:
        return subprocess.check_output(arguments, text=True, stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None


def writable(path):
    return path is not None and os.access(path, os.W_OK | os.X_OK)


def health():
    release = RELEASE
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
        "code": {name: hashlib.sha256((release / name).read_bytes()).hexdigest() for name in CODE},
        "supported_revision": REVISION,
        "config_path": attempted,
        "config_error": CONFIG_ERROR,
        "checks": checks,
    }
    if CONFIG_ERROR:
        reason = "The remote configuration is invalid; see config_error"
        return {**result, "revision": None, "compatible": False, "compatibility_reason": reason}
    revision, reason = fusion_compatibility(APP)  # The check GPU workers run at startup.
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
        "compatible": reason is None,
        "compatibility_reason": reason,
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


# Preflight (DESIGN.md, Preflight). Cheap checks read files and run FFprobe, git and nvidia-smi
# only: no Fusion import, model loading, GPU lock or job. The optional GPU stage queues ordinary
# reference-analysis jobs. A preflight reserves nothing; submission revalidates everything.
SEVERITY = {
    "passed": "info",
    "not_checked": "info",
    "pending": "info",
    "warning": "warning",
    "blocked": "error",
}
WORST = ("blocked", "pending", "warning", "passed", "not_checked")
READINESS = {"blocked": "blocked", "pending": "pending", "warning": "warnings"}
HEADROOM = 1024**3  # Free space within this of the estimate is reported as low.
PREVIEW_BYTES = 64 * 1024**2  # Preview artifacts, report and log.
# Inputs a cheap check reports on; the rest are skipped when the request is malformed.
INPUT_CHECKS = ("media_path", "references", "settings", "model_assets", "media", "storage")


def sha256_of(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def finding(name, status, code, message, evidence=None, actions=()):
    return {
        "check": name,
        "status": status,
        "severity": SEVERITY[status],
        "code": code,
        "message": message,
        "evidence": evidence or {},
        "actions": list(actions),
    }


def act(description, tool=None, **arguments):
    """A corrective action: a tool call to make, or (without tool) one for the operator."""
    return {"description": description, **({"tool": tool, "arguments": arguments} if tool else {})}


def refusal_code(error):
    """The stable code of a refused input; missing and unreadable files have their own."""
    if isinstance(error, FileNotFoundError):
        return "path_missing"
    return getattr(error, "code", "path_unreadable" if isinstance(error, OSError) else "invalid")


def gpu_observation():
    """Compute capability, VRAM class and memory use of the configured GPU, or None."""
    text = output(
        [
            "nvidia-smi",
            f"--id={GPU}",
            "--query-gpu=compute_cap,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        capability, total, used = (v.strip() for v in text.strip().splitlines()[0].split(","))
        total, used = int(float(total)), int(float(used))
    except (AttributeError, IndexError, ValueError):
        return None
    return {
        "compute_capability": capability,
        "vram_class": f"{round(total / 1024)} GB",
        "vram_total_mib": total,
        "vram_used_mib": used,
    }


def positive(mapping):
    """Indexes of the sources a mapping reads: those with a positive (or omitted) weight."""
    count = len(mapping["source_faces"])
    weights = mapping.get("source_weights")
    weights = weights if isinstance(weights, list) and len(weights) == count else [1] * count
    return [i for i, w in enumerate(weights) if type(w) in (int, float) and w > 0]


def combinations(request, control, parameters):
    """Each mapping's model choices (common.model_choices) with its weighted source count, for
    the base settings and then for each segment that overrides them (segment None: the base)."""
    defaults = {key: spec["default"] for key, spec in parameters.items()}
    settings = {
        "controls": initial_controls(control, request.get("controls") or {}),
        "parameters": [{**defaults, **(m.get("parameters") or {})} for m in request["mappings"]],
    }
    found = []
    for segment, indexes, variant in setting_variants(settings, request.get("segments")):
        for index, choice in zip(indexes, model_choices(variant)):
            mapping = request["mappings"][index]
            choice = {**choice, "sources": len(positive(mapping))}
            found.append({"mapping": index, "segment": segment, "choice": choice})
    return found


def differences(record, mode, choice, mappings, hardware):
    """What of a mapping's request a qualification record does not cover; empty if covered."""
    found = []
    if record["fusion_revision"] != REVISION:
        found.append("fusion_revision")
    if hardware is None:
        found.append("hardware (not observed)")
    elif any(hardware[k] != record["hardware"][k] for k in ("compute_capability", "vram_class")):
        found.append("hardware")
    if mode not in record["modes"]:
        found.append("mode")
    for key in ("detector", "swap_model", "restorer", "expression", "occluder", "xseg"):
        if choice[key] not in record[key]:
            found.append(key)
    if mappings > record["max_mappings"]:
        found.append("mappings")
    if choice["sources"] > record["max_sources"]:
        found.append("sources")
    return found


def media_check(request, source):
    """The worker's own render or preview media checks (engine.clip_range, preview_source);
    timestamps are read only through the requested frames. Returns the check and its evidence."""
    import engine  # Its module level is the standard library; Fusion loads only in workers.

    try:
        if request["kind"] == "render":
            metadata, stream, fps, start, frames = engine.clip_range(source, request)
            width, height = int(stream["width"]), int(stream["height"])
            # Bytes per output frame from the source's bit rate or the resolution, whichever is
            # more; the checkpoint's chunks and the joined result coexist until completion.
            count = int(stream.get("nb_frames") or 0)
            count = count or round(float(stream.get("duration") or 0) * fps)
            per_frame = max(width * height // 20, source.stat().st_size // count if count else 0)
            evidence = {
                "kind": "video",
                "width": width,
                "height": height,
                "fps": str(fps),
                "start_frame": start,
                "frames": frames,
                "frames_checked": start + frames,
                "audio": any(s["codec_type"] == "audio" for s in metadata["streams"]),
                "estimated_bytes": 2 * per_frame * frames + PREVIEW_BYTES,
            }
            message = f"Render media passed the worker's checks over frames 0-{start + frames - 1}"
        else:
            frame = request.get("frame", 0)
            stream = engine.preview_source(source, frame)
            kind = "image" if stream is None else "video"
            evidence = {"kind": kind, "frame": frame, "estimated_bytes": PREVIEW_BYTES}
            message = f"Preview frame {frame} is readable"
    except (ValueError, OSError, subprocess.SubprocessError, KeyError, ZeroDivisionError) as error:
        actions = []
        if getattr(error, "normalize", False):
            description = (
                "Queue a render-ready copy (CPU only), then preflight and submit with the "
                "completed job's result media_path; nothing is converted unless you do this"
            )
            actions.append(act(description, "normalize_media", media_path=request["media_path"]))
        code = getattr(error, "code", "media_unreadable")
        message = str(error) or type(error).__name__
        return finding("media", "blocked", code, message, {}, actions), None
    return finding("media", "passed", "media_ready", message, evidence), evidence


def analysis_check(request, digests, enabled):
    """Queue (or find) one reference-analysis job per distinct set of sources the swap reads."""
    if not enabled:
        action = act("Rerun preflight_swap with analyze_references=true to queue it")
        message = "The optional GPU reference analysis was not requested"
        return finding("reference_analysis", "not_checked", "not_requested", message, {}, [action])
    jobs = {}
    for index, mapping in enumerate(request["mappings"]):
        used = positive(mapping)
        analysis = {
            "kind": "analyze",
            "references": [mapping["source_faces"][i] for i in used],
            "reference_sha256": [digests[index][i] for i in used],
        }
        if request.get("controls"):
            analysis["controls"] = request["controls"]
        model = (mapping.get("parameters") or {}).get("SwapModelSelection")
        if model is not None:
            analysis["swap_model"] = model
        # Content addressed: a repeated preflight finds the same job, and changed bytes,
        # controls or swap model make a new one.
        job_id = "preflight-" + sha256_of(analysis)[:40]
        jobs.setdefault(job_id, (analysis, []))[1].append(index)
    entries, statuses, actions = [], [], []
    for job_id, (analysis, mappings) in jobs.items():
        entry = {"job_id": job_id, "mappings": mappings}
        entries.append(entry)
        try:
            state = submit({"job_id": job_id, **analysis})
        except Superseded:
            raise
        except Exception as error:
            entry.update(status="not_queued", error=str(error))
            statuses.append("blocked")
            continue
        entry["status"] = state["status"]
        if state["status"] in ("queued", "running"):
            statuses.append("pending")
            actions.append(act("Wait for it", "job_status", job_id=job_id, wait_seconds=50))
        elif state["status"] == "completed":
            result = state.get("result") or {}
            entry.update({key: result.get(key) for key in ("usable", "errors", "warnings")})
            entry["findings"] = [
                {"path": reference["path"], **item}
                for reference in result.get("references", [])
                for item in reference.get("findings", [])
            ]
            entry["consistency"] = result.get("consistency")
            if result.get("errors") or result.get("warnings"):
                statuses.append("blocked" if result.get("errors") else "warning")
                actions.append(
                    act("Review the references", "view_artifact", job_id=job_id, name="contact.jpg")
                )
            else:
                statuses.append("passed")
        else:
            entry["error"] = state.get("error")
            statuses.append("blocked" if state["status"] == "failed" else "not_checked")
            description = (
                f"The analysis is {state['status']}; delete it, then preflight again with "
                "analyze_references=true to repeat it"
            )
            actions.append(act(description, "delete_job", job_id=job_id))
    status = min(statuses, key=WORST.index)
    code, message = {
        "blocked": (
            "reference_problems",
            "Reference analysis found sources submit_swap refuses, or did not run",
        ),
        "pending": (
            "analysis_pending",
            "Reference analysis is queued or running on the GPU; preflight again when it ends",
        ),
        "warning": ("reference_warnings", "Reference analysis flagged sources to review"),
        "passed": ("references_usable", "Every source the swap reads is usable"),
        "not_checked": ("analysis_cancelled", "Reference analysis was cancelled"),
    }[status]
    return finding("reference_analysis", status, code, message, {"jobs": entries}, actions)


def path_checks(request, checks):
    """Media and reference access. Returns the resolved media path (None if refused), the
    reference evidence and each mapping's source digests."""
    try:
        source = input_path(request["media_path"])
        if not os.access(source, os.R_OK):
            raise PermissionError(f"{source} is not readable")
    except Exception as error:
        source = None
        evidence = {"path": request["media_path"]}
        checks.append(finding("media_path", "blocked", refusal_code(error), str(error), evidence))
    else:
        evidence = {"path": str(source), "bytes": source.stat().st_size}
        message = "The media file is an accepted input"
        checks.append(finding("media_path", "passed", "path_ok", message, evidence))
    items, digests = [], []
    for index, mapping in enumerate(request["mappings"]):
        sources = mapping["source_faces"]
        given = mapping.get("source_sha256")
        given = given if isinstance(given, list) and len(given) == len(sources) else []
        roles = [("target", mapping["target_face"], mapping.get("target_sha256"))]
        given = given or [None] * len(sources)
        roles += [("source", path, digest) for path, digest in zip(sources, given)]
        digests.append([])
        for role, value, digest in roles:
            digest = digest if isinstance(digest, str) else None
            item = {"mapping": index, "role": role, "path": value}
            try:
                path = reference_path(value, digest)  # Hashes the file when a digest is given.
                item.update(sha256=digest or file_sha256(path), bytes=path.stat().st_size)
            except Exception as error:
                item.update(code=refusal_code(error), error=str(error))
            items.append(item)
            if role == "source":
                digests[-1].append(item.get("sha256"))
    problems = [item for item in items if "error" in item]
    evidence = {"references": items}
    if problems:
        action = act("Fix or replace the listed images; analyze changed ones again")
        message = f"{len(problems)} reference image(s) would be refused: {problems[0]['error']}"
        checks.append(
            finding("references", "blocked", problems[0]["code"], message, evidence, [action])
        )
    else:
        message = "Every target and source is an accepted input; pin them with these sha256 values"
        checks.append(finding("references", "passed", "references_ok", message, evidence))
    return source, items, digests


def settings_checks(request, checks):
    """Controls and parameters, then the model files they load. Returns each mapping's model
    choices (None when Fusion's settings metadata is unreadable) and the model evidence."""
    try:
        control, parameters = layouts()
    except ValueError as error:
        checks.append(finding("settings", "blocked", "layouts_unreadable", str(error)))
        checks.append(
            finding("model_assets", "not_checked", "prerequisite_blocked", "Settings unreadable")
        )
        return None, []
    scopes = [("controls", request.get("controls") or {}, control, CONTROLS)]
    for index, mapping in enumerate(request["mappings"]):
        scopes.append((f"mapping {index}", mapping.get("parameters") or {}, parameters, PARAMETERS))
    errors = []
    for scope, values, definitions, allowed in scopes:
        try:
            validate_settings(values, definitions, allowed)
        except ValueError as error:
            errors.append({"scope": scope, "code": refusal_code(error), "error": str(error)})
    if errors:
        action = act("Use names, types, options and limits from get_settings", "get_settings")
        message = f"{errors[0]['scope']}: {errors[0]['error']}"
        evidence = {"errors": errors}
        checks.append(
            finding("settings", "blocked", errors[0]["code"], message, evidence, [action])
        )
    else:
        message = "Controls and parameters are supported values"
        checks.append(finding("settings", "passed", "settings_ok", message))
    combos = combinations(request, control, parameters)
    files = model_catalog()[0]
    if not files:
        message = "Fusion's model list (app/processors/models_data.py) is unreadable"
        checks.append(finding("model_assets", "not_checked", "model_list_unreadable", message))
        return combos, []
    needed, unknown, models, absent = set(), [], [], []
    for entry in combos:
        found, unmapped = required_models(entry["choice"])
        needed |= found
        unknown += [item for item in unmapped if item not in unknown]
    for name in sorted(needed):
        path = files.get(name, (None, None))[0]
        entry = {"model": name, "file": path, "present": False}
        if path is None:
            unknown.append(f"model {name}")
        else:
            with contextlib.suppress(OSError):
                info = Path(path).stat()
                entry.update(
                    present=info.st_size > 0, bytes=info.st_size, mtime_ns=info.st_mtime_ns
                )
            if not entry["present"]:
                absent.append(path)
        models.append(entry)
    evidence = {"models": models, "unmapped": unknown}
    if absent:
        action = act(
            "Install the missing files in the Fusion checkout, for example with Fusion's "
            "download_models.py in its environment (operator)"
        )
        message = f"Missing model files: {', '.join(absent)}"
        checks.append(
            finding("model_assets", "blocked", "model_missing", message, evidence, [action])
        )
    elif unknown:
        message = f"Model files of {', '.join(unknown)} are not known to this adapter"
        checks.append(finding("model_assets", "not_checked", "model_unmapped", message, evidence))
    else:
        message = f"All {len(needed)} model files the selected settings load are present"
        checks.append(finding("model_assets", "passed", "models_present", message, evidence))
    return combos, models


def storage_check(media):
    usage = shutil.disk_usage(STATE)
    needed = media["estimated_bytes"] if media else PREVIEW_BYTES
    evidence = {
        "free_bytes": usage.free,
        "estimated_bytes": needed,
        "basis": "estimated from the source bit rate and resolution; nothing is reserved",
    }
    prune = act("Report old jobs and uploads to remove", "prune_storage", dry_run=True)
    if not (writable(STATE) and writable(LOCK_DIR)):
        message = "state_dir or lock_dir is not writable"
        return finding("storage", "blocked", "state_not_writable", message, evidence)
    if usage.free < needed:
        message = f"{usage.free} bytes free, below the estimated {needed}"
        return finding("storage", "blocked", "insufficient_space", message, evidence, [prune])
    if usage.free < needed + HEADROOM:
        message = f"{usage.free} bytes free leaves under 1 GiB beyond the estimated {needed}"
        return finding("storage", "warning", "low_space", message, evidence, [prune])
    return finding("storage", "passed", "space_ok", "Free space exceeds the estimate", evidence)


def qualification_check(mode, combos, mappings, hardware):
    entries, covered = [], True
    for combo in combos:
        choice, matches, nearest = combo["choice"], [], None
        for record in QUALIFICATIONS:
            missing = differences(record, mode, choice, mappings, hardware)
            if not missing:
                matches.append({key: record[key] for key in ("id", "date", "source", "hardware")})
            elif nearest is None or len(missing) < len(nearest["differs_in"]):
                nearest = {"record": record["id"], "differs_in": missing}
        entry = {"mapping": combo["mapping"], "combination": choice, "qualified_by": matches}
        if combo["segment"] is not None:
            entry["segment"] = combo["segment"]
        if not matches:
            covered = False
            entry["nearest"] = nearest
        entries.append(entry)
    evidence = {"hardware": hardware, "mappings": entries}
    if covered:
        message = "Recorded GPU qualification covers these settings on this hardware class"
        return finding("qualification", "passed", "qualified", message, evidence)
    message = (
        "Unqualified: no recorded GPU qualification covers this combination on this hardware "
        "class. It is allowed, not refused; preview it and inspect the result first."
    )
    action = act("Preview a representative frame and inspect it before a long render")
    return finding("qualification", "warning", "unqualified", message, evidence, [action])


def well_formed(request):
    mappings = request.get("mappings")
    return (
        isinstance(request.get("media_path"), str)
        and isinstance(request.get("controls") or {}, dict)
        and isinstance(mappings, list)
        and all(
            isinstance(m, dict)
            and isinstance(m.get("target_face"), str)
            and isinstance(m.get("source_faces"), list)
            and all(isinstance(path, str) for path in m["source_faces"])
            and isinstance(m.get("parameters") or {}, dict)
            for m in mappings
        )
    )


def preflight(payload):
    """Request-specific readiness of a submit_swap request without starting it (README)."""
    request = payload.get("request")
    if not isinstance(request, dict) or request.get("kind") not in ("preview", "render"):
        raise ValueError("preflight checks a submit_swap request: kind preview or render")
    analyze = payload.get("analyze", False)
    if not isinstance(analyze, bool):
        raise ValueError("analyze must be a boolean")
    checked_at, checks = time.time(), []

    def skip(name, reason):
        checks.append(finding(name, "not_checked", "prerequisite_blocked", reason))

    if CONFIG_ERROR:
        checks.append(finding("configuration", "blocked", "config_invalid", CONFIG_ERROR))
        for name in ("fusion_revision", "deployed_code", "job_id", *INPUT_CHECKS):
            skip(name, "The remote configuration is invalid")
        for name in ("gpu", "qualification", "request", "reference_analysis"):
            skip(name, "The remote configuration is invalid")
        return {
            "readiness": "blocked",
            "checked_at": checked_at,
            "fingerprint": None,
            "not_checked": [c["check"] for c in checks if c["status"] == "not_checked"],
            "checks": checks,
        }
    checks.append(finding("configuration", "passed", "config_valid", "The configuration is valid"))
    revision, reason = fusion_compatibility(APP)  # The check health and GPU workers make.
    evidence = {"revision": revision, "supported_revision": REVISION, "release": RELEASE.name}
    if reason:
        action = act(
            "Return the Fusion checkout to the supported revision without tracked changes, or "
            "qualify the adapter for the new one (README, Fusion revision guard)"
        )
        checks.append(
            finding("fusion_revision", "blocked", "fusion_incompatible", reason, evidence, [action])
        )
    else:
        message = "Fusion is at the supported revision without tracked changes"
        checks.append(finding("fusion_revision", "passed", "fusion_compatible", message, evidence))
    code, client = code_identity()["code"], payload.get("client_code")
    if not isinstance(client, dict):
        message = "The client sent no code hashes to compare"
        checks.append(finding("deployed_code", "not_checked", "client_code_unknown", message))
    elif client != code:
        action = act("Deploy this checkout with scripts/deploy.sh, then preflight again")
        message = "The deployed worker code differs from the client's checkout"
        checks.append(finding("deployed_code", "warning", "code_mismatch", message, {}, [action]))
    else:
        message = "The deployed worker code matches the client's checkout"
        checks.append(finding("deployed_code", "passed", "code_matches", message))

    request = dict(request)
    job_id = request.pop("job_id", None)  # Submission stores the request without it.
    try:
        existing = None if job_id is None else job_path(job_id) / "request.json"
        if existing is None:
            message = "No job_id given; submit_swap generates a new one"
            checks.append(finding("job_id", "passed", "job_id_generated", message))
        elif not existing.exists():
            checks.append(finding("job_id", "passed", "job_id_free", f"No job {job_id} exists"))
        elif canonical(read_json(existing)) == canonical(request):
            message = f"Job {job_id} has this request; submit_swap returns it rather than rerun"
            evidence = {"status": current(existing.parent)["status"]}
            checks.append(finding("job_id", "warning", "job_exists", message, evidence))
        else:
            message = f"Job {job_id} exists with a different request; submit_swap refuses it"
            action = act("Choose another job_id")
            checks.append(finding("job_id", "blocked", "job_id_conflict", message, {}, [action]))
    except (OSError, ValueError, KeyError) as error:
        checks.append(finding("job_id", "blocked", "job_id_invalid", str(error)))

    source = media = combos = None
    references, digests, models = [], [], []
    if well_formed(request):
        source, references, digests = path_checks(request, checks)
        combos, models = settings_checks(request, checks)
        if source is None:
            skip("media", "The media path is not an accepted input")
        else:
            check, media = media_check(request, source)
            checks.append(check)
        checks.append(storage_check(media))
    else:
        for name in INPUT_CHECKS:
            skip(name, "The request is not a well-formed submit_swap request")

    hardware = gpu_observation()
    if hardware is None:
        message = "nvidia-smi did not report the configured GPU"
        checks.append(finding("gpu", "not_checked", "gpu_unobserved", message))
    else:
        message = "GPU class and memory use as observed now; nothing is reserved"
        checks.append(finding("gpu", "passed", "gpu_observed", message, hardware))
    if combos is None:
        skip("qualification", "The settings, and so the model choices, are unknown")
    else:
        checks.append(
            qualification_check(request["kind"], combos, len(request["mappings"]), hardware)
        )

    try:
        validate_request(copy.deepcopy(request))  # Exactly what submission runs first.
    except Exception as error:
        message = f"Submission would refuse the request now: {error}"
        checks.append(finding("request", "blocked", refusal_code(error), message))
    else:
        message = "submit_swap's validation accepts the request now; submission validates again"
        checks.append(finding("request", "passed", "request_valid", message))

    status = {check["check"]: check["status"] for check in checks}
    if not analyze or (status["fusion_revision"] == status["request"] == "passed"):
        checks.append(analysis_check(request, digests, analyze))
    else:
        skip("reference_analysis", "Fusion or the request is blocked; nothing was queued")

    worst = min((check["status"] for check in checks), key=WORST.index)
    media_identity = None
    with contextlib.suppress(AttributeError, OSError):  # No accepted media, or it vanished.
        info = source.stat()
        media_identity = [str(source), info.st_size, info.st_mtime_ns]
    inputs = {
        "media": media_identity,
        "references": [[item["path"], item.get("sha256")] for item in references],
    }
    configuration = {
        "config_sha256": SETTINGS.sha256,
        "revision": revision,
        "code": code,
        "models": [[m["model"], m.get("bytes"), m.get("mtime_ns")] for m in models],
    }
    fingerprint = {
        "request": sha256_of(request),
        "inputs": sha256_of(inputs),
        "configuration": sha256_of(configuration),
    }
    fingerprint["sha256"] = sha256_of(fingerprint)
    return {
        "readiness": READINESS.get(worst, "ready"),
        "checked_at": checked_at,
        "fingerprint": fingerprint,
        "not_checked": [check["check"] for check in checks if check["status"] == "not_checked"],
        "checks": checks,
        "notice": "A preflight reserves nothing. submit_swap validates the request, inputs and "
        "settings again, and the worker rechecks Fusion, the media and the reference bytes.",
    }


def dispatch(action, request):
    if action == "health":
        return health()
    if action == "preflight":
        return preflight(request)
    if CONFIG_ERROR:
        raise ValueError(CONFIG_ERROR)
    if action == "settings":
        controls, parameters = layouts()
        for key, value in ADAPTER_CONTROLS.items():
            if key in controls:
                controls[key]["default"] = value
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
    if action == "resume":
        return resume_render(request)
    if action == "list":
        return list_jobs(request.get("status"), request.get("kind"), request.get("limit", 50))
    if action == "artifact":
        descriptor, path, size = open_artifact(request["job_id"], request["name"])
        os.close(descriptor)
        return {"path": str(path), "size": size}
    if action == "delete":
        return delete_job(request["job_id"])
    if action == "prune":
        return prune(request["older_than_days"], request.get("dry_run", True))
    if action == "find_upload":
        return find_upload(request["sha256"], request["suffix"], request["size"])
    if action == "promote":
        return promote(request.get("force", False))
    if action == "recipe_save":
        return save_recipe(request)
    if action == "recipe_get":
        return get_recipe(request)
    if action == "recipe_run":
        return run_recipe(request)
    if action == "recipe_import":
        return import_recipe(request)
    if action == "recipe_delete":
        return delete_recipe(request["name"], request.get("revision"))
    raise ValueError(f"Unknown action {action}")


def open_artifact(job_id, name):
    """Open a completed job's artifact through one job-directory descriptor.

    State, size and bytes then belong to one job generation, even if the job is deleted and its
    ID reused while a transfer runs. Symlinks and names other than one plain component are refused.
    """
    directory = job_path(job_id)
    if not isinstance(name, str) or name in ("", ".", "..") or "/" in name or "\0" in name:
        raise ValueError("Artifact path escapes the job")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptors = []
    try:
        try:
            descriptors.append(os.open(directory, flags | os.O_DIRECTORY))
            with open(os.open("state.json", flags | os.O_NONBLOCK, dir_fd=descriptors[0])) as state:
                completed = json.load(state).get("status") == "completed"
            if not completed:
                raise ValueError("Artifacts become available after successful completion")
            descriptors.append(os.open("artifacts", flags | os.O_DIRECTORY, dir_fd=descriptors[0]))
            descriptor = os.open(name, flags | os.O_NONBLOCK, dir_fd=descriptors[1])
        except OSError as error:
            if error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise ValueError("Artifact path escapes the job") from None
            raise
    finally:
        for value in descriptors:
            os.close(value)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise ValueError("Artifact path escapes the job")
    return descriptor, directory / "artifacts" / name, info.st_size


def stream_artifact(job_id, name, limit=None):
    """Write a JSON header line with the artifact's size, then exactly that many of its bytes."""
    descriptor, path, size = open_artifact(job_id, name)
    with open(descriptor, "rb") as source:
        if limit is not None and size > int(limit):
            raise ValueError(f"Artifact is larger than the {int(limit)}-byte limit")
        stdout = sys.stdout.buffer
        stdout.write(json.dumps({"result": {"path": str(path), "size": size}}).encode() + b"\n")
        stdout.flush()
        remaining = size
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Artifact shrank during transfer")
            stdout.write(chunk)
            remaining -= len(chunk)
        stdout.flush()


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
    # Prune decides under the same lock: it either sees this refresh or removed the file first.
    with retention_lock():
        try:
            info = destination.lstat()
            if destination.is_symlink() or not destination.is_file() or info.st_size != size:
                return None
            os.utime(destination)  # Reuse counts as recent use for retention pruning.
        except FileNotFoundError:
            return None
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
        # Prune checks age and removes under this lock, so it never removes a fresh publication.
        with retention_lock():
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), "bytes": size, "sha256": digest.hexdigest(), "reused": False}


# Recipes (DESIGN.md, Recipes): immutable revisions in <state>/recipes/<name>/<revision>.json.
RECIPE_FORMAT, RECIPE_VERSION = "visomaster-recipe", 1
RECIPE_KEYS = {
    "format",
    "version",
    "name",
    "revision",
    "created_at",
    "note",
    "origin",
    "request",
    "frame_convention",
    "provenance",
    "retained",
    "sha256",
}
GENERATED = {"format", "version", "name", "revision", "sha256"}  # Set by publish_recipe.
PROVENANCE_KEYS = {"settings", "fusion_revision", "adapter", "config_sha256", "models"}
MAPPING_KEYS = (
    "target_face",
    "target_sha256",
    "source_faces",
    "source_sha256",
    "source_weights",
    "parameters",
)
RANGE = {"preview": ("frame",), "render": ("start_frame", "max_frames")}
# Request fields a recipe carries besides its inputs and range; priority only schedules.
CARRIED = ("controls", "report", "priority", "segments")
INTENT_KEYS = {"kind", "media_path", "mappings", "frame", "start_frame", "max_frames", *CARRIED}
RUN_KEYS = {
    "name",
    "revision",
    "job_id",
    "mode",
    "media_path",
    "targets",
    "range",
    "report",
    "priority",
    "accept_changes",
}
FRAME_CONVENTION = (
    "Zero-based frame indices in decoded order. A preview processes frame; a render processes "
    "start_frame through start_frame + max_frames - 1, or through the last frame when "
    "max_frames is null."
)
REVISION_FILE = re.compile(r"([1-9][0-9]{0,8})\.(json|deleted)")


def recipe_directory(name):
    if not isinstance(name, str) or not JOB_ID.fullmatch(name):
        raise ValueError(
            "Recipe names are 1–64 letters, digits, underscores or hyphens, starting with a "
            "letter or digit"
        )
    return STATE / "recipes" / name


def revisions(directory):
    """Stored revision numbers, and the highest number ever used (deleted ones included)."""
    stored, used = [], 0
    for path in listing(directory):
        match = REVISION_FILE.fullmatch(path.name)
        if match:
            used = max(used, int(match[1]))
            if match[2] == "json":
                stored.append(int(match[1]))
    return sorted(stored), used


def recipe_digest(document):
    body = {key: value for key, value in document.items() if key != "sha256"}
    return hashlib.sha256(canonical(body).encode()).hexdigest()


def check_recipe(document, where):
    """Schema and integrity of a recipe document; raises ValueError saying what to do."""
    if not isinstance(document, dict) or document.get("format") != RECIPE_FORMAT:
        raise ValueError(f"{where} is not a VisoMaster recipe (format {RECIPE_FORMAT!r})")
    if document.get("version") != RECIPE_VERSION:
        raise ValueError(
            f"{where} has recipe format version {document.get('version')!r}, but this release "
            f"reads version {RECIPE_VERSION} only; use a release that reads it, or save the "
            "recipe again with this one"
        )
    if document.get("sha256") != recipe_digest(document):
        raise ValueError(
            f"{where} fails its integrity check: its content does not match its sha256, so it "
            "was edited or damaged after it was written. Export it again from its source"
        )
    if set(document) != RECIPE_KEYS:
        fields = sorted(set(document) ^ RECIPE_KEYS)
        raise ValueError(f"{where} has unexpected or missing fields: {', '.join(fields)}")
    recipe_directory(document["name"])
    request, record = document["request"], document["provenance"]
    if type(document["revision"]) is not int or document["revision"] < 1:
        raise ValueError(f"{where} has an invalid revision")
    if not isinstance(record, dict) or set(record) != PROVENANCE_KEYS:
        raise ValueError(f"{where} has malformed provenance")
    try:
        allowed = {"kind", "media_sha256", "media_path", "mappings", *RANGE[request["kind"]]}
        inputs = recorded_inputs(request)
    except (KeyError, TypeError, AttributeError):
        raise ValueError(f"{where} has a malformed request") from None
    if set(request) - allowed - set(CARRIED) or not request["mappings"]:
        raise ValueError(f"{where} has a malformed request")
    for place, entry in inputs.items():
        path, digest = entry["path"], entry["sha256"]
        if not isinstance(path, str) or not os.path.isabs(path):
            raise ValueError(f"{where}: {place} must be an absolute path")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError(f"{where}: {place} has no valid SHA-256")
    return document


def read_recipe(name, revision=None):
    directory = recipe_directory(name)
    stored, _ = revisions(directory)
    if not stored:
        raise ValueError(f"No recipe named {name}; get_recipe without a name lists them")
    revision = stored[-1] if revision is None else revision
    if revision not in stored or type(revision) is not int:
        raise ValueError(f"Recipe {name} has no revision {revision}; it has {stored}")
    where = f"Recipe {name} revision {revision}"
    try:
        document = read_json(directory / f"{revision}.json")
    except ValueError:
        raise ValueError(f"{where} is not valid JSON; delete it with delete_recipe") from None
    check_recipe(document, where)
    if (document["name"], document["revision"]) != (name, revision):
        raise ValueError(f"{where} names another recipe or revision; it was moved or copied")
    return document, stored


def publish_recipe(name, fields):
    """Write the next revision of `name`. A complete file is hard-linked into place, which fails
    rather than replace an existing revision, so concurrent writers each get their own."""
    directory = recipe_directory(name)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    while True:
        revision = revisions(directory)[1] + 1
        document = {"format": RECIPE_FORMAT, "version": RECIPE_VERSION, **fields}
        document.update(name=name, revision=revision)
        document["sha256"] = recipe_digest(document)
        fd, temporary = tempfile.mkstemp(dir=directory, prefix=".write-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(document, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, directory / f"{revision}.json")
            return document
        except FileExistsError:
            continue
        finally:
            os.unlink(temporary)


def recipe_request(request):
    """The explicit swap request a recipe stores: resolved inputs, range and carried fields."""
    kind = request["kind"]
    template = {"kind": kind, "media_path": request["media_path"]}
    if request.get("media_sha256") is not None:
        template["media_sha256"] = request["media_sha256"]
    for key in RANGE[kind]:
        template[key] = request.get(key, None if key == "max_frames" else 0)
    for key in CARRIED:
        if request.get(key) is not None:
            template[key] = copy.deepcopy(request[key])
    template["mappings"] = [
        {key: copy.deepcopy(mapping[key]) for key in MAPPING_KEYS if mapping.get(key) is not None}
        for mapping in request["mappings"]
    ]
    return template


def apply_digests(request, inputs):
    """Set the request's input fingerprints to those of `inputs` (see recorded_inputs)."""
    request["media_sha256"] = inputs["media_path"]["sha256"]
    for index, mapping in enumerate(request["mappings"]):
        place = f"mappings[{index}]"
        mapping["target_sha256"] = inputs[f"{place}.target_face"]["sha256"]
        mapping["source_sha256"] = [
            inputs[f"{place}.source_faces[{number}]"]["sha256"]
            for number in range(len(mapping["source_faces"]))
        ]


def set_input(request, place, path):
    if place == "media_path":
        request["media_path"] = path
        return
    match = re.fullmatch(r"mappings\[(\d+)\]\.(?:target_face|source_faces\[(\d+)\])", place)
    mapping = request["mappings"][int(match[1])]
    if match[2] is None:
        mapping["target_face"] = path
    else:
        mapping["source_faces"][int(match[2])] = path


def pin_settings(request, settings):
    """Pass every adjustable setting explicitly at its effective value, so a later default
    change cannot alter what the recipe runs with."""
    controls, parameters = layouts()
    request["controls"] = {
        key: settings["controls"][key] for key in sorted(CONTROLS) if key in controls
    }
    for mapping, effective in zip(request["mappings"], settings["parameters"]):
        mapping["parameters"] = {
            key: effective[key] for key in sorted(PARAMETERS) if key in parameters
        }


def unavailable(inputs):
    return [
        f"{place} {entry['path']}" for place, entry in inputs.items() if entry["sha256"] is None
    ]


def retain(request):
    """Copy the request's face references into the content-addressed upload store, pointing the
    request at the copies; returns {copy: original} and the staged (temporary, copy) pairs, which
    publish_retained publishes under the retention lock. References whose type has no upload
    name stay where they are, unretained."""
    retained, staged = {}, []

    def keep(path, digest):
        source = Path(path)
        try:
            destination, _ = upload_destination(digest, source.suffix.lower(), 1)
        except ValueError:
            return path
        if source == destination or str(destination) in retained:
            retained.setdefault(str(destination), path)
            return str(destination)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = destination.with_name(f"{destination.name}.{os.urandom(4).hex()}.partial")
        staged.append((temporary, destination))
        hasher = hashlib.sha256()
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                hasher.update(chunk)
                writer.write(chunk)
        if hasher.hexdigest() != digest:
            raise ValueError(f"Reference {path} changed while it was copied; save the recipe again")
        retained[str(destination)] = path
        return str(destination)

    try:
        for mapping in request["mappings"]:
            mapping["target_face"] = keep(mapping["target_face"], mapping["target_sha256"])
            mapping["source_faces"] = [
                keep(path, digest)
                for path, digest in zip(mapping["source_faces"], mapping["source_sha256"])
            ]
    except BaseException:
        discard(staged)
        raise
    return retained, staged


def publish_retained(staged):
    """Publish staged copies; the caller holds the retention lock, so prune sees them."""
    for temporary, destination in staged:
        if destination.is_file() and not destination.is_symlink():
            os.utime(destination)  # Identical bytes by name: reuse counts as recent use.
        else:
            os.replace(temporary, destination)


def discard(staged):
    for temporary, _ in staged:
        temporary.unlink(missing_ok=True)


def store_recipe(name, request, fields):
    """Retain the request's references and write the next revision under the retention lock."""
    retained, staged = retain(request)
    try:
        with retention_lock():
            publish_retained(staged)
            document = publish_recipe(
                name,
                {
                    "created_at": time.time(),
                    "frame_convention": FRAME_CONVENTION,
                    "request": request,
                    "retained": retained,
                    **fields,
                },
            )
    finally:
        discard(staged)
    return {"recipe": document, "retention": retention_report(document["request"])}


def retention_report(request):
    """Per input: retained (a managed file retention keeps while a recipe names it) or
    unretained (outside state storage: a change or removal is detected, not prevented), and
    missing when the path no longer exists."""
    report = {}
    for place, entry in recorded_inputs(request).items():
        path = Path(entry["path"])
        kept = STATE is not None and path.is_relative_to(STATE)
        report[place] = ("retained" if kept else "unretained") + (
            "" if path.is_file() else ", missing"
        )
    return report


def job_recipe(job_id):
    """A completed swap job's request, as resolved and fingerprinted at submission, with the
    release, revision, configuration and models of its execution."""
    directory = job_path(job_id)
    resolved = directory / "resolved.json"
    try:
        state = read_json(directory / "state.json")
        request = read_json(resolved if resolved.exists() else directory / "request.json")
        if state.get("status") != "completed" or request.get("kind") not in RANGE:
            raise ValueError(
                f"Job {job_id} is a {state.get('status')} {request.get('kind')} job; recipes "
                "come from completed preview or render jobs"
            )
        result = read_json(directory / "result.json")
    except (OSError, AttributeError, json.JSONDecodeError):
        raise ValueError(f"Job {job_id} has no readable state, request and result") from None
    if not resolved.exists():
        request = resolve_inputs(request)  # Submitted by an earlier release.
    release = state.get("release")
    recorded = {
        "fusion_revision": result.get("revision"),
        "adapter": code_identity(release) if release else {"release": None, "code": None},
        "config_sha256": state.get("config_sha256"),
    }
    origin = {"kind": "job", "job_id": job_id, "finished_at": state.get("finished_at")}
    if isinstance(request.get("recipe"), dict):
        parent = request["recipe"]
        origin["recipe"] = {key: parent.get(key) for key in ("name", "revision", "sha256")}
    return recipe_request(request), recorded, result.get("models") or [], origin


def save_recipe(request):
    """Save a new revision of recipe `name` from a completed job or an explicit request."""
    name, note = request.get("name"), request.get("note")
    recipe_directory(name)
    if note is not None and (not isinstance(note, str) or len(note) > 2000):
        raise ValueError("note must be text of at most 2000 characters")
    intent = {key: value for key, value in request.items() if key not in {"name", "note"}}
    if request.get("job_id") is not None:
        if set(intent) != {"job_id"}:
            raise ValueError("Save a recipe from either job_id or an explicit request, not both")
        template, recorded, models, origin = job_recipe(request["job_id"])
    else:
        intent.pop("job_id", None)
        extra = sorted(set(intent) - INTENT_KEYS)
        if extra:
            raise ValueError(f"Recipes do not accept {', '.join(extra)}")
        if intent.get("kind") not in RANGE:
            raise ValueError("A recipe is a preview or render request with explicit mappings")
        validate_request(intent)
        template, recorded, models = recipe_request(resolve_inputs(intent)), {}, []
        origin = {"kind": "request"}
    current = provenance(template, models)
    missing = unavailable(current["inputs"])
    if missing:
        raise ValueError(f"Inputs are unavailable: {'; '.join(missing)}")
    for place, entry in recorded_inputs(template).items():
        if entry["sha256"] not in (None, current["inputs"][place]["sha256"]):
            raise ValueError(
                f"{place} {entry['path']} no longer has the bytes the job used (SHA-256 "
                f"{entry['sha256']}); the recipe cannot record them. Save it from an explicit "
                "request instead, or rerun the job"
            )
    if recorded and recorded["fusion_revision"] != current["fusion_revision"]:
        raise ValueError(
            f"Job ran Fusion {recorded['fusion_revision']}, but the workstation now has "
            f"{current['fusion_revision']}, so its effective defaults cannot be reconstructed; "
            "save the recipe from an explicit request instead"
        )
    apply_digests(template, current["inputs"])
    pin_settings(template, current["settings"])
    record = {key: current[key] for key in PROVENANCE_KEYS}
    record.update(recorded)
    return store_recipe(name, template, {"note": note, "origin": origin, "provenance": record})


def get_recipe(request):
    """List recipes, or read one revision; with verify, hash its inputs and models now."""
    name = request.get("name")
    if name is None:
        return {"recipes": list_recipes()}
    document, stored = read_recipe(name, request.get("revision"))
    result = {
        "recipe": document,
        "revisions": stored,
        "retention": retention_report(document["request"]),
    }
    if request.get("verify"):
        template = document["request"]
        recorded = {**document["provenance"], "inputs": recorded_inputs(template)}
        current = provenance(template, document["provenance"]["models"])
        found = provenance_changes(recorded, current)
        result["verification"] = {"unchanged": not found, "changes": found}
    return result


def list_recipes():
    entries = []
    for directory in listing(STATE / "recipes"):
        if not JOB_ID.fullmatch(directory.name) or not directory.is_dir():
            continue
        stored, _ = revisions(directory)
        if not stored:
            continue
        entry = {"name": directory.name, "revisions": stored}
        try:
            document, _ = read_recipe(directory.name)
        except (OSError, ValueError) as error:
            entry["error"] = str(error)
        else:
            entry.update(
                kind=document["request"]["kind"],
                media_path=document["request"]["media_path"],
                created_at=document["created_at"],
                origin=document["origin"]["kind"],
                note=document["note"],
            )
        entries.append(entry)
    return entries


def delete_recipe(name, revision=None):
    """Delete one revision, or all; a tombstone keeps its number from being reused. Inputs are
    never deleted here: retention may remove them once no recipe or active job names them."""
    directory = recipe_directory(name)
    with retention_lock():
        stored, _ = revisions(directory)
        chosen = stored if revision is None else [revision]
        if not stored or not set(chosen) <= set(stored):
            raise ValueError(f"Recipe {name} has no revision {revision}; it has {stored}")
        for number in chosen:
            (directory / f"{number}.deleted").touch()
            (directory / f"{number}.json").unlink()
    return {"name": name, "deleted": chosen, "remaining": sorted(set(stored) - set(chosen))}


def instantiate(template, request):
    """The recipe's request with the caller's explicit substitutions, and their record."""
    new, substitutions = copy.deepcopy(template), {}
    mode = request.get("mode") or template["kind"]
    if mode not in RANGE:
        raise ValueError("mode must be preview or render")
    if mode != template["kind"]:
        substitutions["mode"] = {"recorded": template["kind"], "used": mode}
        new["kind"] = mode
        for key in RANGE[template["kind"]]:
            new.pop(key)
        for key in ("report", "segments") if mode == "preview" else ():
            if key in new:
                substitutions[key] = {"recorded": new.pop(key), "used": None}
    if request.get("media_path") is not None:
        path = str(input_path(request["media_path"]))
        if path != template["media_path"]:
            new["media_path"] = path
            substitutions["media_path"] = {"recorded": template["media_path"], "used": path}
    targets = request.get("targets")
    if targets is not None:
        count = len(new["mappings"])
        if not isinstance(targets, list) or len(targets) != count:
            raise ValueError(f"targets must list one target_face per recipe mapping ({count})")
        for index, (mapping, target) in enumerate(zip(new["mappings"], targets)):
            path = str(input_path(target))
            if path != mapping["target_face"]:
                substitutions.setdefault("targets", []).append(
                    {"mapping": index, "recorded": mapping["target_face"], "used": path}
                )
                mapping["target_face"] = path
    frames = request.get("range")
    if frames is not None:
        if not isinstance(frames, dict) or set(frames) != set(RANGE[mode]):
            raise ValueError(f"range for a {mode} gives exactly {' and '.join(RANGE[mode])}")
        recorded = {key: template[key] for key in RANGE[template["kind"]]}
        new.update(frames)
        if frames != recorded:
            substitutions["range"] = {"recorded": recorded, "used": frames}
    elif "mode" in substitutions:
        raise ValueError(
            f"A {mode} of a {template['kind']} recipe needs an explicit range "
            f"({' and '.join(RANGE[mode])})"
        )
    if request.get("report") is not None:
        if new.get("report") != request["report"]:
            substitutions["report"] = {"recorded": new.get("report"), "used": request["report"]}
        new["report"] = request["report"]
    if request.get("priority") is not None:
        new["priority"] = request["priority"]
    return new, substitutions


def run_recipe(request):
    """Submit a new swap job from a recipe revision, with explicit, recorded substitutions."""
    extra = sorted(set(request) - RUN_KEYS)
    if extra:
        raise ValueError(f"run_recipe does not accept {', '.join(extra)}")
    accept = request.get("accept_changes", False)
    if not isinstance(accept, bool):
        raise ValueError("accept_changes must be a boolean")
    job_id = request["job_id"]
    directory = job_path(job_id)
    invocation = {key: request.get(key) for key in sorted(RUN_KEYS - {"job_id"})}
    # A retry returns the job before anything is hashed, so it answers after inputs are pruned.
    if (directory / "request.json").exists():
        try:
            previous = read_json(directory / "request.json")["recipe"]["invocation"]
        except (OSError, ValueError, KeyError, TypeError):
            previous = None
        if previous is None or canonical(previous) != canonical(invocation):
            raise ValueError("job_id already exists with a different request")
        return status(job_id)
    document, _ = read_recipe(request["name"], request.get("revision"))
    template = document["request"]
    new, substitutions = instantiate(template, request)
    current = provenance(new, document["provenance"]["models"])
    missing = unavailable(current["inputs"])
    if missing:
        raise ValueError(
            f"Recipe inputs are unavailable: {'; '.join(missing)}. Restore or re-upload them "
            "(upload_media stores identical bytes at uploads/<sha256><suffix>), or substitute "
            "media_path or targets explicitly"
        )
    media = current["inputs"]["media_path"]["sha256"]
    if "media_path" in substitutions:
        substitutions["media_path"].update(recorded_sha256=template["media_sha256"], sha256=media)
        expected = (template.get("report") or {}).get("expected_visible")
        needed = [key for key in ("targets", "range") if request.get(key) is None]
        needed += ["report"] if expected and request.get("report") is None else []
        if media != template["media_sha256"] and template.get("segments"):
            raise ValueError(
                "media_path has different content from the recipe's clip, and the recipe's "
                "segments name frames of that clip; save a recipe for the new clip with its own "
                "segments"
            )
        if media != template["media_sha256"] and needed:
            raise ValueError(
                "media_path has different content from the recipe's clip, so applying the "
                f"recipe to it needs explicit {', '.join(needed)}: targets lists one "
                "target_face per mapping from a detection on the new clip, range the frames to "
                "process, and report replaces expected_visible ranges of the recipe's clip. A "
                "target from another clip is not assumed to be valid"
            )
    recorded = {**document["provenance"], "inputs": recorded_inputs(template)}
    found = provenance_changes(recorded, current)
    if found and not accept:
        raise ValueError(
            "Recipe "
            f"{document['name']} revision {document['revision']} no longer matches the "
            "workstation:\n- "
            + "\n- ".join(change["message"] for change in found)
            + "\nReview the changes (get_recipe with verify), then rerun with "
            "accept_changes=true to run with the current inputs, models and code, or save a "
            "new revision"
        )
    apply_digests(new, current["inputs"])
    new["recipe"] = {
        "name": document["name"],
        "revision": document["revision"],
        "sha256": document["sha256"],
        "invocation": invocation,
        "substitutions": substitutions,
        "accepted_changes": found,
    }
    return submit({"job_id": job_id, **new})


def import_recipe(request):
    """Store an exported recipe as a new revision here, after checking its schema, integrity,
    inputs (allowed paths with their recorded bytes) and settings."""
    document = request.get("document")
    check_recipe(document, "The imported recipe")
    name = request.get("name") or document["name"]
    recipe_directory(name)
    template = copy.deepcopy(document["request"])
    relocated, missing = [], []
    for place, entry in recorded_inputs(template).items():
        if current_digest(entry["path"]) == entry["sha256"]:
            continue
        # Uploads are named by content, so identical bytes uploaded here are found by digest.
        upload = STATE / "uploads" / (entry["sha256"] + Path(entry["path"]).suffix.lower())
        if current_digest(upload) == entry["sha256"]:
            set_input(template, place, str(upload))
            relocated.append({"input": place, "recorded": entry["path"], "used": str(upload)})
        else:
            missing.append(f"{place} {entry['path']} (SHA-256 {entry['sha256']})")
    if missing:
        raise ValueError(
            "Cannot import: these inputs are not available here with their recorded bytes: "
            + "; ".join(missing)
            + ". Upload each file with upload_media (identical bytes are stored as "
            "uploads/<sha256><suffix>, which import finds) or place it at its recorded path, "
            "then import again"
        )
    validate_request(template)  # Allowed paths, reference bytes and installed settings.
    source = {key: document[key] for key in ("name", "revision", "sha256", "created_at")}
    origin = {"kind": "import", "source": source, "source_origin": document["origin"]}
    origin["relocated"] = relocated
    fields = {key: document[key] for key in ("note", "provenance")}
    return store_recipe(name, template, {**fields, "origin": origin})


def rerun(release, payload):
    """Repeat this command from `release`, with the same request on standard input."""
    with tempfile.TemporaryFile() as stream:
        stream.write(payload)
        stream.seek(0)
        os.dup2(stream.fileno(), 0)
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, str(release / "control.py"), *sys.argv[1:]])


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "supervise":
        supervise(sys.argv[2], sys.argv[3])
    elif len(sys.argv) in (4, 5) and sys.argv[1] == "artifact":
        try:
            if CONFIG_ERROR:
                raise ValueError(CONFIG_ERROR)
            stream_artifact(*sys.argv[2:])
        except Exception as error:
            # After the header this trails partial bytes; the client sees the exit status.
            print(json.dumps({"error": str(error)}))
            sys.exit(1)
    else:
        try:
            if len(sys.argv) == 5 and sys.argv[1] == "upload":
                if CONFIG_ERROR:
                    raise ValueError(CONFIG_ERROR)
                result = upload(*sys.argv[2:])
            else:
                raw = sys.stdin.buffer.read()
                payload = json.loads(raw)
                result = dispatch(payload.pop("action"), payload)
            print(json.dumps({"result": result}))
        except Superseded as moved:
            rerun(moved.args[0], raw)
        except Exception as error:
            print(json.dumps({"error": str(error)}))
            sys.exit(1)

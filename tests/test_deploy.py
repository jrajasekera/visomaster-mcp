"""Deployment: the active-job guard, promotion under the submit lock and per-command release pins.

Race boundaries are explicit pause points. A sitecustomize hook, armed through a file, pauses the
first process that reaches the armed point until the test resumes it: just before (`before`) or
just after (`after`) taking the submit lock, or as submission publishes a job (`publish`). The SSH
shim forwards the hook to remote commands.
"""

import json
import os
import shutil
import subprocess
import sys
import time

import pytest
from conftest import ROOT, write_config
from test_portability import ALIAS, ssh_log

SHIM = """#!/bin/sh
printf '%s\\036' "$*" >> "$SHIM_LOG"
for command; do :; done
exec env -i PATH=/usr/bin:/bin HOME="$HOME" ${PYTHONPATH:+"PYTHONPATH=$PYTHONPATH"} \\
  ${VM_HOOK_DIR:+"VM_HOOK_DIR=$VM_HOOK_DIR"} /bin/sh -c "$command"
"""
HOOK = """import fcntl, os, time

_flock, _replace = fcntl.flock, os.replace


def _pause(point):
    directory = os.environ.get("VM_HOOK_DIR")
    if not directory:
        return
    armed = os.path.join(directory, "armed")
    try:
        with open(armed) as stream:
            if stream.read() != point:
                return
        os.rename(armed, armed + ".taken")  # Only the first process claims the pause.
    except FileNotFoundError:
        return
    open(os.path.join(directory, "paused"), "w").close()
    deadline = time.monotonic() + 60
    while not os.path.exists(os.path.join(directory, "resume")) and time.monotonic() < deadline:
        time.sleep(0.01)


def flock(file, operation):
    if not str(getattr(file, "name", "")).endswith("submit.lock") or operation != fcntl.LOCK_EX:
        return _flock(file, operation)
    _pause("before")
    _flock(file, operation)
    _pause("after")


def replace(source, destination, **options):
    if os.path.basename(str(source)).startswith("prepare-"):
        _pause("publish")  # Submission is about to publish its job directory.
    return _replace(source, destination, **options)


fcntl.flock, os.replace = flock, replace
"""
# A control script that answers each list query from responses.json: status -> [stdout, exit].
STUB = """import json, sys
from pathlib import Path
status = json.load(sys.stdin)["status"]
responses = json.loads((Path(__file__).parent / "responses.json").read_text())
stdout, code = responses.get(status, ['{"result": []}', 0])
sys.stdout.write(stdout)
sys.exit(code)
"""
CODE = ("common.py", "control.py", "engine.py")


@pytest.fixture
def site(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "face.png").write_bytes(b"not decoded by the stub runner")
    deployment = tmp_path / "deploy"
    deployment.mkdir()
    write_config(deployment / "remote-config.json", media, state_dir=str(media / "state"))
    shim = tmp_path / "ssh-shim"
    shim.write_text(SHIM)
    shim.chmod(0o755)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(HOOK)
    base = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG") if k in os.environ}
    base["PYTHONPATH"] = str(hooks)
    environment = dict(
        base,
        VISOMASTER_HOST=ALIAS,
        VISOMASTER_REMOTE=str(deployment),
        VISOMASTER_SSH=str(shim),
        VISOMASTER_REMOTE_PYTHON=sys.executable,
        SHIM_LOG=str(tmp_path / "ssh.log"),
    )
    site = {
        "media": media,
        "deployment": deployment,
        "state": media / "state",
        "base": base,
        "env": environment,
        "log": tmp_path / "ssh.log",
        "tmp": tmp_path,
    }
    yield site
    # The stub runner of each submitted job finishes once the job holds a release marker.
    submitted = [job.parent for job in (site["state"] / "jobs").glob("*/worker.log")]
    for job in submitted:
        (job / "release").touch()
    until(lambda: all(state(site, job.name)["status"] in TERMINAL for job in submitted), 20)


TERMINAL = {"completed", "failed", "cancelled"}


def until(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def state(site, job_id):
    return json.loads((site["state"] / "jobs" / job_id / "state.json").read_text())


def releases(site):
    return sorted(path.name for path in (site["deployment"] / "releases").iterdir())


def selected(site):
    return os.readlink(site["deployment"] / "remote")


def install(site, name, source=ROOT / "remote"):
    release = site["deployment"] / "releases" / name
    release.mkdir(parents=True)
    for file in CODE:
        shutil.copy(source / file, release)
    link = site["deployment"] / "remote"
    link.unlink(missing_ok=True)
    link.symlink_to(f"releases/{name}")
    return release.resolve()


def hook(site, name, mode):
    """Arm a one-time pause at the next submit-lock acquisition of processes using this hook."""
    directory = site["tmp"] / f"hook-{name}"
    directory.mkdir()
    (directory / "armed").write_text(mode)
    return directory


def paused(directory):
    until(lambda: (directory / "paused").exists())


def resume(directory):
    (directory / "resume").touch()


def checkout(site, name, marker=""):
    """A copy of the files deploy.sh publishes; a marker makes its candidate distinguishable."""
    root = site["tmp"] / name
    for path in (*(f"remote/{file}" for file in CODE), "scripts/deploy.sh"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / path, root / path)
    (root / "src/visomaster_mcp").mkdir(parents=True)
    shutil.copy(ROOT / "src/visomaster_mcp/transport.py", root / "src/visomaster_mcp")
    with (root / "remote/engine.py").open("a") as stream:
        stream.write(f"# {marker}\n")
    return root


def start_deploy(site, *arguments, root=ROOT, hook_dir=None):
    environment = dict(site["env"])
    if hook_dir:
        environment["VM_HOOK_DIR"] = str(hook_dir)
    return subprocess.Popen(
        ["bash", str(root / "scripts/deploy.sh"), *arguments],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def deploy(site, *arguments, **options):
    process = start_deploy(site, *arguments, **options)
    stdout, stderr = process.communicate(timeout=120)
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def deployed(result):
    """The release name a successful deploy reports."""
    assert result.returncode == 0, result.stderr
    return result.stdout.split("Deployed ", 1)[1].split()[0]


def start_submit(site, job_id, hook_dir=None):
    """Submit through the deployment's remote link, as the MCP server does."""
    environment = dict(site["base"])
    if hook_dir:
        environment["VM_HOOK_DIR"] = str(hook_dir)
    payload = {
        "action": "submit",
        "job_id": job_id,
        "kind": "detect",
        "media_path": str(site["media"] / "face.png"),
    }
    request = site["tmp"] / f"submit-{job_id}.json"
    request.write_text(json.dumps(payload))
    with request.open() as stdin:
        return subprocess.Popen(
            [sys.executable, str(site["deployment"] / "remote/control.py")],
            env=environment,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def finish_submit(process):
    stdout, stderr = process.communicate(timeout=60)
    assert process.returncode == 0, (stdout, stderr)
    return json.loads(stdout)["result"]


def engine(site, job_id):
    """The engine path the job's worker was started with."""
    entered = site["state"] / "jobs" / job_id / "entered"
    until(entered.exists)
    return entered.read_text()


# qpn.22: the unlocked early check must establish both job states before anything is copied.


def stub_install(site, responses):
    release = site["deployment"] / "releases/stub"
    release.mkdir(parents=True)
    (release / "control.py").write_text(STUB)
    (release / "responses.json").write_text(json.dumps(responses))
    (site["deployment"] / "remote").symlink_to("releases/stub")


def assert_nothing_copied(site, link="releases/stub"):
    assert releases(site) == ["stub"]
    assert selected(site) == link
    log = ssh_log(site)
    assert len(log) == 2 and not any("tar" in line or "mkdir" in line for line in log)


@pytest.mark.parametrize(
    "responses, message",
    [
        # The reviewed failure: a failed queued query, then an empty successful running query.
        ({"queued": ["", 1]}, "(queued query): exit status 1: no output"),
        ({"running": ["", 1]}, "(running query): exit status 1: no output"),
        ({"queued": ["Traceback: broken", 0]}, "(queued query): invalid response: Traceback"),
        ({"running": ['{"error": "state unreadable"}', 1]}, "(running query): remote error: state"),
        ({"queued": ["", 0]}, "(queued query): invalid response: no output"),
        ({"running": ['{"result": {"jobs": []}}', 0]}, "(running query): invalid response"),
        ({"queued": ['{"result": [{"status": "queued"}]}', 0]}, "(queued query): invalid"),
    ],
)
def test_failed_or_malformed_job_query_aborts_before_any_copy(site, responses, message):
    stub_install(site, responses)
    result = deploy(site)
    assert result.returncode == 1
    assert "Cannot list active jobs " + message in result.stderr and "--force" in result.stderr
    assert_nothing_copied(site)


def test_active_jobs_from_either_query_are_reported(site):
    stub_install(
        site,
        {
            "queued": ['{"result": [{"job_id": "waiting", "status": "queued"}]}', 0],
            "running": ['{"result": [{"job_id": "busy", "status": "running"}]}', 0],
        },
    )
    result = deploy(site)
    assert result.returncode == 1
    assert "Refusing to deploy while jobs are active: waiting busy" in result.stderr
    assert_nothing_copied(site)


def test_remote_entry_without_control_script_is_not_a_first_install(site):
    (site["deployment"] / "remote").symlink_to("releases/pruned")
    result = deploy(site)
    assert result.returncode == 1 and "has no control.py" in result.stderr
    assert ssh_log(site) and not (site["deployment"] / "releases").exists()


def test_force_skips_the_early_check_and_still_promotes(site):
    stub_install(site, {"queued": ["", 1], "running": ["", 1]})
    name = deployed(deploy(site, "--force"))
    assert selected(site) == f"releases/{name}" and releases(site) == sorted(["stub", name])


# qpn.23: promotion, submission and cleanup coordinate through the submit lock.


@pytest.mark.parametrize("layout", ["first-install", "ordinary", "legacy"])
def test_deploy_selects_new_release_in_every_layout(site, layout):
    if layout == "ordinary":
        install(site, "r1")
    elif layout == "legacy":
        legacy = site["deployment"] / "remote"
        legacy.mkdir()
        for file in CODE:
            shutil.copy(ROOT / "remote" / file, legacy)
    result = deploy(site)
    name = deployed(result)
    assert (layout == "first-install") == ("No release is installed" in result.stderr)
    assert selected(site) == f"releases/{name}"
    for file in CODE:
        assert (site["deployment"] / "remote" / file).read_bytes() == (
            ROOT / "remote" / file
        ).read_bytes()
    others = [release for release in releases(site) if release != name]
    if layout == "ordinary":
        assert others == ["r1"]
    elif layout == "legacy":
        assert len(others) == 1 and others[0].startswith("legacy-")
        assert (site["deployment"] / "releases" / others[0] / "engine.py").is_file()
    else:
        assert others == []
    assert not list(site["deployment"].glob("remote.next*"))
    # The selected release serves the MCP server's commands.
    job = finish_submit(start_submit(site, "after-deploy"))
    assert job["status"] in {"queued", "running"}
    release = (site["deployment"] / "releases" / name).resolve()
    assert state(site, "after-deploy")["release"] == str(release)


def test_job_submitted_after_early_check_blocks_promotion(site):
    install(site, "r1")
    gate = hook(site, "deployer", "before")
    deployer = start_deploy(site, hook_dir=gate)
    paused(gate)  # Early check passed and the candidate is uploaded; promotion waits for the lock.
    candidate = [name for name in releases(site) if name != "r1"]
    assert len(candidate) == 1
    finish_submit(start_submit(site, "late"))
    resume(gate)
    stdout, stderr = deployer.communicate(timeout=120)
    assert deployer.returncode == 1, stdout
    assert "Refusing to deploy while jobs are active: late" in stderr
    assert "not promoted; remote selects releases/r1" in stderr
    assert selected(site) == "releases/r1" and releases(site) == ["r1"]
    release = site["deployment"] / "releases/r1"
    assert engine(site, "late") == str(release.resolve() / "engine.py")


def test_submission_waiting_for_the_lock_runs_the_newly_promoted_release(site):
    old = install(site, "r1")
    gate = hook(site, "submitter", "before")
    submitter = start_submit(site, "racing", hook_dir=gate)
    paused(gate)  # Pinned to r1 and about to take the submit lock.
    name = deployed(deploy(site))
    resume(gate)
    finish_submit(submitter)
    new = (site["deployment"] / "releases" / name).resolve()
    assert new != old
    assert state(site, "racing")["release"] == str(new)
    assert engine(site, "racing") == str(new / "engine.py")


@pytest.mark.parametrize("point, expected", [("after", "r2"), ("publish", "r1")])
def test_switch_during_submission_uses_one_release(site, point, expected):
    install(site, "r2")
    install(site, "r1")
    gate = hook(site, "pinned", point)
    submitter = start_submit(site, "pinned", hook_dir=gate)
    paused(gate)  # Holds the submit lock; started through the link while it selected r1.
    # An unlocked switch (a manual rollback, or a deployer from an earlier version).
    staging = site["deployment"] / "remote.manual"
    staging.symlink_to("releases/r2")
    staging.replace(site["deployment"] / "remote")
    resume(gate)
    finish_submit(submitter)
    # Before the selection check the submission retries from r2; after it, r1 stays pinned.
    release = (site["deployment"] / "releases" / expected).resolve()
    assert state(site, "pinned")["release"] == str(release)
    assert engine(site, "pinned") == str(release / "engine.py")


def test_concurrent_deploys_promote_their_own_candidates(site):
    install(site, "r1")
    first, second = checkout(site, "first", "first"), checkout(site, "second", "second")
    holding = hook(site, "first", "after")
    waiting = hook(site, "second", "before")
    one = start_deploy(site, root=first, hook_dir=holding)
    paused(holding)  # The first deployer holds the submit lock inside promotion.
    two = start_deploy(site, root=second, hook_dir=waiting)
    paused(waiting)  # The second deployer reached promotion and now waits for the lock.
    resume(waiting)
    resume(holding)
    results = [one.communicate(timeout=120), two.communicate(timeout=120)]
    assert one.returncode == 0 and two.returncode == 0, results
    names = [stdout.split("Deployed ", 1)[1].split()[0] for stdout, _ in results]
    for name, marker in zip(names, ("first", "second")):
        engine_file = site["deployment"] / "releases" / name / "engine.py"
        assert engine_file.read_text().endswith(f"# {marker}\n")
    # The second promotion followed the first, so it is selected; both candidates remain.
    assert selected(site) == f"releases/{names[1]}"
    assert releases(site) == sorted(["r1", *names])
    assert not list(site["deployment"].glob("remote.next*"))


def test_failed_copy_keeps_previous_release(site):
    install(site, "r1")
    directory = site["deployment"] / "releases"
    directory.chmod(0o555)
    try:
        result = deploy(site)
    finally:
        directory.chmod(0o755)
    assert result.returncode != 0
    assert selected(site) == "releases/r1" and releases(site) == ["r1"]


def queued_job(site, job_id, release):
    job = site["state"] / "jobs" / job_id
    (job / "artifacts").mkdir(parents=True)
    (job / "request.json").write_text(json.dumps({"kind": "detect"}))
    record = {"job_id": job_id, "status": "queued", "created_at": time.time()}
    if release is not None:
        record["release"] = str(release)
    (job / "state.json").write_text(json.dumps(record))


@pytest.mark.parametrize("recorded", [True, False])
def test_forced_cleanup_keeps_releases_active_jobs_need(site, recorded):
    paths = [install(site, f"r{index}") for index in range(7)]
    for index, path in enumerate(paths):
        stamp = time.time() - 1000 + index
        os.utime(path, (stamp, stamp))
    queued_job(site, "old-code", paths[0] if recorded else None)
    result = deploy(site, "--force")
    name = deployed(result)
    assert "active jobs: old-code" in result.stdout
    assert selected(site) == f"releases/{name}"
    if recorded:
        # Five newest kept (the new release, r6..r3), plus r0 for the queued job.
        assert releases(site) == sorted([name, "r6", "r5", "r4", "r3", "r0"])
        assert "removed releases: r2 r1" in result.stdout
    else:
        # A job from an earlier release does not say which release it needs: keep them all.
        assert releases(site) == sorted([name, *(f"r{index}" for index in range(7))])

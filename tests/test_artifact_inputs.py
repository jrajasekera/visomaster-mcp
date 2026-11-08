"""Job artifacts are inputs only once their job has completed (vmcp-n88)."""

import fcntl
import shutil

import common
import control
import pytest
import test_submit
from test_submit import complete, upload, wait_file, wait_for

# Real validation at submission and in the stub worker; normalization runs the real engine.
make_site, library = test_submit.make_site, test_submit.library


def producer(job_id, status, names=("result.mp4", "face-0.png")):
    """A job in the given state whose artifacts hold (possibly partial) files."""
    job = common.job_path(job_id)
    (job / "artifacts").mkdir(parents=True)
    common.atomic_json(job / "request.json", {"kind": "render"})
    common.atomic_json(job / "state.json", {"job_id": job_id, "status": status, "created_at": 0})
    for name in names:
        source = common.BASE / ("clip.mp4" if name.endswith(".mp4") else "face.png")
        shutil.copy(source, job / "artifacts" / name)
    return {name: str(job / "artifacts" / name) for name in names}


def consumer(site, media=None, target=None, source=None, job_id="consumer", kind="preview"):
    target = target or site.face
    mapping = {"target_face": target, "source_faces": [source or site.source], "parameters": {}}
    request = {"job_id": job_id, "kind": kind, "media_path": media or site.clip, "frame": 0}
    return {**request, "mappings": [mapping], "controls": {}}


@pytest.mark.parametrize(
    "status, message",
    [
        ("queued", "Job made is queued; wait for it to complete before using its artifacts"),
        ("running", "Job made is running; wait for it to complete before using its artifacts"),
        ("failed", r"Job made did not complete \(failed\); .*rerun it under a new job_id"),
        ("cancelled", r"Job made did not complete \(cancelled\); .*rerun it under a new job_id"),
        (None, "Job made has no readable state"),
    ],
)
@pytest.mark.parametrize("role", ["media", "target", "source"])
@pytest.mark.parametrize("external", [False, True])
def test_artifacts_of_unfinished_jobs_are_refused_at_submission(
    make_site, status, message, role, external
):
    site = make_site(external=external)
    made = producer("made", status or "completed")
    if status is None:
        (common.job_path("made") / "state.json").unlink()
    artifact = made["result.mp4" if role == "media" else "face-0.png"]
    request = consumer(site, **{role: artifact})
    with pytest.raises(ValueError, match=message):
        common.validate_request({k: v for k, v in request.items() if k != "job_id"})
    with pytest.raises(ValueError, match=message):
        control.submit(request)
    assert not common.job_path("consumer").exists()
    assert list(site.state.glob("prepare-*")) == []
    if status:
        control.current(common.job_path("made"))  # Supervisor-less: reconciled for teardown.


@pytest.mark.parametrize("external", [False, True])
def test_completed_artifacts_are_inputs_in_every_role(make_site, external):
    site = make_site(external=external)
    made = producer("made", "completed")
    request = consumer(site, made["result.mp4"], made["face-0.png"], made["face-0.png"])
    control.submit(request)
    assert complete("consumer")["result"] == {"validated": True}


@pytest.mark.parametrize("external", [False, True])
def test_running_normalization_output_becomes_usable_once_completed(make_site, external):
    """A real normalization: refused while running, accepted (and rerun by the real engine's
    revalidation) once it has completed."""
    site = make_site(external=external)
    (site.state / "hold").touch()  # The stub runner holds normalization before the engine.
    control.submit({"job_id": "norm", "kind": "normalize", "media_path": site.clip})
    wait_file(common.job_path("norm") / "entered")
    assert control.status("norm")["status"] == "running"
    partial = common.job_path("norm") / "artifacts/normalized.mp4"
    shutil.copy(site.clip, partial)  # Output a running job has begun to write.
    for request in (
        consumer(site, media=str(partial), kind="render"),
        {"job_id": "consumer", "kind": "normalize", "media_path": str(partial)},
    ):
        with pytest.raises(ValueError, match="Job norm is running; wait for it to complete"):
            control.submit(request)
    with pytest.raises(ValueError, match="Job norm is running"):
        control.dispatch("inspect", {"media_path": str(partial)})
    assert not common.job_path("consumer").exists()
    (site.state / "hold").unlink()
    produced = wait_for("norm", {"completed", "failed"})
    assert produced["status"] == "completed", produced
    assert produced["result"]["media_path"] == str(partial)
    assert control.dispatch("inspect", {"media_path": str(partial)})["streams"]
    # The consumer's worker is the real engine, whose run() revalidates the resolved request.
    control.submit({"job_id": "renorm", "kind": "normalize", "media_path": str(partial)})
    renormalized = wait_for("renorm", {"completed", "failed"})
    assert renormalized["status"] == "completed", renormalized


@pytest.mark.parametrize("external", [False, True])
def test_queued_consumer_of_completed_producer_runs_and_keeps_it(make_site, monkeypatch, external):
    site = make_site(external=external)
    control.submit({"job_id": "norm", "kind": "normalize", "media_path": site.clip})
    produced = wait_for("norm", {"completed", "failed"})
    assert produced["status"] == "completed", produced
    normalized = produced["result"]["media_path"]
    crops = producer("crops", "completed", names=("face-0.png",))["face-0.png"]
    face = upload(monkeypatch, site.face)  # Uploads stay inputs alongside artifacts.
    request = consumer(site, normalized, crops, face, kind="render")
    # Holding the GPU lock keeps the consumer queued, so its worker validates later.
    with (common.LOCK_DIR / f"gpu-{common.GPU}.lock").open("a") as gpu:
        fcntl.flock(gpu, fcntl.LOCK_EX)
        assert control.submit(dict(request))["status"] == "queued"
        assert control.submit(dict(request))["status"] == "queued"  # Identical retry.
        # The producers can be neither deleted nor replaced under the same ID meanwhile.
        for job_id in ("norm", "crops"):
            with pytest.raises(ValueError, match="uses this job's artifacts as input"):
                control.delete_job(job_id)
        with pytest.raises(ValueError, match="different request"):
            control.submit({"job_id": "norm", "kind": "normalize", "media_path": site.face})
        pruned = control.prune(0, dry_run=False)
        assert {"norm", "crops"}.isdisjoint(entry["job_id"] for entry in pruned["jobs"])
        assert control.status("consumer")["status"] == "queued"
    # The worker revalidates the same artifacts against their still completed producers.
    assert complete("consumer")["result"] == {"validated": True}
    assert control.delete_job("norm")["deleted"] and control.delete_job("crops")["deleted"]

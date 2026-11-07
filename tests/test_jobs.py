"""Regressions for concurrent clients, durable jobs, cancellation and failed workers."""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import common
import control
import pytest
from conftest import apply_config, write_config


@pytest.fixture
def backend(tmp_path, monkeypatch):
    state = tmp_path / "runtime"
    config = write_config(tmp_path / "remote-config.json", tmp_path, state_dir=str(state))
    apply_config(monkeypatch, config)
    monkeypatch.setattr(control, "validate_request", lambda request: request)
    yield state
    for job in (state / "jobs").glob("*"):
        (job / "cancel").touch()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        jobs = list((state / "jobs").glob("*/state.json"))
        if all(json.loads(p.read_text())["status"] in common.TERMINAL for p in jobs):
            break
        time.sleep(0.1)


def wait_for(job_id, statuses):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        result = control.status(job_id)
        if result["status"] in statuses:
            return result
        time.sleep(0.05)
    raise AssertionError(control.status(job_id))


def test_queue_reconnect_idempotency_and_cancellation(backend):
    control.submit({"job_id": "first", "kind": "detect"})
    wait_for("first", {"running"})
    control.submit({"job_id": "second", "kind": "detect"})
    assert control.submit({"job_id": "first", "kind": "detect"})["job_id"] == "first"
    with pytest.raises(ValueError, match="different request"):
        control.submit({"job_id": "first", "kind": "render"})
    time.sleep(0.3)
    assert control.status("second")["status"] == "queued"
    control.dispatch("cancel", {"job_id": "second"})
    assert wait_for("second", {"cancelled"})["status"] == "cancelled"
    assert not (backend / "jobs/second/entered").exists()
    # A brand new command process can observe a detached job.
    result = subprocess.run(
        [sys.executable, control.__file__],
        input=json.dumps({"action": "status", "job_id": "first"}),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout)["result"]["status"] == "running"
    control.dispatch("cancel", {"job_id": "first"})
    wait_for("first", {"cancelled"})
    control.submit({"job_id": "third", "kind": "detect"})
    wait_for("third", {"running"})
    (backend / "jobs/third/release").touch()
    assert wait_for("third", {"completed"})["result"] == {"ok": True}


def test_failure_is_terminal_and_artifacts_are_withheld(backend):
    control.submit({"job_id": "broken", "kind": "detect", "fail": True})
    assert "9" in wait_for("broken", {"failed"})["error"]
    with pytest.raises(ValueError, match="successful completion"):
        control.dispatch("artifact", {"job_id": "broken", "name": "result.mp4"})


def test_worker_exception_message_becomes_job_error(backend):
    control.submit({"job_id": "raised", "kind": "detect", "error": "No faces in reference"})
    assert wait_for("raised", {"failed"})["error"] == "No faces in reference"


def test_engine_persists_exception_and_clears_stale_error(tmp_path, monkeypatch):
    import engine

    def fail(directory):
        raise ValueError("Face reference must be an image")

    monkeypatch.setattr(engine, "run", fail)
    with pytest.raises(ValueError):
        engine.main(tmp_path)
    assert common.read_json(tmp_path / "error.json") == {
        "type": "ValueError",
        "message": "Face reference must be an image",
    }
    monkeypatch.setattr(engine, "run", lambda directory: None)
    engine.main(tmp_path)
    assert not (tmp_path / "error.json").exists()


def test_cancel_after_worker_success_keeps_completion(backend, monkeypatch):
    job = common.job_path("late-cancel")
    (job / "artifacts").mkdir(parents=True)
    common.atomic_json(job / "request.json", {"kind": "detect"})
    common.atomic_json(
        job / "state.json", {"job_id": "late-cancel", "status": "queued", "created_at": 0}
    )
    (job / "release").touch()

    class CancelOnExit(subprocess.Popen):
        def poll(self):
            code = super().poll()
            if code is not None:
                (job / "cancel").touch()
            return code

    monkeypatch.setattr(control.subprocess, "Popen", CancelOnExit)
    control.supervise(str(job), control.SETTINGS.sha256)
    assert (job / "cancel").exists()
    result = control.status("late-cancel")
    assert result["status"] == "completed"
    assert result["result"] == {"ok": True}


def test_orphaned_job_is_reconciled(backend):
    job = common.job_path("orphan")
    job.mkdir(parents=True)
    common.atomic_json(
        job / "state.json", {"job_id": "orphan", "status": "running", "created_at": 0}
    )
    assert control.status("orphan")["status"] == "failed"


def test_paths_reject_traversal_and_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "BASE", tmp_path / "media")
    common.BASE.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("secret")
    (common.BASE / "link.png").symlink_to(outside)
    with pytest.raises(ValueError):
        common.input_path(str(common.BASE / "link.png"))
    with pytest.raises(ValueError):
        common.job_path("../escape")
    monkeypatch.setattr(common, "STATE", tmp_path)
    artifacts = tmp_path / "jobs/ok/artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "link").symlink_to(outside)
    with pytest.raises(ValueError):
        common.artifact_path("ok", "link")


def test_settings_enforce_type_bounds_and_known_keys():
    definitions = {"CountSlider": {"default": 10, "min_value": "1", "max_value": "20"}}
    for value in (True, -1, 21, 2.5, float("nan"), "5"):
        with pytest.raises(ValueError):
            common.validate_settings({"CountSlider": value}, definitions, {"CountSlider"})
    common.validate_settings({"CountSlider": 5}, definitions, {"CountSlider"})
    with pytest.raises(ValueError):
        common.validate_settings({"Unknown": 1}, definitions, {"CountSlider"})


def test_timestamp_validation_catches_vfr_hidden_by_metadata(monkeypatch):
    from fractions import Fraction

    import engine

    monkeypatch.setattr(
        engine.subprocess, "check_output", lambda *a, **kw: "0.000000\n0.040000\n0.090000\n"
    )
    with pytest.raises(ValueError, match="Variable"):
        engine.validate_timing("video.mp4", Fraction(25), {"time_base": "1/12800"})
    monkeypatch.setattr(engine.subprocess, "check_output", lambda *a, **kw: "0.000\n0.042\n0.083\n")
    assert engine.validate_timing("video.mkv", Fraction(24), {"time_base": "1/1000"}) == 3


def test_timestamp_validation_sorts_packets_and_bounds_reads(monkeypatch):
    from fractions import Fraction

    import engine

    calls = []
    # Decode-order packets with B-frames, an edit-list discard and a frame beyond the range.
    packets = "-0.040,K_D\n0.000,K__\n0.120,___\n0.040,___\n0.080,___\n0.160,___\n"

    def check_output(command, **kwargs):
        calls.append(command)
        return packets

    monkeypatch.setattr(engine.subprocess, "check_output", check_output)
    stream = {"time_base": "1/1000"}
    assert engine.validate_timing("video.mp4", Fraction(25), stream, 4) == 4
    assert calls[-1][calls[-1].index("-read_intervals") + 1] == "%+#36"
    assert engine.validate_timing("video.mp4", Fraction(25), stream) == 5
    assert "-read_intervals" not in calls[-1]
    packets = "1.000,K__\n1.040,___\n"
    with pytest.raises(ValueError, match="start at zero"):
        engine.validate_timing("video.mp4", Fraction(25), stream)


def test_decode_seeks_on_cfr_timeline_and_frames_raw_output(monkeypatch):
    import io
    from types import SimpleNamespace

    import engine

    commands = []

    class Decoder:
        def __init__(self, command, stdout):
            commands.append(command)
            self.stdout = io.BytesIO(bytes(range(12)) * 2 + b"partial")

        def poll(self):
            return 0

        def wait(self):
            return 0

    frame = SimpleNamespace(reshape=lambda *shape: shape)
    numpy = SimpleNamespace(uint8="uint8", frombuffer=lambda data, dtype: frame)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setattr(engine.subprocess, "Popen", Decoder)
    stream = {"width": 2, "height": 2, "avg_frame_rate": "25/1"}
    assert list(engine.decode("video.mp4", stream, 10, 5)) == [(2, 2, 3), (2, 2, 3)]
    assert commands[-1][commands[-1].index("-ss") + 1] == "0.38"
    assert commands[-1][commands[-1].index("-frames:v") + 1] == "5"
    assert len(list(engine.decode("video.mp4", stream, 0, 1))) == 1
    assert "-ss" not in commands[-1]
    with pytest.raises(ValueError, match="Rotated"):
        engine.video_stream(
            {"streams": [{"codec_type": "video", "side_data_list": [{"rotation": -90}]}]}
        )


def test_encoder_preserves_source_color_and_depth():
    from fractions import Fraction

    import engine

    def option(command, name):
        return command[command.index(name) + 1]

    def formats(command):
        return [command[i + 1] for i, value in enumerate(command) if value == "-pix_fmt"]

    stream = {"width": 4, "height": 2, "pix_fmt": "yuv420p10le", "color_range": "tv"}
    stream.update(color_space="bt709", color_transfer="bt709", color_primaries="bt709")
    command = engine.encode_command(stream, Fraction(24000, 1001), "out.mp4")
    assert formats(command) == ["bgr24", "yuv420p10le"]
    assert option(command, "-vf").startswith("scale=out_color_matrix=bt709:out_range=tv:")
    assert option(command, "-colorspace") == "bt709"
    assert "setparams=color_primaries=bt709:color_trc=bt709" in option(command, "-vf")
    assert option(command, "-color_range") == "tv"
    assert command[-1] == "out.mp4"

    full = {"width": 4, "height": 2, "pix_fmt": "yuvj420p", "color_space": "smpte170m"}
    command = engine.encode_command(full, Fraction(25), "out.mp4")
    assert formats(command) == ["bgr24", "yuv420p"]
    assert option(command, "-vf").startswith("scale=out_color_matrix=smpte170m:out_range=pc:")
    assert option(command, "-color_range") == "pc"
    assert "setparams" not in option(command, "-vf")

    # Untagged input keeps swscale's default matrix on both sides and stays untagged.
    assert "color_matrix" not in engine.scale_filter({"pix_fmt": "yuv420p"}, "in")
    with pytest.raises(ValueError, match="HDR"):
        engine.encode_command({**stream, "color_transfer": "smpte2084"}, Fraction(25), "o.mp4")


def test_render_progress_reports_throughput_and_eta(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace

    import engine

    frame = SimpleNamespace(shape=(2, 2, 3), tobytes=lambda: b"")
    stream = {"codec_type": "video", "width": 2, "height": 2}
    stream.update(avg_frame_rate="25/1", r_frame_rate="25/1")
    progress = []
    encoder = SimpleNamespace(stdin=io.BytesIO(), wait=lambda timeout: 1, poll=lambda: 1)
    monkeypatch.setattr(engine, "input_path", lambda path: path)
    monkeypatch.setattr(engine, "probe", lambda path: {"streams": [stream]})
    monkeypatch.setattr(engine, "validate_timing", lambda *a: 10)
    monkeypatch.setattr(engine, "decode", lambda *a: (f for f in [frame] * 3))
    monkeypatch.setattr(engine, "process", lambda worker, bgr, index: frame)
    monkeypatch.setattr(engine, "save_image", lambda *a: None)
    monkeypatch.setattr(engine, "atomic_json", lambda path, value: progress.append(value))
    monkeypatch.setattr(engine.subprocess, "Popen", lambda *a, **kw: encoder)
    request = {"media_path": "video.mp4", "start_frame": 7}
    with pytest.raises(RuntimeError, match="encoding failed"):
        engine.render_video(None, request, tmp_path, SimpleNamespace(swap_count=1))
    assert [p["frames"] for p in progress] == [1, 2, 3]
    assert progress[-1]["eta_seconds"] == 0
    assert all(p["frames_per_second"] > 0 for p in progress)


def test_render_worker_consumes_fusion_sequential_detections(monkeypatch):
    from types import ModuleType, SimpleNamespace

    import engine

    calls = []

    class SequentialDetector:
        def __init__(self, window):
            calls.append("init")

        def reset_state(self):
            calls.append("reset")

        def run(self, frame, control, parameters, frame_number):
            calls.append(("detect", frame_number, frame, parameters))
            return "boxes", "kps5", "kps", "kps203"

    class FrameWorker:
        def __init__(self, window, is_single_frame):
            self.main_window, self.is_single_frame = window, is_single_frame
            self.precomputed_bboxes = None

        def process_frame(self, control, stop_event):
            calls.append(("process", self.is_single_frame, self.precomputed_bboxes))
            return self.frame

    for name, attributes in {
        "numpy": {"ascontiguousarray": lambda array: f"rgb:{array}"},
        "app": {},
        "app.processors": {},
        "app.processors.workers": {},
        "app.processors.workers.frame_worker": {"FrameWorker": FrameWorker},
        "app.processors.video_utils": {},
        "app.processors.video_utils.sequential_detector": {
            "SequentialDetector": SequentialDetector
        },
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    window = SimpleNamespace(control={"Detector": 1}, parameters={"0": {}})
    frame = type("Frame", (), {"__getitem__": lambda self, key: "flipped"})()

    single = engine.make_worker(window)
    engine.process(single, frame, 3)
    assert calls == [("process", True, None)]

    calls.clear()
    sequential = engine.make_worker(window, sequential=True)
    for index in (7, 8):
        engine.process(sequential, frame, index)
    assert calls == [
        "init",
        "reset",
        ("detect", 7, "rgb:flipped", window.parameters),
        ("process", False, "boxes"),
        ("detect", 8, "rgb:flipped", window.parameters),
        ("process", False, "boxes"),
    ]


def test_incomplete_upload_never_publishes_file(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace

    monkeypatch.setattr(control, "STATE", tmp_path)
    monkeypatch.setattr(control.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"short")))
    with pytest.raises(ValueError, match="interrupted"):
        control.upload("a" * 32, ".png", 10)
    assert list((tmp_path / "uploads").iterdir()) == []


def test_status_completion_race_preserves_success(backend, monkeypatch):
    job = common.job_path("finishing")
    job.mkdir(parents=True)
    common.atomic_json(
        job / "state.json", {"job_id": "finishing", "status": "running", "created_at": 0}
    )
    original = control.fcntl.flock

    def complete_before_lease(fd, operation):
        common.atomic_json(
            job / "state.json", {"job_id": "finishing", "status": "completed", "created_at": 0}
        )
        return original(fd, operation)

    monkeypatch.setattr(control.fcntl, "flock", complete_before_lease)
    assert control.status("finishing")["status"] == "completed"


def test_submitted_job_keeps_its_release_after_deploy(backend, tmp_path, monkeypatch):
    import shutil

    releases = tmp_path / "releases"
    for name in ("one", "two"):
        (releases / name).mkdir(parents=True)
        for source in Path(control.__file__).parent.glob("*.py"):
            shutil.copy(source, releases / name)
    link = tmp_path / "remote"
    link.symlink_to("releases/one")
    monkeypatch.setattr(control, "RELEASE", (releases / "one").resolve())
    control.submit({"job_id": "pinned", "kind": "detect"})
    # deploy.sh swaps the symlink while the job is still queued or starting.
    (tmp_path / "next").symlink_to("releases/two")
    (tmp_path / "next").replace(link)
    wait_for("pinned", {"running"})
    entered = backend / "jobs/pinned/entered"
    deadline = time.monotonic() + 5
    while not entered.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert entered.read_text() == str((releases / "one/engine.py").resolve())


def test_status_long_poll_returns_on_change_or_timeout(backend):
    job = common.job_path("waiting")
    job.mkdir(parents=True)
    common.atomic_json(
        job / "state.json", {"job_id": "waiting", "status": "running", "created_at": time.time()}
    )
    common.atomic_json(job / "progress.json", {"stage": "loading_models"})
    (job / "worker.log").write_text("model loaded\n")
    # Holding the lease stands in for a live supervisor.
    with (job / "lease").open("a") as lease:
        control.fcntl.flock(lease, control.fcntl.LOCK_EX)
        started = time.monotonic()
        result = control.status("waiting", wait=0.6)
        assert 0.5 < time.monotonic() - started < 2
        assert result["progress"] == {"stage": "loading_models"} and "log_tail" not in result
        assert "model loaded" in control.status("waiting", include_log=True)["log_tail"]

        def advance(progress):
            time.sleep(0.3)
            common.atomic_json(job / "progress.json", progress)

        # Frame counts within a stage keep waiting; a stage change ends the wait.
        threading.Thread(target=advance, args=({"stage": "loading_models", "n": 1},)).start()
        assert control.status("waiting", wait=0.8)["progress"]["n"] == 1
        threading.Thread(target=advance, args=({"stage": "rendering", "frames": 1},)).start()
        started = time.monotonic()
        assert control.status("waiting", wait=10)["progress"]["stage"] == "rendering"
        assert time.monotonic() - started < 2
        threading.Thread(target=advance, args=({"stage": "rendering", "frames": 2},)).start()
        common.atomic_json(
            job / "state.json", common.read_json(job / "state.json") | {"status": "completed"}
        )
        started = time.monotonic()
        assert control.status("waiting", wait=10)["status"] == "completed"
        assert time.monotonic() - started < 0.2
    common.atomic_json(
        job / "state.json", common.read_json(job / "state.json") | {"status": "failed"}
    )
    result = control.dispatch("status", {"job_id": "waiting", "wait_seconds": 50})
    assert result["status"] == "failed" and "model loaded" in result["log_tail"]


def test_list_summarizes_and_filters_jobs(backend):
    for index, (job_id, kind, state) in enumerate(
        [("a", "detect", "completed"), ("b", "render", "failed"), ("c", "render", "completed")]
    ):
        job = common.job_path(job_id)
        job.mkdir(parents=True)
        request = {"kind": kind, "media_path": "/m/clip.mp4", "mappings": [{"x": 1}], "frame": 3}
        common.atomic_json(job / "request.json", request | {"start_frame": 5, "max_frames": 9})
        common.atomic_json(
            job / "state.json",
            {"job_id": job_id, "status": state, "created_at": 100.0, "finished_at": 112.5},
        )
        (job / "worker.log").write_text("noise")
        os.utime(job / "state.json", (index, index))
    (backend / "jobs/partial").mkdir()
    jobs = control.dispatch("list", {})
    assert [job["job_id"] for job in jobs] == ["c", "b", "a"]
    assert jobs[2] == {
        "job_id": "a",
        "kind": "detect",
        "status": "completed",
        "media_path": "/m/clip.mp4",
        "frame": 3,
        "created_at": 100.0,
        "started_at": None,
        "finished_at": 112.5,
        "elapsed_seconds": 12.5,
        "error": None,
        "progress": None,
    }
    assert jobs[0]["start_frame"] == 5 and "frame" not in jobs[0] and "mappings" not in jobs[0]
    filtered = control.dispatch("list", {"status": "completed", "kind": "render", "limit": 5})
    assert [job["job_id"] for job in filtered] == ["c"]
    assert [job["job_id"] for job in control.dispatch("list", {"limit": 1})] == ["c"]


def test_delete_refuses_active_jobs_and_removes_terminal_ones(backend):
    control.submit({"job_id": "busy", "kind": "detect"})
    wait_for("busy", {"running"})
    control.submit({"job_id": "waiting", "kind": "detect"})
    for job_id in ("busy", "waiting"):
        with pytest.raises(ValueError, match="active|can be deleted"):
            control.dispatch("delete", {"job_id": job_id})
    (backend / "jobs/busy/release").touch()
    wait_for("busy", {"completed"})
    (backend / "jobs/busy/artifacts/result.mp4").write_bytes(b"x" * 1000)
    deleted = control.dispatch("delete", {"job_id": "busy"})
    assert deleted["deleted"] and deleted["status"] == "completed" and deleted["bytes"] >= 1000
    assert not (backend / "jobs/busy").exists() and not list(backend.glob("trash-*"))
    with pytest.raises(FileNotFoundError):
        control.status("busy")
    # The queued job proceeds unaffected and becomes deletable once terminal.
    wait_for("waiting", {"running"})
    control.dispatch("cancel", {"job_id": "waiting"})
    wait_for("waiting", {"cancelled"})
    assert control.dispatch("delete", {"job_id": "waiting"})["status"] == "cancelled"


def test_delete_keeps_artifacts_used_by_active_jobs(backend):
    control.submit({"job_id": "normalized", "kind": "detect"})
    wait_for("normalized", {"running"})
    (backend / "jobs/normalized/release").touch()
    wait_for("normalized", {"completed"})
    media = backend / "jobs/normalized/artifacts/normalized.mp4"
    media.write_bytes(b"x")
    control.submit({"job_id": "consumer", "kind": "render", "media_path": str(media)})
    wait_for("consumer", {"running"})
    with pytest.raises(ValueError, match="artifacts as input"):
        control.dispatch("delete", {"job_id": "normalized"})
    assert control.prune(0, dry_run=False)["jobs"] == [] and media.exists()
    control.dispatch("cancel", {"job_id": "consumer"})
    wait_for("consumer", {"cancelled"})
    assert control.dispatch("delete", {"job_id": "normalized"})["deleted"]


def test_prune_respects_age_references_and_dry_run(backend):
    import os

    old = time.time() - 10 * 86400
    uploads = backend / "uploads"
    uploads.mkdir(parents=True)
    names = ["a" * 64 + ".mp4", "b" * 32 + ".png", "c" * 64 + ".png", "d" * 64 + ".jpg"]
    partial = "e" * 64 + ".mp4.1234abcd.partial"
    for name in names + [partial]:
        (uploads / name).write_bytes(b"u" * 100)
    for name in names[:3] + [partial]:
        os.utime(uploads / name, (old, old))
    for job_id, finished in (("ancient", old), ("recent", time.time())):
        job = common.job_path(job_id)
        (job / "artifacts").mkdir(parents=True)
        common.atomic_json(job / "request.json", {"kind": "detect"})
        common.atomic_json(
            job / "state.json",
            {"job_id": job_id, "status": "completed", "created_at": 0, "finished_at": finished},
        )
        (job / "artifacts/result.mp4").write_bytes(b"r" * 1000)
    # An active job keeps the old uploads it references.
    control.submit(
        {
            "job_id": "active",
            "kind": "preview",
            "media_path": str(uploads / names[0]),
            "mappings": [{"target_face": str(uploads / names[2]), "source_faces": []}],
        }
    )
    wait_for("active", {"running"})
    (backend / "prepare-crashed").mkdir()

    preview = control.dispatch("prune", {"older_than_days": 7})
    assert preview["dry_run"]
    assert [job["job_id"] for job in preview["jobs"]] == ["ancient"]
    assert [Path(item["path"]).name for item in preview["uploads"]] == [names[1]]
    assert {Path(item["path"]).name for item in preview["stale"]} == {partial, "prepare-crashed"}
    assert preview["total_bytes"] >= 1000 + 100 + 100
    assert (backend / "jobs/ancient").exists() and (uploads / names[1]).exists()

    pruned = control.dispatch("prune", {"older_than_days": 7, "dry_run": False})
    assert pruned["total_bytes"] == preview["total_bytes"]
    assert not (backend / "jobs/ancient").exists() and (backend / "jobs/recent").exists()
    assert sorted(path.name for path in uploads.iterdir()) == sorted([names[0], names[2], names[3]])
    assert not (backend / "prepare-crashed").exists()
    assert control.status("active")["status"] == "running"
    with pytest.raises(ValueError):
        control.dispatch("prune", {"older_than_days": -1})


def test_content_addressed_upload_verifies_hash_and_is_found_again(tmp_path, monkeypatch):
    import hashlib
    import io
    from types import SimpleNamespace

    monkeypatch.setattr(control, "STATE", tmp_path)
    data = b"media" * 1000
    digest = hashlib.sha256(data).hexdigest()
    assert control.find_upload(digest, ".png", len(data)) is None
    monkeypatch.setattr(control.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"x" * len(data))))
    with pytest.raises(ValueError, match="SHA-256"):
        control.upload(digest, ".png", len(data))
    assert list((tmp_path / "uploads").iterdir()) == []
    monkeypatch.setattr(control.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(data)))
    stored = control.upload(digest, ".png", len(data))
    assert stored["path"] == str(tmp_path / "uploads" / f"{digest}.png") and not stored["reused"]
    found = control.find_upload(digest, ".png", len(data))
    assert found["path"] == stored["path"] and found["reused"]
    assert control.find_upload(digest, ".png", len(data) + 1) is None
    for arguments in (("../" + digest[3:], ".png", 5), (digest, ".txt", 5), (digest, ".png", 0)):
        with pytest.raises(ValueError):
            control.find_upload(*arguments)


def test_cluster_faces_groups_by_fusion_similarity_and_separates_same_frame():
    import engine

    a, a2, b, c = [1.0, 0.0, 0.0], [0.9, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    # Fusion score 50 + 50 * cosine: a/a2 ≈ 97, a/b = 50, so a threshold of 60 separates them.
    labels = engine.cluster_faces([a, b, a2, c, b], [0, 0, 5, 5, 9], 60)
    assert labels == [0, 1, 0, 2, 1]
    # Two faces in one frame are never the same identity, however similar.
    assert engine.cluster_faces([a, a2], [3, 3], 60) == [0, 1]
    assert engine.cluster_faces([a, a2], [3, 4], 99) == [0, 1]
    assert engine.cluster_faces([], [], 60) == []


def test_multi_frame_detection_requests_are_bounded(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    monkeypatch.setattr(common, "BASE", tmp_path)
    monkeypatch.setattr(common, "layouts", lambda: ({}, {}))
    base = {"kind": "detect", "media_path": str(media)}
    for extra in (
        {"sample_frames": 0},
        {"sample_frames": 65},
        {"sample_frames": True},
        {"frames": []},
        {"frames": list(range(65))},
        {"frames": [1, 1]},
        {"frames": [-1]},
        {"frames": "1,2"},
        {"frames": [1], "sample_frames": 2},
        {"frame": 3, "sample_frames": 2},
        {"kind": "preview", "sample_frames": 2},
    ):
        with pytest.raises(ValueError):
            common.validate_request({**base, **extra})
    common.validate_request({**base, "sample_frames": 64})
    common.validate_request({**base, "frames": [0, 10, 5]})
    common.validate_request(base)


def test_normalize_runs_alongside_gpu_jobs(backend):
    control.submit({"job_id": "gpu", "kind": "render"})
    wait_for("gpu", {"running"})
    control.submit({"job_id": "cpu", "kind": "normalize"})
    wait_for("cpu", {"running"})
    (backend / "jobs/cpu/release").touch()
    assert wait_for("cpu", {"completed"})["result"] == {"ok": True}
    assert control.status("gpu")["status"] == "running"
    assert control.list_jobs(only_kind="normalize")[0]["fps"] is None


def test_normalize_request_validation(monkeypatch):
    monkeypatch.setattr(common, "layouts", lambda: ({}, {}))
    monkeypatch.setattr(common, "input_path", lambda path: path)
    request = {"kind": "normalize", "media_path": "/m/clip.mp4"}
    assert common.validate_request(request) == request
    for fps in (25, 29.97, "30000/1001", "24"):
        common.validate_request({**request, "fps": fps})
    for fps in (0, -1, 241, "0/0", "abc", float("nan"), True, "1/1002", [25]):
        with pytest.raises(ValueError, match="fps"):
            common.validate_request({**request, "fps": fps})
    with pytest.raises(ValueError, match="mapping"):
        common.validate_request({**request, "kind": "render"})


def test_normalized_rate_snaps_to_standard_rates():
    from fractions import Fraction

    import engine

    rate = engine.normalized_rate
    assert rate({"avg_frame_rate": "24000/1001"}) == Fraction(24000, 1001)
    assert rate({"avg_frame_rate": "1798200/60001"}) == Fraction(30000, 1001)
    assert rate({"avg_frame_rate": "3050/100"}) == 30
    assert rate({"avg_frame_rate": "0/0", "r_frame_rate": "25/1"}) == 25
    assert rate({"avg_frame_rate": "1000/90"}) == Fraction(100, 9)
    assert rate({"avg_frame_rate": "25/1"}, "29.97") == Fraction(2997, 100)
    with pytest.raises(ValueError, match="pass fps"):
        rate({"avg_frame_rate": "0/0", "r_frame_rate": "0/0"})


def test_normalize_command_rebases_time_and_pads():
    from fractions import Fraction

    import engine

    stream = {"start_time": "1.5", "duration": "4.0"}
    command = engine.normalize_command("in.mov", "out.mp4", stream, Fraction(24000, 1001))
    assert "-copyts" in command and command[-1] == "out.mp4"
    video = command[command.index("-vf") + 1]
    assert video.startswith("setpts=PTS-STARTPTS,fps=24000/1001,pad=ceil(iw/2)*2:ceil(ih/2)*2,")
    # Audio joins afterwards, cut to the encoded video (see test_audio).
    assert "-af" not in command and "0:a:0" not in command
    assert command[command.index("-crf") + 1] == "18" and "yuv420p" in command
    assert command[command.index("-map_metadata") + 1] == "-1"
    # Normalizing must not undo bit depth or color tags that renders preserve.
    deep = {"pix_fmt": "yuv420p10le", "color_space": "bt709", "color_range": "pc"}
    command = engine.normalize_command("in.mov", "out.mp4", deep, Fraction(25))
    assert command[command.index("-pix_fmt") + 1] == "yuv420p10le"
    assert command[command.index("-colorspace") + 1] == "bt709"
    assert command[command.index("-color_range") + 1] == "pc"
    with pytest.raises(ValueError, match="HDR"):
        engine.normalize_command("in.mov", "out.mp4", {"color_transfer": "smpte2084"}, 25)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is not installed")
def test_normalize_video_makes_render_ready_output(tmp_path, monkeypatch):
    import engine

    # Odd dimensions, a one-second start offset and rotation metadata.
    source, plain = tmp_path / "odd.mp4", tmp_path / "plain.mp4"
    generate = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=63x47:r=24:d=2"]
    generate += ["-f", "lavfi", "-i", "sine=d=2.5", "-c:v", "libx264", "-pix_fmt", "yuv444p"]
    subprocess.run(generate + ["-c:a", "aac", str(plain)], check=True)
    rotate = ["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-itsoffset", "1"]
    subprocess.run(rotate + ["-i", str(plain), "-c", "copy", str(source)], check=True)
    monkeypatch.setattr(engine, "input_path", lambda path: Path(path))
    (tmp_path / "artifacts").mkdir()
    result = engine.normalize_video({"media_path": str(source)}, tmp_path)
    output = tmp_path / "artifacts/normalized.mp4"
    assert result["media_path"] == str(output) and output.is_file()
    assert (result["width"], result["height"], result["fps"]) == (48, 64, "24")
    assert result["frames"] == 48 and result["audio"] == "AAC"
    assert result["source"]["start_time"] >= 1 and abs(result["rotation_applied"]) == 90
    stream = engine.video_stream(common.probe(output))  # Rotation metadata is gone.
    assert engine.validate_timing(output, engine.Fraction(24), stream) == 48
    assert common.read_json(tmp_path / "progress.json")["stage"] == "normalizing"

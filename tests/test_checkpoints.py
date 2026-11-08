"""Render checkpoints and explicit resume (vmcp-harden-pqj.7).

Engine tests render test_timing's numbered clip (luma = source frame number, a tone at frames 16
and 17) through real FFmpeg in 4-frame chunks with a scripted worker that inverts each frame it
processes, then compare uninterrupted, interrupted and resumed attempts exactly: libx264 is
deterministic, so equal frames encode to equal chunks. Failures are injected at each step of
chunk publication, in assembly and in the inherited files. Control tests use the stub runner.
"""

import asyncio
import copy
import fcntl
import json
import threading
import time
from fractions import Fraction
from types import SimpleNamespace

import common
import control
import engine
import pytest
import test_segments
import test_timing
from conftest import apply_config, write_config
from test_timing import TONE, audible, sequential

library, media = test_timing.library, test_timing.media  # Fixtures.
START, END = 3, 25  # early.mkv has 25 frames; chunks are [3, 7), [7, 11), ... [23, 25).
CODE = {name: "0" * 64 for name in common.CODE}
RECORD = {
    "inputs": {},
    "settings": {"controls": {}, "parameters": [{}, {}]},
    "fusion_revision": "f" * 40,
    "adapter": {"release": "r1", "code": CODE},
    "config_sha256": "c" * 64,
    "models": {"YoloFace8n": {"path": "/m/yolo.onnx", "declared_sha256": None, "sha256": "a" * 64}},
}


class Worker(test_segments.Scripted):
    """Inverts every frame it processes (so each output frame is distinct and checkable) and
    swaps one face per bound mapping, except in `misses`; fails at frame `crash`."""

    def __init__(self, crash=None, misses=()):
        super().__init__(misses)
        self.crash = crash

    def process(self, worker, bgr, index):
        if index == self.crash:
            raise RuntimeError(f"Injected failure at frame {index}")
        super().process(worker, bgr, index)
        inverted = type(bgr)(bytes(255 - value for value in bgr))
        inverted.shape = bgr.shape
        return inverted


@pytest.fixture
def site(media, tmp_path, monkeypatch):
    config = write_config(tmp_path / "remote-config.json", media, state_dir=str(tmp_path / "state"))
    apply_config(monkeypatch, config)
    monkeypatch.setattr(engine, "CHUNK_FRAMES", 4)
    images = {}
    monkeypatch.setattr(engine, "save_image", lambda path, image: images.update({path: image}))
    monkeypatch.setattr(engine, "review_image", lambda path, *a: images.update({path: a}))
    return media


def render_request(site, **changes):
    request = {
        **test_segments.swap_request(),
        "media_path": str(site / "early.mkv"),
        "start_frame": START,
        "max_frames": None,
        "report": {"review_images": 2},
    }
    return {**request, **changes}


def attempt(monkeypatch, job_id, request, worker=None, record=RECORD):
    """Run one render attempt in job directory `job_id` and publish its terminal state."""
    directory = common.job_path(job_id)
    (directory / "artifacts").mkdir(parents=True, exist_ok=True)
    for name in ("request.json", "resolved.json"):
        common.atomic_json(directory / name, request)
    worker = worker or Worker()
    monkeypatch.setattr(engine, "process", worker.process)
    state = {"job_id": job_id, "status": "completed", "created_at": 0, "finished_at": 1}
    try:
        checkpoint = engine.Checkpoint(directory, request, record)
        outcome = engine.render_video(None, request, directory, worker, checkpoint)
    except Exception as error:
        outcome = error
        state.update(status="failed", error=str(error))
    common.atomic_json(directory / "state.json", state)
    return worker, outcome


def resume(monkeypatch, job_id, origin, worker=None, record=RECORD, **changes):
    """A resumed attempt as resume_render submits it: the earlier request with a resume link,
    and the earlier checkpoint inherited under the submit lock."""
    request = common.read_json(common.job_path(origin) / "resolved.json")
    request = {key: value for key, value in request.items() if key != "resume"}
    request["resume"] = {"from_job_id": origin, "attempt": 2}
    with control.retention_lock():
        control.inherit(request, common.job_path(job_id))
    request.update(changes)  # As if an input alias moved after submission.
    return attempt(monkeypatch, job_id, request, worker, record)


def manifest(job_id):
    return common.read_json(common.job_path(job_id) / "checkpoint/manifest.json")


def output(job_id):
    return sequential(common.job_path(job_id) / "artifacts/result.mp4", False)


def report(job_id):
    return common.read_json(common.job_path(job_id) / "artifacts/swap-report.json")


@pytest.fixture
def reference(site, monkeypatch):
    """An uninterrupted render, with a dropout that crosses the boundary at frame 11."""
    request = render_request(site)
    worker, result = attempt(monkeypatch, "whole", request, Worker(misses={10, 11, 12}))
    assert sorted(worker.seen) == list(range(START, END))
    return request, result


def same_as_reference(job_id, result, reference):
    assert output(job_id) == output("whole")
    assert report(job_id) == report("whole")
    assert {**result, "checkpoint": None} == {**reference[1], "checkpoint": None}
    # Equal frames encode to equal chunks, so the output joins the same bytes.
    digests = [entry["sha256"] for entry in manifest(job_id)["chunks"]]
    assert digests == [entry["sha256"] for entry in manifest("whole")["chunks"]]


# The checkpoint of an uninterrupted render


def test_uninterrupted_render_publishes_validated_chunks_and_a_manifest(site, reference):
    request, result = reference
    document = manifest("whole")
    assert (document["format"], document["version"]) == ("visomaster-checkpoint", 1)
    assert document["range"] == {"start_frame": 3, "end_frame": 25, "frames": 22, "fps": "25"}
    assert document["chunk_frames"] == 4 and "stream copy" in document["chunking"]
    assert document["provenance"] == RECORD and document["resumed_from"] is None
    assert document["intent_sha256"] == common.intent_digest(request)
    assert document["runtime"] == common.ffmpeg_version() and document["temporal"] is False
    assert document["codec"]["codec_name"] == "h264"
    assert document["codec"]["extradata_hash"].startswith("SHA256:")
    chunks = document["chunks"]
    assert [(c["start_frame"], c["end_frame"]) for c in chunks] == [
        (3, 7),
        (7, 11),
        (11, 15),
        (15, 19),
        (19, 23),
        (23, 25),  # The last chunk is clipped to the range.
    ]
    directory = common.job_path("whole") / "checkpoint/chunks"
    for number, chunk in enumerate(chunks):
        path = directory / chunk["file"]
        assert chunk["index"] == number and chunk["file"] == f"{number:05d}.mp4"
        assert chunk["sha256"] == common.file_sha256(path) and chunk["bytes"] == path.stat().st_size
        state = common.read_json(directory / chunk["state"])
        assert chunk["state_sha256"] == common.file_sha256(directory / chunk["state"])
        assert state["end_frame"] == chunk["end_frame"] and chunk["job_id"] == "whole"
        assert len(sequential(path, False)) == chunk["frames"]
    assert not list(directory.glob("*.partial.mp4"))
    assert result["checkpoint"] == {
        "chunks": 6,
        "chunk_frames": 4,
        "reused_chunks": 0,
        "reused_frames": 0,
        "resumed_from": None,
    }
    # Stream copy keeps source order, count, rate, duration and the audio cut of the range.
    source = sequential(site / "early.mkv")
    frames = output("whole")
    assert (result["frames"], result["fps"], result["duration"]) == (22, "25", 0.88)
    assert len(frames) == 22
    for offset, frame in enumerate(frames):
        inverted = [255 - value for value in source[START + offset]]
        assert max(abs(a - b) for a, b in zip(frame, inverted)) <= 2, offset
    first, last = audible(common.job_path("whole") / "artifacts/result.mp4")
    assert abs(first - (TONE[0] - 3 / 25)) < 0.025 and abs(last - (TONE[1] - 3 / 25)) < 0.025
    intervals = report("whole")["mappings"][0]["intervals"]
    assert ("dropout", 10, 13) in [(i["kind"], i["start_frame"], i["end_frame"]) for i in intervals]


def test_ntsc_chunks_join_on_exact_frame_times(site, tmp_path, monkeypatch):
    # FFmpeg's own sum of microsecond chunk durations drifts at 30000/1001; the listed ones
    # keep each chunk on its exact frame time, which the output's timestamp check requires.
    monkeypatch.setattr(engine, "CHUNK_FRAMES", 1)
    request = render_request(site, media_path=str(site / "ntsc.mp4"), start_frame=0)
    _, result = attempt(monkeypatch, "ntsc", request)
    assert result["frames"] == 20 and result["checkpoint"]["chunks"] == 20
    checkpoint = engine.Checkpoint(tmp_path, {})
    checkpoint.start, checkpoint.end, checkpoint.size = 0, 3000, 1
    checkpoint.fps, checkpoint.root = Fraction(30000, 1001), tmp_path
    checkpoint.manifest = {
        "chunks": [{"index": n, "end_frame": n + 1, "file": f"{n:05d}.mp4"} for n in range(3000)]
    }
    checkpoint.finish()
    lines = (tmp_path / "concat.txt").read_text().splitlines()
    total = 0
    for number, line in enumerate(line for line in lines if line.startswith("duration ")):
        seconds, _, micros = line.split()[1].partition(".")
        total += int(seconds) * 10**6 + int(micros)
        assert total == round(Fraction(number + 1) * Fraction(1001, 30000) * 10**6)


# Interruption and resume


def test_interrupted_and_resumed_render_matches_the_uninterrupted_one(site, reference, monkeypatch):
    request = reference[0]
    _, error = attempt(monkeypatch, "broken", request, Worker(crash=13, misses={10, 11, 12}))
    assert "Injected failure at frame 13" in str(error)
    assert [c["end_frame"] for c in manifest("broken")["chunks"]] == [7, 11]
    assert not (common.job_path("broken") / "artifacts/result.mp4").exists()
    worker, result = resume(monkeypatch, "again", "broken", Worker(misses={10, 11, 12}))
    assert sorted(worker.seen) == list(range(11, END))  # Only frames after the reused chunks.
    same_as_reference("again", result, reference)
    assert result["checkpoint"] == {
        "chunks": 6,
        "chunk_frames": 4,
        "reused_chunks": 2,
        "reused_frames": 8,
        "resumed_from": "broken",
    }
    document = manifest("again")
    assert document["resumed_from"] == "broken"
    assert [c["job_id"] for c in document["chunks"]] == ["broken"] * 2 + ["again"] * 4
    # The open dropout crossed the interruption and is reported once, as uninterrupted.
    assert report("again")["mappings"][0]["intervals"][0]["start_frame"] == 10
    # The earlier attempt keeps its state, files and manifest.
    assert common.read_json(common.job_path("broken") / "state.json")["status"] == "failed"
    assert [c["end_frame"] for c in manifest("broken")["chunks"]] == [7, 11]


def fail_once(monkeypatch, module, name, when):
    original, done = getattr(module, name), []

    def wrapper(*args, **kwargs):
        if not done and when(*args):
            done.append(True)
            raise OSError(f"Injected failure in {name}")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)


def committed_chunks(path, value):
    return path.name == "manifest.json" and len(value["chunks"]) == 3


@pytest.mark.parametrize(
    "window, reused",
    [
        # Chunk 2 is encoded but not yet named: its partial file is left behind.
        ("before_rename", 2),
        # Chunk 2 is named, but its accounting is not written: the manifest omits it.
        ("before_state", 2),
        # Chunk 2 and its accounting are written, but not the manifest entry.
        ("before_manifest", 2),
        # The manifest commits chunk 2, and the worker dies right after.
        ("after_manifest", 3),
    ],
)
def test_crash_windows_around_chunk_publication(site, reference, monkeypatch, window, reused):
    request, crash = reference[0], None
    if window == "before_rename":
        fail_once(monkeypatch, engine, "file_sha256", lambda path: path.name == "00002.partial.mp4")
    elif window == "before_state":
        fail_once(monkeypatch, engine, "atomic_json", lambda path, value: path.name == "00002.json")
    elif window == "before_manifest":
        fail_once(monkeypatch, engine, "atomic_json", committed_chunks)
    else:
        crash = 15
    _, error = attempt(monkeypatch, "broken", request, Worker(crash=crash, misses={10, 11, 12}))
    assert isinstance(error, (OSError, RuntimeError))
    left = {path.name for path in (common.job_path("broken") / "checkpoint/chunks").iterdir()}
    expected = {f"0000{n}.{kind}" for n in range(reused) for kind in ("mp4", "json")}
    expected |= {
        "before_rename": {"00002.partial.mp4"},
        "before_state": {"00002.mp4"},
        "before_manifest": {"00002.mp4", "00002.json"},
        "after_manifest": set(),
    }[window]
    assert left == expected
    worker, result = resume(monkeypatch, "again", "broken", Worker(misses={10, 11, 12}))
    assert (
        min(worker.seen) == START + 4 * reused and result["checkpoint"]["reused_chunks"] == reused
    )
    same_as_reference("again", result, reference)


@pytest.mark.parametrize(
    "damage",
    [
        "flip_byte",  # Same size, other bytes.
        "truncate",
        "remove",
        "state",  # The accounting file, not the video.
        "entry",  # A manifest entry naming other frames.
    ],
)
def test_corrupt_chunks_are_never_trusted_and_later_ones_are_redone(
    site, reference, monkeypatch, damage
):
    request = reference[0]
    attempt(monkeypatch, "broken", request, Worker(crash=20, misses={10, 11, 12}))
    directory = common.job_path("broken") / "checkpoint"
    chunk = directory / "chunks/00001.mp4"
    if damage == "flip_byte":
        data = bytearray(chunk.read_bytes())
        data[len(data) // 2] ^= 1
        chunk.write_bytes(bytes(data))
    elif damage == "truncate":
        chunk.write_bytes(chunk.read_bytes()[:-1])
    elif damage == "remove":
        chunk.unlink()
    elif damage == "state":
        (directory / "chunks/00001.json").write_text('{"end_frame": 11, "swap_count": 0}')
    else:
        document = common.read_json(directory / "manifest.json")
        document["chunks"][1]["end_frame"] = 12
        common.atomic_json(directory / "manifest.json", document)
    damaged = chunk.read_bytes() if chunk.exists() else None
    worker, result = resume(monkeypatch, "again", "broken", Worker(misses={10, 11, 12}))
    # Chunk 1 is redone, and chunks 2 and 3 after it, whose accounting builds on it.
    assert min(worker.seen) == 7 and result["checkpoint"]["reused_chunks"] == 1
    same_as_reference("again", result, reference)
    # The replacement is the new job's own file: the earlier attempt's stays as it was.
    assert (chunk.read_bytes() if chunk.exists() else None) == damaged


def test_an_assembly_crash_resumes_without_rendering_again(site, reference, monkeypatch):
    request = reference[0]
    fail_once(monkeypatch, engine, "mux", lambda *a: True)
    _, error = attempt(monkeypatch, "broken", request, Worker(misses={10, 11, 12}))
    assert "Injected failure in mux" in str(error)
    assert len(manifest("broken")["chunks"]) == 6
    worker, result = resume(monkeypatch, "again", "broken", Worker(misses={10, 11, 12}))
    assert worker.seen == {} and result["checkpoint"]["reused_frames"] == 22
    same_as_reference("again", result, reference)


def test_a_resume_of_a_resume_continues_the_chain(site, reference, monkeypatch):
    request = reference[0]
    attempt(monkeypatch, "first", request, Worker(crash=9, misses={10, 11, 12}))
    resume(monkeypatch, "second", "first", Worker(crash=18, misses={10, 11, 12}))
    assert [c["job_id"] for c in manifest("second")["chunks"]] == ["first", "second", "second"]
    worker, result = resume(monkeypatch, "third", "second", Worker(misses={10, 11, 12}))
    assert min(worker.seen) == 15 and result["checkpoint"]["resumed_from"] == "second"
    same_as_reference("third", result, reference)


@pytest.mark.parametrize("resumed", [False, True])
def test_progress_rate_and_eta_count_only_this_attempts_frames(site, monkeypatch, resumed):
    # On a scripted clock the decoder takes 100 s to reach its first frame (FFmpeg start-up
    # and, when resuming, the seek past the reused chunks), the attempt's first frame takes 50 s
    # to process (model warm-up) and each later one 1 s.
    request, clock = render_request(site), [0.0]
    if resumed:
        attempt(monkeypatch, "broken", request, Worker(crash=13))
    monkeypatch.setattr(engine, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
    real_decode, real_json, records = engine.decode, engine.atomic_json, []

    def decode(*args):
        clock[0] += 100
        yield from real_decode(*args)

    def atomic_json(path, value):
        if path.name == "progress.json":
            records.append(value)
        real_json(path, value)

    class Timed(Worker):
        def process(self, worker, bgr, index):
            clock[0] += 1 if self.seen else 50
            return super().process(worker, bgr, index)

    monkeypatch.setattr(engine, "decode", decode)
    monkeypatch.setattr(engine, "atomic_json", atomic_json)
    if resumed:
        _, result = resume(monkeypatch, "timed", "broken", Timed())
    else:
        _, result = attempt(monkeypatch, "timed", request, Timed())
    assert not isinstance(result, Exception), result
    first = 11 if resumed else START  # The resume reuses chunks [3, 7) and [7, 11).
    assert [r["frames"] for r in records] == list(range(first - START + 1, END - START + 1))
    for record in records:
        assert record["total_frames"] == END - START
        assert record["reused_frames"] == first - START
    # No rate is known until a frame after the first is done.
    assert records[0]["frames_per_second"] is records[0]["eta_seconds"] is None
    for index, record in zip(range(first + 1, END), records[1:]):
        # One frame per second, however many frames came before this attempt.
        assert record["frames_per_second"] == 1.0
        assert record["eta_seconds"] == END - index - 1


@pytest.mark.parametrize(
    "changes, advice",
    [({}, "Analyze it again and resubmit"), ({"resume": {"from_job_id": "x"}}, "submit a fresh")],
)
def test_a_reference_changed_after_submission_fails_with_advice_for_the_job(
    site, tmp_path, monkeypatch, changes, advice
):
    face = str(site / "face.png")
    request = render_request(site, start_frame=0, max_frames=8, **changes)
    # The SHA-256 recorded at submission no longer matches, as after an edit.
    request["mappings"] = [
        {"target_face": face, "source_faces": [face], "source_sha256": ["0" * 64]}
    ]
    (tmp_path / "job").mkdir()
    with pytest.raises(common.Refusal, match=f"content changed.*{advice}") as refused:
        test_timing.run_job(tmp_path / "job", monkeypatch, request)
    assert refused.value.code == "reference_changed"
    assert ("Analyze" in str(refused.value)) == (not changes)


def test_segment_accounting_is_restored_at_the_resume_point(site, monkeypatch):
    request = render_request(site, segments=test_segments.MULTI_SHOT)
    _, whole = attempt(monkeypatch, "whole", request, Worker())
    _, error = attempt(monkeypatch, "broken", request, Worker(crash=17))
    assert isinstance(error, RuntimeError)
    worker, result = resume(monkeypatch, "again", "broken", Worker())
    assert min(worker.seen) == 16  # Frames 12-15 are passthrough; chunk [15, 19) is redone.
    assert result["plan"] == whole["plan"] and output("again") == output("whole")
    # Entering the resumed run binds its settings, as an uninterrupted run had them.
    assert worker.seen[16] == (["0", "1"], [60, 60], 30)


# Compatibility checks before reuse


@pytest.mark.parametrize(
    "change, message",
    [
        ({"record": "models"}, "Model YoloFace8n has different bytes"),
        ({"record": "adapter"}, "Adapter engine.py of release r1"),
        ({"record": "revision"}, "Fusion revision"),
        ({"record": "tracking"}, "FaceTrackingEnableToggle carries tracking state"),
        ({"request": {"report": {"review_images": 3}}}, "request differs"),
        ({"runtime": "ffmpeg version 0.1"}, "FFmpeg changed"),
    ],
)
def test_incompatible_checkpoints_are_refused_with_the_reasons(
    site, reference, monkeypatch, change, message
):
    attempt(monkeypatch, "broken", reference[0], Worker(crash=13))
    record = copy.deepcopy(RECORD)
    if change.get("record") == "models":
        record["models"]["YoloFace8n"]["sha256"] = "b" * 64
    elif change.get("record") == "adapter":
        record["adapter"] = {"release": "r2", "code": {**CODE, "engine.py": "1" * 64}}
    elif change.get("record") == "revision":
        record["fusion_revision"] = "e" * 40
    elif change.get("record") == "tracking":
        record["settings"]["controls"]["FaceTrackingEnableToggle"] = True
    if "runtime" in change:
        monkeypatch.setattr(common, "ffmpeg_version", lambda: change["runtime"])
    worker = Worker()
    _, error = resume(monkeypatch, "again", "broken", worker, record, **change.get("request", {}))
    assert isinstance(error, common.Refusal) and error.code == "checkpoint_incompatible"
    assert message in str(error) and "Submit a fresh render" in str(error)
    assert worker.seen == {}


@pytest.mark.parametrize(
    "edit, message",
    [
        ({"version": 2}, "version 2; this release reads version 1 only"),
        ({"format": "other"}, "unknown format"),
        ({"extra": True}, "unexpected or missing fields"),
    ],
)
def test_damaged_or_newer_inherited_manifests_are_refused(
    site, reference, monkeypatch, edit, message
):
    attempt(monkeypatch, "broken", reference[0], Worker(crash=13))
    path = common.job_path("again") / "checkpoint"
    path.mkdir(parents=True)
    document = {**manifest("broken"), **edit}
    common.atomic_json(path / "inherited.json", document)
    request = {**reference[0], "resume": {"from_job_id": "broken"}}
    with pytest.raises(common.Refusal, match=message) as raised:
        engine.Checkpoint(common.job_path("again"), request, RECORD)
    assert raised.value.code == "checkpoint_unusable"


def test_chunks_that_cannot_join_without_re_encoding_fail_the_render(site, reference, monkeypatch):
    attempt(monkeypatch, "broken", reference[0], Worker(crash=13))
    path = common.job_path("broken") / "checkpoint/manifest.json"
    document = common.read_json(path)
    document["codec"]["level"] = 51  # As if an earlier FFmpeg build chose another level.
    common.atomic_json(path, document)
    _, error = resume(monkeypatch, "again", "broken")
    assert "Chunk 2 differs from the earlier chunks in level" in str(error)


def test_the_worker_records_provenance_and_refuses_a_changed_input(site, tmp_path, monkeypatch):
    # engine.run fingerprints a render before Fusion loads and records it with the chunks.
    request = render_request(site, start_frame=0, max_frames=8)
    face = str(site / "face.png")
    request["mappings"] = [{"target_face": face, "source_faces": [face]}]
    directory = tmp_path / "whole"
    directory.mkdir()
    test_timing.run_job(directory, monkeypatch, request)
    recorded = common.read_json(directory / "checkpoint/manifest.json")["provenance"]
    assert recorded["inputs"]["media_path"]["sha256"] == common.file_sha256(site / "early.mkv")
    assert recorded == common.provenance(request)
    # A resume whose input bytes differ from the recorded ones fails before anything loads.
    resumed = tmp_path / "again"
    (resumed / "checkpoint").mkdir(parents=True)
    changed = copy.deepcopy(common.read_json(directory / "checkpoint/manifest.json"))
    changed["provenance"]["inputs"]["media_path"]["sha256"] = "0" * 64
    common.atomic_json(resumed / "checkpoint/inherited.json", changed)
    with pytest.raises(common.Refusal, match="media_path .* has different bytes"):
        test_timing.run_job(resumed, monkeypatch, {**request, "resume": {"from_job_id": "x"}})
    assert common.read_json(resumed / "progress.json") == {"stage": "fingerprinting"}


# Submission, retention and supervision


@pytest.fixture
def backend(tmp_path, monkeypatch):
    """Stub-runner jobs (conftest.WORKER waits for a release file) and a failed render with a
    checkpoint whose provenance matches the workstation now."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    (media_root / "clip.mp4").write_bytes(b"video")
    (media_root / "face.png").write_bytes(b"face")
    config = write_config(
        tmp_path / "remote-config.json", media_root, state_dir=str(tmp_path / "s")
    )
    apply_config(monkeypatch, config)
    monkeypatch.setattr(control, "validate_request", lambda request: request)
    yield media_root
    settle(common.STATE)


def settle(state):
    for job in (state / "jobs").glob("*"):
        (job / "cancel").touch()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        statuses = [
            json.loads(p.read_text())["status"] for p in (state / "jobs").glob("*/state.json")
        ]
        if all(status in common.TERMINAL for status in statuses):
            return
        time.sleep(0.05)


def failed_render(media_root, job_id="broken", status="failed", **changes):
    """A terminal render with two committed chunks, as its worker would have left it."""
    request = {
        "kind": "render",
        "media_path": str(media_root / "clip.mp4"),
        "start_frame": 0,
        "max_frames": None,
        "mappings": [{"target_face": str(media_root / "face.png"), "source_faces": []}],
        **changes,
    }
    directory = common.job_path(job_id)
    chunks = directory / "checkpoint/chunks"
    chunks.mkdir(parents=True)
    for name in ("request.json", "resolved.json"):
        common.atomic_json(directory / name, request)
    entries = []
    for number in range(2):
        (chunks / f"{number:05d}.mp4").write_bytes(b"chunk %d" % number)
        (chunks / f"{number:05d}.json").write_text("{}")
        entries.append(
            {"index": number, "file": f"{number:05d}.mp4", "state": f"{number:05d}.json"}
        )
    document = {
        "format": "visomaster-checkpoint",
        "version": 1,
        "job_id": job_id,
        "created_at": 0,
        "resumed_from": None,
        "intent_sha256": common.intent_digest(request),
        "range": {},
        "chunk_frames": 250,
        "chunking": "",
        "temporal": False,
        "runtime": common.ffmpeg_version(),
        "codec": None,
        "provenance": common.provenance(request),
        "chunks": entries,
    }
    common.atomic_json(directory / "checkpoint/manifest.json", document)
    state = {"job_id": job_id, "status": status, "created_at": 0, "finished_at": 1}
    common.atomic_json(directory / "state.json", state)
    return request


def wait_for(job_id, statuses):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = control.status(job_id)
        if state["status"] in statuses:
            return state
        time.sleep(0.05)
    raise AssertionError(control.status(job_id))


def test_resume_is_a_new_linked_job_with_its_own_links(backend):
    request = failed_render(backend)
    before = common.read_json(common.job_path("broken") / "state.json")
    state = control.dispatch("resume", {"job_id": "again", "from_job_id": "broken"})
    assert state["job_id"] == "again" and state["resume_from"] == "broken"
    assert state["priority"] == "normal"  # Scheduled as a render.
    directory = common.job_path("again")
    stored = common.read_json(directory / "request.json")
    assert stored == {
        **request,
        "resume": {
            "from_job_id": "broken",
            "attempt": 2,
            "invocation": {"from_job_id": "broken", "priority": None},
        },
    }
    assert common.read_json(directory / "checkpoint/inherited.json") == manifest("broken")
    for name in ("00000.mp4", "00000.json", "00001.mp4", "00001.json"):
        own, theirs = directory / "checkpoint/chunks" / name, common.job_path("broken")
        assert own.stat().st_ino == (theirs / "checkpoint/chunks" / name).stat().st_ino
    assert common.read_json(common.job_path("broken") / "state.json") == before
    listed = {job["job_id"]: job for job in control.list_jobs()}
    assert listed["again"]["resume_from"] == "broken" and "resume_from" not in listed["broken"]


def test_identical_resume_retries_are_idempotent_and_changed_intent_conflicts(backend):
    failed_render(backend)
    failed_render(backend, "other")
    first = control.dispatch("resume", {"job_id": "again", "from_job_id": "broken"})
    retry = control.dispatch("resume", {"job_id": "again", "from_job_id": "broken"})
    assert retry["job_id"] == first["job_id"] and retry["created_at"] == first["created_at"]
    for changed in ({"from_job_id": "other"}, {"from_job_id": "broken", "priority": "normal"}):
        with pytest.raises(ValueError, match="different request"):
            control.dispatch("resume", {"job_id": "again", **changed})
    # A retry answers from the persisted job even after the earlier attempt is gone.
    settle(common.STATE)
    control.delete_job("broken")
    assert control.dispatch("resume", {"job_id": "again", "from_job_id": "broken"})["job_id"]
    with pytest.raises(ValueError, match="give it a job_id of its own"):
        control.dispatch("resume", {"job_id": "other", "from_job_id": "other"})
    with pytest.raises(ValueError, match="does not accept media_path"):
        control.dispatch("resume", {"job_id": "x", "from_job_id": "other", "media_path": "/m"})
    # A raw submission cannot pair another request with the earlier attempt's chunks.
    changed = {**common.read_json(common.job_path("other") / "resolved.json"), "start_frame": 2}
    with pytest.raises(ValueError, match="request differs from job other's"):
        control.submit({"job_id": "raw", **changed, "resume": {"from_job_id": "other"}})
    assert not common.job_path("raw").exists()


@pytest.mark.parametrize(
    "status, changes, message",
    [
        ("completed", {}, "completed; its result is final"),
        ("running", {}, "is running; wait for it to end, or cancel it"),
        ("failed", {"kind": "preview"}, "is a preview job; only renders resume"),
        ("failed", {"manifest": None}, "has no checkpoint manifest"),
        ("failed", {"controls": {"KPSSmoothingEnableToggle": True}}, "KPSSmoothingEnableToggle"),
        ("failed", {"media": b"edited"}, "media_path .* has different bytes"),
    ],
)
def test_attempts_that_cannot_be_resumed_are_refused(backend, status, changes, message):
    edits = {key: value for key, value in changes.items() if key in ("kind", "controls")}
    failed_render(backend, status=status, **edits)
    directory = common.job_path("broken")
    (directory / "lease").touch()
    if "manifest" in changes:
        (directory / "checkpoint/manifest.json").unlink()
    if "media" in changes:
        (backend / "clip.mp4").write_bytes(changes["media"])
    with open(directory / "lease") as lease:
        if status == "running":
            fcntl.flock(lease, fcntl.LOCK_EX)  # Held by a live supervisor.
        with pytest.raises(ValueError, match=message):
            control.dispatch("resume", {"job_id": "again", "from_job_id": "broken"})
    assert not common.job_path("again").exists()
    with pytest.raises(ValueError, match="does not exist"):
        control.dispatch("resume", {"job_id": "again", "from_job_id": "missing"})


def test_pruning_the_earlier_attempt_cannot_remove_what_a_resume_uses(backend):
    failed_render(backend)
    control.dispatch("resume", {"job_id": "again", "from_job_id": "broken"})
    chunks = common.job_path("again") / "checkpoint/chunks"
    before = {path.name: path.read_bytes() for path in chunks.iterdir()}
    pruned = control.prune(0, dry_run=False)
    assert [job["job_id"] for job in pruned["jobs"]] == ["broken"]  # The resume is queued.
    assert not common.job_path("broken").exists()
    assert {path.name: path.read_bytes() for path in chunks.iterdir()} == before
    # The resume job itself is protected while active, its checkpoint with it.
    with pytest.raises(ValueError, match="still active|Only completed, failed or cancelled"):
        control.delete_job("again")


def test_concurrent_resumes_each_get_their_own_links(backend):
    failed_render(backend)
    source = common.job_path("broken") / "checkpoint/chunks"
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    errors = []

    def submit(job_id):
        try:
            control.dispatch("resume", {"job_id": job_id, "from_job_id": "broken"})
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=submit, args=(f"again-{n}",)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    for number in range(4):
        own = common.job_path(f"again-{number}") / "checkpoint/chunks"
        assert {path.name for path in own.iterdir()} == set(before)
        # A worker replaces a chunk by renaming over its own link; the shared file is untouched.
        common.atomic_json(own / "00001.json", {"replaced": number})
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before


def supervised(job_id, request, checkpoint=True):
    directory = common.job_path(job_id)
    (directory / "artifacts").mkdir(parents=True)
    common.atomic_json(directory / "request.json", request)
    state = {"job_id": job_id, "status": "queued", "created_at": time.time()}
    common.atomic_json(directory / "state.json", state)
    if checkpoint:
        (directory / "checkpoint/chunks").mkdir(parents=True)
        (directory / "checkpoint/chunks/00000.mp4").write_bytes(b"chunk")
    (directory / "release").touch()
    control.supervise(str(directory), control.SETTINGS.sha256)
    return directory


def test_completion_removes_chunks_and_failure_keeps_them_for_resume(backend):
    done = supervised("done", {"kind": "render"})
    assert control.status("done")["status"] == "completed"
    assert not (done / "checkpoint").exists()
    failed = supervised("failed", {"kind": "render", "fail": True})
    assert control.status("failed")["status"] == "failed"
    assert (failed / "checkpoint/chunks/00000.mp4").exists()


def test_cancellation_never_starts_another_attempt(backend):
    control.submit({"job_id": "render", "kind": "render"})
    wait_for("render", {"running"})
    control.cancel("render")
    assert wait_for("render", {"cancelled"})["status"] == "cancelled"
    time.sleep(0.3)  # Let any (wrongly) restarted attempt appear.
    assert [job["job_id"] for job in control.list_jobs()] == ["render"]


def test_resume_render_tool_sends_its_arguments(monkeypatch):
    from visomaster_mcp import server

    sent = []

    async def command(action, **values):
        sent.append((action, values))
        return {}

    monkeypatch.setattr(server, "command", command)
    asyncio.run(server.resume_render("broken", job_id="again"))
    assert sent[-1] == (
        "resume",
        {"timeout": server.HASHING_TIMEOUT, "job_id": "again", "from_job_id": "broken"},
    )
    asyncio.run(server.resume_render("broken", priority="interactive"))
    assert sent[-1][1]["priority"] == "interactive" and len(sent[-1][1]["job_id"]) == 32

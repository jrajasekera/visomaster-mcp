"""Variant comparisons: validation and limits, durable lifecycle, settings inheritance, identical
inputs, manifests, artifacts, partial failure and cancellation.

Engine tests run engine.run on test_timing's FFmpeg fixtures (25 fps clips whose frame luma is
the source frame number) with Fusion replaced by recorders. Control tests use real submission
with the stub runners of test_submit and test_cancel.
"""

import asyncio
import os
import sys
from fractions import Fraction
from types import SimpleNamespace

import common
import control
import engine
import pytest
import test_cancel
import test_submit
import test_timing
from conftest import apply_config, write_config
from test_cancel import assert_stopped, started, wait_for
from test_scheduling import gpu_held
from test_timing import TONE, audible, sequential

library, media = test_timing.library, test_timing.media  # FFmpeg fixtures.
backend = test_cancel.backend  # Stub runner with a TERM-ignoring descendant.

CONTROL_SPECS = {
    "DetectorModelSelection": {"default": "RetinaFace", "options": ["RetinaFace", "Yolov8"]},
    "DetectorScoreSlider": {"default": 50, "min_value": "1", "max_value": "100"},
    "MaxFacesToDetectSlider": {"default": 5, "min_value": "1", "max_value": "50"},
    "FaceTrackingEnableToggle": {"default": False},
    "KPSSmoothingEnableToggle": {"default": False},
    "SwapOnlyBestMatchEnableToggle": {"default": False},
    "RecognitionModelSelection": {"default": "Inswapper128ArcFace"},
}
PARAMETER_SPECS = {
    "SwapModelSelection": {"default": "Inswapper128", "options": ["Inswapper128", "SimSwap512"]},
    "SimilarityThresholdSlider": {"default": 60, "min_value": "1", "max_value": "100"},
    "FaceRestorerEnableToggle": {"default": False},
    "NotAdjustable": {"default": 1},
}
LAYOUTS = (CONTROL_SPECS, PARAMETER_SPECS)


def variants(*extra):
    return [
        {"name": "base", "parameters": {}, "controls": {}},
        {
            "name": "strict",
            "parameters": {"SimilarityThresholdSlider": 75, "FaceRestorerEnableToggle": True},
            "controls": {"DetectorScoreSlider": 70},
        },
        *extra,
    ]


def comparison(media_path, face, **fields):
    mapping = {
        "target_face": face,
        "source_faces": [face],
        "parameters": {"SimilarityThresholdSlider": 55},
    }
    request = {
        "kind": "compare",
        "media_path": media_path,
        "mappings": [mapping],
        "controls": {"DetectorScoreSlider": 40},
        "variants": variants(),
    }
    return {**request, **fields}


# Validation and limits, before anything is queued.


@pytest.fixture
def validation(monkeypatch):
    monkeypatch.setattr(common, "layouts", lambda: LAYOUTS)
    monkeypatch.setattr(common, "input_path", lambda path: path)
    monkeypatch.setattr(common, "reference_path", lambda path, expected=None: path)
    return comparison("/m/clip.mp4", "/m/face.png", frames=[3])


def test_valid_still_and_clip_requests_pass(validation):
    assert common.validate_request(validation) == validation
    clip = {k: v for k, v in validation.items() if k != "frames"}
    common.validate_request({**clip, "start_frame": 10, "max_frames": 120})
    six = variants(
        *(
            {
                "name": f"v{n}",
                "parameters": {"SwapModelSelection": "SimSwap512"},
                "controls": {"MaxFacesToDetectSlider": n},
            }
            for n in range(1, 5)
        )
    )
    common.validate_request({**validation, "variants": six, "frames": [0, 1, 2, 3]})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"frames": None}, "either explicit frames or a clip"),
        ({"max_frames": 5}, "either explicit frames or a clip"),
        ({"frames": [1, 2, 3, 4, 5]}, "frames must list 1–4 distinct"),
        ({"frames": [2, 2]}, "frames must list 1–4 distinct"),
        ({"frames": [-1]}, "frames must list"),
        ({"frames": [1.0]}, "frames must list"),
        ({"frames": []}, "frames must list"),
        ({"start_frame": 4}, "start_frame applies only to clip"),
        ({"frames": None, "max_frames": 121}, "max_frames of a clip comparison must be 1–120"),
        ({"frames": None, "max_frames": 0}, "1–120"),
        ({"frames": None, "max_frames": True}, "1–120"),
        ({"report": {}}, "Comparisons do not accept report"),
        ({"frame": 2}, "do not accept frame"),
        ({"variants": variants()[:1]}, "variants must list 2–6"),
        ({"variants": variants(*[{"name": f"x{n}"} for n in range(5)])}, "variants must list 2–6"),
        (
            {"variants": variants({"name": "BASE", "controls": {"DetectorScoreSlider": 1}})},
            "Duplicate variant name BASE",
        ),
        ({"variants": variants({"name": "bad name"})}, "Variant names are 1–20"),
        ({"variants": variants({"name": "x" * 21})}, "Variant names are 1–20"),
        ({"variants": variants({"name": "-x"})}, "Variant names"),
        ({"variants": variants({"parameters": {}})}, "Each variant has a name"),
        ({"variants": variants({"name": "x", "report": {}})}, "Each variant has a name"),
        ({"variants": variants({"name": "x", "parameters": []})}, "must be objects"),
        ({"variants": variants({"name": "copy", "controls": {}})}, "repeats another variant"),
        (
            {
                "variants": variants(
                    {"name": "high", "parameters": {"SimilarityThresholdSlider": 101}}
                )
            },
            "Variant high: Invalid value for SimilarityThresholdSlider",
        ),
        (
            {"variants": variants({"name": "odd", "parameters": {"NotAdjustable": 2}})},
            "Variant odd: Unsupported setting NotAdjustable",
        ),
        (
            {"variants": variants({"name": "dfm", "parameters": {"SwapModelSelection": "X"}})},
            "Variant dfm: Invalid value for SwapModelSelection",
        ),
        (
            {"variants": variants({"name": "ctl", "controls": {"DetectorScoreSlider": 0}})},
            "Variant ctl: Invalid value for DetectorScoreSlider",
        ),
        ({"mappings": []}, "Variant base: At least one explicit"),
        ({"mappings": [{"source_faces": ["/m/face.png"]}]}, "Variant base: invalid request"),
        ({"mappings": "x"}, "mappings must list"),
        ({"controls": {"Bogus": 1}}, "Variant base: Unsupported setting Bogus"),
        ({"priority": "urgent"}, "priority must be"),
    ],
)
def test_invalid_comparisons_are_rejected(validation, changes, message):
    request = {**validation, **changes}
    request = {key: value for key, value in request.items() if value is not None}
    with pytest.raises(ValueError, match=message):
        common.validate_request(request)


def test_a_common_setting_that_one_variant_breaks_names_that_variant(validation):
    # Merged settings are what is validated: the common value is fine, one override is not.
    request = {
        **validation,
        "variants": variants({"name": "t", "controls": {"DetectorScoreSlider": 200}}),
    }
    with pytest.raises(ValueError, match="^Variant t: Invalid value for DetectorScoreSlider$"):
        common.validate_request(request)


def test_variant_requests_inherit_everything_but_their_overrides(validation):
    strict = validation["variants"][1]
    single = common.variant_request(validation, strict)
    assert single == {
        "kind": "preview",
        "media_path": "/m/clip.mp4",
        "mappings": [
            {
                "target_face": "/m/face.png",
                "source_faces": ["/m/face.png"],
                "parameters": {"SimilarityThresholdSlider": 75, "FaceRestorerEnableToggle": True},
            }
        ],
        "controls": {"DetectorScoreSlider": 70},
        "frame": 3,
    }
    assert validation["mappings"][0]["parameters"] == {"SimilarityThresholdSlider": 55}
    clip = {**validation, "start_frame": 8, "max_frames": 4}
    del clip["frames"]
    assert common.variant_request(clip, strict)["kind"] == "render"
    assert common.variant_request(clip, strict)["start_frame"] == 8
    base = common.variant_request(clip, validation["variants"][0])
    assert base["controls"] == {"DetectorScoreSlider": 40}
    assert base["mappings"][0]["parameters"] == {"SimilarityThresholdSlider": 55}


def test_still_comparisons_are_interactive_and_clips_normal():
    assert common.job_priority({"kind": "compare", "frames": [1]}) == "interactive"
    assert common.job_priority({"kind": "compare", "max_frames": 5}) == "normal"
    assert common.job_priority({"kind": "compare", "max_frames": 5, "priority": "interactive"}) == (
        "interactive"
    )
    assert common.PRIORITIES["compare"] == "interactive"


def test_long_polls_end_when_the_compared_variant_changes():
    state = {"status": "running", "progress": {"stage": "comparing", "variant": 1}}
    moved = {"status": "running", "progress": {"stage": "comparing", "variant": 2, "frames": 9}}
    assert control.observed(state) != control.observed(moved)
    more = {"status": "running", "progress": {"stage": "comparing", "variant": 1, "frames": 9}}
    assert control.observed(state) == control.observed(more)


# Durable submission with real validation and the stub runner that revalidates (test_submit).


@pytest.fixture
def site(tmp_path, monkeypatch):
    """Media, Fusion layout files with real settings, and test_submit's revalidating runner."""
    media = tmp_path / "media"
    media.mkdir()
    for name in ("clip.mp4", "face.png"):
        (media / name).write_bytes(name.encode())
    runner = tmp_path / "bin/runner"
    runner.parent.mkdir()
    runner.write_text(test_submit.RUNNER)
    runner.chmod(0o755)
    config = write_config(
        tmp_path / "remote-config.json",
        media,
        runner=runner,
        fusion_dir=str(tmp_path / "fusion"),
        state_dir=str(media / "state"),
    )
    settings = apply_config(monkeypatch, config)
    widgets = settings.app / test_submit.WIDGETS
    (widgets / "settings_layout_data.py").write_text(f"LAYOUT = {{'S': {CONTROL_SPECS!r}}}\n")
    (widgets / "common_layout_data.py").write_text(f"LAYOUT = {{'P': {PARAMETER_SPECS!r}}}\n")
    yield SimpleNamespace(
        state=settings.state,
        config=config,
        clip=str(settings.base / "clip.mp4"),
        face=str(settings.base / "face.png"),
    )
    test_submit.settle(settings.state)


def submission(site, job_id="cmp", **fields):
    return {"job_id": job_id, **comparison(site.clip, site.face, **{"frames": [0], **fields})}


def test_durable_retries_conflicts_and_rejection_before_queueing(site, monkeypatch):
    calls = test_submit.counting_popen(monkeypatch)
    first = control.submit(submission(site))
    assert first["priority"] == "interactive" and len(calls) == 1
    # One job, one supervisor: variants are not child jobs.
    assert [p.name for p in (site.state / "jobs").iterdir()] == ["cmp"]
    retried = control.submit(submission(site))
    assert retried["created_at"] == first["created_at"] and len(calls) == 1
    changed = submission(site)
    changed["variants"][1]["parameters"]["SimilarityThresholdSlider"] = 80
    for request in (changed, {**submission(site), "variants": variants()[::-1]}):
        with pytest.raises(ValueError, match="different request"):
            control.submit(request)
    duplicate = submission(site, "dup", variants=variants({"name": "Strict"}))
    with pytest.raises(ValueError, match="Duplicate variant name Strict"):
        control.submit(duplicate)
    invalid = submission(site, "bad", variants=variants({"name": "x", "parameters": {"X": 1}}))
    with pytest.raises(ValueError, match="Variant x: Unsupported setting X"):
        control.submit(invalid)
    over = submission(site, "over", frames=[0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="frames must list 1–4"):
        control.submit(over)
    assert sorted(p.name for p in (site.state / "jobs").iterdir()) == ["cmp"]
    assert len(calls) == 1 and list(site.state.glob("prepare-*")) == []
    # A fresh control process (another client) sees the job and its progress.
    test_submit.wait_file(common.job_path("cmp") / "entered")
    common.atomic_json(
        common.job_path("cmp") / "progress.json",
        {"stage": "comparing", "variant": 2, "variants": 2, "name": "strict"},
    )
    seen = test_submit.control_command(site, {"action": "status", "job_id": "cmp"})["result"]
    assert seen["status"] == "running" and seen["progress"]["name"] == "strict"
    (listed,) = control.list_jobs(only_kind="compare")
    assert listed["variants"] == ["base", "strict"] and listed["frames"] == [0]
    assert "start_frame" not in listed and listed["priority"] == "interactive"
    assert test_submit.complete("cmp")["result"] == {"validated": True}
    clip = {k: v for k, v in submission(site, "clip").items() if k != "frames"}
    assert control.submit({**clip, "max_frames": 5})["priority"] == "normal"
    listed = {job["job_id"]: job for job in control.list_jobs(only_kind="compare")}
    assert listed["clip"]["max_frames"] == 5 and "frames" not in listed["clip"]
    assert test_submit.complete("clip")["result"] == {"validated": True}


def test_comparison_waits_for_the_gpu_and_cancel_stops_its_owned_group(backend):
    request = {"job_id": "cmp", "kind": "compare", "frames": [0], "descendant": "ignore"}
    with gpu_held(backend):
        control.submit(request)
        assert control.status("cmp")["queue"]["reason"] == "gpu_busy"
        assert control.status("cmp")["priority"] == "interactive"
        assert not (common.job_path("cmp") / "entered").exists()
    job = common.job_path("cmp")
    started(job)
    control.dispatch("cancel", {"job_id": "cmp"})
    assert wait_for("cmp", {"cancelled"})["status"] == "cancelled"
    assert_stopped(job)
    with pytest.raises(ValueError, match="successful completion"):
        control.dispatch("artifact", {"job_id": "cmp", "name": "compare.json"})


# Engine: every variant over the same decoded frames, with Fusion replaced by recorders.


class Picture(bytes):
    """A decoded still standing in for a cv2 image."""

    shape = (16, 32, 3)

    def tobytes(self):
        return bytes(self)


@pytest.fixture
def fusion(media, tmp_path, monkeypatch):
    """Scripted Fusion: `script(parameters, frame)` returns "swap" or "miss", or raises."""
    record = SimpleNamespace(configured=[], seen=[], workers=[], saved={}, sheets=[])
    record.script = lambda parameters, frame: "swap"
    monkeypatch.setattr(common, "layouts", lambda: LAYOUTS)
    monkeypatch.setattr(engine, "layouts", lambda: LAYOUTS)
    detector = object()

    def initialize(controls):
        defaults = {key: spec["default"] for key, spec in PARAMETER_SPECS.items()}
        record.window = SimpleNamespace(
            control=engine.initial_controls(CONTROL_SPECS, controls),
            default_parameters=defaults,
            parameters={},
            target_faces={},
            function_worker=SimpleNamespace(run_detect=detector),
        )
        return record.window

    def configure_faces(window, mappings):
        record.configured.append({"controls": dict(window.control), "mappings": mappings})
        return [{"mapping": i, "target_sha256": m["target_sha256"]} for i, m in enumerate(mappings)]

    def make_worker(window, sequential=False):
        # The real one wraps run_detect; each worker must start from the unwrapped detector.
        assert window.function_worker.run_detect is detector
        window.function_worker.run_detect = ("counted", detector)
        record.workers.append(sequential)
        return SimpleNamespace(swap_count=0, telemetry=engine.Telemetry({"0": 60.0}))

    def process(worker, bgr, index):
        parameters = record.configured[-1]["mappings"][0]["parameters"]
        record.seen.append((len(record.configured), index, bytes(bgr)))
        outcome = record.script(parameters, index)
        worker.telemetry.begin(index)
        worker.telemetry.detected(1)
        worker.telemetry.match(object(), "0" if outcome == "swap" else None, {"0": 90.0})
        if outcome == "swap":
            worker.telemetry.swap("0")
            worker.swap_count += 1
        worker.telemetry.end()
        return bgr

    def save_image(path, image):
        path.write_bytes(b"image")
        record.saved[path.name] = bytes(image)

    monkeypatch.setattr(engine, "initialize", initialize)
    monkeypatch.setattr(engine, "configure_faces", configure_faces)
    monkeypatch.setattr(engine, "make_worker", make_worker)
    monkeypatch.setattr(engine, "process", process)
    monkeypatch.setattr(engine, "save_image", save_image)
    monkeypatch.setattr(engine, "thumbnail", lambda image, aspect: image)
    monkeypatch.setattr(engine, "blank", lambda shape: ("blank", shape))
    monkeypatch.setattr(engine, "contact_sheet", lambda *args: record.sheets.append(args))
    monkeypatch.setenv("VM_SUPERVISOR_PID", str(os.getppid()))
    monkeypatch.setenv("VM_CONFIG_SHA256", "identity")
    monkeypatch.setattr(engine, "SETTINGS", SimpleNamespace(sha256="identity"))
    monkeypatch.setattr(engine, "CONFIG_ERROR", None)
    prctl = SimpleNamespace(prctl=lambda *a: 0)
    monkeypatch.setattr(engine, "ctypes", SimpleNamespace(CDLL=lambda name: prctl))
    record.media, record.job = media, tmp_path / "job"
    return record


def run(fusion, request):
    """Submit-time files (request.json and the fingerprinted resolved.json), then engine.run."""
    job = fusion.job
    (job / "artifacts").mkdir(parents=True)
    common.atomic_json(job / "request.json", request)
    common.atomic_json(job / "resolved.json", control.fingerprint(control.resolve_inputs(request)))
    engine.main(job)
    return common.read_json(job / "result.json")


def stills(fusion, name="early.mkv", **fields):
    face = str(fusion.media / "face.png")
    return comparison(str(fusion.media / name), face, **{"frames": [20, 3], **fields})


def test_stills_give_every_variant_the_same_frames_and_inherited_settings(fusion):
    loose = {"name": "loose", "parameters": {"SimilarityThresholdSlider": 50}}
    request = stills(fusion, variants=variants(loose))
    result = run(fusion, request)
    frames = sequential(fusion.media / "early.mkv")
    # Ascending frames, variants in request order, each on identical decoded bytes.
    assert [(variant, index) for variant, index, _ in fusion.seen] == [
        (v, f) for v in (1, 2, 3) for f in (3, 20)
    ]
    assert all(data == frames[index] for _, index, data in fusion.seen)
    # Common settings, then each variant's overrides only; nothing leaks between variants.
    controls = [c["controls"]["DetectorScoreSlider"] for c in fusion.configured]
    thresholds = [c["mappings"][0]["parameters"] for c in fusion.configured]
    assert controls == [40, 70, 40]
    assert thresholds == [
        {"SimilarityThresholdSlider": 55},
        {"SimilarityThresholdSlider": 75, "FaceRestorerEnableToggle": True},
        {"SimilarityThresholdSlider": 50},
    ]
    assert all(c["controls"]["SwapOnlyBestMatchEnableToggle"] for c in fusion.configured)
    assert fusion.workers == [False] * 6  # One single-frame worker per compared frame.
    assert result["kind"] == "stills" and result["complete"] and result["failed_variants"] == []
    assert result["source"]["frames"] == [3, 20]
    assert result["source"]["frame_sha256"] == {
        str(n): engine.hashlib.sha256(frames[n]).hexdigest() for n in (3, 20)
    }
    assert result["source"]["sample_aspect_ratio"] == "1:1"
    names = [v["name"] for v in result["variants"]]
    assert names == ["base", "strict", "loose"] and [v["index"] for v in result["variants"]] == [
        1,
        2,
        3,
    ]
    assert len({v["input_sha256"] for v in result["variants"]}) == 1
    strict = result["variants"][1]
    assert strict["status"] == "completed" and strict["error"] is None
    assert strict["overrides"] == {
        "parameters": {"SimilarityThresholdSlider": 75, "FaceRestorerEnableToggle": True},
        "controls": {"DetectorScoreSlider": 70},
    }
    assert strict["settings"]["mappings"] == [
        {
            "FaceRestorerEnableToggle": True,
            "SimilarityThresholdSlider": 75,
            "SwapModelSelection": "Inswapper128",
        }
    ]
    assert strict["settings"]["controls"] == {
        "DetectorModelSelection": "Yolov8",
        "DetectorScoreSlider": 70,
        "FaceTrackingEnableToggle": False,
        "KPSSmoothingEnableToggle": False,
        "MaxFacesToDetectSlider": 20,
        "SwapOnlyBestMatchEnableToggle": True,
    }
    assert strict["swapped_faces"] == 2
    assert [cell["frame"] for cell in strict["frames"]] == [3, 20]
    assert strict["frames"][0]["artifact"] == "v2-strict-f3.png"
    assert strict["frames"][0]["swap_report"]["artifact"] == "v2-strict-swap-report.json"
    reports = common.read_json(fusion.job / "artifacts/v2-strict-swap-report.json")
    assert [(r["kind"], r["start_frame"], r["end_frame"]) for r in reports] == [
        ("preview", 3, 4),
        ("preview", 20, 21),
    ]
    assert result["artifacts"] == [
        "compare-f3.jpg",
        "compare-f20.jpg",
        "compare.json",
        "original-f3.png",
        "original-f20.png",
        *(
            name
            for prefix in ("v1-base", "v2-strict", "v3-loose")
            for name in (f"{prefix}-f3.png", f"{prefix}-f20.png", f"{prefix}-swap-report.json")
        ),
    ]
    for name in result["artifacts"]:
        assert (fusion.job / "artifacts" / name).exists() or name.endswith(".jpg")
    assert fusion.saved["original-f20.png"] == frames[20]
    assert common.read_json(fusion.job / "artifacts/compare.json") == result
    # Labelled grids: the original, then each variant by index and name, at full verification.
    assert [args[0].name for args in fusion.sheets] == ["compare-f3.jpg", "compare-f20.jpg"]
    path, tiles, size, columns = fusion.sheets[0]
    assert (size, columns) == (engine.COMPARE_TILE, engine.COMPARE_COLUMNS)
    assert [label for label, _, _ in tiles] == ["original f3", "1 base", "2 strict", "3 loose"]
    assert all(verified for _, _, verified in tiles)
    assert all(bytes(image) == frames[3] for _, image, _ in tiles)
    assert common.read_json(fusion.job / "progress.json")["variant"] == 3


def follow_up_payload(arguments, monkeypatch):
    """What submit_swap sends for the manifest's arguments, through the client's own models."""
    from visomaster_mcp import server

    sent = []

    async def command(action, **values):
        sent.append(values)
        return {}

    monkeypatch.setattr(server, "command", command)
    mappings = [server.FaceMapping(**m) for m in arguments["mappings"]]
    asyncio.run(server.submit_swap(**{**arguments, "mappings": mappings, "job_id": "next"}))
    return sent[0]


def test_manifest_follow_up_is_a_valid_single_variant_submission(fusion, monkeypatch):
    request = stills(fusion)
    result = run(fusion, request)
    resolved = common.read_json(fusion.job / "resolved.json")
    for variant, entry in zip(request["variants"], result["variants"]):
        arguments = entry["submit_swap"]
        assert arguments["mode"] == "preview" and arguments["frame"] == 3
        payload = follow_up_payload(arguments, monkeypatch)
        payload.pop("job_id")
        common.validate_request(payload)
        expected = common.variant_request(request, variant)
        assert payload["kind"] == "preview" and payload["frame"] == expected["frame"]
        assert payload["controls"] == expected["controls"]
        for sent, wanted, pinned in zip(
            payload["mappings"], expected["mappings"], resolved["mappings"]
        ):
            assert sent == {**wanted, "source_sha256": pinned["source_sha256"]}
    clip = {k: v for k, v in request.items() if k != "frames"}
    arguments = engine.follow_up(
        {**clip, "start_frame": 14, "max_frames": 5}, resolved, variants()[1]
    )
    assert (arguments["mode"], arguments["start_frame"], arguments["max_frames"]) == (
        "render",
        14,
        5,
    )
    payload = follow_up_payload(arguments, monkeypatch)
    assert (payload["kind"], payload["start_frame"], payload["max_frames"]) == ("render", 14, 5)


def test_failed_variants_are_visible_and_never_a_complete_comparison(fusion):
    def script(parameters, frame):
        if parameters.get("SwapModelSelection") == "SimSwap512" and frame == 20:
            raise RuntimeError("Fusion reported a per-face processing error at source frame 20")
        return "miss" if parameters.get("SimilarityThresholdSlider") == 99 else "swap"

    fusion.script = script
    broken = {"name": "broken", "parameters": {"SwapModelSelection": "SimSwap512"}}
    empty = {"name": "empty", "parameters": {"SimilarityThresholdSlider": 99}}
    result = run(fusion, stills(fusion, variants=variants()[:1] + [broken, empty]))
    assert not result["complete"] and result["failed_variants"] == ["broken", "empty"]
    assert "2 of 3 variants failed" in result["warning"]
    assert (
        "broken (Fusion reported a per-face processing error at source frame 20)"
        in (result["warning"])
    )
    base, failed, missing = result["variants"]
    assert base["status"] == "completed"
    assert failed["status"] == "failed" and "source frame 20" in failed["error"]
    assert missing["error"] == "No faces matched the target references in any frame"
    for entry in (failed, missing):
        assert entry["artifacts"] == [] and entry["input_sha256"] is None
        assert "frames" not in entry and "swapped_faces" not in entry
        assert entry["settings"] is not None and entry["submit_swap"]["mode"] == "preview"
    # Partial artifacts of failed variants are removed and not listed.
    on_disk = {p.name for p in (fusion.job / "artifacts").iterdir()}
    assert not any(name.startswith(("v2-", "v3-")) for name in on_disk | set(result["artifacts"]))
    _, tiles, _, _ = fusion.sheets[1]
    assert [(label, verified) for label, _, verified in tiles] == [
        ("original f20", True),
        ("1 base", True),
        ("2 broken FAILED", False),
        ("3 empty FAILED", False),
    ]
    assert tiles[2][1] == ("blank", (16, 32, 3))
    # Per-frame misses do not fail a variant that swapped somewhere.
    fusion.script = lambda parameters, frame: "miss" if frame == 3 else "swap"
    fusion.job = fusion.job.parent / "again"
    result = run(fusion, stills(fusion))
    assert result["complete"] and result["variants"][0]["frames"][0]["swapped_faces"] == 0


def test_every_variant_failing_fails_the_job(fusion):
    fusion.script = lambda parameters, frame: "miss"
    with pytest.raises(RuntimeError, match="Every variant failed: base: No faces matched"):
        run(fusion, stills(fusion))
    error = common.read_json(fusion.job / "error.json")["message"]
    assert error.startswith("Every variant failed: base: ") and "; strict: " in error
    assert not (fusion.job / "result.json").exists()


@pytest.mark.parametrize(
    ("fields", "message"),
    [({"frames": [3, 25]}, "Frame 25 is out of range"), ({"frames": [3]}, "at most 100 pixels")],
)
def test_source_problems_fail_before_any_variant(fusion, monkeypatch, fields, message):
    monkeypatch.setattr(engine, "MAX_COMPARE_PIXELS", 100 if "pixels" in message else 10**8)
    with pytest.raises(ValueError, match=message):
        run(fusion, stills(fusion, **fields))
    assert fusion.seen == [] and fusion.configured == []
    assert list((fusion.job / "artifacts").iterdir()) == []


def test_image_stills_compare_frame_zero(fusion, monkeypatch):
    picture = Picture(b"\x07" * (16 * 32 * 3))
    monkeypatch.setattr(sys.modules["cv2"], "imread", lambda path: picture)
    result = run(fusion, stills(fusion, "face.png", frames=[0]))
    assert result["source"]["frames"] == [0] and result["source"]["width"] == 32
    assert {data for _, _, data in fusion.seen} == {bytes(picture)}
    fusion.job = fusion.job.parent / "again"
    with pytest.raises(ValueError, match="Images only have frame 0"):
        run(fusion, stills(fusion, "face.png", frames=[1]))


def clip(fusion, name="early.mkv", start=14, count=5, **fields):
    request = stills(fusion, name, **fields)
    del request["frames"]
    return {**request, "start_frame": start, "max_frames": count}


def test_clip_variants_share_range_frames_duration_and_audio(fusion):
    track = {"name": "track", "controls": {"FaceTrackingEnableToggle": True}}
    result = run(fusion, clip(fusion, variants=variants(track)))
    frames = sequential(fusion.media / "early.mkv")
    assert [(variant, index) for variant, index, _ in fusion.seen] == [
        (v, f) for v in (1, 2, 3) for f in range(14, 19)
    ]
    assert all(data == frames[index] for _, index, data in fusion.seen)
    assert fusion.workers == [False, False, True]  # Tracking follows each variant's controls.
    source = result["source"]
    assert source["frames"] == [14, 16, 18]
    assert (source["start_frame"], source["end_frame"], source["frame_count"]) == (14, 19, 5)
    assert (source["fps"], source["start_time"], source["duration"]) == ("25", 0.56, 0.2)
    assert result["kind"] == "clip" and result["complete"]
    assert "same audio" in result["audio_policy"]
    assert len({v["input_sha256"] for v in result["variants"]}) == 1
    artifacts = fusion.job / "artifacts"
    for entry in result["variants"]:
        prefix = f"v{entry['index']}-{entry['name']}"
        assert entry["clip"] == {
            "artifact": f"{prefix}.mp4",
            "frames": 5,
            "fps": "25",
            "duration": 0.2,
            "audio": "AAC",
        }
        assert entry["frames"] == [
            {"frame": n, "artifact": f"{prefix}-f{n}.png"} for n in (14, 16, 18)
        ]
        assert entry["swap_report"]["artifact"] == f"{prefix}-swap-report.json"
        assert entry["submit_swap"]["start_frame"] == 14
        output = artifacts / f"{prefix}.mp4"
        video = engine.video_stream(common.probe(output))
        assert int(video["nb_frames"]) == 5 and video["avg_frame_rate"] == "25/1"
        # The same source interval on every clip: the tone starts 14 frames earlier in each.
        first, last = audible(output)
        assert abs(first - (TONE[0] - 14 / 25)) < 0.025
        assert abs(last - (TONE[1] - 14 / 25)) < 0.025
        report = common.read_json(artifacts / f"{prefix}-swap-report.json")
        assert (report["kind"], report["start_frame"], report["end_frame"]) == ("render", 14, 19)
        assert report["options"]["review_images"] == 0 and report["review_images"] == []
        assert fusion.saved[f"{prefix}-f16.png"] == frames[16]
    assert fusion.saved["original-f18.png"] == frames[18]
    assert not list(fusion.job.glob("*.video-only.mp4"))
    assert [args[0].name for args in fusion.sheets] == [f"compare-f{n}.jpg" for n in (14, 16, 18)]
    progress = common.read_json(fusion.job / "progress.json")
    assert (progress["stage"], progress["variant"], progress["name"]) == ("comparing", 3, "track")
    assert (progress["frames"], progress["total_frames"]) == (5, 5)


def test_clip_keeps_a_rational_rate_and_start_offset(fusion):
    result = run(fusion, clip(fusion, "ntsc.mp4", start=7, count=3))
    source = result["source"]
    assert source["fps"] == "30000/1001"
    assert source["start_time"] == float(Fraction(7 * 1001, 30000))
    assert source["duration"] == float(Fraction(3 * 1001, 30000))
    for entry in result["variants"]:
        assert entry["clip"]["fps"] == "30000/1001" and entry["clip"]["audio"] == "none"
        assert entry["clip"]["duration"] == source["duration"]


def test_failed_clip_variant_leaves_no_partial_files(fusion):
    def script(parameters, frame):
        if parameters.get("FaceRestorerEnableToggle") and frame == 17:
            raise RuntimeError("Fusion changed its mind")
        return "swap"

    fusion.script = script
    result = run(fusion, clip(fusion))
    base, strict = result["variants"]
    assert base["status"] == "completed" and strict["error"] == "Fusion changed its mind"
    assert not result["complete"] and result["failed_variants"] == ["strict"]
    leftovers = [p.name for p in fusion.job.rglob("v2-*")]
    assert leftovers == [] and not any(n.startswith("v2-") for n in result["artifacts"])
    assert "v1-base.mp4" in result["artifacts"]


def test_thumbnails_keep_display_aspect_without_upscaling(monkeypatch):
    cv2 = SimpleNamespace(INTER_AREA=3, resize=lambda image, size, interpolation: size)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    cases = [
        ((1080, 1440, 3), Fraction(4, 3), (480, 270)),  # 4:3 pixels: 16:9 display.
        ((16, 32, 3), Fraction(1), (32, 16)),  # Never enlarged.
        ((100, 100, 3), Fraction(2), (200, 100)),
        ((3840, 2160, 3), Fraction(1), (270, 480)),  # Portrait fits the cell's height.
    ]
    for shape, aspect, size in cases:
        assert engine.thumbnail(SimpleNamespace(shape=shape), aspect) == size


def test_client_sends_only_the_chosen_selection(monkeypatch):
    from pydantic import ValidationError

    from visomaster_mcp import server

    sent = []

    async def command(action, **values):
        sent.append(values)
        return {}

    monkeypatch.setattr(server, "command", command)
    mapping = server.FaceMapping(target_face="/m/t.png", source_faces=["/m/s.png"])
    chosen = [server.Variant(name="a"), server.Variant(name="b", parameters={"X": 1})]
    asyncio.run(server.compare_variants("/m/c.mp4", [mapping], chosen, frames=[4], job_id="s"))
    assert sent[-1] == {
        "job_id": "s",
        "kind": "compare",
        "media_path": "/m/c.mp4",
        "mappings": [{"target_face": "/m/t.png", "source_faces": ["/m/s.png"], "parameters": {}}],
        "variants": [
            {"name": "a", "parameters": {}, "controls": {}},
            {"name": "b", "parameters": {"X": 1}, "controls": {}},
        ],
        "controls": {},
        "frames": [4],
    }
    asyncio.run(
        server.compare_variants(
            "/m/c.mp4", [mapping], chosen, start_frame=3, max_frames=9, priority="normal"
        )
    )
    assert (sent[-1]["start_frame"], sent[-1]["max_frames"], sent[-1]["priority"]) == (
        3,
        9,
        "normal",
    )
    assert "frames" not in sent[-1]
    for name in ("bad name", "", "x" * 21, "_x"):
        with pytest.raises(ValidationError):
            server.Variant(name=name)
    with pytest.raises(ValidationError):
        server.Variant(name="x", report={})

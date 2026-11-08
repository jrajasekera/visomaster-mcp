"""Render segments: validation, plan resolution, per-frame application, passthrough and recipes.

Validation runs common's checks with stubbed paths and layouts. Application runs the engine's
Segments over test_swap_report's Fusion stub. Renders decode and encode test_timing's numbered
clip (frame luma = source frame number, a tone at frames 16 and 17) through real FFmpeg with a
scripted worker that whitens every frame it processes. Recipes are in test_segment_recipes.
"""

import asyncio
import copy
from types import SimpleNamespace

import common
import engine
import pytest
import test_swap_report
import test_timing
from test_swap_report import A, B, Image, feed, person
from test_timing import TONE, audible, sequential

library, media = test_timing.library, test_timing.media  # Fixtures.
fusion = test_swap_report.fusion

CONTROL_SPECS = {
    "DetectorModelSelection": {"default": "Yolov8", "options": ["Yolov8", "RetinaFace"]},
    "DetectorScoreSlider": {"default": 50, "min_value": 1, "max_value": 100},
    "FaceTrackingEnableToggle": {"default": False},
}
PARAMETER_SPECS = {
    "SwapModelSelection": {"default": "Inswapper128", "options": ["Inswapper128"]},
    "SimilarityThresholdSlider": {"default": 60, "min_value": 1, "max_value": 100},
    "FaceRestorerEnableToggle": {"default": False},
}


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(common, "input_path", lambda path: path)
    monkeypatch.setattr(common, "reference_path", lambda path, expected=None: path)
    monkeypatch.setattr(common, "layouts", lambda: (CONTROL_SPECS, PARAMETER_SPECS))


def render_request(segments, start=10, count=20, mappings=2):
    return {
        "kind": "render",
        "media_path": "/m/clip.mp4",
        "mappings": [
            {"target_face": f"/m/t{i}.png", "source_faces": ["/m/s.png"]} for i in range(mappings)
        ],
        "start_frame": start,
        "max_frames": count,
        "segments": segments,
    }


def segment(name, low, high, **changes):
    return {"id": name, "start_frame": low, "end_frame": high, **changes}


THRESHOLD = {"parameters": {"SimilarityThresholdSlider": 75}}


# Validation at submission


def test_valid_segments_pass_and_uniform_requests_are_unchanged(stubbed):
    segments = [
        segment("intro", 0, 12, **THRESHOLD),  # Begins before the render: clipped.
        segment("solo", 12, 15, mappings=[1]),
        segment("title", 15, 18, passthrough=True),
        segment("wide", 25, 40, controls={"DetectorScoreSlider": 30}),  # Ends after it.
    ]
    request = render_request(segments)
    assert common.validate_fusion_request(copy.deepcopy(request)) == request
    uniform = render_request(None)
    del uniform["segments"]
    assert "segments" not in common.validate_fusion_request(dict(uniform))


@pytest.mark.parametrize(
    "segments, message",
    [
        ([], "segments must list 1–100 segments; omit it"),
        ({}, "segments must list 1–100"),
        ([segment("a", 10, 12, **THRESHOLD)] * 101, "1–100"),
        ([{"id": "a", "start_frame": 10}], "needs id, start_frame and end_frame"),
        ([segment("a", 10, 12, swap=True)], "Unknown segment fields swap"),
        ([segment("-a", 10, 12, **THRESHOLD)], "Segment ids are"),
        ([segment(3, 10, 12, **THRESHOLD)], "Segment ids are"),
        ([segment("a", 10, 12, **THRESHOLD), segment("a", 12, 13, **THRESHOLD)], "a is used twice"),
        ([segment("a", 12, 12, **THRESHOLD)], "0 <= start_frame < end_frame"),
        ([segment("a", -1, 12, **THRESHOLD)], "0 <= start_frame < end_frame"),
        ([segment("a", 10.0, 12, **THRESHOLD)], "integer frames"),
        ([segment("a", 10, True, **THRESHOLD)], "integer frames"),
        (
            [segment("a", 10, 14, **THRESHOLD), segment("b", 13, 16, **THRESHOLD)],
            r"a \[10, 14\) and b \[13, 16\) overlap",
        ),
        (
            [segment("a", 14, 16, **THRESHOLD), segment("b", 10, 12, **THRESHOLD)],
            "ascending start_frame order",
        ),
        ([segment("a", 0, 10, **THRESHOLD)], r"\[0, 10\) ends before the rendered frames start"),
        ([segment("a", 30, 31, **THRESHOLD)], r"lies outside the rendered frames \[10, 30\)"),
        ([segment("a", 10, 12, passthrough=False)], "passthrough must be true"),
        (
            [segment("a", 10, 12, passthrough=True, mappings=[0])],
            "is passthrough, so it cannot also set mappings",
        ),
        ([segment("a", 10, 12)], "changes nothing"),
        ([segment("a", 10, 12, parameters={}, controls={})], "changes nothing"),
        ([segment("a", 10, 12, mappings=[2])], r"distinct indices into mappings \(0–1\)"),
        ([segment("a", 10, 12, mappings=[0, 0])], "distinct indices"),
        ([segment("a", 10, 12, mappings=[True])], "distinct indices"),
        ([segment("a", 10, 12, mappings="0")], "distinct indices"),
        ([segment("a", 10, 12, parameters={"Hidden": 1})], "Unsupported setting Hidden"),
        ([segment("a", 10, 12, parameters={"SimilarityThresholdSlider": 0})], "Invalid value"),
        ([segment("a", 10, 12, parameters=[1])], "Settings must be an object"),
        (
            [segment("a", 10, 12, parameters={"SwapModelSelection": "Inswapper128"})],
            "cannot set SwapModelSelection: model selections and tracking toggles hold",
        ),
        (
            [segment("a", 10, 12, controls={"DetectorModelSelection": "RetinaFace"})],
            "cannot set DetectorModelSelection",
        ),
        (
            [segment("a", 10, 12, controls={"FaceTrackingEnableToggle": True})],
            "cannot set FaceTrackingEnableToggle",
        ),
        (
            [segment("a", 0, 40, passthrough=True)],
            r"Every rendered frame \[10, 30\) is passthrough",
        ),
        (
            [segment("a", 10, 20, passthrough=True), segment("b", 20, 30, passthrough=True)],
            "nothing to swap",
        ),
    ],
)
def test_invalid_segments_are_refused_before_work_starts(stubbed, segments, message):
    with pytest.raises(ValueError, match=message):
        common.validate_fusion_request(render_request(segments))


def test_segments_apply_only_to_renders_and_not_to_comparisons(stubbed):
    request = {**render_request([segment("a", 10, 12, **THRESHOLD)]), "kind": "preview"}
    with pytest.raises(ValueError, match="segments apply only to renders"):
        common.validate_fusion_request(request)
    comparison = {
        "kind": "compare",
        "media_path": "/m/clip.mp4",
        "mappings": [],
        "frames": [0],
        "variants": [
            {"name": "a"},
            {"name": "b", "parameters": {"FaceRestorerEnableToggle": True}},
        ],
        "segments": [segment("a", 0, 1, **THRESHOLD)],
    }
    with pytest.raises(ValueError, match="Comparisons do not accept segments"):
        common.validate_request(comparison)


def test_an_open_ended_render_checks_the_far_end_in_the_worker(stubbed):
    # Without max_frames the video's length decides; a late segment passes submission.
    request = render_request([segment("late", 500, 600, **THRESHOLD)], count=None)
    del request["max_frames"]
    assert common.validate_fusion_request(request)
    with pytest.raises(ValueError, match=r"late \[500, 600\) lies outside the rendered frames"):
        common.segment_plan(request["segments"], 10, 25, 2)


# Plan resolution


def test_plan_covers_every_frame_with_clipped_segments_and_base_gaps():
    segments = [
        segment("intro", 0, 12, **THRESHOLD),
        segment("solo", 14, 16, mappings=[1]),
        segment("title", 16, 18, passthrough=True),
        segment("tail", 25, 40, controls={"DetectorScoreSlider": 30}),
    ]
    runs = common.segment_plan(segments, 10, 30, 2)
    assert [(r["segment"], r["start_frame"], r["end_frame"]) for r in runs] == [
        ("intro", 10, 12),  # The render begins inside intro.
        (None, 12, 14),
        ("solo", 14, 16),
        ("title", 16, 18),
        (None, 18, 25),
        ("tail", 25, 30),  # And ends inside tail.
    ]
    assert [r["mappings"] for r in runs] == [[0, 1], [0, 1], [1], [], [0, 1], [0, 1]]
    assert [r["passthrough"] for r in runs] == [False, False, False, True, False, False]
    assert runs[0]["parameters"] == THRESHOLD["parameters"] and runs[1]["parameters"] == {}
    assert runs[5]["controls"] == {"DetectorScoreSlider": 30}
    # Adjacent segments leave no gap between them.
    adjacent = [segment("a", 0, 5, **THRESHOLD), segment("b", 5, 10, mappings=[0])]
    assert [r["segment"] for r in common.segment_plan(adjacent, 0, 10, 1)] == ["a", "b"]


# Application per frame, through the Fusion stub


def test_settings_apply_to_exactly_their_frames(fusion):
    near_a = [1.0, 1.0, 0.0]  # Scores 85.4 against mapping 0's target.
    worker = engine.make_worker(fusion.window)
    runs = common.segment_plan(
        [
            segment("strict", 2, 4, parameters={"SimilarityThresholdSlider": 90}),
            segment("only-b", 4, 6, mappings=[1]),
        ],
        1,
        8,
        2,
    )
    plan = engine.Segments(worker, runs)
    for frame in range(1, 8):
        fusion.scenes[frame] = [person(near_a), person(B)]
        number = plan.at(frame)
        swaps = worker.swap_count
        engine.process(worker, Image(), frame)
        plan.count(number, worker.swap_count - swaps)
    worker.telemetry.finish(8)
    first, second = engine.swap_report(swap_request(), worker.telemetry, None, 1)["mappings"]
    assert first["frames"]["swapped"] == 3  # Frames 1, 6 and 7 (base threshold 60).
    assert first["frames"]["below_threshold"] == 2  # Frames 2 and 3 (threshold 90).
    assert first["frames"]["not_requested"] == 2  # Frames 4 and 5.
    assert second["frames"]["swapped"] == 7 and second["frames"]["not_requested"] == 0
    assert first["coverage"]["requested"] == 0.6
    # Runs: gap 1, strict 2-3 (mapping 1 only clears 90), only-b 4-5, gap 6-7.
    assert [s["swaps"] for s in plan.stats] == [2, 2, 2, 4]
    # Leaving a segment restores the base threshold and every mapping.
    assert sorted(fusion.window.target_faces) == ["0", "1"]
    assert fusion.window.parameters["0"]["SimilarityThresholdSlider"] == 60
    assert worker.telemetry.thresholds == {"0": 60.0, "1": 60.0}


def swap_request():
    return {
        "kind": "render",
        "media_path": "/m/clip.mp4",
        "mappings": [{"target_face": "/m/a.png"}, {"target_face": "/m/b.png"}],
    }


def test_tracking_state_resets_at_every_boundary_and_controls_follow_runs(fusion, monkeypatch):
    fusion.window.control.update(DetectorScoreSlider=50)
    worker = engine.make_worker(fusion.window, sequential=True)
    resets, seen = [], []
    worker.detector.reset_state = lambda: resets.append(worker.frame_number)
    run_detect = fusion.window.function_worker.run_detect

    def detect(frame, control):
        seen.append((frame, control["DetectorScoreSlider"]))
        return run_detect(frame, control)

    fusion.window.function_worker.run_detect = detect
    runs = common.segment_plan(
        [
            segment("low", 2, 4, controls={"DetectorScoreSlider": 30}),
            segment("title", 4, 5, passthrough=True),
            segment("same", 5, 6, controls={"DetectorScoreSlider": 30}),
        ],
        0,
        8,
        2,
    )
    plan = engine.Segments(worker, runs)
    worker.frame_number = None
    for frame in range(8):
        fusion.scenes[frame] = [person(A, matches=True)]
        if not plan.runs[plan.at(frame)]["passthrough"]:
            engine.process(worker, Image(), frame)
        else:
            worker.telemetry.begin(frame, ())
            worker.telemetry.end()
    # Entering a run resets before its first frame is processed: the gap at 0, low at 2,
    # same at 5 (after passthrough at 4) and the gap at 6. Never inside a run.
    assert len(resets) == 4 and resets[1:] == [1, 3, 5]
    assert seen == [(0, 50), (1, 50), (2, 30), (3, 30), (5, 30), (6, 50), (7, 50)]
    assert worker.telemetry.mappings["0"].frames["not_requested"] == 1


# Renders through real FFmpeg


class Scripted:
    """A worker whose process whitens each frame and swaps one face per bound mapping, unless
    the frame is in `misses`; it records what each processed frame was bound with."""

    def __init__(self, misses=()):
        self.swap_count, self.detector, self.misses, self.seen = 0, None, set(misses), {}
        targets = {face_id: object() for face_id in ("0", "1")}
        self.main_window = SimpleNamespace(
            target_faces=targets,
            parameters={face_id: {"SimilarityThresholdSlider": 60} for face_id in targets},
            control={"DetectorScoreSlider": 50},
        )
        self.telemetry = engine.Telemetry({"0": 60.0, "1": 60.0})

    def process(self, worker, bgr, index):
        window = self.main_window
        active = sorted(window.target_faces)
        thresholds = [window.parameters[k]["SimilarityThresholdSlider"] for k in active]
        self.seen[index] = (active, thresholds, window.control["DetectorScoreSlider"])
        swaps = [] if index in self.misses else active
        faces = [(face_id, {face_id: 90.0}) for face_id in active]
        feed(self.telemetry, index, len(faces), faces, swaps, window.target_faces)
        self.swap_count += len(swaps)
        white = type(bgr)(b"\xff" * len(bgr))
        white.shape = bgr.shape
        return white


def render(media, directory, monkeypatch, segments, start=3, count=None, misses=()):
    worker = Scripted(misses)
    monkeypatch.setattr(engine, "process", worker.process)
    monkeypatch.setattr(engine, "save_image", lambda path, image: None)
    monkeypatch.setattr(engine, "review_image", lambda *a: None)
    (directory / "artifacts").mkdir(parents=True)
    request = {
        **swap_request(),
        "media_path": str(media / "early.mkv"),
        "start_frame": start,
        "max_frames": count,
        "segments": segments,
    }
    return worker, engine.render_video(None, request, directory, worker)


MULTI_SHOT = [
    segment("intro", 0, 6, **THRESHOLD),  # The render starts at 3, inside intro.
    segment("solo", 9, 12, mappings=[1]),
    segment("title", 12, 16, passthrough=True),
    segment("tone", 16, 30, controls={"DetectorScoreSlider": 30}),  # Past the 25-frame end.
]


def test_multi_shot_render_keeps_the_timeline_and_passthrough_pixels(media, tmp_path, monkeypatch):
    worker, result = render(media, tmp_path, monkeypatch, MULTI_SHOT)
    base, strict = (["0", "1"], [60, 60], 50), (["0", "1"], [75, 75], 50)
    expected = {3: strict, 4: strict, 5: strict, 6: base, 7: base, 8: base}
    expected.update({9: (["1"], [60], 50), 10: (["1"], [60], 50), 11: (["1"], [60], 50)})
    expected.update({frame: (["0", "1"], [60, 60], 30) for frame in range(16, 25)})
    assert worker.seen == expected  # Frames 12-15 are never processed.
    # 22 frames, in order, with the passthrough frames as decoded and the others processed.
    assert (result["frames"], result["fps"], result["duration"]) == (22, "25", 0.88)
    source = sequential(media / "early.mkv")
    output = sequential(tmp_path / "artifacts/result.mp4", False)
    assert len(output) == 22
    for offset, frame in enumerate(output):
        number = 3 + offset
        if 12 <= number < 16:
            assert max(abs(a - b) for a, b in zip(frame, source[number])) <= 2, number
        else:
            assert min(frame) >= 250, number
    # The tone at source frames 16-17 stays at the same output time.
    assert result["audio"] == "AAC"
    first, last = audible(tmp_path / "artifacts/result.mp4")
    assert abs(first - (TONE[0] - 3 / 25)) < 0.025 and abs(last - (TONE[1] - 3 / 25)) < 0.025
    # The manifest: segments with the frames they cover, and every run's resolved settings.
    plan = common.read_json(tmp_path / "artifacts/plan.json")
    assert (plan["start_frame"], plan["end_frame"]) == (3, 25)
    assert [(s["id"], s["applied"], s["clipped"]) for s in plan["segments"]] == [
        ("intro", [3, 6], True),
        ("solo", [9, 12], False),
        ("title", [12, 16], False),
        ("tone", [16, 25], True),
    ]
    assert plan["segments"][0]["parameters"] == THRESHOLD["parameters"]
    runs = plan["runs"]
    assert [(r["segment"], r["start_frame"], r["end_frame"]) for r in runs] == [
        ("intro", 3, 6),
        (None, 6, 9),
        ("solo", 9, 12),
        ("title", 12, 16),
        ("tone", 16, 25),
    ]
    assert runs[0]["effective"]["parameters"]["0"] == {"SimilarityThresholdSlider": 75}
    assert runs[2]["effective"]["parameters"] == {"1": {"SimilarityThresholdSlider": 60}}
    assert runs[3]["effective"] is None
    assert runs[4]["effective"]["controls"] == {"DetectorScoreSlider": 30}
    assert "half-open" in plan["conventions"] and "base" in plan["gaps"]
    assert "reset" in plan["boundaries"]
    # Result and report: per-run counts; passthrough is not failed coverage.
    assert "plan.json" in result["artifacts"]
    summary = result["plan"]["runs"]
    assert [(r["segment"], r["frames"], r["swaps"]) for r in summary] == [
        ("intro", 3, 6),
        (None, 3, 6),
        ("solo", 3, 3),
        ("title", 4, 0),
        ("tone", 9, 18),
    ]
    report = common.read_json(tmp_path / "artifacts/swap-report.json")
    assert report["plan"] == result["plan"] and report["warnings"] == []
    assert (report["start_frame"], report["end_frame"]) == (3, 25)
    first, second = report["mappings"]
    assert first["frames"]["not_requested"] == 7 and first["frames"]["requested"] == 15
    assert second["frames"]["not_requested"] == 4 and second["coverage"]["requested"] == 1.0
    assert first["intervals"] == [] and second["intervals"] == []


def test_a_render_ending_inside_a_segment_has_no_missing_or_extra_frames(
    media, tmp_path, monkeypatch
):
    worker, result = render(
        media, tmp_path, monkeypatch, [segment("title", 5, 20, passthrough=True)], 0, 8
    )
    assert sorted(worker.seen) == [0, 1, 2, 3, 4]
    assert result["frames"] == 8 and len(sequential(tmp_path / "artifacts/result.mp4", 0)) == 8
    assert result["plan"]["runs"][-1] == {
        "segment": "title",
        "start_frame": 5,
        "end_frame": 8,
        "passthrough": True,
        "mappings": [],
        "frames": 3,
        "swaps": 0,
        "swapped_frames": 0,
    }


def test_a_segment_without_swaps_warns_and_a_plan_without_any_fails(media, tmp_path, monkeypatch):
    segments = [segment("solo", 9, 12, mappings=[1]), segment("title", 12, 16, passthrough=True)]
    _, result = render(media, tmp_path, monkeypatch, segments, misses={9, 10, 11})
    message = "Segment solo [9, 12) swapped no face in its 3 frames"
    assert message in result["swap_report"]["warnings"]
    report = common.read_json(tmp_path / "artifacts/swap-report.json")
    assert {"segment": "solo", "code": "segment_never_swapped", "message": message} in (
        report["warnings"]
    )
    # Passthrough frames cannot hide a render whose requested frames swapped nothing.
    late = tmp_path / "none"
    misses = set(range(25))
    with pytest.raises(RuntimeError, match="matched .* range outside passthrough segments"):
        render(media, late, monkeypatch, segments, misses=misses)
    assert not (late / "artifacts/result.mp4").exists()


@pytest.mark.parametrize(
    "segments, message",
    [
        ([segment("title", 0, 40, passthrough=True)], r"Every rendered frame \[3, 25\)"),
        ([segment("late", 30, 40, **THRESHOLD)], r"late \[30, 40\) lies outside"),
    ],
)
def test_plans_known_only_from_the_video_fail_before_processing(
    media, tmp_path, monkeypatch, segments, message
):
    worker = Scripted()
    with pytest.raises(ValueError, match=message):
        render(media, tmp_path, monkeypatch, segments)
    assert worker.seen == {} and not (tmp_path / "artifacts/plan.json").exists()


# The client


def test_submit_swap_sends_segments_only_when_given(monkeypatch):
    from visomaster_mcp import server

    sent = []

    async def command(action, **values):
        sent.append(values)
        return {}

    monkeypatch.setattr(server, "command", command)
    mapping = server.FaceMapping(target_face="/m/t.png", source_faces=["/m/s.png"])
    asyncio.run(server.submit_swap("/m/clip.mp4", [mapping], job_id="plain"))
    assert "segments" not in sent[-1]
    segments = [
        server.Segment(id="close", start_frame=1, end_frame=3, mappings=[0]),
        server.Segment(id="title", start_frame=3, end_frame=4, passthrough=True),
    ]
    asyncio.run(server.submit_swap("/m/clip.mp4", [mapping], mode="render", segments=segments))
    assert sent[-1]["segments"] == [
        {"id": "close", "start_frame": 1, "end_frame": 3, "mappings": [0]},
        {"id": "title", "start_frame": 3, "end_frame": 4, "passthrough": True},
    ]
    with pytest.raises(ValueError):
        server.Segment(id="x", start_frame=0, end_frame=1, passthrough=False)
    asyncio.run(
        server.save_recipe(
            "cut", media_path="/m/clip.mp4", mappings=[mapping], mode="render", segments=segments
        )
    )
    assert sent[-1]["segments"][1]["passthrough"] is True

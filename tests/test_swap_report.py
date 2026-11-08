"""Per-mapping swap telemetry, reports and review images, with Fusion's matching stubbed.

Telemetry tests feed observations directly. The worker tests run engine.make_worker and
engine.process over a FrameWorker stub that follows Fusion's swap-only-best-match loop. The
render tests decode and encode real FFmpeg fixtures from test_timing with a scripted worker.
"""

import asyncio
import json
import math
import sys
from fractions import Fraction
from types import ModuleType, SimpleNamespace

import common
import engine
import pytest
import test_timing
from test_timing import sequential

library, media = test_timing.library, test_timing.media  # Fixtures.

A, B, C = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]


def feed(telemetry, frame, detected=0, faces=(), swaps=(), requested=None):
    """One frame: `faces` are (Fusion's best mapping or None, {mapping: score})."""
    telemetry.begin(frame, requested)
    telemetry.detected(detected)
    for best, scores in faces:
        telemetry.match(object(), best, scores)
    for face_id in swaps:
        telemetry.swap(face_id)
    telemetry.end()


def two_mappings():
    return engine.Telemetry({"0": 60.0, "1": 60.0})


def request(count=2, **report):
    mappings = [
        {"target_face": f"/m/target-{i}.png", "source_faces": ["/m/source.png"]}
        for i in range(count)
    ]
    fields = {"report": report} if report else {}
    return {"kind": "render", "media_path": "/m/clip.mp4", "mappings": mappings, **fields}


def report_of(telemetry, end, fps=Fraction(25), start=0, **options):
    telemetry.finish(end)
    return engine.swap_report(request(len(telemetry.mappings), **options), telemetry, fps, start)


def test_unused_mapping_gets_a_per_identity_warning():
    telemetry = two_mappings()
    for frame in range(10):
        target = ("0", {"0": 80.0, "1": 40.0})
        other = (None, {"0": 30.0, "1": 52.0 - frame})
        feed(telemetry, frame, 2, [target, other], ["0"])
    report = report_of(telemetry, 10)
    first, second = report["mappings"]
    assert first["frames"]["swapped"] == 10 and first["coverage"]["requested"] == 1.0
    assert first["scores"]["swapped_mean"] == 80.0
    assert second["frames"]["swapped"] == 0 and second["frames"]["below_threshold"] == 10
    assert second["scores"]["best_unswapped"] == 52.0
    assert second["scores"]["best_unswapped_frame"] == 0
    assert second["coverage"] == {"requested": 0.0, "with_faces": 0.0, "expected_visible": None}
    assert [(w["mapping"], w["code"]) for w in report["warnings"]] == [(1, "never_swapped")]
    assert "best observed score 52 against threshold 60" in report["warnings"][0]["message"]
    assert report["frames"] == {
        "processed": 10,
        "with_faces": 10,
        "faces_detected": 20,
        "faces_scored": 20,
    }
    assert engine.review_frames(report, 6) == [(1, 0, "never_swapped")]
    summary = engine.report_summary(report)
    assert summary["mappings"][1] == {
        "mapping": 1,
        "swapped_frames": 0,
        "requested_frames": 10,
        "coverage": 0.0,
        "ambiguous_frames": 0,
        "suspicious_intervals": 0,
    }
    assert summary["warnings"] == [report["warnings"][0]["message"]]


def test_reasons_distinguish_absence_unscored_faces_thresholds_and_claims():
    telemetry = two_mappings()
    feed(telemetry, 0)  # No face at all.
    feed(telemetry, 1, 2, [(None, {"0": 40.0, "1": 30.0})])  # One face too small to score.
    feed(telemetry, 2, 1, [(None, {"0": 40.0, "1": 30.0})])  # Scored, below both thresholds.
    feed(telemetry, 3, 1, [("0", {"0": 90.0, "1": 65.0})], ["0"])  # Clears both; 0 wins.
    feed(telemetry, 4, 1, [("1", {"0": 20.0, "1": 70.0})])  # Matched 1, but nothing swapped.
    reasons = [m["frames"] for m in report_of(telemetry, 5)["mappings"]]
    expected = {"no_faces": 1, "unscored": 1, "below_threshold": 2, "swapped": 1}
    assert {k: reasons[0][k] for k in expected} == expected
    assert reasons[0]["requested"] == 5 and reasons[0]["with_faces"] == 4
    expected = {"no_faces": 1, "unscored": 1, "below_threshold": 1, "claimed_by_other": 1}
    assert {k: reasons[1][k] for k in expected} == expected
    assert reasons[1]["not_swapped"] == 1 and reasons[1]["error"] == 0


def test_ambiguous_candidates_follow_observed_scores():
    telemetry = two_mappings()
    feed(telemetry, 0, 1, [("0", {"0": 75.0, "1": 70.0})], ["0"])  # Clears both thresholds.
    feed(telemetry, 1, 2, [("0", {"0": 80.0, "1": 10.0}), ("0", {"0": 65.0, "1": 5.0})], ["0"])
    feed(telemetry, 2, 2, [("0", {"0": 80.0, "1": 10.0}), ("1", {"0": 5.0, "1": 85.0})])
    first, second = report_of(telemetry, 3)["mappings"]
    assert first["ambiguous_frames"] == 2 and second["ambiguous_frames"] == 1
    assert [(i["kind"], i["start_frame"], i["end_frame"]) for i in first["intervals"]] == [
        ("ambiguous", 0, 2),
        ("not_swapped", 2, 3),
    ]
    assert [(i["kind"], i["start_frame"], i["end_frame"]) for i in second["intervals"]] == [
        ("ambiguous", 0, 1),
        ("claimed_by_other", 0, 1),
        ("not_swapped", 2, 3),
    ]
    # The swapped face's score is reported, not the competing look-alike's.
    assert first["scores"]["swapped_min"] == 75.0 and first["scores"]["swapped_max"] == 80.0


def test_dropouts_need_faces_and_swaps_on_both_sides():
    telemetry = engine.Telemetry({"0": 60.0})
    below = [(None, {"0": 40.0})]
    plan = ["swap", "swap", "below", "below", "swap", "none", "none", "swap", "none", "below"]
    plan += ["swap", "below", "below"]
    for frame, kind in enumerate(plan):
        if kind == "swap":
            feed(telemetry, frame, 1, [("0", {"0": 90.0})], ["0"])
        elif kind == "below":
            feed(telemetry, frame, 1, below)
        else:
            feed(telemetry, frame)
    (mapping,) = report_of(telemetry, len(plan))["mappings"]
    intervals = [(i["kind"], i["start_frame"], i["end_frame"]) for i in mapping["intervals"]]
    # 2-3 had faces; 5-6 had none, so they are not asserted; 8-9 mixes both; 11-12 trail.
    assert intervals == [("dropout", 2, 4), ("dropout", 8, 10)]
    assert mapping["intervals"][1]["reasons"] == {"no_faces": 1, "below_threshold": 1}
    assert mapping["intervals"][0]["frames"] == 2
    assert mapping["suspicious"]["dropout"] == 2
    assert [w["code"] for w in report_of(telemetry, len(plan))["warnings"]] == [
        "suspicious_intervals"
    ]


def test_intervals_use_rational_times_from_a_nonzero_start():
    fps, start = Fraction(30000, 1001), 1000
    telemetry = engine.Telemetry({"0": 60.0})
    for frame in range(start, start + 10):
        if frame in (1003, 1004):
            feed(telemetry, frame, 1, [(None, {"0": 50.0})])
        else:
            feed(telemetry, frame, 1, [("0", {"0": 90.0})], ["0"])
    report = report_of(telemetry, start + 10, fps, start)
    (interval,) = report["mappings"][0]["intervals"]
    assert (interval["start_frame"], interval["end_frame"], interval["frames"]) == (1003, 1005, 2)
    assert interval["start_time"] == round(1003 * 1001 / 30000, 6)
    assert interval["end_time"] == round(1005 * 1001 / 30000, 6)
    assert interval["output_start_time"] == round(3 * 1001 / 30000, 6)
    assert (report["start_frame"], report["end_frame"], report["fps"]) == (1000, 1010, "30000/1001")
    assert engine.review_frames(report, 1) == [(0, 1003, "dropout")]


def test_interval_and_review_limits(monkeypatch):
    monkeypatch.setattr(engine, "MAX_INTERVALS", 3)
    telemetry = engine.Telemetry({"0": 60.0, "1": 60.0})
    for frame in range(20):
        # Every other frame is ambiguous for mapping 0; mapping 1 never wins.
        scores = {"0": 80.0, "1": 70.0 if frame % 2 else 10.0}
        feed(telemetry, frame, 1, [("0", scores)], ["0"])
    report = report_of(telemetry, 20)
    first = report["mappings"][0]
    assert len(first["intervals"]) == 3 and first["intervals_omitted"] == 7
    assert first["ambiguous_frames"] == 10
    assert engine.report_summary(report)["mappings"][0]["suspicious_intervals"] == 10
    assert "7 more" in report["warnings"][0]["message"]
    # The never-swapped mapping's best frame comes first, then intervals by kind and length;
    # a frame already shown for a mapping is not repeated.
    assert engine.review_frames(report, 2) == [
        (1, 1, "never_swapped"),
        (1, 3, "claimed_by_other"),
    ]
    assert len(engine.review_frames(report, 24)) == 5
    assert engine.review_frames(report, 0) == []


def test_not_requested_frames_leave_coverage_and_break_dropouts():
    telemetry = two_mappings()
    swap0 = [("0", {"0": 90.0, "1": 10.0})]
    feed(telemetry, 0, 1, swap0, ["0"])
    feed(telemetry, 1, 1, [(None, {"0": 10.0, "1": 10.0})], requested={"1"})
    feed(telemetry, 2, 1, [(None, {"0": 10.0, "1": 10.0})])
    feed(telemetry, 3, 1, swap0, ["0"], requested={"0"})
    first, second = report_of(telemetry, 4)["mappings"]
    assert first["frames"]["not_requested"] == 1 and first["frames"]["requested"] == 3
    assert first["coverage"]["requested"] == round(2 / 3, 4)
    assert first["intervals"] == []  # Passthrough is no dropout.
    assert second["frames"]["requested"] == 3 and second["frames"]["not_requested"] == 1


def test_expected_visible_ranges_report_missing_frames_and_strict_coverage():
    telemetry = two_mappings()
    telemetry.expect({"1": [(2, 6)]})
    for frame in range(8):
        if frame in (2, 3):
            feed(telemetry, frame, 1, [("1", {"0": 10.0, "1": 80.0})], ["1"])
        elif frame == 4:
            feed(telemetry, frame)  # Asserted visible, so even a no-face frame counts.
        else:
            feed(telemetry, frame, 1, [("0", {"0": 80.0, "1": 10.0})], ["0"])
    report = report_of(telemetry, 8, min_coverage=0.75)
    second = report["mappings"][1]
    assert second["expected_visible"] == [[2, 6]]
    assert second["frames"]["expected_visible"] == 4
    assert second["coverage"]["expected_visible"] == 0.5
    missing = [i for i in second["intervals"] if i["kind"] == "expected_missing"]
    assert [(i["start_frame"], i["end_frame"], i["reasons"]) for i in missing] == [
        (4, 6, {"no_faces": 1, "below_threshold": 1})
    ]
    messages = [w["message"] for w in report["warnings"]]
    assert "Mapping 1 was not swapped in 2 of its 4 expected-visible frames" in messages
    assert engine.report_summary(report)["mappings"][1]["expected_visible_coverage"] == 0.5
    # Mapping 0 has no ranges, so all requested frames are its denominator: 5 of 8.
    assert engine.coverage_shortfalls(report) == [
        "mapping 0 5 of 8 requested frames",
        "mapping 1 2 of 4 expected-visible frames",
    ]
    assert report["options"]["min_coverage"] == 0.75
    assert engine.coverage_shortfalls(report_of(two_mappings(), 0)) == []


# A Fusion stub: detections per frame are people (embedding, or None when too small to
# recognize); FrameWorker follows Fusion's swap-only-best-match loop and threshold rule.
SMALL = None


@pytest.fixture
def fusion(monkeypatch):
    scenes, calls = {}, []

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        return 100 - (1 - dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))) * 50

    function_worker = SimpleNamespace(
        run_detect=lambda frame, control: (scenes[frame], scenes[frame], scenes[frame]),
        findCosineDistance=cosine,
    )

    class FrameWorker:
        def __init__(self, window, is_single_frame):
            self.main_window, self.is_single_frame = window, is_single_frame
            self.function_worker = window.function_worker

        def _find_best_target_match(self, embedding, control, snapshot, tracked=False):
            calls.append("match")
            best, params, high = None, None, -1.0
            for face_id, target in snapshot.items():
                score = self.function_worker.findCosineDistance(
                    embedding, target.get_embedding(control["RecognitionModelSelection"])
                )
                threshold = self.parameters[face_id]["SimilarityThresholdSlider"]
                if score >= threshold and score > high:
                    best, params, high = target, self.parameters[face_id], score
            return best, params, high

        def swap_core(self, image, face, *args, t_e=None, **kwargs):
            if face.get("fail"):
                raise RuntimeError("swap failed")
            return image, None, None

        def process_frame(self, control, stop_event):
            if self.is_single_frame:
                people = self.function_worker.run_detect(self.frame_number, control)[0]
            else:
                people = self.precomputed_bboxes
            faces = [dict(p) for p in people if p["embedding"] is not SMALL]
            snapshot = dict(self.main_window.target_faces)
            for target in snapshot.values():
                chosen, high = None, -1.0
                for face in faces:
                    match, _, score = self._find_best_target_match(
                        face["embedding"], control, snapshot
                    )
                    if match is target and score > high:
                        chosen, high = face, score
                if chosen is not None:
                    try:
                        self.swap_core(self.frame, chosen, t_e=target.get_embedding("ArcFace"))
                    except Exception as error:
                        print(f"[ERROR] Standard mode swap_core failed for best face: {error}")
            return self.frame

    class SequentialDetector:
        """Detects through the window's FunctionWorker, then drops unmatched faces."""

        def __init__(self, window):
            self.window = window

        def reset_state(self):
            pass

        def run(self, frame, control, parameters, frame_number):
            people = self.window.function_worker.run_detect(frame_number, control)[0]
            kept = [p for p in people if p.get("matches")]
            return kept, None, None, None

    modules = {
        "numpy": {"ascontiguousarray": lambda array: array},
        "app": {},
        "app.processors": {},
        "app.processors.workers": {},
        "app.processors.workers.frame_worker": {"FrameWorker": FrameWorker},
        "app.processors.video_utils": {},
        "app.processors.video_utils.sequential_detector": {
            "SequentialDetector": SequentialDetector
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    targets = {}
    for face_id, embedding in (("0", A), ("1", B)):
        arcface = list(embedding)  # The array the swap step receives as t_e.
        targets[face_id] = engine.TargetFace(face_id, {"ArcFace": arcface}, {})
    window = SimpleNamespace(
        control={"RecognitionModelSelection": "ArcFace"},
        parameters={face_id: {"SimilarityThresholdSlider": 60} for face_id in targets},
        target_faces=targets,
        function_worker=function_worker,
    )
    return SimpleNamespace(window=window, scenes=scenes, calls=calls)


class Image:
    """A frame stand-in: the engine flips channels with [..., ::-1]."""

    def __getitem__(self, key):
        return self


def person(embedding, **flags):
    return {"embedding": embedding, **flags}


def test_worker_attributes_fusion_matches_and_swaps_to_mappings(fusion):
    worker = engine.make_worker(fusion.window)
    # A swaps as mapping 0; C matches nobody; a face too small to recognize is unscored.
    fusion.scenes[10] = [person(A), person(C)]
    fusion.scenes[11] = [person(C), person(SMALL)]
    fusion.scenes[12] = [person(B)]
    fusion.scenes[13] = []
    for frame in (10, 11, 12, 13):
        engine.process(worker, Image(), frame)
    # Fusion asked about each face once per target; each face was scored once.
    assert fusion.calls.count("match") == 2 * 2 + 2 * 1 + 2 * 1
    assert worker.swap_count == 2
    telemetry = worker.telemetry
    assert (telemetry.processed, telemetry.detections, telemetry.scored_faces) == (4, 5, 4)
    report = report_of(telemetry, 14, start=10)
    first, second = (m["frames"] for m in report["mappings"])
    assert (first["swapped"], first["unscored"], first["below_threshold"]) == (1, 1, 1)
    assert (second["swapped"], second["unscored"], second["below_threshold"]) == (1, 1, 1)
    assert first["no_faces"] == second["no_faces"] == 1
    assert report["mappings"][0]["scores"]["swapped_max"] == 100.0
    assert report["mappings"][1]["scores"]["best_unswapped"] == 50.0
    assert report["detection_path"] == "per_frame"


def test_caught_fusion_swap_error_fails_with_frame_and_mapping(fusion, capsys):
    worker = engine.make_worker(fusion.window)
    fusion.scenes[12] = [person(A), person(B, fail=True)]
    with pytest.raises(RuntimeError, match=r"at source frame 12 \(mapping 1\); see worker.log"):
        engine.process(worker, Image(), 12)
    assert "[ERROR]" in capsys.readouterr().out
    assert worker.telemetry.processed == 0  # The failed frame never completes a report.


def test_sequential_path_counts_faces_the_detector_filtered_as_unscored(fusion):
    worker = engine.make_worker(fusion.window, sequential=True)
    fusion.scenes[5] = [person(A, matches=True), person(C)]
    engine.process(worker, Image(), 5)
    report = report_of(worker.telemetry, 6, start=5)
    assert report["detection_path"] == "sequential"
    assert "SequentialDetector" in report["evidence"]["detection"]
    assert report["frames"]["faces_detected"] == 2 and report["frames"]["faces_scored"] == 1
    assert report["mappings"][0]["frames"]["swapped"] == 1
    assert report["mappings"][1]["frames"]["unscored"] == 1


# Renders through real FFmpeg (test_timing's 25 fps clips, frame luma = source frame number).


def scripted(plan, telemetry):
    """A worker whose process feeds `plan[frame]` = (faces, swaps) into its telemetry."""
    worker = SimpleNamespace(swap_count=0, telemetry=telemetry)

    def process(worker, bgr, index):
        faces, swaps = plan.get(index, ([], []))
        feed(worker.telemetry, index, len(faces), faces, swaps)
        worker.swap_count += len(swaps)
        return bgr

    return worker, process


def render(media, directory, monkeypatch, plan, start=14, count=5, processed=None, **options):
    images, processed = [], [] if processed is None else processed
    worker, process = scripted(plan, two_mappings())

    def recording(worker, bgr, index):
        processed.append(index)
        return process(worker, bgr, index)

    monkeypatch.setattr(engine, "process", recording)
    monkeypatch.setattr(engine, "save_image", lambda path, image: None)
    monkeypatch.setattr(engine, "review_image", lambda *a: images.append(a))
    (directory / "artifacts").mkdir(parents=True)
    job = {**request(**options), "media_path": str(media / "early.mkv")}
    job.update(start_frame=start, max_frames=count)
    return engine.render_video(None, job, directory, worker), images


SWAP0 = ([("0", {"0": 90.0, "1": 10.0})], ["0"])
MISS0 = ([(None, {"0": 40.0, "1": 55.0})], [])


def test_render_publishes_report_summary_and_labelled_review_images(media, tmp_path, monkeypatch):
    plan = {14: SWAP0, 15: MISS0, 16: MISS0, 17: SWAP0, 18: SWAP0}
    result, images = render(media, tmp_path, monkeypatch, plan)
    artifacts = tmp_path / "artifacts"
    assert result["artifacts"] == [
        "result.mp4",
        "preview.jpg",
        "before.jpg",
        "swap-report.json",
        "review-01.jpg",
        "review-02.jpg",
    ]
    summary = result["swap_report"]
    assert summary["artifact"] == "swap-report.json" and summary["version"] == 1
    assert summary["review_images"] == ["review-01.jpg", "review-02.jpg"]
    assert [m["swapped_frames"] for m in summary["mappings"]] == [3, 0]
    assert summary["warnings"][0].startswith("Mapping 0 has suspicious intervals (1 dropout")
    assert summary["warnings"][1].startswith("Mapping 1 was never swapped in 5 frames")
    report = common.read_json(artifacts / "swap-report.json")
    assert report["complete"] and (report["start_frame"], report["end_frame"]) == (14, 19)
    (dropout,) = report["mappings"][0]["intervals"]
    assert (dropout["start_frame"], dropout["end_frame"]) == (15, 17)
    assert (dropout["start_time"], dropout["end_time"], dropout["output_start_time"]) == (
        0.6,
        0.68,
        0.04,
    )
    assert report["review_images"] == [
        {
            "artifact": "review-01.jpg",
            "mapping": 1,
            "frame": 15,
            "time": 0.6,
            "output_time": 0.04,
            "reason": "never_swapped",
        },
        {
            "artifact": "review-02.jpg",
            "mapping": 0,
            "frame": 15,
            "time": 0.6,
            "output_time": 0.04,
            "reason": "dropout",
        },
    ]
    # Each pair is the source frame and the published output's frame at that source index.
    source, output = sequential(media / "early.mkv"), sequential(artifacts / "result.mp4", False)
    paths = [path for path, *_ in images]
    assert paths == [artifacts / "review-01.jpg", artifacts / "review-02.jpg"]
    assert all(path.parent == artifacts for path in paths)
    for _, before, after, label in images:
        assert before == source[15] and after == output[1]
    assert images[0][3] == "mapping 1 | frame 15 | 0.600 s | never_swapped | before, after"


def test_review_image_count_is_configurable(media, tmp_path, monkeypatch):
    plan = {14: SWAP0, 15: MISS0, 16: MISS0, 17: SWAP0, 18: SWAP0}
    result, images = render(media, tmp_path, monkeypatch, plan, review_images=0)
    assert result["swap_report"]["review_images"] == [] and images == []
    assert result["artifacts"][-1] == "swap-report.json"


def test_strict_coverage_fails_without_publishing_output_or_report(media, tmp_path, monkeypatch):
    plan = {frame: SWAP0 for frame in range(14, 19)}
    with pytest.raises(RuntimeError, match="below min_coverage 0.5: mapping 1 0 of 5 requested"):
        render(media, tmp_path, monkeypatch, plan, min_coverage=0.5)
    assert not (tmp_path / "artifacts/swap-report.json").exists()
    assert not (tmp_path / "artifacts/result.mp4").exists()


def test_zero_swaps_still_fail_without_a_report(media, tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="No faces matched the target references"):
        render(media, tmp_path, monkeypatch, {15: MISS0})
    assert not (tmp_path / "artifacts/swap-report.json").exists()


def test_expected_range_past_the_video_fails_before_processing(media, tmp_path, monkeypatch):
    late, processed = [{"mapping": 1, "start_frame": 20, "end_frame": 26}], []
    with pytest.raises(ValueError, match="ends after the rendered range, which ends at frame 25"):
        render(media, tmp_path, monkeypatch, {}, 14, None, processed, expected_visible=late)
    assert processed == []


def test_report_options_are_validated(monkeypatch):
    monkeypatch.setattr(common, "input_path", lambda path: path)
    monkeypatch.setattr(common, "reference_path", lambda path, expected=None: path)
    monkeypatch.setattr(common, "layouts", lambda: ({}, {}))
    base = {**request(), "start_frame": 10, "max_frames": 20}
    good = {
        "review_images": 24,
        "min_coverage": 1,
        "expected_visible": [{"mapping": 1, "start_frame": 10, "end_frame": 30}],
    }
    assert common.validate_fusion_request({**base, "report": good})["report"] == good
    bad = [
        ([], "must be an object"),
        ({"images": 1}, "Unknown report options images"),
        ({"review_images": 25}, "review_images must be an integer from 0 to 24"),
        ({"review_images": True}, "review_images"),
        ({"min_coverage": 0}, "min_coverage"),
        ({"min_coverage": 1.5}, "min_coverage"),
        ({"min_coverage": float("nan")}, "min_coverage"),
        ({"expected_visible": {}}, "at most 100 ranges"),
        ({"expected_visible": [{"mapping": 0, "start_frame": 10}]}, "needs mapping"),
        ({"expected_visible": [{"mapping": 2, "start_frame": 10, "end_frame": 11}]}, "2 does"),
        ({"expected_visible": [{"mapping": 0, "start_frame": 9, "end_frame": 11}]}, "within"),
        ({"expected_visible": [{"mapping": 0, "start_frame": 12, "end_frame": 12}]}, "within"),
        ({"expected_visible": [{"mapping": 0, "start_frame": 12, "end_frame": 31}]}, "within"),
        ({"expected_visible": [{"mapping": 0, "start_frame": 1.0, "end_frame": 12}]}, "integers"),
    ]
    for options, message in bad:
        with pytest.raises(ValueError, match=message):
            common.validate_fusion_request({**base, "report": options})
    with pytest.raises(ValueError, match="apply only to renders"):
        common.validate_fusion_request({**base, "kind": "preview", "report": {}})


def test_submit_swap_sends_report_options_only_when_given(monkeypatch):
    from visomaster_mcp import server

    sent = []

    async def command(action, **values):
        sent.append(values)
        return {}

    monkeypatch.setattr(server, "command", command)
    mapping = server.FaceMapping(target_face="/m/t.png", source_faces=["/m/s.png"])
    asyncio.run(server.submit_swap("/m/clip.mp4", [mapping], job_id="plain"))
    assert "report" not in sent[-1]
    options = server.SwapReportOptions(
        expected_visible=[server.VisibleRange(mapping=0, start_frame=0, end_frame=5)]
    )
    asyncio.run(server.submit_swap("/m/clip.mp4", [mapping], mode="render", report=options))
    assert sent[-1]["report"] == {
        "review_images": 6,
        "expected_visible": [{"mapping": 0, "start_frame": 0, "end_frame": 5}],
    }
    assert json.dumps(sent[-1]["report"])

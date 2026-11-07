"""Detection references, identity clustering and best-match swapping, with Fusion stubbed.

A scene is a set of people with boxes and embeddings. The stub detector finds a person whose
box centre lies inside the (cropped) image, unless that person is marked as missed in crops,
as a small or blurred face can be beside a sharper neighbour.
"""

import asyncio
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import common
import control
import engine
import pytest
from conftest import apply_config, write_config

A, B = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
# Scores 55 against A on Fusion's scale (50 + 50 * cosine): a profile view of the same person.
PROFILE = [0.1, math.sqrt(0.99), 0.0]
SPECS = {"SimilarityThresholdSlider": {"default": 60, "min_value": "1", "max_value": "100"}}


class Person:
    def __init__(self, box, embedding, in_crops=True):
        self.box, self.embedding, self.in_crops = box, embedding, in_crops


class Picture:
    """An image as a window (x1, y1, x2, y2) on a scene; slicing crops the window."""

    def __init__(self, people, bounds, full=False):
        self.people, self.bounds, self.full = people, bounds, full
        self.shape = (bounds[3] - bounds[1], bounds[2] - bounds[0], 3)

    def __getitem__(self, index):
        rows, columns = index
        x, y = self.bounds[:2]
        bounds = (x + columns.start, y + rows.start, x + columns.stop, y + rows.stop)
        return Picture(self.people, bounds)

    def copy(self):
        return self


def scene(*people, width=640, height=480):
    return Picture(people, (0, 0, width, height), full=True)


class Vector(list):
    def ravel(self):
        return self


@pytest.fixture
def fusion(tmp_path, monkeypatch):
    """Stub Fusion: records detector calls, saved images and contact-sheet tiles."""
    calls, saved, sheets = [], {}, []

    def detect(window, image, max_faces=None):
        calls.append((image.bounds, max_faces))
        x, y = image.bounds[:2]
        found = []
        for person in image.people:
            cx, cy = [(a + b) / 2 for a, b in zip(person.box[:2], person.box[2:])]
            inside = (
                image.bounds[0] <= cx < image.bounds[2] and image.bounds[1] <= cy < image.bounds[3]
            )
            if inside and (image.full or person.in_crops):
                found.append(person)
        boxes = [[v - shift for v, shift in zip(p.box, (x, y, x, y))] for p in found]
        return image, boxes, found

    def recognize(tensor, person, mode, model):
        assert model == "ArcFace"
        return (None if person.embedding is None else Vector(person.embedding)), None

    window = SimpleNamespace(
        control={"RecognitionModelSelection": "ArcFace"},
        function_worker=SimpleNamespace(run_recognize_direct=recognize),
    )
    frames = {}
    monkeypatch.setattr(engine, "detect", detect)
    monkeypatch.setattr(engine, "layouts", lambda: ({}, SPECS))
    monkeypatch.setattr(engine, "input_path", Path)
    monkeypatch.setattr(engine, "load_frame", lambda path, number=0, timing=None: frames[number])
    monkeypatch.setattr(engine, "save_image", lambda path, image: saved.__setitem__(path, image))
    monkeypatch.setattr(engine, "contact_sheet", lambda path, tiles: sheets.append(tiles))
    monkeypatch.setattr(engine, "timeline", lambda path, end=None: None)
    monkeypatch.setitem(
        sys.modules, "cv2", SimpleNamespace(imread=lambda path: saved.get(Path(path)))
    )
    (tmp_path / "artifacts").mkdir()
    return SimpleNamespace(
        window=window, frames=frames, calls=calls, saved=saved, sheets=sheets, directory=tmp_path
    )


def single(fusion, picture):
    fusion.frames[0] = picture
    return engine.detect_frame(fusion.window, {"media_path": "/m/clip.mp4"}, fusion.directory)


def multi(fusion, pictures, **fields):
    fusion.frames.update(pictures)
    request = {"media_path": "/m/clip.mp4", "frames": sorted(pictures), **fields}
    return engine.detect_identities(fusion.window, request, fusion.directory)


def saved(fusion, entry):
    return fusion.saved[fusion.directory / "artifacts" / entry["artifact"]]


def configure(fusion, monkeypatch, reference):
    """Run submit_swap's configure_faces on a saved reference; returns the target embedding."""
    helpers = ModuleType("app.helpers.miscellaneous")
    helpers.ParametersDict = lambda values, defaults: values
    for name, module in (("app", ModuleType("app")), ("app.helpers", ModuleType("app.helpers"))):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "app.helpers.miscellaneous", helpers)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(mean=lambda values, axis: values[0]))
    window = fusion.window
    window.default_parameters = {"SwapModelSelection": "Inswapper128"}
    window.parameters, window.target_faces = {}, {}
    window.function_worker.get_arcface_model = lambda model: "ArcFace"
    engine.configure_faces(window, [{"target_face": reference, "source_faces": [reference]}])
    return list(window.target_faces["0"].embeddings["ArcFace"])


# Person A with a small neighbour B whose centre lies in A's 30% padding but not in a 10% one.
NEAR = Person([205, 120, 245, 160], B)


def test_padded_crop_with_two_people_is_never_the_reference(fusion, monkeypatch):
    a = Person([100, 100, 200, 200], A)
    result = single(fusion, scene(a, NEAR))
    face = result["faces"][0]
    assert face["reference_verified"] and "warning" not in result
    # The two-face padded crop failed; the tighter crop of A alone became the reference.
    assert saved(fusion, face).bounds == (90, 90, 210, 210)
    assert ((70, 70, 230, 230), 20) in fusion.calls
    assert configure(fusion, monkeypatch, face["reference_path"]) == A
    # submit_swap would refuse the padded crop itself.
    padded = Picture((a, NEAR), (70, 70, 230, 230))
    fusion.saved[Path("padded")] = padded
    with pytest.raises(ValueError, match=r"exactly one detected face.*\(2\)"):
        configure(fusion, monkeypatch, "padded")


def test_neighbour_inside_every_crop_leaves_the_face_unverified(fusion):
    a, b = Person([100, 100, 200, 200], A), Person([180, 130, 220, 170], B)
    result = single(fusion, scene(a, b))
    unverified = result["faces"][0]
    assert not unverified["reference_verified"] and unverified["reference_path"] is None
    assert saved(fusion, unverified).bounds == (70, 70, 230, 230)
    assert result["faces"][1]["reference_verified"]
    assert [tile[0] for tile in fusion.sheets[0]] == ["face 0 *", "face 1"]


def test_lone_neighbour_redetection_is_unverified_and_a_later_candidate_wins(fusion, monkeypatch):
    # In frame 0 the detector finds A only in the full frame; its crops show just B.
    missed = Person([100, 100, 220, 220], A, in_crops=False)
    near = Person([215, 140, 255, 180], B)
    alone = Person([300, 100, 390, 190], A)
    single_result = single(fusion, scene(missed, near))
    face = single_result["faces"][0]
    assert not face["reference_verified"] and face["reference_path"] is None
    # Such a crop would pass submit_swap's count but bind the neighbour's identity.
    assert configure(fusion, monkeypatch, str(fusion.directory / "artifacts/face-0.png")) == B

    fusion.saved.clear()
    result = multi(fusion, {0: scene(missed, near), 5: scene(alone)})
    first = result["identities"][0]
    assert first["appearances"] == 2 and first["frames"] == [0, 5]
    assert first["reference_verified"] and (first["frame"], first["box"]) == (5, alone.box)
    assert saved(fusion, first).bounds == (273, 73, 417, 217)
    assert configure(fusion, monkeypatch, first["reference_path"]) == A
    assert result["identities"][1]["reference_verified"] and "warning" not in result


def test_overlapping_redetection_of_another_identity_is_unverified(fusion):
    # A crop's only face sits on A's box but is someone else, so the embeddings disagree.
    a = Person([100, 100, 200, 200], A, in_crops=False)
    other = Person([105, 105, 205, 205], B)
    result = single(fusion, scene(a, other))
    assert not result["faces"][0]["reference_verified"]
    assert result["faces"][1]["reference_verified"]


def test_clean_crops_are_verified_in_both_modes(fusion, monkeypatch):
    a, b = Person([100, 100, 200, 200], A), Person([400, 100, 480, 180], B)
    result = single(fusion, scene(a, b))
    assert [f["reference_verified"] for f in result["faces"]] == [True, True]
    face = result["faces"][0]
    assert face["reference_path"] == str(fusion.directory / "artifacts/face-0.png")
    assert saved(fusion, face).bounds == (70, 70, 230, 230)
    assert configure(fusion, monkeypatch, face["reference_path"]) == A
    assert result["artifacts"] == ["frame.jpg", "contact.jpg", "face-0.png", "face-1.png"]
    assert [tile[0] for tile in fusion.sheets[0]] == ["face 0", "face 1"]

    result = multi(fusion, {0: scene(a, b), 7: scene(b)})
    identities = result["identities"]
    assert [(i["appearances"], i["reference_verified"]) for i in identities] == [
        (2, True),
        (1, True),
    ]
    assert configure(fusion, monkeypatch, identities[0]["reference_path"]) == B
    assert configure(fusion, monkeypatch, identities[1]["reference_path"]) == A
    # Verification uses submit_swap's detector limit on every crop.
    assert {limit for bounds, limit in fusion.calls if bounds[:2] != (0, 0)} == {20}


def test_all_unverified_results_say_so_and_keep_the_crops(fusion):
    a = Person([100, 100, 200, 200], A, in_crops=False)
    b = Person([180, 130, 220, 170], B, in_crops=False)
    result = single(fusion, scene(a, b))
    assert result["warning"] == engine.UNVERIFIED
    assert [f["reference_path"] for f in result["faces"]] == [None, None]
    assert result["artifacts"] == ["frame.jpg", "contact.jpg", "face-0.png", "face-1.png"]
    assert all(saved(fusion, face) for face in result["faces"])
    assert [(text, verified) for text, _, verified in fusion.sheets[0]] == [
        ("face 0 *", False),
        ("face 1 *", False),
    ]

    result = multi(fusion, {0: scene(a, b), 3: scene(a, b)})
    assert result["warning"] == engine.UNVERIFIED
    assert all(i["reference_path"] is None for i in result["identities"])
    assert result["artifacts"] == ["contact.jpg", "identity-0.png", "identity-1.png"]
    assert [tile[0] for tile in fusion.sheets[1]] == ["id 0: 2x *", "id 1: 2x *"]


def test_no_faces_carries_no_warning(fusion):
    assert single(fusion, scene()) == {"faces": [], "artifacts": ["frame.jpg"]}


def test_unrecognized_faces_are_kept_but_unverified(fusion):
    result = single(fusion, scene(Person([100, 100, 200, 200], None)))
    assert result["faces"][0]["reference_path"] is None


def test_clustering_threshold_merges_profile_views(fusion):
    front, profile = Person([100, 100, 200, 200], A), Person([100, 100, 190, 190], PROFILE)
    pictures = {0: scene(front), 4: scene(profile)}
    default = multi(fusion, pictures)
    assert default["similarity_threshold"] == 60 and len(default["identities"]) == 2
    merged = multi(fusion, pictures, similarity_threshold=50)
    assert merged["similarity_threshold"] == 50
    assert [i["frames"] for i in merged["identities"]] == [[0, 4]]
    assert merged["identities"][0]["reference_verified"]


@pytest.fixture
def media(tmp_path, monkeypatch):
    apply_config(monkeypatch, write_config(tmp_path / "remote-config.json", tmp_path))
    (tmp_path / "clip.mp4").write_bytes(b"x")
    monkeypatch.setattr(common, "layouts", lambda: ({}, SPECS))
    return {"kind": "detect", "media_path": str(tmp_path / "clip.mp4")}


def test_similarity_threshold_is_validated_on_fusions_scale(media):
    for value in (0, 101, 60.5, True, "60"):
        with pytest.raises(ValueError, match="integer from 1 to 100"):
            common.validate_request({**media, "sample_frames": 4, "similarity_threshold": value})
    for extra in ({}, {"frame": 2}):
        with pytest.raises(ValueError, match="only to multi-frame"):
            common.validate_request({**media, **extra, "similarity_threshold": 50})
    with pytest.raises(ValueError, match="only to multi-frame"):
        common.validate_request(
            {**media, "kind": "preview", "mappings": [], "similarity_threshold": 50}
        )
    for value in (1, 50, 100):
        common.validate_request({**media, "frames": [0, 3], "similarity_threshold": value})


def test_list_jobs_reports_detection_sampling(media):
    requests = {
        "one": {**media, "frame": 3},
        "sampled": {**media, "frame": 0, "sample_frames": 16, "similarity_threshold": 50},
        "chosen": {**media, "frame": 0, "frames": [2, 9]},
    }
    for index, (job_id, request) in enumerate(requests.items()):
        job = common.job_path(job_id)
        job.mkdir(parents=True)
        common.atomic_json(job / "request.json", request)
        common.atomic_json(job / "state.json", {"job_id": job_id, "status": "completed"})
        os.utime(job / "state.json", (index, index))
    jobs = {job["job_id"]: job for job in control.list_jobs()}
    assert jobs["one"]["frame"] == 3 and "sample_frames" not in jobs["one"]
    assert (jobs["sampled"]["sample_frames"], jobs["sampled"]["similarity_threshold"]) == (16, 50)
    assert jobs["chosen"]["frames"] == [2, 9]
    assert not {"frame", "frames"} & jobs["sampled"].keys()


def test_detect_tool_forwards_the_threshold_only_when_set(monkeypatch):
    from visomaster_mcp import server

    sent = []

    async def command(action, **values):
        sent.append(values)
        return {}

    monkeypatch.setattr(server, "command", command)
    asyncio.run(server.detect_faces("/m/clip.mp4", sample_frames=8, similarity_threshold=50))
    asyncio.run(server.detect_faces("/m/clip.mp4", job_id="j"))
    assert sent[0]["similarity_threshold"] == 50 and sent[0]["sample_frames"] == 8
    assert sent[1] == {
        "job_id": "j",
        "kind": "detect",
        "media_path": "/m/clip.mp4",
        "frame": 0,
        "controls": {},
    }
    schema = asyncio.run(server.mcp.list_tools())
    detect = next(t for t in schema if t.name == "detect_faces").inputSchema["properties"]
    bounds = detect["similarity_threshold"]["anyOf"][0]
    assert (bounds["minimum"], bounds["maximum"]) == (1, 100)


def test_best_match_is_an_allowed_control_and_on_by_default(tmp_path, monkeypatch):
    apply_config(monkeypatch, write_config(tmp_path / "remote-config.json", tmp_path))
    widgets = Path(common.APP) / "app/ui/widgets"
    settings = {
        "Swap settings": {
            "SwapOnlyBestMatchEnableToggle": {"default": False, "help": "best match only"},
            "ForceSwapToggle": {"default": True},
        },
        "Detectors": {"DetectorModelSelection": {"default": "RetinaFace", "options": ["Yolov8"]}},
    }
    (widgets / "settings_layout_data.py").write_text(f"LAYOUT = {settings!r}\n")
    swapper = {"Swap": {"SwapModelSelection": {"default": "Inswapper128", "options": ["X"]}}}
    (widgets / "swapper_layout_data.py").write_text(f"LAYOUT = {swapper!r}\n")
    reported = control.dispatch("settings", {})["controls"]
    assert reported["SwapOnlyBestMatchEnableToggle"]["default"] is True
    assert reported["DetectorModelSelection"]["default"] == "Yolov8"
    assert "ForceSwapToggle" not in reported
    specs = {k: v for group in settings.values() for k, v in group.items()}
    control_specs = {**specs, "LandmarkDetectToggle": {"default": True}}
    assert engine.initial_controls(control_specs, {})["SwapOnlyBestMatchEnableToggle"] is True
    chosen = engine.initial_controls(control_specs, {"SwapOnlyBestMatchEnableToggle": False})
    assert chosen["SwapOnlyBestMatchEnableToggle"] is False
    assert not chosen["ForceSwapToggle"] and not chosen["LandmarkDetectToggle"]
    common.validate_settings({"SwapOnlyBestMatchEnableToggle": False}, specs, common.CONTROLS)
    with pytest.raises(ValueError, match="Invalid value"):
        common.validate_settings({"SwapOnlyBestMatchEnableToggle": 1}, specs, common.CONTROLS)

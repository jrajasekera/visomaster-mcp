"""Reference analysis, explicit source weights and reference fingerprints (vmcp-harden-pqj.3).

Fusion is stubbed: an image is a set of faces, each a box and the embedding its recognition
returns, plus the sharpness its face box measures.
"""

import fcntl
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import common
import control
import engine
import pytest
import test_server
import test_submit
from conftest import apply_config, write_config
from test_submit import complete, preview, wait_for

make_site, library = test_submit.make_site, test_submit.library
server, call = test_server.server, test_server.call

A = [1.0, 0.0, 0.0, 0.0]
NEAR_A = [0.95, 0.31, 0.0, 0.0]  # Scores 97.5 against A: nearly the same image.
OTHER_VIEW = [0.8, 0.0, 0.6, 0.0]  # Scores 90 against A: the same person, another photo.
SOFT = [0.6, 0.6, 0.5, 0.0]
STRANGER = [0.0, 0.0, 0.0, 1.0]  # Scores 50 against every other reference.
SPECS = {
    "SimilarityThresholdSlider": {"default": 60, "min_value": "1", "max_value": "100"},
    "SwapModelSelection": {"default": "Inswapper128", "options": ["Inswapper128", "SimSwap512"]},
}
MODELS = {"Inswapper128": "Inswapper128ArcFace", "SimSwap512": "SimSwapArcFace"}


class Vector(list):
    def ravel(self):
        return self


class Image:
    """Faces are (box, embedding or None); slicing (cropping) keeps the picture."""

    def __init__(self, *faces, size=(400, 300), sharpness=200.0):
        self.faces, self.sharpness = faces, sharpness
        self.shape = (size[1], size[0], 3)

    def __getitem__(self, index):
        return self


def face(embedding, box=(100, 80, 260, 260)):
    return (list(box), embedding)


@pytest.fixture
def fusion(tmp_path, monkeypatch):
    """Stub Fusion for the engine: references, detection, recognition and image output."""
    references, reads, saved, sheets = {}, [], {}, []

    def reference_image(path, digest=None):
        reads.append(path)
        sha256, image = references[path]
        if digest is not None and digest != sha256:
            raise ValueError(f"Reference {path} changed after submission")
        return sha256, image

    def recognize(tensor, embedding, mode, model):
        return (None if embedding is None else Vector(embedding)), None

    def detect(window, image):
        return image, [box for box, _ in image.faces], [embedding for _, embedding in image.faces]

    window = SimpleNamespace(
        control={"RecognitionModelSelection": "Inswapper128ArcFace"},
        function_worker=SimpleNamespace(
            run_recognize_direct=recognize, get_arcface_model=MODELS.__getitem__
        ),
    )
    monkeypatch.setattr(engine, "reference_image", reference_image)
    monkeypatch.setattr(engine, "detect_reference", detect)
    monkeypatch.setattr(engine, "face_sharpness", lambda image, box: image.sharpness)
    monkeypatch.setattr(engine, "layouts", lambda: ({}, SPECS))
    monkeypatch.setattr(engine, "save_image", lambda path, image: saved.__setitem__(path, image))
    monkeypatch.setattr(engine, "contact_sheet", lambda path, tiles: sheets.append(tiles))
    numpy = SimpleNamespace(
        full=lambda shape, value, dtype: "blank",
        uint8="uint8",
        asarray=lambda values, dtype: values,
        float32="float32",
    )
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    (tmp_path / "artifacts").mkdir()

    def add(name, image, sha256=None):
        path = f"/media/{name}"
        references[path] = (sha256 or hashlib.sha256(name.encode()).hexdigest(), image)
        return path

    return SimpleNamespace(
        window=window,
        add=add,
        reads=reads,
        saved=saved,
        sheets=sheets,
        directory=tmp_path,
        references=references,
    )


def analyze(fusion, paths, **fields):
    request = {"kind": "analyze", "references": paths, **fields}
    return engine.analyze_references(fusion.window, request, fusion.directory)


def codes(entry):
    return [finding["code"] for finding in entry["findings"]]


# Analysis findings


def test_collection_reports_every_reference_without_removing_any(fusion):
    clear = fusion.add("clear.png", Image(face(A)), sha256="a" * 64)
    paths = [
        clear,
        fusion.add("copy.png", Image(face(A)), sha256="a" * 64),
        fusion.add("resaved.jpg", Image(face(NEAR_A))),
        fusion.add("other-view.png", Image(face(OTHER_VIEW))),
        fusion.add("stranger.png", Image(face(STRANGER))),
        fusion.add("group.png", Image(face(A), face(STRANGER, (300, 80, 380, 180)))),
        fusion.add("landscape.png", Image()),
        fusion.add("notes.png", None),
        fusion.add("tiny.png", Image(face(SOFT, (10, 10, 50, 60)), sharpness=12.5)),
        fusion.add("unrecognized.png", Image(face(None))),
    ]
    result = analyze(fusion, paths)
    entries = result["references"]
    # Every reference is reported, in order, under a stable id; nothing is dropped or edited.
    assert [e["id"] for e in entries] == [f"r{n}" for n in range(10)]
    assert [e["path"] for e in entries] == paths
    assert fusion.reads == paths and fusion.references[clear][0] == "a" * 64
    assert [e["usable"] for e in entries] == [True] * 5 + [False] * 3 + [True, False]
    assert [codes(e) for e in entries] == [
        [],
        ["exact_duplicate"],
        ["near_duplicate"],
        [],
        ["outlier"],
        ["multiple_faces"],
        ["no_face"],
        ["unreadable"],
        ["small_face", "blurry"],
        ["unrecognized"],
    ]
    assert [e["faces"] for e in entries] == [1, 1, 1, 1, 1, 2, 0, None, 1, 1]
    assert entries[5]["boxes"] == [[100, 80, 260, 260], [300, 80, 380, 180]]
    assert all(f["severity"] == "error" for e in entries[5:8] for f in e["findings"])
    assert entries[9]["findings"][0]["severity"] == "error"
    assert (result["usable"], result["errors"], result["warnings"]) == (6, 4, 5)
    # Face size and sharpness of a usable face, with the box they were measured on.
    tiny = entries[8]
    assert (tiny["box"], tiny["face_pixels"], tiny["sharpness"]) == ([10, 10, 50, 60], 40, 12.5)
    assert tiny["image_size"] == [400, 300]
    assert "40 px" in tiny["findings"][0]["message"] and "12.5" in tiny["findings"][1]["message"]
    assert entries[7]["sha256"] and entries[7]["image_size"] is None
    # Duplicates name the earlier reference they repeat.
    assert result["duplicates"] == [
        {"reference": "r1", "duplicate_of": "r0", "kind": "identical_bytes", "similarity": None},
        {"reference": "r2", "duplicate_of": "r0", "kind": "near_identical", "similarity": 97.5},
    ]
    # Outlier scores state their model and basis, and exist only for usable references.
    consistency = result["consistency"]
    assert consistency["available"] and consistency["model"] == "Inswapper128ArcFace"
    assert "Leave-one-out" in consistency["basis"]
    assert entries[4]["consistency_score"] == 50.0
    assert all(entries[n]["consistency_score"] > 80 for n in (0, 1, 2, 3, 8))
    assert all(entries[n]["consistency_score"] is None for n in (5, 6, 7, 9))
    assert "Review its crop; nothing was removed" in entries[4]["findings"][0]["message"]
    assert (result["swap_model"], result["model"]) == ("Inswapper128", "Inswapper128ArcFace")
    assert result["thresholds"] == {
        "min_face_pixels": 112,
        "min_sharpness": 40.0,
        "duplicate_similarity": 95.0,
        "outlier_similarity": 60,
    }
    assert {"face_pixels", "sharpness", "similarity"} <= result["metrics"].keys()
    assert "squared luma levels" in result["metrics"]["sharpness"]


def test_crops_contact_sheet_and_report_artifact(fusion):
    paths = [
        fusion.add("clear.png", Image(face(A))),
        fusion.add("group.png", Image(face(A), face(STRANGER, (300, 80, 380, 180)))),
        fusion.add("notes.png", None),
        fusion.add("tiny.png", Image(face(OTHER_VIEW, (10, 10, 50, 60)), sharpness=3.0)),
    ]
    result = analyze(fusion, paths)
    assert [e["artifact"] for e in result["references"]] == [
        "reference-0.png",
        None,
        None,
        "reference-3.png",
    ]
    assert result["artifacts"] == [
        "contact.jpg",
        "analysis.json",
        "reference-0.png",
        "reference-3.png",
    ]
    artifacts = fusion.directory / "artifacts"
    assert set(fusion.saved) == {artifacts / "reference-0.png", artifacts / "reference-3.png"}
    labels = [(label, verified) for label, _, verified in fusion.sheets[0]]
    assert labels == [("r0 ok", True), ("r1 2 faces", False), ("r2 unreadable", False)] + [
        ("r3 small+", True)
    ]
    assert fusion.sheets[0][2][1] == "blank"
    report = json.loads((artifacts / "analysis.json").read_text())
    assert report["revision"] == engine.REVISION
    assert report["references"] == result["references"]
    # Embeddings stay internal: neither the result nor the report exposes them.
    assert all("embedding" not in entry for entry in report["references"])


def test_single_reference_is_valid_without_consistency_conclusions(fusion):
    result = analyze(fusion, [fusion.add("only.png", Image(face(A)))])
    (entry,) = result["references"]
    assert entry["usable"] and codes(entry) == [] and entry["consistency_score"] is None
    consistency = result["consistency"]
    assert not consistency["available"] and "at least 2" in consistency["reason"]
    assert "pair_similarity" not in consistency
    # Byte-identical copies add no evidence either.
    copies = [fusion.add(f"copy-{n}.png", Image(face(A)), sha256="c" * 64) for n in range(3)]
    result = analyze(fusion, copies)
    assert not result["consistency"]["available"]
    assert [codes(e) for e in result["references"]] == [[], ["exact_duplicate"]] + [
        ["exact_duplicate"]
    ]


@pytest.mark.parametrize("second, flagged", [(OTHER_VIEW, False), (STRANGER, True)])
def test_two_references_report_their_agreement_but_no_outlier(fusion, second, flagged):
    paths = [fusion.add("a.png", Image(face(A))), fusion.add("b.png", Image(face(second)))]
    result = analyze(fusion, paths)
    consistency = result["consistency"]
    assert not consistency["available"] and "cannot be attributed" in consistency["reason"]
    assert consistency["pair_similarity"] == (50.0 if flagged else 90.0)
    expected = ["inconsistent_pair"] if flagged else []
    assert [codes(e) for e in result["references"]] == [expected, expected]
    assert all(e["consistency_score"] is None for e in result["references"])


def test_thresholds_are_configurable_and_reported(fusion):
    paths = [
        fusion.add("a.png", Image(face(A), sharpness=30.0)),
        fusion.add("b.png", Image(face(NEAR_A, (0, 0, 90, 100)))),
        fusion.add("c.png", Image(face(OTHER_VIEW))),
    ]
    assert [codes(e) for e in analyze(fusion, paths)["references"]] == [
        ["blurry"],
        ["small_face", "near_duplicate"],
        [],
    ]
    thresholds = {
        "min_face_pixels": 64,
        "min_sharpness": 25,
        "duplicate_similarity": 99,
        "outlier_similarity": 92,
    }
    result = analyze(fusion, paths, thresholds=thresholds, swap_model="SimSwap512")
    assert result["thresholds"] == thresholds
    assert result["model"] == result["consistency"]["model"] == "SimSwapArcFace"
    # Only the reference least like the others falls below the stricter outlier threshold.
    scores = [e["consistency_score"] for e in result["references"]]
    assert scores[2] < 92 <= min(scores[:2])
    assert [codes(e) for e in result["references"]] == [[], [], ["outlier"]]


def test_laplacian_variance_measures_sharpness():
    flat = [[128] * 8 for _ in range(8)]
    assert engine.laplacian_variance(flat) == 0.0
    # A checkerboard of 0 and 255 gives Laplacians of -1020 and +1020 (variance 1020 squared).
    board = [[255 * ((x + y) % 2) for x in range(6)] for y in range(6)]
    assert engine.laplacian_variance(board) == 1020.0**2
    # Smoothing the same edge lowers it.
    edge = [[0] * 4 + [255] * 4 for _ in range(6)]
    ramp = [[0, 0, 0, 85, 170, 255, 255, 255] for _ in range(6)]
    assert engine.laplacian_variance(ramp) < engine.laplacian_variance(edge)
    assert engine.laplacian_variance([[1, 2]]) == 0.0


# Weighted merge


def close(a, b, tolerance=1e-9):
    return all(math.isclose(x, y, abs_tol=tolerance) for x, y in zip(a, b, strict=True))


def test_weighted_merge_matches_fusions_mean_then_normalization():
    a, b = [3.0, 4.0, 0.0], [0.0, 0.0, 10.0]
    # Equal weights are Fusion's Mean merge followed by L2 normalization.
    mean = [(x + y) / 2 for x, y in zip(a, b)]
    assert close(engine.merge_embeddings([a, b], [1, 1]), engine.unit(mean))
    assert math.isclose(sum(v * v for v in engine.merge_embeddings([a, b], [1, 1])), 1.0)
    # Only ratios matter, and weight 2 counts an image as two copies would.
    assert close(
        engine.merge_embeddings([a, b], [2, 1]), engine.merge_embeddings([a, a, b], [1] * 3)
    )
    assert close(
        engine.merge_embeddings([a, b], [0.5, 0.25]), engine.merge_embeddings([a, b], [2, 1])
    )
    # One selected reference reproduces the single-reference embedding.
    single = engine.merge_embeddings([a], [1])
    assert close(single, [0.6, 0.8, 0.0])
    assert close(engine.merge_embeddings([a, b], [1, 0]), single)
    assert close(engine.merge_embeddings([a, b], [1e-300, 0]), single)


@pytest.fixture
def configure(fusion, monkeypatch):
    """Run configure_faces against the stub; returns (used, merged swap embedding)."""
    helpers = ModuleType("app.helpers.miscellaneous")
    helpers.ParametersDict = lambda values, defaults: values
    for name, module in (("app", ModuleType("app")), ("app.helpers", ModuleType("app.helpers"))):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "app.helpers.miscellaneous", helpers)
    window = fusion.window
    window.default_parameters = {"SwapModelSelection": "Inswapper128"}

    def run(**mapping):
        window.parameters, window.target_faces = {}, {}
        used = engine.configure_faces(window, [mapping])
        return used, window.target_faces["0"].assigned_input_embedding["Inswapper128ArcFace"]

    return run


def test_configure_weights_sources_and_skips_excluded_ones(fusion, configure):
    target = fusion.add("target.png", Image(face(A)))
    a = fusion.add("a.png", Image(face([3.0, 4.0, 0.0, 0.0])))
    b = fusion.add("b.png", Image(face([0.0, 0.0, 10.0, 0.0])))
    group = fusion.add("group.png", Image(face(A), face(A)))
    used, equal = configure(target_face=target, source_faces=[a, b])
    assert close(equal, engine.unit([1.5, 2.0, 5.0, 0.0]))
    assert used[0]["weighted"] is False
    assert [(s["weight"], s["share"]) for s in used[0]["sources"]] == [(1, 0.5), (1, 0.5)]
    used, weighted = configure(target_face=target, source_faces=[a, b], source_weights=[3, 1])
    assert close(weighted, engine.unit([3 * 0.75 / 1, 4 * 0.75, 2.5, 0.0]))
    assert used[0]["weighted"] and [s["share"] for s in used[0]["sources"]] == [0.75, 0.25]
    # An excluded reference is never read, so even a multi-face image may stay listed.
    fusion.reads.clear()
    used, alone = configure(target_face=target, source_faces=[a, group], source_weights=[1, 0])
    assert group not in fusion.reads and close(alone, [0.6, 0.8, 0.0, 0.0])
    _, single = configure(target_face=target, source_faces=[a])
    assert close(alone, single)
    sources = used[0]["sources"]
    assert sources[1] == {"path": group, "sha256": None, "weight": 0, "share": 0.0}
    assert sources[0]["sha256"] == fusion.references[a][0]
    assert used[0]["target_sha256"] == fusion.references[target][0]
    # Selected references must still have exactly one face.
    with pytest.raises(ValueError, match=r"exactly one detected face.*\(2\)"):
        configure(target_face=target, source_faces=[a, group], source_weights=[1, 1])


def test_configure_refuses_references_changed_after_submission(fusion, configure):
    target = fusion.add("target.png", Image(face(A)))
    source = fusion.add("source.png", Image(face(A)))
    digest = fusion.references[source][0]
    used, _ = configure(target_face=target, source_faces=[source], source_sha256=[digest])
    assert used[0]["sources"][0]["sha256"] == digest
    with pytest.raises(ValueError, match="changed after submission"):
        configure(target_face=target, source_faces=[source], source_sha256=["f" * 64])


def test_reference_image_decodes_exactly_the_bytes_it_hashed(tmp_path, monkeypatch):
    decoded = []
    cv2 = SimpleNamespace(
        IMREAD_COLOR=1, imdecode=lambda buffer, flags: decoded.append(buffer) or "image"
    )
    numpy = SimpleNamespace(frombuffer=lambda data, dtype: bytes(data), uint8="uint8")
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setattr(engine, "input_path", Path)
    path = tmp_path / "face.png"
    path.write_bytes(b"first")
    digest = hashlib.sha256(b"first").hexdigest()
    assert engine.reference_image(str(path), digest) == (digest, "image")
    assert engine.reference_image(str(path)) == (digest, "image")
    assert decoded == [b"first", b"first"]
    path.write_bytes(b"second")
    with pytest.raises(ValueError, match="changed after submission.*expected " + digest):
        engine.reference_image(str(path), digest)
    assert len(decoded) == 2


# Validation and fingerprints at submission


SWAPPER_LAYOUT = (
    "LAYOUT = {'Swap': {'SwapModelSelection': {'default': 'Inswapper128', "
    "'options': ['Inswapper128', 'SimSwap512', 'AlphaFace']}}}\n"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_without_id(site, **mapping):
    request = preview(site)
    del request["job_id"]
    request["mappings"][0].update(mapping)
    return request


@pytest.mark.parametrize(
    "count, weights, message",
    [
        (1, [0], "positive, finite sum"),
        (2, [0.0, 0], "positive, finite sum"),
        (2, [-1, 2], "finite nonnegative"),
        (2, [float("nan"), 1], "finite nonnegative"),
        (2, [float("inf"), 1], "finite nonnegative"),
        (2, [1e308, 1e308], "positive, finite sum"),
        (2, [True, 1], "finite nonnegative"),
        (2, ["1", 1], "finite nonnegative"),
        (2, [1], "one weight per source"),
        (2, [1, 1, 1], "one weight per source"),
        (1, {"a": 1}, "one weight per source"),
        (1, 1, "one weight per source"),
    ],
)
def test_invalid_weights_are_refused(make_site, count, weights, message):
    site = make_site()
    sources = [site.source, site.face][:count]
    request = request_without_id(site, source_faces=sources, source_weights=weights)
    with pytest.raises(ValueError, match=message):
        common.validate_request(request)


@pytest.mark.parametrize("weights", [[1, 0], [0.25, 3], [2, 2], None])
def test_valid_weights_are_accepted(make_site, weights):
    site = make_site()
    request = request_without_id(site, source_faces=[site.source, site.face])
    request["mappings"][0]["source_weights"] = weights
    assert common.validate_request(request) is request


def test_supplied_fingerprints_must_match_current_bytes(make_site):
    site = make_site()
    digest = sha(site.source)
    common.validate_request(request_without_id(site, source_sha256=[digest]))
    for value, message in (
        ([digest, digest], "one 64-hex-digit SHA-256 per reference"),
        ([digest.upper()], "one 64-hex-digit SHA-256 per reference"),
        (digest, "one 64-hex-digit SHA-256 per reference"),
        (["0" * 64], "no longer has SHA-256 0{64}: its content changed"),
    ):
        with pytest.raises(ValueError, match=message):
            common.validate_request(request_without_id(site, source_sha256=value))
    with pytest.raises(ValueError, match="target_sha256 must be"):
        common.validate_request(request_without_id(site, target_sha256=["0" * 64]))
    with pytest.raises(ValueError, match="no longer has SHA-256"):
        common.validate_request(request_without_id(site, target_sha256="0" * 64))


def test_analysis_request_validation(make_site):
    site = make_site()
    (site.fusion / test_submit.WIDGETS / "swapper_layout_data.py").write_text(SWAPPER_LAYOUT)
    base = {"kind": "analyze", "references": [site.face, site.source, site.face]}
    assert common.validate_request(dict(base)) == base
    thresholds = {"min_face_pixels": 64, "min_sharpness": 0, "outlier_similarity": 70.5}
    common.validate_request({**base, "thresholds": thresholds, "swap_model": "SimSwap512"})
    invalid = [
        ({"references": []}, "references must list 1–20"),
        ({"references": [site.face] * 21}, "references must list 1–20"),
        ({"references": site.face}, "references must list 1–20"),
        ({"media_path": site.clip}, "does not accept media_path"),
        ({"mappings": []}, "does not accept mappings"),
        ({"thresholds": {"min_face_pixels": 1.5}}, "min_face_pixels must be an integer"),
        ({"thresholds": {"min_face_pixels": 0}}, "min_face_pixels must be an integer"),
        ({"thresholds": {"min_sharpness": -1}}, "min_sharpness must be a number"),
        ({"thresholds": {"duplicate_similarity": 40}}, "duplicate_similarity must be"),
        ({"thresholds": {"outlier_similarity": float("nan")}}, "outlier_similarity must be"),
        ({"thresholds": {"outlier_similarity": True}}, "outlier_similarity must be"),
        ({"thresholds": {"blur": 1}}, "Unknown threshold blur"),
        ({"thresholds": []}, "thresholds must be an object"),
        ({"swap_model": "AlphaFace"}, "embedding-based swappers only"),
        ({"swap_model": "Nope"}, "Invalid value for SwapModelSelection"),
        ({"controls": {"Bogus": 1}}, "Unsupported setting Bogus"),
        ({"reference_sha256": ["0" * 64] * 3}, "no longer has SHA-256"),
    ]
    for changes, message in invalid:
        with pytest.raises(ValueError, match=message):
            common.validate_request({**base, **changes})
    with pytest.raises(ValueError, match="Media must be"):
        common.validate_request({**base, "references": [str(site.fusion / ".git")]})


def test_oversized_reference_is_refused(make_site, monkeypatch):
    site = make_site()
    monkeypatch.setattr(common, "MAX_REFERENCE_BYTES", 10)
    with pytest.raises(ValueError, match="images up to 256 MiB"):
        common.validate_request({"kind": "analyze", "references": [site.face]})


def test_submission_fingerprints_references_and_keeps_the_request(make_site):
    site = make_site()
    alias = site.media / "alias.png"
    alias.symlink_to(site.face)
    request = preview(site, job_id="weighted")
    request["mappings"][0].update(source_faces=[site.source, str(alias)], source_weights=[3, 0.5])
    control.submit(json.loads(json.dumps(request)))
    job = common.job_path("weighted")
    stored = json.loads((job / "request.json").read_text())
    assert stored == {k: v for k, v in request.items() if k != "job_id"}
    (mapping,) = json.loads((job / "resolved.json").read_text())["mappings"]
    assert mapping["source_faces"] == [site.source, os.path.realpath(site.face)]
    assert mapping["source_weights"] == [3, 0.5]
    assert mapping["source_sha256"] == [sha(site.source), sha(site.face)]
    assert mapping["target_sha256"] == sha(site.face)
    assert complete("weighted")["result"] == {"validated": True}
    # Identical retries stay idempotent; the recorded weights are part of the request.
    assert control.submit(json.loads(json.dumps(request)))["status"] == "completed"
    request["mappings"][0]["source_weights"] = [3, 1]
    with pytest.raises(ValueError, match="different request"):
        control.submit(request)


def test_supplied_fingerprint_mismatch_publishes_nothing(make_site):
    site = make_site()
    request = preview(site, job_id="stale")
    request["mappings"][0]["source_sha256"] = ["0" * 64]
    with pytest.raises(ValueError, match="content changed"):
        control.submit(request)
    assert not (site.state / "jobs/stale").exists()


@pytest.mark.parametrize("kind", ["preview", "analyze"])
def test_reference_changed_while_queued_fails_the_job(make_site, kind):
    site = make_site()
    if kind == "analyze":
        request = {"job_id": "queued", "kind": kind, "references": [site.face, site.source]}
    else:
        request = preview(site, job_id="queued")
    lock = site.state / "gpu-0.lock"
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # Another GPU job is running.
        control.submit(request)
        resolved = json.loads((common.job_path("queued") / "resolved.json").read_text())
        if kind == "analyze":
            assert resolved["reference_sha256"] == [sha(site.face), sha(site.source)]
            assert control.request_paths(resolved) >= {Path(site.face), Path(site.source)}
        assert control.status("queued")["status"] == "queued"
        Path(site.source).write_bytes(Path(site.source).read_bytes() + b"edited")
    failed = wait_for("queued", {"failed"})
    assert "no longer has SHA-256" in failed["error"] and site.source in failed["error"]


def test_engine_runs_analysis_and_reports_used_references(tmp_path, monkeypatch):
    config = write_config(tmp_path / "remote-config.json", tmp_path)
    settings = apply_config(monkeypatch, config)
    monkeypatch.setattr(engine, "CONFIG_ERROR", None)
    monkeypatch.setattr(engine, "SETTINGS", settings)
    monkeypatch.setattr(engine.sys, "platform", "test")  # No parent-death signal for pytest.
    monkeypatch.setenv("VM_SUPERVISOR_PID", str(os.getppid()))
    monkeypatch.setenv("VM_CONFIG_SHA256", settings.sha256)
    face, clip = tmp_path / "face.png", tmp_path / "clip.mp4"
    face.write_bytes(b"face")
    clip.write_bytes(b"clip")
    used = [{"mapping": 0, "sources": []}]
    calls = []
    monkeypatch.setattr(engine, "initialize", lambda controls: calls.append(controls) or "window")
    monkeypatch.setattr(
        engine, "analyze_references", lambda window, request, directory: {"analyzed": request}
    )
    monkeypatch.setattr(engine, "configure_faces", lambda window, mappings: used)
    monkeypatch.setattr(
        engine,
        "make_worker",
        lambda window, sequential=False: SimpleNamespace(
            swap_count=1, main_window=window, telemetry=engine.Telemetry({"0": 60.0})
        ),
    )
    monkeypatch.setattr(engine, "load_frame", lambda path, number=0: "frame")
    monkeypatch.setattr(engine, "process", lambda worker, frame, index: frame)
    monkeypatch.setattr(engine, "save_image", lambda path, image: None)
    job = common.job_path("analysis")
    (job / "artifacts").mkdir(parents=True)
    request = {"kind": "analyze", "references": [str(face)]}
    common.atomic_json(job / "request.json", request)
    common.atomic_json(job / "resolved.json", control.fingerprint(control.resolve_inputs(request)))
    engine.run(job)
    result = common.read_json(job / "result.json")
    assert result["analyzed"]["reference_sha256"] == [sha(face)]
    assert result["revision"] == engine.REVISION and calls == [{}]
    # Previews report what each mapping used.
    mapping = {"target_face": str(face), "source_faces": [str(face)], "source_weights": [2]}
    request = {"kind": "preview", "media_path": str(clip), "mappings": [mapping]}
    common.atomic_json(job / "resolved.json", control.fingerprint(control.resolve_inputs(request)))
    window = SimpleNamespace(
        control={"FaceTrackingEnableToggle": False, "KPSSmoothingEnableToggle": False}
    )
    monkeypatch.setattr(engine, "initialize", lambda controls: window)
    engine.run(job)
    assert common.read_json(job / "result.json")["references"] == used


# Public tools over stdio


def test_tools_submit_analysis_and_weighted_swaps_over_stdio(server, tmp_path):
    for name in ("clip.mp4", "face.png", "source.png", "other.png"):
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "fusion" / test_submit.WIDGETS / "swapper_layout_data.py").write_text(
        SWAPPER_LAYOUT
    )
    face, source, other = (str(tmp_path / n) for n in ("face.png", "source.png", "other.png"))
    clip = str(tmp_path / "clip.mp4")
    digest = sha(source)
    plain = {"target_face": face, "source_faces": [source, other]}
    weighted = {**plain, "source_weights": [1, 0], "source_sha256": [digest, sha(other)]}
    results = call(
        server,
        ("analyze_references", {"references": [source, other, source], "job_id": "refs"}),
        (
            "analyze_references",
            {
                "references": [source],
                "thresholds": {"min_sharpness": 25, "outlier_similarity": 70},
                "swap_model": "Inswapper128",
                "job_id": "tuned",
            },
        ),
        ("analyze_references", {"references": [source], "thresholds": {"blur": 1}}),
        ("analyze_references", {"references": []}),
        ("submit_swap", {"media_path": clip, "mappings": [plain], "job_id": "plain"}),
        ("submit_swap", {"media_path": clip, "mappings": [weighted], "job_id": "weighted"}),
        (
            "submit_swap",
            {"media_path": clip, "mappings": [{**plain, "source_weights": [-1, 2]}]},
        ),
        (
            "submit_swap",
            {"media_path": clip, "mappings": [{**plain, "source_weights": [0, 0]}]},
        ),
        ("submit_swap", {"media_path": clip, "mappings": [{**plain, "source_weights": [1]}]}),
        (
            "submit_swap",
            {"media_path": clip, "mappings": [{**plain, "source_sha256": ["0" * 64] * 2}]},
        ),
        ("submit_swap", {"media_path": clip, "mappings": [{**plain, "source_sha256": ["x"]}]}),
        ("list_jobs", {"kind": "analyze"}),
    )
    refs, tuned, unknown, empty, plain_job, weighted_job, negative, zero, short = results[:9]
    changed, malformed, listed = results[9:]
    assert not refs.isError and refs.structuredContent["job_id"] == "refs"
    jobs = tmp_path / "state/jobs"
    assert json.loads((jobs / "refs/request.json").read_text()) == {
        "kind": "analyze",
        "references": [source, other, source],
    }
    resolved = json.loads((jobs / "refs/resolved.json").read_text())
    assert resolved["reference_sha256"] == [digest, sha(other), digest]
    assert json.loads((jobs / "tuned/request.json").read_text()) == {
        "kind": "analyze",
        "references": [source],
        "thresholds": {"min_sharpness": 25.0, "outlier_similarity": 70.0},
        "swap_model": "Inswapper128",
    }
    assert unknown.isError and empty.isError
    # Omitted weights and fingerprints keep the earlier request form.
    stored = json.loads((jobs / "plain/request.json").read_text())["mappings"]
    assert stored == [{**plain, "parameters": {}}]
    assert not weighted_job.isError
    stored = json.loads((jobs / "weighted/request.json").read_text())["mappings"]
    assert stored[0]["source_weights"] == [1.0, 0.0]
    assert stored[0]["source_sha256"] == [digest, sha(other)]
    assert negative.isError and "greater than or equal to 0" in negative.content[0].text
    assert zero.isError and "positive, finite sum" in zero.content[0].text
    assert short.isError and "one weight per source" in short.content[0].text
    assert changed.isError and "content changed" in changed.content[0].text
    assert malformed.isError and "pattern" in malformed.content[0].text
    assert {job["job_id"] for job in listed.structuredContent["result"]} == {"refs", "tuned"}
    entry = listed.structuredContent["result"][0]
    assert entry["kind"] == "analyze" and "references" in entry
    test_submit.settle(tmp_path / "state")

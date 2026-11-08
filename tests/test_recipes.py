"""Recipes: provenance, immutable revisions, instantiation, import and retention (pqj.5)."""

import copy
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import common
import control
import engine
import pytest
import test_server
import test_submit
from conftest import apply_config, write_config
from test_submit import complete, preview, settle

make_site, library = test_submit.make_site, test_submit.library
server, call = test_server.server, test_server.call
OLD = time.time() - 10 * 86400
SETTINGS_LAYOUT = """LAYOUT = {'Detection': {
    'DetectorModelSelection': {'default': 'Yolov8', 'options': ['Yolov8', 'RetinaFace']},
    'RecognitionModelSelection': {'default': 'Inswapper128ArcFace',
        'options': ['Inswapper128ArcFace', 'GhostArcFace']},
    'MaxFacesToDetectSlider': {'default': '20', 'min_value': '1', 'max_value': '50'},
    'SwapOnlyBestMatchEnableToggle': {'default': False},
    'HiddenSlider': {'default': '%s', 'min_value': '1', 'max_value': '10'},
}}
"""
SWAPPER_LAYOUT = """LAYOUT = {'Swap': {
    'SwapModelSelection': {'default': 'Inswapper128', 'options': ['Inswapper128', 'GhostFace-v1']},
    'SimilarityThresholdSlider': {'default': '60', 'min_value': '1', 'max_value': '100'},
    'HiddenDecimalSlider': {'default': '0.5', 'min_value': '0', 'max_value': '1'},
}}
"""
MODEL_FILES = {
    "Inswapper128": "inswapper_128.fp16.onnx",
    "GhostFacev1": "ghost_unet_1_block.onnx",
    "Inswapper128ArcFace": "w600k_r50.onnx",
    "GhostArcFace": "ghost_arcface_backbone.onnx",
    "YoloFace8n": "yoloface_8n.onnx",
    "GFPGANv1.4": "GFPGANv1.4.onnx",
}
MODELS_DATA = """from pathlib import Path
import numpy as np
models_dir = Path(__file__).resolve().parent.parent.parent / "model_assets"
arcface_mapping_model_dict = {"Inswapper128": "Inswapper128ArcFace", "GhostFace-v1": "GhostArcFace"}
detection_model_mapping = {"Yolov8": "YoloFace8n", "RetinaFace": "RetinaFace"}
models_list = [
%s]
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fusion_files(site, hidden=5):
    widgets = site.fusion / test_submit.WIDGETS
    (widgets / "settings_layout_data.py").write_text(SETTINGS_LAYOUT % hidden)
    (widgets / "swapper_layout_data.py").write_text(SWAPPER_LAYOUT)
    assets = site.fusion / "model_assets"
    assets.mkdir(exist_ok=True)
    entries = ""
    for name, file in MODEL_FILES.items():
        (assets / file).write_bytes(name.encode() * 100)
        entries += (
            f'    {{"model_name": "{name}", "local_path": f"{{models_dir}}/{file}", '
            f'"hash": "{sha(assets / file)}"}},\n'
        )
    path = site.fusion / "app/processors/models_data.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MODELS_DATA % entries)


@pytest.fixture
def site(make_site):
    site = make_site()
    fusion_files(site)
    site.uploads = site.state / "uploads"
    with open(site.source, "ab") as stream:
        stream.write(b"another person")  # Distinct bytes from the target image.
    return site


def intent(site, name="shot", **changes):
    mapping = {
        "target_face": site.face,
        "source_faces": [site.source, site.face],
        "source_weights": [2, 1],
        "parameters": {"SimilarityThresholdSlider": 70},
    }
    request = {
        "name": name,
        "kind": "preview",
        "media_path": site.clip,
        "frame": 3,
        "mappings": [mapping],
        "controls": {"DetectorModelSelection": "Yolov8"},
    }
    return {**request, **changes}


def saved(site, **changes):
    return control.save_recipe(intent(site, **changes))["recipe"]


def run(name="shot", job_id="from-recipe", **values):
    return control.run_recipe({"name": name, "job_id": job_id, **values})


def job_request(job_id):
    return common.read_json(common.job_path(job_id) / "request.json")


def resign(document):
    document = copy.deepcopy(document)
    document["sha256"] = control.recipe_digest(document)
    return document


# Format: effective settings, fingerprints and provenance


def test_recipe_records_effective_settings_fingerprints_and_provenance(site):
    result = control.save_recipe(intent(site, note="hero shot"))
    document = result["recipe"]
    assert document["format"] == "visomaster-recipe" and document["version"] == 1
    assert (document["name"], document["revision"], document["note"]) == ("shot", 1, "hero shot")
    assert document["origin"] == {"kind": "request"}
    assert document["frame_convention"].startswith("Zero-based frame indices in decoded order")
    request = document["request"]
    assert request["kind"] == "preview" and request["frame"] == 3
    assert "start_frame" not in request and "max_frames" not in request
    assert request["media_path"] == site.clip and request["media_sha256"] == sha(site.clip)
    # Every adjustable setting is pinned at its effective value: the adapter's own defaults too.
    assert request["controls"] == {
        "DetectorModelSelection": "Yolov8",
        "MaxFacesToDetectSlider": 20,
        "SwapOnlyBestMatchEnableToggle": True,
    }
    mapping = request["mappings"][0]
    assert mapping["parameters"] == {
        "SimilarityThresholdSlider": 70,
        "SwapModelSelection": "Inswapper128",
    }
    assert mapping["source_weights"] == [2, 1]
    # References are retained as content-addressed uploads; their originals are recorded.
    face, source = sha(site.face), sha(site.source)
    assert mapping["target_face"] == str(site.uploads / f"{face}.png")
    assert mapping["source_faces"] == [str(site.uploads / f"{source}.png"), mapping["target_face"]]
    assert (mapping["target_sha256"], mapping["source_sha256"]) == (face, [source, face])
    assert document["retained"] == {
        mapping["target_face"]: site.face,
        mapping["source_faces"][0]: site.source,
    }
    assert result["retention"] == {
        "media_path": "unretained",
        "mappings[0].target_face": "retained",
        "mappings[0].source_faces[0]": "retained",
        "mappings[0].source_faces[1]": "retained",
    }
    record = document["provenance"]
    assert record["settings"]["controls"]["HiddenSlider"] == 5
    assert record["settings"]["controls"]["LandmarkDetectToggle"] is False
    assert record["settings"]["parameters"] == [
        {
            "SwapModelSelection": "Inswapper128",
            "SimilarityThresholdSlider": 70,
            "HiddenDecimalSlider": 0.5,
            "DFMModelSelection": "",
        }
    ]
    assert record["adapter"] == common.code_identity()
    assert record["config_sha256"] == common.SETTINGS.sha256
    assets = site.fusion.resolve() / "model_assets"
    assert set(record["models"]) == {"Inswapper128", "Inswapper128ArcFace", "YoloFace8n"}
    model = record["models"]["Inswapper128"]
    assert model == {
        "path": str(assets / MODEL_FILES["Inswapper128"]),
        "declared_sha256": sha(assets / MODEL_FILES["Inswapper128"]),
        "sha256": sha(assets / MODEL_FILES["Inswapper128"]),
    }
    # The stored file round-trips to the same document, which verifies unchanged.
    stored = json.loads((site.state / "recipes/shot/1.json").read_text())
    assert stored == document and stored["sha256"] == control.recipe_digest(stored)
    read = control.get_recipe({"name": "shot", "verify": True})
    assert read["recipe"] == document and read["revisions"] == [1]
    assert read["verification"] == {"unchanged": True, "changes": []}


def test_swap_selection_names_its_model_and_arcface(site):
    document = saved(
        site,
        mappings=[
            {
                "target_face": site.face,
                "source_faces": [site.source],
                "parameters": {"SwapModelSelection": "GhostFace-v1"},
            }
        ],
    )
    assert set(document["provenance"]["models"]) == {
        "GhostFacev1",
        "GhostArcFace",
        "Inswapper128ArcFace",
        "YoloFace8n",
    }


def test_recipe_from_completed_job_records_its_execution(site):
    request = preview(site, job_id="done", frame=1)
    request["mappings"][0]["source_weights"] = [3]
    control.submit(request)
    complete("done")
    job = common.job_path("done")
    common.atomic_json(job / "result.json", {"revision": None, "models": ["GFPGANv1.4"]})
    document = control.save_recipe({"name": "from-job", "job_id": "done"})["recipe"]
    assert document["origin"]["kind"] == "job" and document["origin"]["job_id"] == "done"
    assert document["request"]["frame"] == 1
    assert document["request"]["mappings"][0]["source_weights"] == [3]
    assert document["request"]["mappings"][0]["target_sha256"] == sha(site.face)
    # Models the job loaded are recorded besides those its settings select.
    assert "GFPGANv1.4" in document["provenance"]["models"]
    release = common.read_json(job / "state.json")["release"]
    assert document["provenance"]["adapter"] == common.code_identity(release)
    assert document["provenance"]["config_sha256"] == common.SETTINGS.sha256
    # A job whose reference bytes changed since it ran cannot be recorded as if it had not.
    Path(site.source).write_bytes(b"edited")
    with pytest.raises(ValueError, match="no longer has the bytes the job used"):
        control.save_recipe({"name": "from-job", "job_id": "done"})
    with pytest.raises(ValueError, match="either job_id or an explicit request"):
        control.save_recipe({"name": "x", "job_id": "done", "frame": 2})


def test_only_completed_swap_jobs_become_recipes(site):
    control.submit(preview(site, job_id="waiting"))
    with pytest.raises(ValueError, match="recipes come from completed preview or render jobs"):
        control.save_recipe({"name": "x", "job_id": "waiting"})
    with pytest.raises(ValueError, match="no readable state"):
        control.save_recipe({"name": "x", "job_id": "absent"})


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"kind": "detect"}, "preview or render request"),
        ({"fps": 25}, "do not accept fps"),
        ({"name": "../escape"}, "Recipe names"),
        ({"note": "x" * 2001}, "at most 2000 characters"),
        ({"controls": {"HiddenSlider": 3}}, "Unsupported setting HiddenSlider"),
        ({"media_path": "/etc/hosts"}, "regular file inside"),
    ],
)
def test_invalid_recipe_requests_are_refused(site, changes, message):
    with pytest.raises(ValueError, match=message):
        control.save_recipe(intent(site, **changes))
    assert not list((site.state / "recipes").glob("*/*.json"))


# Immutable revisions and atomic writes


def test_revisions_are_immutable_and_never_reused(site):
    first = saved(site)
    second = saved(site, frame=5)
    assert (first["revision"], second["revision"]) == (1, 2)
    assert control.get_recipe({"name": "shot", "revision": 1})["recipe"] == first
    assert control.get_recipe({"name": "shot"})["recipe"]["request"]["frame"] == 5
    assert control.delete_recipe("shot", 2) == {"name": "shot", "deleted": [2], "remaining": [1]}
    assert saved(site, frame=6)["revision"] == 3  # The deleted number stays used.
    assert control.get_recipe({"name": "shot", "revision": 1})["recipe"] == first
    with pytest.raises(ValueError, match="has no revision 2"):
        control.get_recipe({"name": "shot", "revision": 2})
    listed = control.get_recipe({})["recipes"]
    assert [(entry["name"], entry["revisions"]) for entry in listed] == [("shot", [1, 3])]


def test_concurrent_writers_get_distinct_complete_revisions(site):
    barrier, documents = threading.Barrier(8), []

    def write(index):
        barrier.wait()
        documents.append(control.publish_recipe("busy", {"note": f"writer {index}"}))

    threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert sorted(document["revision"] for document in documents) == list(range(1, 9))
    directory = site.state / "recipes/busy"
    for document in documents:
        stored = json.loads((directory / f"{document['revision']}.json").read_text())
        assert stored == document and stored["sha256"] == control.recipe_digest(stored)
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        f"{number}.json" for number in range(1, 9)
    )


def test_a_writer_that_lost_the_race_takes_the_next_number(site, monkeypatch):
    existing = saved(site)
    original, stale = control.revisions, []

    def outdated(directory):
        if not stale:
            stale.append(True)
            return [], 0  # As if listed before revision 1 was written.
        return original(directory)

    monkeypatch.setattr(control, "revisions", outdated)
    fields = {key: existing[key] for key in control.RECIPE_KEYS - control.GENERATED}
    document = control.publish_recipe("shot", fields)
    assert document["revision"] == 2 and stale
    assert json.loads((site.state / "recipes/shot/1.json").read_text()) == existing
    # A crash before the link leaves only a temporary file, which is never a revision.
    (site.state / "recipes/shot/.write-crashed").write_text('{"partial":')
    monkeypatch.setattr(control, "revisions", original)
    assert control.get_recipe({"name": "shot"})["revisions"] == [1, 2]


# Inspect, export and import: schema, integrity and paths


def test_tampered_or_unsupported_documents_are_refused(site):
    saved(site)
    path = site.state / "recipes/shot/1.json"
    document = json.loads(path.read_text())
    edited = copy.deepcopy(document)
    edited["request"]["mappings"][0]["source_weights"] = [1, 1]
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="revision 1 fails its integrity check"):
        control.get_recipe({"name": "shot"})
    with pytest.raises(ValueError, match="fails its integrity check"):
        control.run_recipe({"name": "shot", "job_id": "tampered"})
    path.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        control.get_recipe({"name": "shot"})
    assert "not valid JSON" in control.get_recipe({})["recipes"][0]["error"]
    future = resign({**document, "version": 2})
    with pytest.raises(ValueError, match="format version 2, but this release reads version 1"):
        control.import_recipe({"document": future})
    with pytest.raises(ValueError, match="integrity check"):
        control.import_recipe({"document": edited})
    with pytest.raises(ValueError, match="not a VisoMaster recipe"):
        control.import_recipe({"document": ["not", "a", "recipe"]})
    extra = resign({**document, "script": "rm"})
    with pytest.raises(ValueError, match="unexpected or missing fields: script"):
        control.import_recipe({"document": extra})
    relative = copy.deepcopy(document)
    relative["request"]["media_path"] = "clip.mp4"
    with pytest.raises(ValueError, match="media_path must be an absolute path"):
        control.import_recipe({"document": resign(relative)})
    unhashed = copy.deepcopy(document)
    del unhashed["request"]["media_sha256"]
    with pytest.raises(ValueError, match="media_path has no valid SHA-256"):
        control.import_recipe({"document": resign(unhashed)})


def test_import_checks_allowed_paths_bytes_and_relocates_uploads(site, tmp_path):
    document = saved(site)
    # Outside the allowed roots: never read, reported as unavailable.
    outside = tmp_path / "private.mp4"
    shutil.copy(site.clip, outside)
    escaped = copy.deepcopy(document)
    escaped["request"]["media_path"] = str(outside)
    with pytest.raises(ValueError, match=r"Cannot import: .*media_path .*private\.mp4"):
        control.import_recipe({"document": resign(escaped)})
    # Missing media: actionable, and nothing is written.
    clip = Path(site.clip)
    data = clip.read_bytes()
    clip.unlink()
    with pytest.raises(ValueError, match="Upload each file with upload_media"):
        control.import_recipe({"document": document, "name": "copy"})
    assert not (site.state / "recipes/copy").exists()
    # The same bytes uploaded here are found by digest, and the relocation is recorded.
    upload = site.uploads / f"{hashlib.sha256(data).hexdigest()}.mp4"
    upload.write_bytes(data)
    imported = control.import_recipe({"document": document, "name": "copy"})["recipe"]
    assert (imported["name"], imported["revision"]) == ("copy", 1)
    assert imported["request"]["media_path"] == str(upload)
    assert imported["origin"]["kind"] == "import"
    assert imported["origin"]["source"]["sha256"] == document["sha256"]
    assert imported["origin"]["relocated"] == [
        {"input": "media_path", "recorded": site.clip, "used": str(upload)}
    ]
    assert imported["provenance"] == document["provenance"]
    # Different bytes at the recorded path are not the recorded input.
    clip.write_bytes(b"other bytes")
    upload.unlink()
    with pytest.raises(ValueError, match="not available here with their recorded bytes"):
        control.import_recipe({"document": document})


def test_import_refuses_settings_the_installation_does_not_accept(site):
    document = copy.deepcopy(saved(site))
    document["request"]["controls"]["DetectorModelSelection"] = "Missing"
    with pytest.raises(ValueError, match="Invalid value for DetectorModelSelection"):
        control.import_recipe({"document": resign(document)})


# Instantiation, substitutions and the parent revision


def test_run_submits_the_explicit_request_and_records_its_parent(site):
    document = saved(site)
    state = run()
    assert state["status"] == "queued"
    request = job_request("from-recipe")
    template = document["request"]
    assert {key: request[key] for key in template} == template
    assert request["recipe"] == {
        "name": "shot",
        "revision": 1,
        "sha256": document["sha256"],
        "invocation": {
            "accept_changes": None,
            "media_path": None,
            "mode": None,
            "name": "shot",
            "priority": None,
            "range": None,
            "report": None,
            "revision": None,
            "targets": None,
        },
        "substitutions": {},
        "accepted_changes": [],
    }
    # The stub worker revalidates the resolved request as the real worker does.
    assert complete("from-recipe")["result"] == {"validated": True}
    # A retry answers from the persisted job, even once inputs are gone.
    Path(site.clip).unlink()
    assert run()["status"] == "completed"
    with pytest.raises(ValueError, match="different request"):
        run(mode="render", range={"start_frame": 0, "max_frames": 2})
    control.submit(preview(site, job_id="plain", media_path=site.face))
    with pytest.raises(ValueError, match="different request"):
        run(job_id="plain")


def test_sources_of_weight_zero_are_not_recipe_inputs(site):
    # Excluded images are never read, so they need not exist and are neither hashed nor kept.
    missing = os.path.realpath(site.media / "never-written.png")
    mapping = {"target_face": site.face, "source_faces": [site.source, missing]}
    document = saved(site, mappings=[{**mapping, "source_weights": [1, 0]}])
    stored = document["request"]["mappings"][0]
    assert stored["source_faces"][1] == missing and stored["source_sha256"][1] is None
    assert "mappings[0].source_faces[1]" not in control.retention_report(document["request"])
    verified = control.get_recipe({"name": "shot", "verify": True})["verification"]
    assert verified == {"unchanged": True, "changes": []}
    assert run()["status"] == "queued"
    resolved = common.read_json(common.job_path("from-recipe") / "resolved.json")
    assert resolved["mappings"][0]["source_sha256"][1] is None
    assert complete("from-recipe")["result"] == {"validated": True}


def test_same_content_at_another_path_needs_no_new_decisions(site):
    saved(site)
    moved = Path(site.media) / "moved.mp4"
    shutil.copy(site.clip, moved)
    run(media_path=str(moved))
    request = job_request("from-recipe")
    assert request["media_path"] == str(moved) and request["frame"] == 3
    digest = sha(site.clip)
    assert request["recipe"]["substitutions"] == {
        "media_path": {
            "recorded": site.clip,
            "used": str(moved),
            "recorded_sha256": digest,
            "sha256": digest,
        }
    }


def test_a_new_clip_needs_explicit_targets_and_range(site, library):
    document = saved(site)
    other = Path(site.media) / "other.mp4"
    test_submit.ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=0.2", "-c:v", "libx264", other
    )
    with pytest.raises(ValueError, match="needs explicit targets, range"):
        run(media_path=str(other))
    with pytest.raises(ValueError, match="needs explicit range"):
        run(media_path=str(other), targets=[site.source])
    with pytest.raises(ValueError, match="needs explicit targets"):
        run(media_path=str(other), range={"frame": 0})
    assert not common.job_path("from-recipe").exists()
    run(media_path=str(other), targets=[site.source], range={"frame": 2})
    request = job_request("from-recipe")
    assert request["media_sha256"] == sha(other) and request["frame"] == 2
    mapping = request["mappings"][0]
    assert mapping["target_face"] == site.source and mapping["target_sha256"] == sha(site.source)
    assert mapping["source_faces"] == document["request"]["mappings"][0]["source_faces"]
    substitutions = request["recipe"]["substitutions"]
    assert substitutions["targets"] == [
        {
            "mapping": 0,
            "recorded": document["request"]["mappings"][0]["target_face"],
            "used": site.source,
        }
    ]
    assert substitutions["range"] == {"recorded": {"frame": 3}, "used": {"frame": 2}}
    assert substitutions["media_path"]["sha256"] == sha(other)


@pytest.mark.parametrize(
    "values, message",
    [
        ({"mode": "render"}, "render of a preview recipe needs an explicit range"),
        ({"mode": "render", "range": {"frame": 1}}, "gives exactly start_frame and max_frames"),
        ({"range": {"frame": 1, "start_frame": 0}}, "gives exactly frame"),
        ({"targets": []}, "one target_face per recipe mapping"),
        ({"targets": ["/etc/hosts"]}, "regular file inside"),
        ({"media_path": "/etc/hosts"}, "regular file inside"),
        ({"range": {"frame": -1}}, "frame must be a nonnegative integer"),
        ({"accept_changes": "yes"}, "accept_changes must be a boolean"),
        ({"fps": 25}, "does not accept fps"),
        ({"revision": 9}, "has no revision 9"),
        ({"name": "missing"}, "No recipe named missing"),
    ],
)
def test_invalid_substitutions_are_refused(site, values, message):
    saved(site)
    with pytest.raises(ValueError, match=message):
        run(**values)
    assert not common.job_path("from-recipe").exists()


def test_mode_range_report_and_priority_are_carried_or_substituted(site):
    report = {"review_images": 2, "expected_visible": [], "min_coverage": 0.5}
    saved(
        site,
        kind="render",
        start_frame=1,
        max_frames=4,
        report=report,
        priority="normal",
    )
    run(job_id="again")
    request = job_request("again")
    assert (request["start_frame"], request["max_frames"]) == (1, 4)
    assert request["report"] == report and request["priority"] == "normal"
    run(job_id="urgent", priority="interactive", range={"start_frame": 0, "max_frames": None})
    request = job_request("urgent")
    assert request["priority"] == "interactive" and request["max_frames"] is None
    assert request["recipe"]["substitutions"] == {
        "range": {
            "recorded": {"start_frame": 1, "max_frames": 4},
            "used": {"start_frame": 0, "max_frames": None},
        }
    }
    run(job_id="still", mode="preview", range={"frame": 2})
    request = job_request("still")
    assert request["kind"] == "preview" and "report" not in request and "start_frame" not in request
    substitutions = request["recipe"]["substitutions"]
    assert substitutions["mode"] == {"recorded": "render", "used": "preview"}
    assert substitutions["report"] == {"recorded": report, "used": None}


# Detecting changed bytes, defaults, models and versions before reuse


def change_media(site):
    data = bytearray(Path(site.clip).read_bytes())
    info = os.stat(site.clip)
    data[-1] ^= 0xFF  # Same size; the modification time is restored below.
    Path(site.clip).write_bytes(bytes(data))
    os.utime(site.clip, ns=(info.st_atime_ns, info.st_mtime_ns))
    return "media_path " + site.clip + " has different bytes"


def change_retained_reference(site):
    path = site.uploads / f"{sha(site.source)}.png"
    path.write_bytes(b"tampered")
    return "source_faces[0] " + str(path) + " has different bytes"


def change_model(site):
    path = site.fusion / "model_assets" / MODEL_FILES["YoloFace8n"]
    path.write_bytes(b"retrained")
    return "Model YoloFace8n has different bytes"


def remove_model(site):
    (site.fusion / "model_assets" / MODEL_FILES["Inswapper128ArcFace"]).unlink()
    return "Model Inswapper128ArcFace is not in this Fusion's model list or its file is missing"


def change_default(site):
    fusion_files(site, hidden=7)
    return "controls.HiddenSlider 5 now 7"


def change_adapter(site, monkeypatch):
    identity = common.code_identity()
    identity["code"]["engine.py"] = "0" * 64
    monkeypatch.setattr(common, "code_identity", lambda directory=None: identity)
    return "Adapter engine.py of release"


def change_fusion(site, monkeypatch):
    monkeypatch.setattr(common, "fusion_compatibility", lambda app: ("a" * 40, None))
    return "Fusion revision None now " + "a" * 40


@pytest.mark.parametrize(
    "change",
    [
        change_media,
        change_retained_reference,
        change_model,
        remove_model,
        change_default,
        change_adapter,
        change_fusion,
    ],
)
def test_changes_are_reported_before_reuse(site, monkeypatch, change):
    saved(site)
    arguments = (site, monkeypatch) if change in (change_adapter, change_fusion) else (site,)
    message = change(*arguments)
    verification = control.get_recipe({"name": "shot", "verify": True})["verification"]
    assert not verification["unchanged"]
    assert any(message in entry["message"] for entry in verification["changes"])
    with pytest.raises(ValueError, match="no longer matches the workstation") as refused:
        run()
    assert message in str(refused.value) and "accept_changes=true" in str(refused.value)
    assert not common.job_path("from-recipe").exists()
    run(accept_changes=True)
    request = job_request("from-recipe")
    accepted = request["recipe"]["accepted_changes"]
    assert any(message in entry["message"] for entry in accepted)
    assert request["media_sha256"] == sha(site.clip)


def test_metadata_alone_is_not_content_identity(site):
    saved(site)
    os.utime(site.clip, (OLD, OLD))  # A new timestamp with the same bytes is no change.
    assert control.get_recipe({"name": "shot", "verify": True})["verification"]["unchanged"]
    change_media(site)  # Same size and timestamp, other bytes, is.
    assert not control.get_recipe({"name": "shot", "verify": True})["verification"]["unchanged"]


def test_missing_inputs_fail_even_when_changes_are_accepted(site):
    saved(site)
    Path(site.clip).unlink()
    assert control.get_recipe({"name": "shot"})["retention"]["media_path"] == "unretained, missing"
    with pytest.raises(ValueError, match="Recipe inputs are unavailable: media_path"):
        run(accept_changes=True)


def test_hashing_happens_only_at_creation_verification_and_instantiation(site, monkeypatch):
    hashed, original = [], common.file_sha256

    def counting(path):
        hashed.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(common, "file_sha256", counting)
    saved(site)
    assert MODEL_FILES["Inswapper128"] in hashed and Path(site.clip).name in hashed
    hashed.clear()
    control.get_recipe({"name": "shot"})
    control.get_recipe({})
    assert hashed == []
    run()
    assert Path(site.clip).name in hashed
    hashed.clear()
    for _ in range(3):
        control.status("from-recipe")
        control.list_jobs()
        control.get_recipe({"name": "shot"})
    assert hashed == []
    control.get_recipe({"name": "shot", "verify": True})
    assert Path(site.clip).name in hashed


# Retention: retained copies, pinned artifacts and recipe deletion


def completed_artifact(job_id, name, data):
    job = common.job_path(job_id)
    (job / "artifacts").mkdir(parents=True)
    common.atomic_json(job / "request.json", {"kind": "normalize"})
    common.atomic_json(
        job / "state.json",
        {"job_id": job_id, "status": "completed", "created_at": 0, "finished_at": OLD},
    )
    path = job / "artifacts" / name
    path.write_bytes(data)
    return str(path)


def test_retention_keeps_what_recipes_name_and_deleting_recipes_deletes_no_inputs(site):
    media = completed_artifact("normalized", "normalized.mp4", Path(site.clip).read_bytes())
    crop = completed_artifact("detection", "face-0.png", Path(site.face).read_bytes())
    first = saved(site, media_path=media, mappings=[{"target_face": crop, "source_faces": [crop]}])
    second = saved(site, name="other", media_path=media)
    target = first["request"]["mappings"][0]["target_face"]
    assert target == str(site.uploads / f"{sha(crop)}.png") and Path(target).read_bytes()
    for path in site.uploads.iterdir():
        os.utime(path, (OLD, OLD))
    # The detection job is not needed: the recipe owns a copy of its crop.
    pruned = control.prune(1, dry_run=False)
    assert [job["job_id"] for job in pruned["jobs"]] == ["detection"]
    assert pruned["uploads"] == []
    # The normalized media is a recipe input, so its job stays.
    with pytest.raises(ValueError, match="saved recipe uses this job's artifacts"):
        control.delete_job("normalized")
    assert control.delete_recipe("shot")["deleted"] == [1]
    # Deleting a recipe deletes nothing else; another recipe still protects shared inputs.
    assert Path(target).exists() and Path(media).exists()
    assert control.prune(1, dry_run=False)["uploads"] == []
    with pytest.raises(ValueError, match="saved recipe uses"):
        control.delete_job("normalized")
    # An active job using a retained copy protects it once no recipe names it.
    shared = second["request"]["mappings"][0]["source_faces"][0]
    control.submit(
        preview(
            site,
            job_id="active",
            media_path=media,
            mappings=[{"target_face": shared, "source_faces": [shared]}],
        )
    )
    control.delete_recipe("other")
    pruned = control.prune(1, dry_run=False)
    assert str(shared) not in [entry["path"] for entry in pruned["uploads"]]
    assert Path(shared).exists()
    with pytest.raises(ValueError, match="queued or running job"):
        control.delete_job("normalized")
    complete("active")
    os.utime(shared, (OLD, OLD))
    pruned = control.prune(1, dry_run=False)
    assert shared in [entry["path"] for entry in pruned["uploads"]]


def test_an_unreadable_recipe_keeps_everything(site):
    saved(site)
    for path in site.uploads.iterdir():
        os.utime(path, (OLD, OLD))
    (site.state / "recipes/shot/1.json").write_text("{damaged")
    assert control.prune(1, dry_run=True)["uploads"] == []
    control.delete_recipe("shot")
    assert len(control.prune(1, dry_run=True)["uploads"]) == 2


def test_unretainable_reference_types_stay_in_place_and_are_reported(site):
    bitmap = Path(site.media) / "face.bmp"
    shutil.copy(site.face, bitmap)
    result = control.save_recipe(
        intent(site, mappings=[{"target_face": str(bitmap), "source_faces": [site.source]}])
    )
    assert result["recipe"]["request"]["mappings"][0]["target_face"] == str(bitmap)
    assert result["retention"]["mappings[0].target_face"] == "unretained"
    assert result["retention"]["mappings[0].source_faces[0]"] == "retained"


# Engine: media verification and the execution manifest


@pytest.fixture
def worker(tmp_path, monkeypatch):
    config = write_config(tmp_path / "remote-config.json", tmp_path)
    settings = apply_config(monkeypatch, config)
    monkeypatch.setattr(engine, "CONFIG_ERROR", None)
    monkeypatch.setattr(engine, "SETTINGS", settings)
    monkeypatch.setattr(engine.sys, "platform", "test")
    monkeypatch.setenv("VM_SUPERVISOR_PID", str(os.getppid()))
    monkeypatch.setenv("VM_CONFIG_SHA256", settings.sha256)
    face, clip = tmp_path / "face.png", tmp_path / "clip.mp4"
    face.write_bytes(b"face")
    clip.write_bytes(b"clip")
    models = {"Inswapper128": object(), "GFPGANv1.4": None, "YoloFace8n": object()}
    window = SimpleNamespace(
        control={"FaceTrackingEnableToggle": False, "KPSSmoothingEnableToggle": False},
        models_processor=SimpleNamespace(models=models),
    )
    loaded = []
    monkeypatch.setattr(engine, "initialize", lambda controls: loaded.append(1) or window)
    monkeypatch.setattr(engine, "configure_faces", lambda window, mappings: [])
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
    job = common.job_path("recipe-job")
    (job / "artifacts").mkdir(parents=True)
    mapping = {"target_face": str(face), "source_faces": [str(face)]}
    request = {"kind": "preview", "media_path": str(clip), "mappings": [mapping]}
    return SimpleNamespace(job=job, request=request, clip=clip, loaded=loaded)


def test_worker_reports_the_parent_recipe_and_loaded_models(worker):
    parent = {"name": "shot", "revision": 2, "sha256": "f" * 64}
    request = {**worker.request, "media_sha256": sha(worker.clip), "recipe": parent}
    common.atomic_json(worker.job / "resolved.json", control.fingerprint(request))
    engine.run(worker.job)
    result = common.read_json(worker.job / "result.json")
    assert result["recipe"] == parent
    assert result["models"] == ["Inswapper128", "YoloFace8n"]


def test_worker_refuses_media_changed_after_submission(worker):
    request = {**worker.request, "media_sha256": sha(worker.clip)}
    common.atomic_json(worker.job / "resolved.json", control.fingerprint(request))
    worker.clip.write_bytes(b"edited")
    with pytest.raises(ValueError, match="changed after submission"):
        engine.run(worker.job)
    assert worker.loaded == []  # Refused before Fusion loads.
    with pytest.raises(ValueError, match="media_sha256 must be"):
        common.validate_request({**request, "media_sha256": "abc"})


# Public tools over stdio


def test_recipe_tools_round_trip_over_stdio(server, tmp_path):
    for name in ("clip.mp4", "face.png", "source.png"):
        (tmp_path / name).write_bytes(name.encode() * 10)
    clip, face, source = (str(tmp_path / n) for n in ("clip.mp4", "face.png", "source.png"))
    mapping = {"target_face": face, "source_faces": [source], "source_weights": [1]}
    saving = {"name": "shot", "media_path": clip, "mappings": [mapping], "frame": 0}
    results = call(
        server,
        ("save_recipe", saving),
        ("save_recipe", {"name": "shot"}),
        ("get_recipe", {}),
        ("get_recipe", {"name": "shot", "verify": True}),
        ("export_recipe", {"name": "shot"}),
    )
    created, incomplete, listed, read, exported = results
    assert not created.isError, created.content[0].text
    document = created.structuredContent["recipe"]
    assert document["revision"] == 1 and document["request"]["mappings"][0]["source_weights"]
    assert incomplete.isError and "Give job_id, or media_path and mappings" in (
        incomplete.content[0].text
    )
    assert [entry["name"] for entry in listed.structuredContent["recipes"]] == ["shot"]
    assert read.structuredContent["verification"]["unchanged"]
    path = Path(exported.structuredContent["path"])
    assert path == tmp_path / "downloads/recipes/shot-r1.json"
    assert json.loads(path.read_text()) == document
    tampered = tmp_path / "tampered.json"
    tampered.write_text(path.read_text().replace('"frame": 0', '"frame": 9'))
    results = call(
        server,
        ("import_recipe", {"local_path": str(path), "name": "copy"}),
        ("import_recipe", {"local_path": str(tampered)}),
        ("run_recipe", {"name": "copy", "job_id": "replay"}),
        ("run_recipe", {"name": "copy", "job_id": "replay"}),
        ("run_recipe", {"name": "copy", "mode": "render"}),
        (
            "run_recipe",
            {
                "name": "copy",
                "job_id": "render",
                "mode": "render",
                "frame_range": {"start_frame": 0, "max_frames": None},
                "priority": "normal",
            },
        ),
        ("delete_recipe", {"name": "shot"}),
        ("get_recipe", {"name": "shot"}),
    )
    imported, refused, replay, retry, needs_range, render, deleted, gone = results
    assert not imported.isError, imported.content[0].text
    assert imported.structuredContent["recipe"]["origin"]["source"]["name"] == "shot"
    assert refused.isError and "integrity check" in refused.content[0].text
    assert not replay.isError, replay.content[0].text
    assert retry.structuredContent["job_id"] == "replay"
    request = json.loads((tmp_path / "state/jobs/replay/request.json").read_text())
    assert request["recipe"]["name"] == "copy" and request["recipe"]["revision"] == 1
    assert needs_range.isError and "needs an explicit range" in needs_range.content[0].text
    assert not render.isError, render.content[0].text
    request = json.loads((tmp_path / "state/jobs/render/request.json").read_text())
    assert request["recipe"]["substitutions"]["range"]["used"] == {
        "start_frame": 0,
        "max_frames": None,
    }
    assert request["priority"] == "normal"
    assert deleted.structuredContent == {"name": "shot", "deleted": [1], "remaining": []}
    assert gone.isError and "No recipe named shot" in gone.content[0].text
    settle(tmp_path / "state")

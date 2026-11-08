"""Recipe provenance covers every model a swap's settings load, from the selection tables
preflight checks: restorer, occluder, XSeg, expression and CSCS adapter models included."""

import json

import common
import control
import pytest
import test_recipes
from test_recipes import MODEL_FILES, intent, resign, saved, sha  # noqa: F401
from test_submit import library, make_site  # noqa: F401

COMMON_LAYOUT = """LAYOUT = {'Restore': {
    'FaceRestorerEnableToggle': {'default': False},
    'FaceRestorerTypeSelection': {'default': 'GFPGAN-v1.4', 'options': ['GFPGAN-v1.4', 'CodeFormer']},
    'OccluderEnableToggle': {'default': False},
}}
"""
EXTRA_FILES = {"CodeFormer": "codeformer_fp16.onnx", "Occluder": "occluder.onnx"}
RESTORERS = {"GFPGANv1.4", "CodeFormer"}
RESTORED = {"FaceRestorerEnableToggle": True, "FaceRestorerTypeSelection": "GFPGAN-v1.4"}


@pytest.fixture
def site(make_site, monkeypatch):  # noqa: F811
    monkeypatch.setattr(test_recipes, "MODEL_FILES", {**MODEL_FILES, **EXTRA_FILES})
    site = make_site()
    test_recipes.fusion_files(site)
    widgets = site.fusion / "app/ui/widgets"
    (widgets / "common_layout_data.py").write_text(COMMON_LAYOUT)
    site.uploads = site.state / "uploads"
    with open(site.source, "ab") as stream:
        stream.write(b"another person")
    site.assets = site.fusion.resolve() / "model_assets"
    return site


def with_parameters(site, parameters, name="shot"):
    request = intent(site, name=name)
    request["mappings"][0]["parameters"] = {**request["mappings"][0]["parameters"], **parameters}
    return control.save_recipe(request)["recipe"]


def verify(name="shot"):
    return control.get_recipe({"name": name, "verify": True})["verification"]


def test_model_names_come_from_the_selection_tables_preflight_checks():
    settings = {
        "controls": {
            "DetectorModelSelection": "Yolov8",
            "RecognitionModelSelection": "Inswapper128ArcFace",
        },
        "parameters": [
            {
                "SwapModelSelection": "CSCS",
                **RESTORED,
                "OccluderEnableToggle": True,
                "DFLXSegEnableToggle": True,
                "FaceExpressionEnableBothToggle": True,
            },
            {"SwapModelSelection": "GhostFace-v1", "FaceRestorerTypeSelection": "CodeFormer"},
        ],
    }
    names = common.model_names(settings)
    expected = {"YoloFace8n", "Inswapper128ArcFace", "CSCS", "CSCSArcFace", "CSCSIDArcFace"}
    expected |= {"GFPGANv1.4", "Occluder", "XSeg", *common.EXPRESSION_MODELS}
    expected |= {"GhostFacev1", "GhostArcFace"}  # The second mapping's restorer is off.
    assert names == expected
    # Preflight's model check uses the same functions.
    for choice in common.model_choices(settings):
        assert common.required_models(choice)[0] <= names


def test_a_changed_restorer_or_mask_model_is_reported_before_reuse(site):
    document = with_parameters(site, {**RESTORED, "OccluderEnableToggle": True})
    models = document["provenance"]["models"]
    assert {"GFPGANv1.4", "Occluder"} <= set(models)
    assert models["Occluder"]["sha256"] == sha(site.assets / "occluder.onnx")
    assert verify() == {"unchanged": True, "changes": []}
    (site.assets / "GFPGANv1.4.onnx").write_bytes(b"retrained restorer")
    (site.assets / "occluder.onnx").write_bytes(b"retrained occluder")
    changes = verify()["changes"]
    assert {(c["kind"], c["item"]) for c in changes} == {
        ("model", "GFPGANv1.4"),
        ("model", "Occluder"),
    }
    assert all("different bytes" in c["message"] for c in changes)
    with pytest.raises(ValueError, match="GFPGANv1.4"):
        control.run_recipe({"name": "shot", "job_id": "reuse"})


def test_settings_without_a_restorer_hash_no_restorer_file(site, monkeypatch):
    hashed, real = [], common.file_sha256
    monkeypatch.setattr(common, "file_sha256", lambda path: hashed.append(str(path)) or real(path))
    document = saved(site)
    assert not RESTORERS & set(document["provenance"]["models"])
    verify()
    restorer_files = {
        str(site.assets / MODEL_FILES["GFPGANv1.4"]),
        str(site.assets / "codeformer_fp16.onnx"),
    }
    assert hashed and not restorer_files & set(hashed)
    hashed.clear()
    with_parameters(site, RESTORED, name="restored")
    assert str(site.assets / MODEL_FILES["GFPGANv1.4"]) in hashed
    assert str(site.assets / "codeformer_fp16.onnx") not in hashed  # Not the selected type.


def test_a_revision_saved_before_restorer_models_were_recorded_reports_them_clearly(site):
    document = with_parameters(site, RESTORED)
    old = json.loads(json.dumps(document))
    del old["provenance"]["models"]["GFPGANv1.4"]  # As earlier releases recorded it.
    path = site.state / "recipes/shot/1.json"
    path.write_text(json.dumps(resign(old)))
    (change,) = verify()["changes"]
    assert (change["kind"], change["item"], change["recorded"]) == ("model", "GFPGANv1.4", None)
    assert "was not recorded when this revision was saved" in change["message"]
    assert "save a new revision" in change["message"]
    # A revision without such settings records the same models as before and still verifies.
    saved(site, name="plain")
    assert verify("plain") == {"unchanged": True, "changes": []}


def test_models_a_segment_enables_are_recorded_and_checked(site):
    segment = {"id": "close-up", "start_frame": 2, "end_frame": 5, "parameters": RESTORED}
    request = intent(site, kind="render", start_frame=0, max_frames=8, segments=[segment])
    del request["frame"]
    document = control.save_recipe(request)["recipe"]
    assert document["request"]["segments"] == [segment]
    models = document["provenance"]["models"]
    # The base settings leave the restorer off; the segment turns it on.
    assert "GFPGANv1.4" in models and "Occluder" not in models
    (site.assets / "GFPGANv1.4.onnx").write_bytes(b"retrained restorer")
    (change,) = verify()["changes"]
    assert (change["kind"], change["item"]) == ("model", "GFPGANv1.4")
    # A passthrough segment selects nothing; an overriding one counts for its mappings.
    passthrough = {"id": "titles", "start_frame": 0, "end_frame": 2, "passthrough": True}
    settings = {"controls": {}, "parameters": [{"SwapModelSelection": "Inswapper128"}]}
    assert "GFPGANv1.4" not in common.model_names(settings, [passthrough])
    restored = {**segment, "mappings": [0]}
    assert "GFPGANv1.4" in common.model_names(settings, [passthrough, restored])

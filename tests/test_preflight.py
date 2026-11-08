"""Swap preflight: readiness, stable check codes, qualification and the optional GPU stage
(vmcp-harden-pqj.4).

Media are FFmpeg-generated; the Fusion checkout is a fake with realistic layout metadata, a
models_data.py in Fusion's format and model files. Workers are stubs, so nothing loads a model.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import common
import control
import engine
import pytest
import test_server
from conftest import apply_config, write_config
from test_submit import control_command, settle, wait_file, wait_for

server, call = test_server.server, test_server.call

LAYOUTS = {
    "settings": {
        "DetectorModelSelection": {
            "default": "RetinaFace",
            "options": ["RetinaFace", "Yolov8", "SCRFD", "Yunet"],
        },
        "DetectorScoreSlider": {"default": "50", "min_value": "1", "max_value": "100"},
        "MaxFacesToDetectSlider": {"default": "20", "min_value": "1", "max_value": "50"},
        "RecognitionModelSelection": {
            "default": "Inswapper128ArcFace",
            "options": ["Inswapper128ArcFace", "SimSwapArcFace"],
        },
        "SwapOnlyBestMatchEnableToggle": {"default": True},
    },
    "swapper": {
        "SwapModelSelection": {
            "default": "Inswapper128",
            "options": ["Inswapper128", "AlphaFace", "SimSwap512", "CSCS"],
        },
        "SimilarityThresholdSlider": {"default": "60", "min_value": "1", "max_value": "100"},
        "OccluderEnableToggle": {"default": False},
    },
    "common": {
        "FaceRestorerEnableToggle": {"default": False},
        "FaceRestorerTypeSelection": {
            "default": "GFPGAN-v1.4",
            "options": ["GFPGAN-v1.4", "CodeFormer", "Future-Restorer"],
        },
        "FaceExpressionEnableBothToggle": {"default": False},
    },
}
MODELS = {
    "YoloFace8n": "yoloface_8n.onnx",
    "RetinaFace": "det_10g.onnx",
    "Inswapper128": "inswapper_128.fp16.onnx",
    "Inswapper128ArcFace": "w600k_r50.onnx",
    "SimSwap512": "simswap_512_unoff.onnx",
    "SimSwapArcFace": "simswap_arcface_model.onnx",
    "CSCS": "cscs_256.onnx",
    "CSCSArcFace": "cscs_arcface_model.onnx",
    "GFPGANv1.4": "GFPGANv1.4.onnx",
    "CodeFormer": "codeformer_fp16.onnx",
    "Occluder": "occluder.onnx",
}
MISSING = {"CSCSIDArcFace": "cscs_id_adapter.onnx"}  # Listed by Fusion, never downloaded here.
# Analyze jobs wait for `release`, then publish STATE/analysis-result.json as their result.
RUNNER = (
    "#!/usr/bin/env python3\n"
    + """
import sys, time
from pathlib import Path
job = Path(sys.argv[2])
(job / "entered").touch()
while not (job / "release").exists():
    time.sleep(0.05)
(job / "result.json").write_text((job.parents[1] / "analysis-result.json").read_text())
"""
)
HARDWARE = "5.2, 24576, 155\n"
FUSION_ONLY = ("torch", "cv2", "numpy", "onnxruntime", "app", "PySide6")


def ffmpeg(*arguments):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *map(str, arguments)], check=True)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("preflight-media")
    source = "testsrc=size=64x48:rate=25:duration=1"
    ffmpeg("-f", "lavfi", "-i", source, "-c:v", "libx264", "-bf", "0", root / "clip.mp4")
    ffmpeg("-i", root / "clip.mp4", "-frames:v", "1", root / "face.png")
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=48x48:rate=1", "-frames:v", "1", root / "other.png")
    # Frames 3-12 dropped with their timestamps kept. MP4 metadata shows it; Matroska's claims
    # 25 fps, so only the timestamps after frame 2 reveal it.
    drop = ["-vf", "select='not(between(n,3,12))'", "-fps_mode", "passthrough"]
    for name in ("vfr.mp4", "vfr.mkv"):
        ffmpeg("-i", root / "clip.mp4", *drop, "-c:v", "libx264", root / name)
    odd = "testsrc=size=63x47:rate=25:duration=0.4"
    ffmpeg("-f", "lavfi", "-i", odd, "-c:v", "libx264", "-pix_fmt", "yuv444p", root / "odd.mp4")
    return root


def fusion_checkout(path):
    """Fill conftest's fake Fusion checkout with layouts, a model list and model files."""
    widgets = path / "app/ui/widgets"
    for name in common.LAYOUT_NAMES:
        layout = {"Group": LAYOUTS[name]} if name in LAYOUTS else {}
        (widgets / f"{name}_layout_data.py").write_text(f"LAYOUT = {layout!r}\n")
    entries = "".join(
        f'    {{"model_name": "{name}", "local_path": f"{{models_dir}}/{file}", "hash": "0"}},\n'
        for name, file in {**MODELS, **MISSING}.items()
    )
    processors = path / "app/processors"
    processors.mkdir(parents=True)
    (processors / "models_data.py").write_text(
        "from pathlib import Path\nimport numpy as np\n\n"
        'models_dir = Path(__file__).resolve().parent.parent.parent / "model_assets"\n'
        f"models_list = [\n{entries}]\n"
    )
    assets = path / "model_assets"
    assets.mkdir()
    for file in MODELS.values():
        (assets / file).write_bytes(b"onnx")
    return path


@pytest.fixture
def site(tmp_path, monkeypatch, library):
    media = tmp_path / "media"
    media.mkdir()
    for name in ("clip.mp4", "face.png", "other.png", "vfr.mp4", "vfr.mkv", "odd.mp4"):
        shutil.copy(library / name, media / name)
    shutil.copy(library / "face.png", media / "source.png")
    runner = tmp_path / "bin/runner"
    runner.parent.mkdir()
    runner.write_text(RUNNER)
    runner.chmod(0o755)
    config = write_config(
        tmp_path / "remote-config.json",
        media,
        runner=runner,
        fusion_dir=str(tmp_path / "fusion"),
        state_dir=str(tmp_path / "state"),
    )
    fusion = fusion_checkout(tmp_path / "fusion")
    settings = apply_config(monkeypatch, config)
    for name in ("SETTINGS", "APP", "CONFIG_ERROR"):
        monkeypatch.setattr(engine, name, getattr(common, name))
    # A git checkout at the supported revision, and one observable GPU of the qualified class.
    monkeypatch.setattr(control, "fusion_compatibility", lambda app: (common.REVISION, None))
    monkeypatch.setattr(control, "output", lambda arguments: HARDWARE)
    yield SimpleNamespace(
        media=settings.base,
        state=settings.state,
        fusion=fusion,
        config=config,
        clip=str(settings.base / "clip.mp4"),
        face=str(settings.base / "face.png"),
        source=str(settings.base / "source.png"),
    )
    settle(settings.state)


def swap(site, mode="render", parameters=None, **fields):
    mapping = {"target_face": site.face, "source_faces": [site.source]}
    mapping["parameters"] = parameters or {}
    request = {
        "kind": mode,
        "media_path": site.clip,
        "mappings": [mapping],
        "frame": 0,
        "start_frame": 0,
        "max_frames": None,
        "controls": {},
    }
    return {**request, **fields}


def preflight(request, **options):
    return control.dispatch("preflight", {"request": request, **options})


def by_name(report):
    names = [check["check"] for check in report["checks"]]
    assert len(names) == len(set(names)), names
    return {check["check"]: check for check in report["checks"]}


def tree(path):
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


# Overall readiness and the response contract


def test_a_ready_render_reports_every_check_with_codes_evidence_and_fingerprint(site):
    before = time.time()
    report = preflight(swap(site, max_frames=10))
    checks = by_name(report)
    assert report["readiness"] == "ready", report
    assert before <= report["checked_at"] <= time.time()
    assert set(checks) == {
        "configuration",
        "fusion_revision",
        "deployed_code",
        "job_id",
        "media_path",
        "references",
        "settings",
        "model_assets",
        "media",
        "storage",
        "gpu",
        "qualification",
        "request",
        "reference_analysis",
    }
    for check in report["checks"]:
        assert set(check) == {
            "check",
            "status",
            "severity",
            "code",
            "message",
            "evidence",
            "actions",
        }
        assert check["severity"] == control.SEVERITY[check["status"]]
    assert {name: checks[name]["code"] for name in checks} == {
        "configuration": "config_valid",
        "fusion_revision": "fusion_compatible",
        "deployed_code": "client_code_unknown",
        "job_id": "job_id_generated",
        "media_path": "path_ok",
        "references": "references_ok",
        "settings": "settings_ok",
        "model_assets": "models_present",
        "media": "media_ready",
        "storage": "space_ok",
        "gpu": "gpu_observed",
        "qualification": "qualified",
        "request": "request_valid",
        "reference_analysis": "not_requested",
    }
    # Skipped checks are explicit, with the action that would run them.
    assert report["not_checked"] == ["deployed_code", "reference_analysis"]
    assert checks["reference_analysis"]["actions"][0]["description"].startswith("Rerun")
    media = checks["media"]["evidence"]
    assert media["frames"] == media["frames_checked"] == 10 and media["fps"] == "25"
    assert media["width"] == 64 and media["estimated_bytes"] > 0
    digest = hashlib.sha256(Path(site.source).read_bytes()).hexdigest()
    references = checks["references"]["evidence"]["references"]
    assert [r["role"] for r in references] == ["target", "source"]
    assert references[1]["sha256"] == digest
    # The detector is the adapter default (Yolov8), recognition and swapper Fusion's defaults.
    models = {m["model"] for m in checks["model_assets"]["evidence"]["models"]}
    assert models == {"YoloFace8n", "Inswapper128", "Inswapper128ArcFace"}
    qualified = checks["qualification"]["evidence"]["mappings"][0]["qualified_by"][0]
    assert qualified["id"] == "baseline-2026-11-07" and qualified["date"] == "2026-11-07"
    assert "VERIFICATION.md" in qualified["source"]
    assert checks["gpu"]["evidence"] == {
        "compute_capability": "5.2",
        "vram_class": "24 GB",
        "vram_total_mib": 24576,
        "vram_used_mib": 155,
    }
    assert set(report["fingerprint"]) == {"request", "inputs", "configuration", "sha256"}
    assert "reserves nothing" in report["notice"]


def test_cheap_checks_import_no_fusion_start_nothing_and_change_nothing(site, monkeypatch):
    for name in FUSION_ONLY:
        monkeypatch.setitem(sys.modules, name, None)  # Importing any of them now raises.
    commands, real = [], subprocess.Popen

    def popen(arguments, *args, **kwargs):
        commands.append(Path(arguments[0]).name)
        return real(arguments, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    inputs = [Path(site.clip), Path(site.face), Path(site.source)]
    stamps = [(p.stat().st_mtime_ns, p.read_bytes()) for p in inputs]
    state = tree(site.state)
    for request in (swap(site, max_frames=5), swap(site, mode="preview", frame=3)):
        report = preflight(request)
        assert report["readiness"] == "ready", report
    assert set(commands) == {"ffprobe"}
    assert tree(site.state) == state  # No job, queue, lock or upload was created.
    assert [(p.stat().st_mtime_ns, p.read_bytes()) for p in inputs] == stamps


def test_the_request_shape_is_submit_swaps_and_submission_accepts_what_preflight_passes(site):
    request = swap(site, job_id="planned", max_frames=4)
    assert preflight(dict(request))["readiness"] == "ready"
    state = control.submit(dict(request))
    assert state["job_id"] == "planned" and state["status"] in ("queued", "running")
    # The same request now names an existing job; a changed one under that ID conflicts.
    checks = by_name(preflight(dict(request)))
    assert checks["job_id"]["code"] == "job_exists" and checks["job_id"]["status"] == "warning"
    changed = by_name(preflight({**request, "max_frames": 5}))["job_id"]
    assert changed["code"] == "job_id_conflict" and changed["status"] == "blocked"
    invalid = by_name(preflight({**request, "job_id": "../x"}))["job_id"]
    assert invalid["code"] == "job_id_invalid"
    control.cancel("planned")


@pytest.mark.parametrize("kind", ["detect", "normalize", "analyze", None])
def test_only_swap_requests_are_preflighted(site, kind):
    with pytest.raises(ValueError, match="preview or render"):
        preflight({**swap(site), "kind": kind})


# Cheap check failures


def test_invalid_and_unsupported_settings_block_with_get_settings_action(site):
    report = preflight(
        swap(
            site,
            controls={"DetectorModelSelection": "Nope"},
            parameters={"SwapModelSelection": "AlphaFace"},
        )
    )
    checks = by_name(report)
    assert report["readiness"] == "blocked"
    settings = checks["settings"]
    assert settings["code"] == "setting_invalid" and settings["severity"] == "error"
    assert [e["code"] for e in settings["evidence"]["errors"]] == [
        "setting_invalid",
        "swapper_unsupported",
    ]
    assert settings["actions"] == [
        {
            "description": "Use names, types, options and limits from get_settings",
            "tool": "get_settings",
            "arguments": {},
        }
    ]
    assert checks["request"]["status"] == "blocked"
    unknown = by_name(preflight(swap(site, parameters={"Bogus": 1})))["settings"]
    assert unknown["code"] == "setting_unsupported"


def test_missing_and_unmapped_model_files(site):
    restored = {"FaceRestorerEnableToggle": True, "FaceRestorerTypeSelection": "CodeFormer"}
    assert by_name(preflight(swap(site, parameters=restored)))["model_assets"]["code"] == (
        "models_present"
    )
    (site.fusion / "model_assets/codeformer_fp16.onnx").unlink()
    report = preflight(swap(site, parameters=restored))
    assets = by_name(report)["model_assets"]
    assert report["readiness"] == "blocked"
    assert assets["status"] == "blocked" and assets["code"] == "model_missing"
    assert "model_assets/codeformer_fp16.onnx" in assets["message"]
    assert "tool" not in assets["actions"][0]  # An operator step, not a tool call.
    # Fusion lists the model but its file was never downloaded.
    cscs = by_name(preflight(swap(site, parameters={"SwapModelSelection": "CSCS"})))
    assert cscs["model_assets"]["code"] == "model_missing"
    assert "cscs_id_adapter.onnx" in cscs["model_assets"]["message"]
    # An option this adapter has no model mapping for is not guessed at.
    future = {"FaceRestorerEnableToggle": True, "FaceRestorerTypeSelection": "Future-Restorer"}
    unmapped = by_name(preflight(swap(site, parameters=future)))["model_assets"]
    assert unmapped["status"] == "not_checked" and unmapped["code"] == "model_unmapped"


def test_missing_media_and_refused_references(site, tmp_path):
    outside = tmp_path / "outside.png"
    shutil.copy(site.face, outside)
    request = swap(site, media_path=str(site.media / "gone.mp4"))
    request["mappings"][0]["source_faces"] = [site.source, str(outside)]
    report = preflight(request)
    checks = by_name(report)
    assert checks["media_path"]["code"] == "path_missing"
    assert checks["media"]["status"] == "not_checked"
    assert checks["media"]["code"] == "prerequisite_blocked"
    assert "media" in report["not_checked"]
    assert checks["references"]["code"] == "path_outside_roots"
    refused = [r for r in checks["references"]["evidence"]["references"] if "error" in r]
    assert [r["path"] for r in refused] == [str(outside)]
    assert checks["request"]["status"] == "blocked" and report["readiness"] == "blocked"


def test_a_changed_reference_is_reported_against_its_pinned_digest(site):
    digest = hashlib.sha256(Path(site.source).read_bytes()).hexdigest()
    request = swap(site)
    request["mappings"][0]["source_sha256"] = [digest]
    assert preflight(request)["readiness"] == "ready"
    shutil.copy(site.media / "other.png", site.source)
    checks = by_name(preflight(request))
    assert checks["references"]["code"] == "reference_changed"
    assert checks["request"]["code"] == "reference_changed"


def test_stale_revision_and_mismatched_code_are_reported(site, monkeypatch):
    reason = "Fusion revision changed to abc, but the adapter supports another; qualify it"
    monkeypatch.setattr(control, "fusion_compatibility", lambda app: ("abc", reason))
    report = preflight(swap(site), client_code={"common.py": "0" * 64})
    checks = by_name(report)
    assert report["readiness"] == "blocked"
    revision = checks["fusion_revision"]
    assert revision["code"] == "fusion_incompatible" and revision["message"] == reason
    assert revision["evidence"]["revision"] == "abc" and revision["actions"]
    assert checks["deployed_code"]["code"] == "code_mismatch"
    matching = preflight(swap(site), client_code=common.code_identity()["code"])
    assert by_name(matching)["deployed_code"]["code"] == "code_matches"


def test_an_invalid_configuration_skips_everything_else(site, monkeypatch):
    monkeypatch.setattr(control, "CONFIG_ERROR", "remote-config.json: version must be 1")
    report = preflight(swap(site))
    assert report["readiness"] == "blocked" and report["fingerprint"] is None
    assert by_name(report)["configuration"]["code"] == "config_invalid"
    assert len(report["not_checked"]) == len(report["checks"]) - 1


def test_a_malformed_request_is_blocked_without_input_checks(site):
    report = preflight({**swap(site), "mappings": [{"target_face": 3}]})
    checks = by_name(report)
    assert checks["request"]["status"] == "blocked"
    assert all(checks[name]["status"] == "not_checked" for name in control.INPUT_CHECKS)


# Media: the worker's own checks, range-limited timing, normalization as a suggestion


def test_vfr_media_suggests_normalization_and_converts_nothing(site):
    metadata = by_name(preflight(swap(site, media_path=str(site.media / "vfr.mp4"))))["media"]
    assert metadata["code"] == "variable_frame_rate"
    assert metadata["actions"][0]["tool"] == "normalize_media"
    vfr = str(site.media / "vfr.mkv")
    before = Path(vfr).read_bytes()
    report = preflight(swap(site, media_path=vfr))
    media = by_name(report)["media"]
    assert media["status"] == "blocked" and media["code"] == "variable_timestamps"
    assert media["actions"][0]["tool"] == "normalize_media"
    assert media["actions"][0]["arguments"] == {"media_path": vfr}
    assert Path(vfr).read_bytes() == before
    assert not (site.state / "jobs").exists()  # No normalization job was queued.
    # A preview selects frames by their timestamps, so the same file is fine there.
    assert by_name(preflight(swap(site, mode="preview", media_path=vfr, frame=5)))["media"][
        "code"
    ] == ("media_ready")


def test_timing_is_read_only_through_the_requested_frames(site, monkeypatch):
    vfr = str(site.media / "vfr.mkv")
    ends, real = [], engine.timeline
    monkeypatch.setattr(
        engine, "timeline", lambda source, end=None: ends.append(end) or real(source, end)
    )
    # Frames 0-2 keep their spacing, so a render of them passes; the gap after them is unread.
    report = preflight(swap(site, media_path=vfr, max_frames=3))
    assert by_name(report)["media"]["code"] == "media_ready" and ends == [3]
    assert by_name(report)["media"]["evidence"]["frames_checked"] == 3
    preflight(swap(site, start_frame=4, max_frames=2))
    assert ends[-1] == 6
    preflight(swap(site, mode="preview", frame=7))
    assert ends[-1] == 8


def test_other_media_refusals_have_codes_and_only_fixable_ones_suggest_normalization(site):
    odd = by_name(preflight(swap(site, media_path=str(site.media / "odd.mp4"))))["media"]
    assert odd["code"] == "odd_dimensions" and odd["actions"][0]["tool"] == "normalize_media"
    empty = by_name(preflight(swap(site, start_frame=500)))["media"]
    assert empty["code"] == "empty_range" and empty["actions"] == []
    beyond = by_name(preflight(swap(site, mode="preview", frame=500)))["media"]
    assert beyond["code"] == "frame_out_of_range"
    image = by_name(preflight(swap(site, mode="preview", media_path=site.face, frame=1)))["media"]
    assert image["code"] == "frame_out_of_range" and image["message"] == "Images only have frame 0"
    report = {"expected_visible": [{"mapping": 0, "start_frame": 0, "end_frame": 30}]}
    visible = by_name(preflight(swap(site, report=report)))["media"]
    assert visible["code"] == "expected_visible_beyond_range"


# Storage: an estimate against observed free space, never a reservation


def test_low_and_insufficient_storage(site, monkeypatch):
    estimate = by_name(preflight(swap(site)))["storage"]["evidence"]["estimated_bytes"]
    usage = shutil.disk_usage(site.state)

    def free(amount):
        monkeypatch.setattr(control.shutil, "disk_usage", lambda path: usage._replace(free=amount))
        return by_name(preflight(swap(site)))["storage"]

    low = free(estimate + control.HEADROOM // 2)
    assert low["status"] == "warning" and low["code"] == "low_space"
    assert low["actions"][0]["tool"] == "prune_storage"
    assert low["actions"][0]["arguments"] == {"dry_run": True}
    short = free(estimate - 1)
    assert short["status"] == "blocked" and short["code"] == "insufficient_space"
    # Submission does not consult the preflight: it neither needs nor reserves its space.
    assert control.submit({"job_id": "unreserved", **swap(site)})["status"] in ("queued", "running")
    control.cancel("unreserved")


# Qualification: evidence for a combination; missing evidence is a warning, not a refusal


def test_unqualified_combinations_are_labelled_but_still_accepted(site, monkeypatch):
    request = swap(site, parameters={"SwapModelSelection": "SimSwap512"})
    report = preflight(request)
    checks = by_name(report)
    assert report["readiness"] == "warnings"
    qualification = checks["qualification"]
    assert qualification["status"] == "warning" and qualification["code"] == "unqualified"
    nearest = qualification["evidence"]["mappings"][0]["nearest"]
    assert nearest == {"record": "baseline-2026-11-07", "differs_in": ["swap_model"]}
    assert checks["request"]["status"] == "passed"
    assert control.submit({"job_id": "unqualified", **request})["status"] in ("queued", "running")
    control.cancel("unqualified")
    two = swap(site)
    two["mappings"][0]["source_faces"] = [site.source, site.face]
    differs = by_name(preflight(two))["qualification"]["evidence"]["mappings"][0]
    assert differs["nearest"]["differs_in"] == ["sources"]
    # Weight 0 excludes a source, so it does not count.
    two["mappings"][0]["source_weights"] = [1, 0]
    assert by_name(preflight(two))["qualification"]["code"] == "qualified"
    # Another hardware class, or none observed, is not covered by this evidence.
    monkeypatch.setattr(control, "output", lambda arguments: "8.6, 12288, 0\n")
    other = by_name(preflight(swap(site)))["qualification"]["evidence"]["mappings"][0]
    assert other["nearest"]["differs_in"] == ["hardware"]
    monkeypatch.setattr(control, "output", lambda arguments: None)
    report = preflight(swap(site))
    assert by_name(report)["gpu"]["code"] == "gpu_unobserved"
    unseen = by_name(report)["qualification"]["evidence"]["mappings"][0]
    assert unseen["nearest"]["differs_in"] == ["hardware (not observed)"]


def test_qualification_records_name_hardware_class_models_date_and_source():
    for record in common.QUALIFICATIONS:
        assert set(record["hardware"]) == {"architecture", "compute_capability", "vram_class"}
        assert record["fusion_revision"] == common.REVISION
        assert time.strptime(record["date"], "%Y-%m-%d")
        assert record["source"].startswith("VERIFICATION.md")
        assert record["swap_model"] and record["detector"] and record["modes"]
        for model in record["swap_model"]:
            assert model in common.SWAPPER_MODELS


# Fingerprints and rechecks


def test_fingerprints_follow_request_inputs_and_configuration(site):
    first = preflight(swap(site))["fingerprint"]
    assert preflight(swap(site))["fingerprint"] == first
    changed_request = preflight(swap(site, max_frames=3))["fingerprint"]
    assert changed_request["request"] != first["request"]
    assert changed_request["inputs"] == first["inputs"]
    shutil.copy(site.media / "other.png", site.source)
    changed_input = preflight(swap(site))["fingerprint"]
    assert changed_input["inputs"] != first["inputs"]
    assert changed_input["request"] == first["request"]
    (site.fusion / "model_assets/w600k_r50.onnx").write_bytes(b"another model")
    changed_model = preflight(swap(site))["fingerprint"]
    assert changed_model["configuration"] != first["configuration"]
    assert len({first["sha256"], changed_input["sha256"], changed_model["sha256"]}) == 3


def test_submission_rechecks_what_changed_after_a_ready_preflight(site):
    request = swap(site)
    digest = hashlib.sha256(Path(site.source).read_bytes()).hexdigest()
    request["mappings"][0]["source_sha256"] = [digest]
    assert preflight(request)["readiness"] == "ready"
    shutil.copy(site.media / "other.png", site.source)
    with pytest.raises(ValueError, match="no longer has SHA-256"):
        control.submit({"job_id": "changed", **request})
    os.unlink(site.clip)
    with pytest.raises(FileNotFoundError):
        control.submit({"job_id": "vanished", **swap(site)})
    assert not (site.state / "jobs/changed").exists()


# The optional GPU stage: ordinary queued analyze jobs


def analysis_job(report):
    jobs = by_name(report)["reference_analysis"]["evidence"]["jobs"]
    assert len(jobs) == 1
    return jobs[0]


def test_reference_analysis_is_durable_idempotent_and_reconnectable(site):
    request = swap(site, controls={"DetectorScoreSlider": 40})
    request["mappings"][0]["source_faces"] = [site.source, site.face]
    request["mappings"][0]["source_weights"] = [1, 0]  # Excluded sources are not analyzed.
    report = preflight(request, analyze=True)
    check = by_name(report)["reference_analysis"]
    assert report["readiness"] == "pending" and check["code"] == "analysis_pending"
    job = analysis_job(report)
    assert job["job_id"].startswith("preflight-") and job["mappings"] == [0]
    assert check["actions"][0]["tool"] == "job_status"
    stored = common.read_json(common.job_path(job["job_id"]) / "request.json")
    digest = hashlib.sha256(Path(site.source).read_bytes()).hexdigest()
    assert stored == {
        "kind": "analyze",
        "references": [site.source],
        "reference_sha256": [digest],
        "controls": {"DetectorScoreSlider": 40},
    }
    # An ordinary GPU job: interactive class, a queue sequence, the shared GPU lock.
    state = common.read_json(common.job_path(job["job_id"]) / "state.json")
    assert state["priority"] == "interactive" and state["sequence"] >= 1
    wait_file(common.job_path(job["job_id"]) / "entered")
    # Repeating the preflight finds the same job, and a new process (a reconnecting client)
    # sees it running.
    assert analysis_job(preflight(request, analyze=True))["job_id"] == job["job_id"]
    assert sorted(p.name for p in (site.state / "jobs").iterdir()) == [job["job_id"]]
    fresh = control_command(site, {"action": "status", "job_id": job["job_id"]})
    assert fresh["result"]["status"] == "running"
    # Results come from the completed job.
    result = {
        "usable": 1,
        "errors": 0,
        "warnings": 1,
        "references": [
            {
                "path": site.source,
                "findings": [{"code": "blurry", "severity": "warning", "message": "soft"}],
            }
        ],
        "consistency": {"available": False},
    }
    (site.state / "analysis-result.json").write_text(json.dumps(result))
    (common.job_path(job["job_id"]) / "release").touch()
    wait_for(job["job_id"], {"completed"})
    report = preflight(request, analyze=True)
    check = by_name(report)["reference_analysis"]
    assert report["readiness"] == "warnings" and check["code"] == "reference_warnings"
    assert analysis_job(report)["findings"] == [
        {"path": site.source, "code": "blurry", "severity": "warning", "message": "soft"}
    ]
    assert check["actions"][0]["arguments"] == {"job_id": job["job_id"], "name": "contact.jpg"}


def test_reference_analysis_results_cancellation_and_rerun(site):
    request = swap(site)
    job_id = analysis_job(preflight(request, analyze=True))["job_id"]
    control.cancel(job_id)
    wait_for(job_id, {"cancelled"})
    check = by_name(preflight(request, analyze=True))["reference_analysis"]
    assert check["status"] == "not_checked" and check["code"] == "analysis_cancelled"
    assert check["actions"][0] == {
        "description": check["actions"][0]["description"],
        "tool": "delete_job",
        "arguments": {"job_id": job_id},
    }
    control.delete_job(job_id)
    result = {"usable": 0, "errors": 1, "warnings": 0, "references": []}
    (site.state / "analysis-result.json").write_text(json.dumps(result))
    assert analysis_job(preflight(request, analyze=True))["job_id"] == job_id
    wait_file(common.job_path(job_id) / "entered")
    (common.job_path(job_id) / "release").touch()
    wait_for(job_id, {"completed"})
    report = preflight(request, analyze=True)
    assert report["readiness"] == "blocked"
    assert by_name(report)["reference_analysis"]["code"] == "reference_problems"
    # Another swap model is another analysis.
    model = swap(site, parameters={"SwapModelSelection": "SimSwap512"})
    other = analysis_job(preflight(model, analyze=True))
    assert other["job_id"] != job_id
    control.cancel(other["job_id"])


def test_reference_analysis_is_not_queued_for_a_blocked_request(site):
    report = preflight(swap(site, controls={"DetectorModelSelection": "Nope"}), analyze=True)
    check = by_name(report)["reference_analysis"]
    assert check["status"] == "not_checked" and check["code"] == "prerequisite_blocked"
    assert not (site.state / "jobs").exists()


# The MCP tool over stdio


def test_stdio_preflight_shares_submit_swaps_schema_and_answers(server, tmp_path, library):
    shutil.copy(library / "clip.mp4", tmp_path / "clip.mp4")
    shutil.copy(library / "face.png", tmp_path / "face.png")
    arguments = {
        "media_path": str(tmp_path / "clip.mp4"),
        "mappings": [
            {
                "target_face": str(tmp_path / "face.png"),
                "source_faces": [str(tmp_path / "face.png")],
            }
        ],
        "mode": "render",
        "max_frames": 5,
    }
    (result,) = call(server, ("preflight_swap", arguments))
    assert not result.isError, result.content
    report = result.structuredContent
    checks = by_name(report)
    assert report["readiness"] == "blocked"  # The fixture's Fusion is no git checkout.
    assert checks["fusion_revision"]["code"] == "fusion_incompatible"
    # The client sent its worker hashes, and the shim runs this checkout's control.py.
    assert checks["deployed_code"]["code"] == "code_matches"
    assert checks["media"]["code"] == "media_ready"
    assert checks["media"]["evidence"]["frames_checked"] == 5
    assert [p.name for p in (tmp_path / "state/jobs").iterdir()] == ["done"]  # Nothing queued.

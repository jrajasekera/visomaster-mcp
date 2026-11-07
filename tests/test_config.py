"""Workstation configuration file: validation, location and the legacy-variable guard."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import common
import pytest
from common import ConfigError, load_config
from conftest import ROOT, fake_fusion, write_config

EXAMPLE = ROOT / "config/remote-config.example.json"


@pytest.fixture
def media(tmp_path, monkeypatch):
    for variable in ("VM_CONFIG", *common.LEGACY):
        monkeypatch.delenv(variable, raising=False)
    root = tmp_path / "media"
    root.mkdir()
    return root


def edit(path, **changes):
    data = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    path.write_text(json.dumps(data))
    return path


def test_valid_file_resolves_defaults_and_identity(media, tmp_path):
    path = write_config(tmp_path / "c.json", media, state_dir=None)
    settings = load_config(path)
    assert settings.base == media.resolve()
    assert settings.state == media.resolve() / "runtime/visomaster-mcp"
    assert settings.state.is_dir() and settings.state.stat().st_mode & 0o777 == 0o700
    assert settings.lock_dir == settings.state and settings.gpu == "0"
    assert settings.runner == (str(media / "bin/runner"),)
    assert len(settings.sha256) == 64 and settings.sha256 == load_config(path).sha256
    edit(path, gpu="1")
    assert load_config(path).sha256 != settings.sha256


def test_fusion_dir_defaults_to_layout_convention(media, tmp_path):
    fake_fusion(media / "tools/visomaster-fusion")
    path = write_config(tmp_path / "c.json", media)
    edit(path, fusion_dir=None)
    assert load_config(path).app == (media / "tools/visomaster-fusion").resolve()


def test_missing_file_names_expected_path(tmp_path):
    with pytest.raises(ConfigError, match=str(tmp_path / "absent.json")):
        load_config(tmp_path / "absent.json")


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"media_root": "relative/dir"}, "media_root must be an absolute"),
        ({"media_root": "/nonexistent/visomaster"}, "not an existing directory"),
        ({"media_root": None}, "media_root is required"),
        ({"fusion_dir": "/nonexistent/fusion"}, "fusion_dir"),
        ({"state_dir": "relative"}, "state_dir must be an absolute"),
        ({"state_dir": "/tmp/outside-media"}, "inside media_root"),
        ({"version": 2}, "version must be 1"),
        ({"extra": 1}, "unknown keys extra"),
        ({"gpu": "0;x"}, "gpu must be"),
        ({"gpu": "../0"}, "gpu must be"),
        ({"gpu": ""}, "gpu must be"),
        ({"gpu": "00"}, "gpu must be"),
        ({"gpu": "GPU-12345678-aaaa-bbbb-cccc-1234567890ab"}, "gpu must be"),
        ({"gpu": 0}, "gpu must be"),
        ({"runner": None}, "runner is required"),
        ({"runner": []}, "runner is required"),
        ({"runner": "relative/runner"}, "absolute path to an executable"),
        ({"lock_dir": "/nonexistent/locks"}, "lock_dir"),
    ],
)
def test_invalid_values_are_rejected(media, tmp_path, changes, message):
    path = edit(write_config(tmp_path / "c.json", media), **changes)
    with pytest.raises(ConfigError, match=message) as error:
        load_config(path)
    assert str(path) in str(error.value) or "VM_" in str(error.value)


def test_state_dir_must_differ_from_media_root_and_avoid_fusion(media, tmp_path):
    path = write_config(tmp_path / "c.json", media)
    with pytest.raises(ConfigError, match="inside media_root"):
        load_config(edit(path, state_dir=str(media)))
    with pytest.raises(ConfigError, match="inside fusion_dir"):
        load_config(edit(path, state_dir=str(media / "fusion/runtime")))


def test_fusion_checkout_layout_files_are_required(media, tmp_path):
    path = write_config(tmp_path / "c.json", media)
    (media / "fusion/app/ui/widgets/swapper_layout_data.py").unlink()
    with pytest.raises(ConfigError, match="swapper_layout_data.py"):
        load_config(path)


def test_runner_must_be_executable_and_may_contain_spaces(media, tmp_path):
    runner = media / "my tools/fusion runner"
    runner.parent.mkdir()
    runner.write_text('#!/bin/sh\nexec python3 "$@"\n')
    path = write_config(tmp_path / "c.json", media, runner=runner)
    with pytest.raises(ConfigError, match="executable"):
        load_config(path)
    runner.chmod(0o755)
    edit(path, runner=[str(runner), "--flag with space"])
    assert load_config(path).runner == (str(runner), "--flag with space")
    edit(path, runner=str(runner))
    assert load_config(path).runner == (str(runner),)


def test_media_root_through_symlink_is_resolved(media, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(media)
    path = write_config(tmp_path / "c.json", media)
    edit(path, media_root=str(link))
    assert load_config(path).base == media.resolve()


def test_legacy_variables_must_match_or_be_absent(media, tmp_path, monkeypatch):
    path = write_config(tmp_path / "c.json", media)
    settings = load_config(path)
    monkeypatch.setenv("VM_BASE", str(media))
    monkeypatch.setenv("VM_GPU", "0")
    assert load_config(path) == settings
    monkeypatch.setenv("VM_STATE", str(tmp_path / "elsewhere"))
    with pytest.raises(ConfigError, match="VM_STATE=.* is no longer read"):
        load_config(path)
    monkeypatch.delenv("VM_STATE")
    monkeypatch.setenv("VM_GPU", "1")
    with pytest.raises(ConfigError, match="VM_GPU"):
        load_config(path)


def test_vm_config_override_and_relative_rejection(media, tmp_path, monkeypatch):
    path = write_config(tmp_path / "c.json", media)
    monkeypatch.setenv("VM_CONFIG", str(path))
    assert common.config_path() == path and load_config().path == path
    monkeypatch.setenv("VM_CONFIG", "relative.json")
    with pytest.raises(ConfigError, match="absolute"):
        common.config_path()


def located(code_directory):
    """config_path() as computed by a copy of common.py living in code_directory."""
    environment = {k: v for k, v in os.environ.items() if k != "VM_CONFIG"}
    return subprocess.run(
        [sys.executable, "-c", "import common; print(common.config_path())"],
        cwd=code_directory,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_location_in_release_legacy_and_symlinked_layouts(tmp_path):
    root = tmp_path / "deploy"
    release = root / "releases/abc"
    legacy = tmp_path / "flat/remote"
    for directory in (release, legacy):
        directory.mkdir(parents=True)
        shutil.copy(ROOT / "remote/common.py", directory)
    (root / "remote").symlink_to("releases/abc")
    expected = str((root / "remote-config.json").resolve())
    assert located(release) == expected
    assert located(root / "remote") == expected
    assert located(legacy) == str((tmp_path / "flat/remote-config.json").resolve())
    assert located(ROOT / "remote") == str(ROOT / "remote-config.json")


def test_import_never_raises_and_records_the_error(tmp_path):
    environment = dict(os.environ, VM_CONFIG=str(tmp_path / "missing.json"))
    result = subprocess.run(
        [sys.executable, "-c", "import common; print(common.CONFIG_ERROR); print(common.STATE)"],
        cwd=ROOT / "remote",
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "not found" in result.stdout and "None" in result.stdout


def test_example_config_loads_once_placeholders_are_mapped(tmp_path, monkeypatch):
    for variable in ("VM_CONFIG", *common.LEGACY):
        monkeypatch.delenv(variable, raising=False)
    text = EXAMPLE.read_text()
    example = json.loads(text)
    root = tmp_path / "srv"
    mapped = json.loads(text.replace("/srv/visomaster", str(root)))
    fake_fusion(Path(mapped["fusion_dir"]))
    runner = Path(mapped["runner"][0])
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("#!/bin/sh\n")
    runner.chmod(0o755)
    (tmp_path / "c.json").write_text(json.dumps(mapped))
    settings = load_config(tmp_path / "c.json")
    assert set(example) <= common.CONFIG_KEYS and settings.base == root.resolve()

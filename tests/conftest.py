"""Shared helpers: workstation configuration files and their in-process application."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "remote"))

LAYOUTS = ("settings", "common", "swapper", "face_editor", "denoiser")
WORKER = (
    "#!/usr/bin/env python3\n"
    + """
import json,sys,time
from pathlib import Path
job = Path(sys.argv[2])
request=json.loads((job/'request.json').read_text())
if request.get('fail'): sys.exit(9)
if request.get('error'):
    (job/'error.json').write_text(json.dumps({'type':'ValueError','message':request['error']}))
    sys.exit(1)
(job/'entered').write_text(sys.argv[1])
while not (job/'release').exists(): time.sleep(.05)
(job/'result.json').write_text('{"ok":true}')
"""
)


def fake_fusion(path):
    """A directory that passes the Fusion checkout checks (git marker and layout files)."""
    widgets = path / "app/ui/widgets"
    widgets.mkdir(parents=True, exist_ok=True)
    (path / ".git").mkdir(exist_ok=True)
    for name in LAYOUTS:
        (widgets / f"{name}_layout_data.py").write_text("{}\n")
    return path


def write_config(path, media_root, runner=None, **values):
    """Write remote-config.json for a media root with a fake Fusion checkout and runner."""
    media_root = Path(media_root)
    fusion = fake_fusion(Path(values.pop("fusion_dir", media_root / "fusion")))
    if runner is None:
        runner = media_root / "bin/runner"
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text(WORKER)
        runner.chmod(0o755)
    config = {
        "version": 1,
        "media_root": str(media_root),
        "fusion_dir": str(fusion),
        "state_dir": str(media_root / "state"),
        "gpu": "0",
        "runner": [str(runner)],
        **values,
    }
    config = {key: value for key, value in config.items() if value is not None}
    Path(path).write_text(json.dumps(config))
    return Path(path)


def apply_config(monkeypatch, path):
    """Point VM_CONFIG at path (for subprocesses) and apply it to the imported modules."""
    import common
    import control

    monkeypatch.setenv("VM_CONFIG", str(path))
    settings = common.load_config(path)
    values = {
        "SETTINGS": settings,
        "CONFIG_ERROR": None,
        "BASE": settings.base,
        "APP": settings.app,
        "STATE": settings.state,
        "GPU": settings.gpu,
        "RUNNER": list(settings.runner),
        "LOCK_DIR": settings.lock_dir,
    }
    for module in (common, control):
        for name, value in values.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, value)
    return settings

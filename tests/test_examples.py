"""Client and workstation examples parse, use placeholders and follow the runner contract."""

import json
import subprocess
import tomllib

from conftest import ROOT

CONFIG = ROOT / "config"
REQUIRED = {"VISOMASTER_HOST": "gpu-host", "VISOMASTER_REMOTE": "/srv/visomaster/mcp"}


def test_client_examples_parse_and_use_placeholders():
    desktop = json.loads((CONFIG / "claude-desktop.json").read_text())["mcpServers"]["visomaster"]
    codex = tomllib.loads((CONFIG / "codex.toml").read_text())["mcp_servers"]["visomaster"]
    for entry in (desktop, codex):
        assert entry["command"] == "<checkout>/.venv/bin/visomaster-mcp"
        assert entry["env"] == REQUIRED
    assert "does not work unmodified" in (CONFIG / "codex.toml").read_text()


def test_remote_example_lists_only_known_keys():
    import common

    example = json.loads((CONFIG / "remote-config.example.json").read_text())
    assert set(example) <= common.CONFIG_KEYS and example["version"] == 1
    assert all(str(v).startswith("/srv/visomaster") for k, v in example.items() if "dir" in k and v)


def test_runner_example_is_valid_sh_and_execs(tmp_path):
    runner = CONFIG / "fusion-runner.example.sh"
    subprocess.run(["sh", "-n", str(runner)], check=True)
    text = runner.read_text()
    assert 'exec "$FUSION_PYTHON" "$@"' in text
    code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert not any("setsid" in line or "nohup" in line for line in code)
    # With a stand-in Fusion layout, the runner execs the interpreter with the original argv.
    fusion = tmp_path / "fusion"
    (fusion / ".venv/bin").mkdir(parents=True)
    (fusion / ".venv/lib/python3.12/site-packages/nvidia/cudnn/lib").mkdir(parents=True)
    python = fusion / ".venv/bin/python"
    python.write_text('#!/bin/sh\nprintf "%s|" "$$" "$LD_LIBRARY_PATH" "$@"\n')
    python.chmod(0o755)
    adapted = tmp_path / "runner"
    adapted.write_text(text.replace("/srv/visomaster/fusion", str(fusion)))
    adapted.chmod(0o755)
    probe = subprocess.run(
        ["sh", "-c", f'echo "$$|"; exec "{adapted}" engine.py "job dir"'],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    shell_pid, rest = probe.stdout.split("|\n", 1)
    pid, libraries, *argv = rest.rstrip("|").split("|")
    assert pid == shell_pid  # exec kept the same process
    assert libraries.endswith("nvidia/cudnn/lib") and argv == ["engine.py", "job dir"]

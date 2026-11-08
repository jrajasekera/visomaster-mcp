"""Client transport settings: validation, SSH argv, interpreter quoting and actionable errors."""

import asyncio
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import pytest
from conftest import ROOT, write_config
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from visomaster_mcp import transport

VALID = {"VISOMASTER_HOST": "gpu-host", "VISOMASTER_REMOTE": "/srv/visomaster/mcp"}
# Records the remote command, then runs it locally (the command is always the last argument).
SHIM = '#!/bin/sh\nfor command; do :; done\nprintf "%s\\n" "$*" >> "$SHIM_LOG"\nexec /bin/sh -c "$command"\n'


def clean_environment(**values):
    # Only what the client needs, so failure output never shows unrelated environment values.
    environment = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG") if k in os.environ}
    return {**environment, **values}


def call(environment, *calls):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "visomaster_mcp.server"], env=environment
    )

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                results = [await session.call_tool(name, arguments) for name, arguments in calls]
                return tools, results

    return asyncio.run(run())


def test_valid_settings_and_defaults():
    settings = transport.load(VALID)
    assert settings.python == "python3" and settings.ssh == "ssh"
    assert settings.downloads == Path("~/Downloads/visomaster-mcp").expanduser()
    prefix = settings.ssh_prefix()
    assert prefix[0] == "ssh" and prefix[-2:] == ["--", "gpu-host"]
    assert "ControlPath=~/.ssh/visomaster-%C" in prefix and "BatchMode=yes" in prefix
    assert settings.control("upload", "x") == [
        "python3",
        "/srv/visomaster/mcp/remote/control.py",
        "upload",
        "x",
    ]
    assert transport.load({**VALID, "VISOMASTER_REMOTE": "/srv/mcp/"}).remote == "/srv/mcp"


@pytest.mark.parametrize(
    "changes, variable",
    [
        ({"VISOMASTER_HOST": ""}, "VISOMASTER_HOST"),
        ({"VISOMASTER_REMOTE": ""}, "VISOMASTER_REMOTE"),
        ({"VISOMASTER_HOST": "-oProxyCommand=touch /tmp/x"}, "VISOMASTER_HOST"),
        ({"VISOMASTER_HOST": "gpu host"}, "VISOMASTER_HOST"),
        ({"VISOMASTER_HOST": "gpu\thost"}, "VISOMASTER_HOST"),
        ({"VISOMASTER_REMOTE": "relative/mcp"}, "VISOMASTER_REMOTE"),
        ({"VISOMASTER_REMOTE": "/srv/../etc"}, "VISOMASTER_REMOTE"),
        ({"VISOMASTER_REMOTE": "/srv/it's"}, "VISOMASTER_REMOTE"),
        ({"VISOMASTER_REMOTE": '/srv/"x"'}, "VISOMASTER_REMOTE"),
        ({"VISOMASTER_REMOTE_PYTHON": "python3 -c"}, "VISOMASTER_REMOTE_PYTHON"),
        ({"VISOMASTER_REMOTE_PYTHON": "python3;id"}, "VISOMASTER_REMOTE_PYTHON"),
        ({"VISOMASTER_REMOTE_PYTHON": "'python3'"}, "VISOMASTER_REMOTE_PYTHON"),
        ({"VISOMASTER_REMOTE_PYTHON": "-python"}, "VISOMASTER_REMOTE_PYTHON"),
        ({"VISOMASTER_REMOTE_PYTHON": "relative/python"}, "VISOMASTER_REMOTE_PYTHON"),
        ({"VISOMASTER_SSH": "ssh\nid"}, "VISOMASTER_SSH"),
        ({"VISOMASTER_DOWNLOADS": "relative/downloads"}, "VISOMASTER_DOWNLOADS"),
    ],
)
def test_invalid_settings_name_the_variable(changes, variable):
    environment = {k: v for k, v in {**VALID, **changes}.items() if v}
    with pytest.raises(transport.TransportError, match=variable):
        transport.load(environment)


def test_custom_interpreter_is_quoted_once():
    settings = transport.load({**VALID, "VISOMASTER_REMOTE_PYTHON": "/opt/py-3.12/bin/python3"})
    command = settings.command(settings.control("upload", "a b", "$(id)"))
    assert shlex.split(command) == [
        "/opt/py-3.12/bin/python3",
        "/srv/visomaster/mcp/remote/control.py",
        "upload",
        "a b",
        "$(id)",
    ]


def test_ssh_setup_failures_get_a_hint():
    settings = transport.load(VALID)
    hinted = settings.hint("ssh: Could not resolve hostname gpu-host: nodename nor servname")
    assert (
        "Check VISOMASTER_HOST (gpu-host)" in hinted and "BatchMode=yes -- gpu-host true" in hinted
    )
    assert settings.hint("Traceback: remote failure") == "Traceback: remote failure"


def test_check_command_prints_validated_values_or_refuses():
    import subprocess

    script = ROOT / "src/visomaster_mcp/transport.py"
    ok = subprocess.run(
        ["python3", str(script), "check"],
        env=clean_environment(**VALID, VISOMASTER_REMOTE_PYTHON="python3.12"),
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0
    assert ok.stdout.splitlines() == ["gpu-host", "/srv/visomaster/mcp", "python3.12", "ssh"]
    bad = subprocess.run(
        ["python3", str(script), "check"],
        env=clean_environment(VISOMASTER_REMOTE="/srv/x"),
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2 and "VISOMASTER_HOST not set" in bad.stderr and not bad.stdout


def test_unconfigured_server_still_lists_tools_and_explains(tmp_path):
    tools, (health, upload) = call(
        clean_environment(),
        ("health_check", {}),
        ("upload_media", {"local_path": str(ROOT / "README.md")}),
    )
    assert len(tools) == 22
    assert (
        health.isError and "VISOMASTER_HOST and VISOMASTER_REMOTE not set" in health.content[0].text
    )
    assert "README" in health.content[0].text and upload.isError


def test_ssh_failure_message_carries_hint(tmp_path):
    shim = tmp_path / "ssh"
    shim.write_text("#!/bin/sh\necho 'ssh: Could not resolve hostname gpu-host' >&2\nexit 255\n")
    shim.chmod(0o755)
    _, (health,) = call(clean_environment(**VALID, VISOMASTER_SSH=str(shim)), ("health_check", {}))
    assert health.isError and "Check VISOMASTER_HOST (gpu-host)" in health.content[0].text


def test_new_upload_uses_custom_interpreter_for_both_commands(tmp_path):
    shim = tmp_path / "ssh"
    shim.write_text(SHIM)
    shim.chmod(0o755)
    wrapper = tmp_path / "bin/remote-python"
    wrapper.parent.mkdir()
    wrapper.write_text(f'#!/bin/sh\necho "$@" >> "{tmp_path}/python.log"\nexec python3 "$@"\n')
    wrapper.chmod(0o755)
    config = write_config(tmp_path / "remote-config.json", tmp_path)
    media = tmp_path / "face.png"
    media.write_bytes(os.urandom(4096))
    environment = clean_environment(
        VISOMASTER_SSH=str(shim),
        VISOMASTER_HOST="custom-alias",
        VISOMASTER_REMOTE=str(ROOT),
        VISOMASTER_REMOTE_PYTHON=str(wrapper),
        VM_CONFIG=str(config),
        SHIM_LOG=str(tmp_path / "ssh.log"),
    )
    _, (uploaded,) = call(environment, ("upload_media", {"local_path": str(media)}))
    assert not uploaded.isError and not uploaded.structuredContent["reused"]
    digest = hashlib.sha256(media.read_bytes()).hexdigest()
    assert Path(uploaded.structuredContent["path"]).name == digest + ".png"
    invocations = (tmp_path / "python.log").read_text().splitlines()
    assert invocations == [
        f"{ROOT}/remote/control.py",
        f"{ROOT}/remote/control.py upload {digest} .png 4096",
    ]
    for line in (tmp_path / "ssh.log").read_text().splitlines():
        assert " -- custom-alias " in f" {line} " and line.split(" -- custom-alias ")[1].startswith(
            shlex.quote(str(wrapper))
        )


def test_new_client_accepts_health_from_an_older_release(monkeypatch):
    from visomaster_mcp import server

    async def command(action):
        return {"release": "older", "code": {}, "revision": "x", "gpu": "0"}

    monkeypatch.setattr(server, "command", command)
    result = asyncio.run(server.health_check())
    assert result["code_matches"] is False and "config_path" not in result
    assert json.dumps(result)

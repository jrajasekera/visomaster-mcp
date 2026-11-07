"""Client transport settings shared by the MCP server and scripts/deploy.sh (stdlib only).

`python3 src/visomaster_mcp/transport.py check` validates the environment exactly as the
server does and prints host, deployment directory, remote interpreter and SSH program, one
per line, or exits with status 2 and the same message the server would return.
"""

import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

SETUP = (
    "Set VISOMASTER_HOST (SSH alias of the GPU workstation) and VISOMASTER_REMOTE "
    "(absolute deployment directory) in the MCP client registration; see README, Setup"
)
HOST = re.compile(r"[A-Za-z0-9._@:%\[\]-]+")
REMOTE = re.compile(r"/[A-Za-z0-9_./-]+")
COMMAND = re.compile(r"[A-Za-z0-9._-]+")
# ssh's own failures (exit 255) that usually mean the alias or its key setup is wrong.
SSH_SETUP_FAILURES = re.compile(
    r"Could not resolve hostname|Permission denied|Host key verification failed|"
    r"Connection refused|Connection timed out|No route to host|Operation timed out|"
    r"Name or service not known|nodename nor servname",
    re.I,
)


class TransportError(ValueError):
    """The client environment does not describe a usable workstation connection."""


@dataclass(frozen=True)
class Transport:
    host: str
    remote: str
    python: str
    ssh: str
    downloads: Path

    def ssh_prefix(self, multiplex=True):
        """SSH argv up to and including the host; "--" ends option parsing before it."""
        options = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if multiplex:
            # Reuse one authenticated connection; %C keeps the socket path under the Unix limit.
            options += [
                "-o",
                "ControlMaster=auto",
                "-o",
                "ControlPersist=120",
                "-o",
                "ControlPath=~/.ssh/visomaster-%C",
            ]
        return [self.ssh, *options, "--", self.host]

    def control(self, *arguments):
        """Remote argv that runs the deployed control.py with the configured interpreter."""
        return [self.python, f"{self.remote}/remote/control.py", *arguments]

    def command(self, arguments):
        return shlex.join(arguments)

    def hint(self, message):
        """Append the connection check to messages that look like SSH setup failures."""
        if SSH_SETUP_FAILURES.search(message):
            return (
                f"{message}\nCheck VISOMASTER_HOST ({self.host}) with: "
                f"{shlex.join([self.ssh, '-T', '-o', 'BatchMode=yes', '--', self.host, 'true'])}"
            )
        return message


def _clean(value):
    return bool(value) and not any(ord(c) < 32 or ord(c) == 127 for c in value)


def load(environ=None):
    """Validate the VISOMASTER_* environment; raise TransportError naming the variable."""
    environ = os.environ if environ is None else environ
    host = environ.get("VISOMASTER_HOST", "")
    remote = environ.get("VISOMASTER_REMOTE", "")
    if not host or not remote:
        missing = [
            n for n, v in (("VISOMASTER_HOST", host), ("VISOMASTER_REMOTE", remote)) if not v
        ]
        raise TransportError(f"{' and '.join(missing)} not set. {SETUP}")
    if host.startswith("-") or not HOST.fullmatch(host):
        raise TransportError(
            "VISOMASTER_HOST must be an SSH alias or host name (letters, digits, ._@:%[]-; "
            "no spaces and no leading '-')"
        )
    if not REMOTE.fullmatch(remote) or ".." in remote.split("/"):
        raise TransportError(
            "VISOMASTER_REMOTE must be an absolute workstation path of letters, digits and "
            "_./- without '..' components"
        )
    remote = remote.rstrip("/") or "/"
    python = environ.get("VISOMASTER_REMOTE_PYTHON") or "python3"
    bare = COMMAND.fullmatch(python) and not python.startswith("-")
    absolute = REMOTE.fullmatch(python) and ".." not in python.split("/")
    if not (bare or absolute):
        raise TransportError(
            "VISOMASTER_REMOTE_PYTHON must be a command name such as python3 or an absolute "
            "path without spaces, quotes or shell characters"
        )
    ssh = environ.get("VISOMASTER_SSH") or "ssh"
    if not _clean(ssh):
        raise TransportError(
            "VISOMASTER_SSH must be a program name or path without control characters"
        )
    downloads = environ.get("VISOMASTER_DOWNLOADS") or str(Path("~/Downloads/visomaster-mcp"))
    downloads = Path(downloads).expanduser()
    if not downloads.is_absolute():
        raise TransportError("VISOMASTER_DOWNLOADS must be an absolute path (or start with ~)")
    return Transport(host, remote, python, ssh, downloads)


def main(argv):
    if argv[1:] != ["check"]:
        print("usage: transport.py check", file=sys.stderr)
        return 2
    try:
        transport = load()
    except TransportError as error:
        print(error, file=sys.stderr)
        return 2
    print("\n".join((transport.host, transport.remote, transport.python, transport.ssh)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

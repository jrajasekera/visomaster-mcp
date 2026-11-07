"""A fresh Beads workspace with the project's actor setting exports no paths or user names."""

import json
import shutil
import subprocess
import sys

import pytest
from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import privacy_audit  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("br") is None, reason="br is not installed")


def br(cwd, *arguments):
    return subprocess.run(
        ["br", *arguments], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_actor_and_scrub_keep_the_export_neutral(tmp_path):
    beads = tmp_path / ".beads"
    beads.mkdir()
    shutil.copy(ROOT / ".beads/config.yaml", beads)
    shutil.copy(ROOT / ".beads/metadata.json", beads)
    (beads / "issues.jsonl").write_text("")
    assert br(tmp_path, "config", "get", "actor") == "maintainer"
    first = br(tmp_path, "q", "first issue")
    second = br(tmp_path, "q", "second issue")
    br(tmp_path, "dep", "add", second, first)
    br(tmp_path, "comments", "add", first, "a comment")
    # Simulate br stamping an absolute checkout path, then scrub it as AGENTS.md instructs.
    br(tmp_path, "update", first, "--source-repo-path", str(tmp_path))
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/beads_scrub.py")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    br(tmp_path, "update", second, "--priority", "1")  # A later mutation and auto-flush.
    text = (beads / "issues.jsonl").read_text()
    records = [json.loads(line) for line in text.splitlines()]
    assert {r["id"] for r in records} == {first, second}
    assert all("source_repo_path" not in r for r in records)
    _, actors = privacy_audit.load_allow()
    assert privacy_audit.scan_beads(".beads/issues.jsonl", text, actors) == []
    assert {r["created_by"] for r in records} == {"maintainer"}

"""Privacy audit rules against synthetic repositories (fixture values are assembled at runtime
so this file itself never matches the rules it tests)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
AUDIT = ROOT / "scripts/privacy_audit.py"
sys.path.insert(0, str(ROOT / "scripts"))
import privacy_audit  # noqa: E402

SECRET = "zz" + "-owner-" + "value"


def git(cwd, *arguments):
    subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True)


def repository(path, files):
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test")
    for name, content in files.items():
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content if isinstance(content, bytes) else content.encode())
    git(path, "add", "-A")
    git(path, "commit", "-qm", "fixture")
    return path


def audit(cwd, *arguments, env=None):
    environment = {k: v for k, v in os.environ.items() if k != "VISOMASTER_AUDIT_PATTERNS"}
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, str(AUDIT), *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=environment,
    )


GENERIC_CASES = {
    "home-path": "path = '/" + "Users/someone/project'\n",
    "email": "contact: someone" + "@" + "mail.test.io\n",
    "private-ipv4": "host 192." + "168.4.20\n",
    "local-hostname": "ssh gpu-box" + ".local\n",
    "private-key": "-----BEGIN " + "OPENSSH PRIVATE KEY-----\n",
    "github-token": "token gh" + "p_" + "a" * 36 + "\n",
    "api-key": "key s" + "k-" + "b" * 32 + "\n",
    "aws-key": "AK" + "IA" + "ABCDEFGHIJKLMNOP\n",
    "slack-token": "xo" + "xb-" + "1234567890-abc\n",
    "bearer-token": "Authorization: Bear" + "er " + "c" * 30 + "\n",
    "credential-url": "https://user:" + "pw" + "@host.test/x\n",
}


@pytest.mark.parametrize("rule", sorted(GENERIC_CASES))
def test_each_generic_rule_flags_without_printing_the_value(tmp_path, rule):
    repo = repository(tmp_path / "repo", {"notes.txt": GENERIC_CASES[rule]})
    result = audit(repo, "--generic-only")
    assert result.returncode == 1
    assert f"notes.txt:1: {rule}" in result.stdout
    assert GENERIC_CASES[rule].strip() not in result.stdout
    shown = audit(repo, "--generic-only", "--show-values")
    assert "notes.txt:1" in shown.stdout and "'" in shown.stdout


def test_clean_repository_and_placeholders_pass(tmp_path):
    repo = repository(
        tmp_path / "repo",
        {
            "README.md": "Use gpu-host and /srv/visomaster; mail admin@example.com.\n"
            "Clone git@github.com:owner/project.git; see AGENTS.local.md\n",
            "docs/images/diagram.png": b"\x89PNG\0",
        },
    )
    result = audit(repo, "--generic-only")
    assert result.returncode == 0, result.stdout


def test_file_rules_cover_media_weights_keys_and_size(tmp_path):
    repo = repository(
        tmp_path / "repo",
        {
            "clip.mp4": b"\0video",
            "model.onnx": b"\0weights",
            "deploy.pem": "x",
            ".env": "A=1",
            ".env.example": "A=",
            "shot.png": b"\x89PNG\0",
            "big.txt": "a" * (privacy_audit.MAX_BYTES + 1),
            "uv.lock": "a" * (privacy_audit.MAX_BYTES + 1),
        },
    )
    out = audit(repo, "--generic-only").stdout
    for name in ("clip.mp4", "model.onnx", "deploy.pem", ".env"):
        assert f"{name}: tracked-media-weight-or-key" in out
    assert "shot.png: tracked-image-outside-docs-images" in out
    assert "big.txt: large-file" in out
    assert ".env.example" not in out and "uv.lock" not in out


def test_beads_export_rules(tmp_path):
    records = [
        {"id": "x-1", "created_by": "maintainer", "source_repo_path": "."},
        {"id": "x-2", "created_by": "maintainer", "source_repo_path": "/" + "srv/checkout"},
        {
            "id": "x-3",
            "created_by": "local-person",
            "dependencies": [{"created_by": "maintainer"}],
            "comments": [{"author": "codex"}],
        },
    ]
    text = "\n".join(json.dumps(r) for r in records) + "\n"
    repo = repository(tmp_path / "repo", {".beads/issues.jsonl": text})
    out = audit(repo, "--generic-only").stdout
    assert ".beads/issues.jsonl:2: beads-source-repo-path" in out
    assert ".beads/issues.jsonl:3: beads-actor" in out
    assert ":1:" not in out and "local-person" not in out


def test_pattern_parsing():
    patterns = privacy_audit.parse_patterns("# comment\n\nLiteral.Value\nre:ab[0-9]+c\n")
    assert len(patterns) == 2
    assert patterns[0].search("x literal.value y") and not patterns[0].search("literalXvalue")
    assert patterns[1].search("ab12c") and not patterns[1].search("abc")


def test_pattern_file_resolves_to_common_dir_in_linked_worktrees(tmp_path, monkeypatch):
    monkeypatch.delenv("VISOMASTER_AUDIT_PATTERNS", raising=False)
    repo = repository(tmp_path / "repo", {"a.txt": f"mentions {SECRET}\n"})
    git(repo, "worktree", "add", "-q", str(tmp_path / "linked"))
    common = Path(
        subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    (common / "info").mkdir(exist_ok=True)
    (common / "info/privacy-patterns").write_text(f"# local\n{SECRET}\n")
    for checkout in (repo, tmp_path / "linked"):
        assert privacy_audit.pattern_file(checkout) == common / "info/privacy-patterns"
        result = audit(checkout)
        assert result.returncode == 1 and "a.txt:1: owner-pattern #1" in result.stdout
        assert SECRET not in result.stdout
    override = tmp_path / "patterns"
    override.write_text("nothing-matches-this\n")
    result = audit(tmp_path / "linked", env={"VISOMASTER_AUDIT_PATTERNS": str(override)})
    assert result.returncode == 0, result.stdout


def test_tracked_tree_passes_generic_rules():
    result = audit(ROOT, "--generic-only")
    assert result.returncode == 0, result.stdout


def test_tracked_tree_passes_owner_patterns_when_present():
    patterns = os.environ.get("VISOMASTER_AUDIT_PATTERNS")
    path = Path(patterns) if patterns else privacy_audit.pattern_file(ROOT)
    if not path.exists():
        pytest.skip("no local owner pattern file (expected in a clean clone)")
    result = audit(ROOT, env={"VISOMASTER_AUDIT_PATTERNS": str(path)})
    assert result.returncode == 0, result.stdout


def test_staged_and_history_scopes(tmp_path):
    repo = repository(tmp_path / "repo", {"a.txt": "clean\n"})
    patterns = tmp_path / "patterns"
    patterns.write_text(SECRET + "\n")
    env = {"VISOMASTER_AUDIT_PATTERNS": str(patterns)}
    (repo / "a.txt").write_text(f"{SECRET}\n")
    assert audit(repo, "--staged", env=env).returncode == 0  # Not staged yet.
    git(repo, "add", "a.txt")
    assert "owner-pattern #1" in audit(repo, "--staged", env=env).stdout
    git(repo, "commit", "-qm", "leak")
    (repo / "a.txt").write_text("clean again\n")
    git(repo, "commit", "-qam", "fix")
    assert audit(repo, env=env).returncode == 0
    history = audit(repo, "--history", env=env)
    assert history.returncode == 1
    assert "commits scanned: 3" in history.stdout
    assert "owner-pattern #1: 1 commits" in history.stdout
    assert "@example.com (6)" in history.stdout and SECRET not in history.stdout

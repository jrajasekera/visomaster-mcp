"""Scan tracked content for identifying values before it is committed or published.

Usage: uv run --locked python scripts/privacy_audit.py [--tree | --staged | --history]
       [--patterns FILE] [--generic-only] [--show-values]

Generic rules (home paths, email addresses, private addresses, token formats, credential
URLs, media/weight/key files, large files and the Beads export) are built in. Owner-specific
values are never committed: they are read from $VISOMASTER_AUDIT_PATTERNS or from
<git common dir>/info/privacy-patterns, one literal per line, or a regex prefixed with "re:".
"#" starts a comment. Literal patterns match case-insensitively.

Findings name the file, line, rule and pattern index, never the matched value, unless
--show-values is given (local use only). The exit status is 1 when anything is found.
A clean run means "no findings for these categories", not proof that nothing remains.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ALLOW_FILE = Path(__file__).with_name("privacy_allow.txt")
MAX_BYTES = 5 * 1024 * 1024
SIZE_EXEMPT = {"uv.lock"}
IMAGE_EXEMPT_PREFIX = "docs/images/"
BEADS_EXPORT = ".beads/issues.jsonl"
MEDIA = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".wav", ".mp3", ".flac", ".aac"}
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic"}
WEIGHTS = {".onnx", ".pth", ".pt", ".ckpt", ".safetensors", ".engine", ".trt", ".dfm", ".bin"}
KEYS = {".pem", ".key", ".p12", ".pfx", ".keystore", ".jks"}
KEY_NAMES = re.compile(r"^(id_(rsa|dsa|ecdsa|ed25519)(\.pub)?|\.env(\..+)?)$")
SAFE_ENV_NAMES = {".env.example"}
EXAMPLE_DOMAINS = re.compile(r"(^|\.)example\.(com|org|net)$", re.I)

# Each rule is (name, compiled regex). Matches are checked against the allowlist.
GENERIC = [
    ("home-path", re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+")),
    (
        "email",
        re.compile(
            r"[A-Za-z0-9._%+-]{1,64}@([A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})"
        ),
    ),
    (
        "private-ipv4",
        re.compile(
            r"(?<![\d.])(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168|169\.254)"
            r"\.\d{1,3}\.\d{1,3}(?![\d.])"
        ),
    ),
    ("local-hostname", re.compile(r"\b[A-Za-z0-9-]+\.local\b(?![.\w-])")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("api-key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}")),
    ("aws-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack-token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("bearer-token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("credential-url", re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@'\"]+:[^/\s@'\"]+@")),
]


def git(*arguments, data=None, cwd=None):
    return subprocess.run(
        ["git", *arguments], input=data, capture_output=True, check=True, cwd=cwd
    ).stdout


def common_dir(cwd=None):
    path = git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=cwd)
    return Path(path.decode().strip())


def pattern_file(cwd=None):
    """$VISOMASTER_AUDIT_PATTERNS, else the clone-wide file shared by all linked worktrees."""
    override = os.environ.get("VISOMASTER_AUDIT_PATTERNS")
    if override:
        return Path(override).expanduser()
    return common_dir(cwd) / "info" / "privacy-patterns"


def parse_patterns(text):
    patterns = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("re:"):
            patterns.append(re.compile(line[3:]))
        else:
            patterns.append(re.compile(re.escape(line), re.I))
    return patterns


def load_allow(path=ALLOW_FILE):
    """Return (allowed literal values, allowed Beads actor names)."""
    values, actors = set(), set()
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            kind, _, value = line.partition(" ")
            if kind == "actor":
                actors.add(value.strip())
            elif kind == "allow":
                values.add(value.strip())
    return values, actors


class Finding:
    def __init__(self, path, line, rule, index=None, value=None):
        self.path, self.line, self.rule, self.index, self.value = path, line, rule, index, value

    def render(self, show):
        where = f"{self.path}:{self.line}" if self.line else self.path
        label = self.rule if self.index is None else f"{self.rule} #{self.index}"
        return f"{where}: {label}" + (f": {self.value!r}" if show and self.value else "")


def scan_text(path, text, owner, allowed):
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        for name, regex in GENERIC:
            for match in regex.finditer(line):
                value = match.group(0)
                if value in allowed:
                    continue
                if name == "email" and EXAMPLE_DOMAINS.search(match.group(1)):
                    continue
                findings.append(Finding(path, number, name, value=value))
        for index, regex in enumerate(owner, 1):
            match = regex.search(line)
            if match:
                findings.append(Finding(path, number, "owner-pattern", index, match.group(0)))
    return findings


def scan_file_name(path, size):
    name = path.rsplit("/", 1)[-1]
    suffix = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    findings = []
    if suffix in MEDIA | WEIGHTS | KEYS or (KEY_NAMES.match(name) and name not in SAFE_ENV_NAMES):
        findings.append(Finding(path, 0, "tracked-media-weight-or-key"))
    elif suffix in IMAGES and not path.startswith(IMAGE_EXEMPT_PREFIX):
        findings.append(Finding(path, 0, "tracked-image-outside-docs-images"))
    if size > MAX_BYTES and path not in SIZE_EXEMPT:
        findings.append(Finding(path, 0, "large-file"))
    return findings


def scan_beads(path, text, actors):
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            findings.append(Finding(path, number, "beads-unparseable"))
            continue
        if record.get("source_repo_path") not in (None, "", "."):
            findings.append(Finding(path, number, "beads-source-repo-path"))
        names = [record.get("created_by"), record.get("assignee"), record.get("owner")]
        names += [c.get("author") for c in record.get("comments") or []]
        names += [d.get("created_by") for d in record.get("dependencies") or []]
        for name in names:
            if name not in (None, "") and name not in actors:
                findings.append(Finding(path, number, "beads-actor", value=name))
    return findings


def scan_blobs(blobs, owner, allowed, actors):
    """blobs: iterable of (path, bytes). Returns findings."""
    findings = []
    for path, data in blobs:
        findings += scan_file_name(path, len(data))
        if b"\0" in data[:8192] or (len(data) > MAX_BYTES and path not in SIZE_EXEMPT):
            continue  # Binary and oversized content is covered by the file-name rules.
        text = data.decode("utf-8", errors="replace")
        findings += scan_text(path, text, owner, allowed)
        if path == BEADS_EXPORT:
            findings += scan_beads(path, text, actors)
    return findings


def tracked_paths(cwd=None):
    return [p for p in git("ls-files", "-z", cwd=cwd).decode().split("\0") if p]


def tree_blobs(root):
    for path in tracked_paths(root):
        file = Path(root) / path
        if file.is_file() and not file.is_symlink():
            yield path, file.read_bytes()


def staged_blobs(root):
    for line in git("ls-files", "-s", "-z", cwd=root).decode().split("\0"):
        if not line:
            continue
        meta, path = line.split("\t", 1)
        mode, sha, _ = meta.split()
        if mode.startswith("100"):
            yield path, git("cat-file", "blob", sha, cwd=root)


def history(root, owner, allowed, actors):
    """Count commits whose tree or message matches each rule; redact identities to domains."""
    commits = git("rev-list", "--all", cwd=root).decode().split()
    cache, counts = {}, Counter()
    for commit in commits:
        rules = set()
        message = git("log", "-1", "--format=%B", commit, cwd=root).decode(errors="replace")
        for finding in scan_text("<message>", message, owner, allowed):
            rules.add((finding.rule, finding.index))
        listing = git("ls-tree", "-r", "-z", commit, cwd=root).decode().split("\0")
        for entry in listing:
            if not entry:
                continue
            meta, path = entry.split("\t", 1)
            mode, kind, sha = meta.split()
            if kind != "blob" or not mode.startswith("100"):
                continue
            key = (path, sha)
            if key not in cache:
                data = git("cat-file", "blob", sha, cwd=root)
                cache[key] = {
                    (f.rule, f.index) for f in scan_blobs([(path, data)], owner, allowed, actors)
                }
            rules |= cache[key]
        counts.update(rules)
    identities = git("log", "--all", "--format=%ae%n%ce", cwd=root).decode().split()
    domains = Counter("@" + address.rsplit("@", 1)[-1] for address in identities)
    return len(commits), counts, domains


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--tree", action="store_true", help="scan tracked working-tree files")
    scope.add_argument("--staged", action="store_true", help="scan the index (pre-commit)")
    scope.add_argument("--history", action="store_true", help="count matches in all commits")
    parser.add_argument("--patterns", type=Path, help="owner pattern file (overrides default)")
    parser.add_argument("--generic-only", action="store_true", help="skip owner patterns")
    parser.add_argument("--show-values", action="store_true", help="print matches (local only)")
    args = parser.parse_args(argv)
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    owner = []
    if not args.generic_only:
        source = args.patterns or pattern_file(root)
        if source.exists():
            owner = parse_patterns(source.read_text())
            print(f"owner patterns: {len(owner)} loaded", file=sys.stderr)
        else:
            print(
                "owner patterns: none (pattern file not found; generic rules only)", file=sys.stderr
            )
    allowed, actors = load_allow()
    if args.history:
        total, counts, domains = history(root, owner, allowed, actors)
        print(f"commits scanned: {total}")
        for (rule, index), count in sorted(counts.items(), key=lambda item: str(item[0])):
            label = rule if index is None else f"{rule} #{index}"
            print(f"{label}: {count} commits")
        print("author/committer domains: " + ", ".join(f"{d} ({n})" for d, n in domains.items()))
        return 1 if counts else 0
    blobs = staged_blobs(root) if args.staged else tree_blobs(root)
    findings = scan_blobs(blobs, owner, allowed, actors)
    for finding in findings:
        print(finding.render(args.show_values))
    print(f"{len(findings)} findings", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

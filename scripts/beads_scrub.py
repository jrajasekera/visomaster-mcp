"""Clear absolute checkout paths that `br create` stamps into Beads issues.

Usage: uv run --locked python scripts/beads_scrub.py [--dry-run]

br records the absolute workspace path as `source_repo_path` on every issue it creates, and
exports it to .beads/issues.jsonl. Clearing the field with `br update --source-repo-path ""`
omits it from the export, and later mutations do not restore it. Run this after creating
issues; scripts/privacy_audit.py flags any issue that still carries a path.
"""

import argparse
import json
import subprocess
import sys


def br(*arguments):
    return subprocess.run(["br", *arguments], capture_output=True, text=True, check=True).stdout


def stamped():
    listing = json.loads(br("list", "--all", "--json", "--limit", "0"))
    issues = listing["issues"] if isinstance(listing, dict) else listing
    return [i["id"] for i in issues if i.get("source_repo_path") not in (None, "", ".")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="list issues without changing them")
    args = parser.parse_args(argv)
    ids = stamped()
    for issue in ids:
        if not args.dry_run:
            br("update", issue, "--source-repo-path", "")
        print(("would clear " if args.dry_run else "cleared ") + issue)
    print(f"{len(ids)} issues {'to clear' if args.dry_run else 'cleared'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

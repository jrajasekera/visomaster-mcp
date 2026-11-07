#!/usr/bin/env bash
# Usage: scripts/deploy.sh [--force]
# Publishes remote/*.py as a new release in $VISOMASTER_REMOTE/releases, checks that the release
# accepts the deployment's remote-config.json, then switches the remote symlink atomically.
# Reads VISOMASTER_HOST, VISOMASTER_REMOTE, VISOMASTER_REMOTE_PYTHON and VISOMASTER_SSH, validated
# exactly as the MCP server validates them. The workstation needs GNU coreutils (mv -T) and tar.
set -euo pipefail
cd "$(dirname "$0")/.."
settings=$(python3 src/visomaster_mcp/transport.py check) || exit 2
{ read -r host; read -r remote; read -r python; read -r ssh; } <<<"$settings"
# Validated values contain no quotes, so single-quoting them in remote commands is safe.
run() { "$ssh" -T -o BatchMode=yes -o ConnectTimeout=10 -- "$host" "$1"; }
files=(common.py control.py engine.py)
if [[ "${1:-}" != --force ]]; then
  # Status filters scan every job; releases without filters fall back to the newest 50.
  active=$(for status in queued running; do
    run "[ ! -e '$remote/remote/control.py' ] || '$python' '$remote/remote/control.py'" \
      <<<"{\"action\": \"list\", \"status\": \"$status\", \"limit\": 200}" || true
  done | python3 -c 'import json, sys
jobs = []
for line in sys.stdin.read().splitlines():
    if not line.strip():
        continue
    try:
        response = json.loads(line)
        jobs += response["result"]
    except (ValueError, KeyError):
        sys.exit("Cannot list active jobs: " + line.strip()[-500:] + " (--force overrides)")
print(" ".join(j["job_id"] for j in jobs if j["status"] not in {"completed", "failed", "cancelled"}))')
  [[ -z "$active" ]] || { echo "Refusing to deploy while jobs are active: $active (--force overrides)" >&2; exit 1; }
fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)
release=$(cd remote && python3 -c 'import hashlib, sys
digest = hashlib.sha256()
for name in sys.argv[1:]:
    with open(name, "rb") as stream:
        digest.update(stream.read())
print(digest.hexdigest()[:12])' "${files[@]}")-$stamp
target="$remote/releases/$release"
# One transport for code and commands; macOS tar must not add AppleDouble or xattr entries.
COPYFILE_DISABLE=1 tar --no-xattrs -C remote -cf - "${files[@]}" |
  run "mkdir -p '$target' && tar -xf - -C '$target'"
# Preflight: the new release must accept the existing remote-config.json before it goes live.
health=$(run "cd '$target' && '$python' control.py" <<<'{"action": "health"}' || true)
if ! verdict=$(python3 -c 'import json, sys
try:
    health = json.loads(sys.argv[1])["result"]
except (ValueError, KeyError, TypeError):
    sys.exit("health check failed: " + sys.argv[1][-1000:])
if health.get("config_error"):
    sys.exit("remote configuration rejected: " + health["config_error"])
if not health.get("compatible"):
    print("warning: Fusion revision " + str(health.get("revision")) + " is not the supported "
          + health["supported_revision"] + "; GPU jobs will refuse to run", file=sys.stderr)
print("ok")' "$health"); then
  run "rm -rf -- '$target'"
  echo "Preflight refused release $release; the active release is unchanged" >&2
  exit 1
fi
# Supervisors start from the resolved release path, so submitted jobs keep the code they began with.
run "set -e; cd '$remote'
if [ -d remote ] && [ ! -L remote ]; then mv -T remote 'releases/legacy-$stamp'; fi
ln -sfn 'releases/$release' remote.next && mv -T remote.next remote
current=\$(readlink remote)
ls -1dt releases/*/ | tail -n +6 | while read -r old; do [ \"\${old%/}\" = \"\$current\" ] || rm -rf -- \"\$old\"; done"
echo "Deployed $release to $host:$remote ($verdict)"

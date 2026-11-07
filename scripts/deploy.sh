#!/usr/bin/env bash
# Usage: scripts/deploy.sh [--force]
# Publishes remote/*.py as a new release in $VISOMASTER_REMOTE/releases, checks that the release
# accepts the deployment's remote-config.json, then has the release select itself through the
# remote symlink under the lock that job submission takes.
# Reads VISOMASTER_HOST, VISOMASTER_REMOTE, VISOMASTER_REMOTE_PYTHON and VISOMASTER_SSH, validated
# exactly as the MCP server validates them. The workstation needs tar.
set -euo pipefail
cd "$(dirname "$0")/.."
settings=$(python3 src/visomaster_mcp/transport.py check) || exit 2
{ read -r host; read -r remote; read -r python; read -r ssh; } <<<"$settings"
# Validated values contain no quotes, so single-quoting them in remote commands is safe.
run() { "$ssh" -T -o BatchMode=yes -o ConnectTimeout=10 -- "$host" "$1"; }
files=(common.py control.py engine.py)
force=false
[[ "${1:-}" != --force ]] || force=true
if ! $force; then
  # An early refusal before anything is copied; promotion repeats the check under the submit lock.
  # Each query must succeed on its own, and only a deployment without a remote entry is a first
  # install. Status filters scan every job; releases without filters fall back to the newest 50.
  checks=()
  for status in queued running; do
    code=0
    response=$(run "if [ -e '$remote/remote/control.py' ]; then exec '$python' '$remote/remote/control.py'
elif [ -e '$remote/remote' ] || [ -L '$remote/remote' ]; then echo '$remote/remote has no control.py'; exit 1
else echo first-install; fi" <<<"{\"action\": \"list\", \"status\": \"$status\", \"limit\": 200}") ||
      code=$?
    checks+=("$status" "$code" "$response")
  done
  active=$(python3 -c 'import json, sys
values, jobs, first = sys.argv[1:], [], set()
for status, code, response in zip(values[::3], values[1::3], values[2::3]):
    text = response.strip()
    if code == "0" and text == "first-install":
        first.add(status)
        continue
    try:
        reply = json.loads(text)
    except ValueError:
        reply = None
    reason = None
    if isinstance(reply, dict) and "error" in reply:
        reason = "remote error: " + str(reply["error"])[-500:]
    elif code != "0":
        reason = "exit status " + code + ": " + (text[-500:] or "no output")
    else:
        result = reply.get("result") if isinstance(reply, dict) else None
        if isinstance(result, list) and all(
            isinstance(job, dict) and isinstance(job.get("job_id"), str)
            and isinstance(job.get("status"), str) for job in result
        ):
            jobs += result
        else:
            reason = "invalid response: " + (text[-500:] or "no output")
    if reason:
        sys.exit("Cannot list active jobs (" + status + " query): " + reason + " (--force overrides)")
if first and len(first) != 2:
    sys.exit("The remote installation changed between job queries; retry")
if first:
    print("No release is installed; deploying the first one", file=sys.stderr)
done = {"completed", "failed", "cancelled"}
print(" ".join(dict.fromkeys(j["job_id"] for j in jobs if j["status"] not in done)))' "${checks[@]}")
  [[ -z "$active" ]] || { echo "Refusing to deploy while jobs are active: $active (--force overrides)" >&2; exit 1; }
fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)
# A random suffix keeps concurrent deployments of the same code in separate directories.
release=$(cd remote && python3 -c 'import hashlib, os, sys
digest = hashlib.sha256()
for name in sys.argv[2:]:
    with open(name, "rb") as stream:
        digest.update(stream.read())
print(digest.hexdigest()[:12] + "-" + sys.argv[1] + "-" + os.urandom(3).hex())' "$stamp" "${files[@]}")
target="$remote/releases/$release"
promoted=false
# Any failure removes the candidate unless it became selected, so the previous release stays.
discard() {
  $promoted || run "cd '$remote' && [ \"\$(readlink remote 2>/dev/null)\" = 'releases/$release' ] ||
    rm -rf -- 'releases/$release'" || true
}
trap discard EXIT
# One transport for code and commands; macOS tar must not add AppleDouble or xattr entries.
COPYFILE_DISABLE=1 tar --no-xattrs -C remote -cf - "${files[@]}" |
  run "mkdir -p '$remote/releases' && mkdir '$target' && tar -xf - -C '$target'"
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
  echo "Preflight refused release $release; the active release is unchanged" >&2
  exit 1
fi
# The candidate rechecks active jobs, selects itself and prunes old releases under the submit lock.
reply=$(run "cd '$target' && '$python' control.py" <<<"{\"action\": \"promote\", \"force\": $force}" ||
  true)
if ! summary=$(python3 -c 'import json, sys
text, release = sys.argv[1:]
try:
    reply = json.loads(text)
except ValueError:
    reply = None
result = reply.get("result") if isinstance(reply, dict) else None
if not isinstance(result, dict):
    error = reply.get("error") if isinstance(reply, dict) else None
    sys.exit("Promotion failed: " + str(error or text.strip()[-1000:] or "no response"))
if result.get("release") != release:
    sys.exit("Promotion reported release " + str(result.get("release")) + ", not " + release)
for key, label in (("active_jobs", "active jobs"), ("removed", "removed releases")):
    if result.get(key):
        print(", " + label + ": " + " ".join(result[key]), end="")' "$reply" "$release"); then
  selected=$(run "readlink '$remote/remote'" || true)
  echo "Release $release was not promoted; remote selects ${selected:-no release}" >&2
  exit 1
fi
promoted=true
echo "Deployed $release to $host:$remote ($verdict$summary)"

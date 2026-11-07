# Agent instructions

## Orientation

This project exposes the existing VisoMaster Fusion installation through MCP for
Claude and Codex. Read `README.md` for setup, `DESIGN.md` for architecture, and
`VERIFICATION.md` for dated evidence and coverage gaps. Check `git status` before
editing and preserve unrelated work. Treat the current code and runtime as the
source of truth when documentation or earlier test results differ.

## Code and runtime boundaries

| Location | Responsibility |
| --- | --- |
| `src/visomaster_mcp/server.py` | Client-side MCP tools, input schemas, SSH commands, uploads and downloads |
| `src/visomaster_mcp/transport.py` | Client settings validation, SSH argv and remote command quoting (also used by `deploy.sh`) |
| `remote/common.py` | Workstation configuration file, Fusion revision, setting validation, atomic JSON and FFprobe |
| `remote/control.py` | Standard-library SSH command handler, durable job state and detached supervisors |
| `remote/engine.py` | Fusion adapter, GPU processing, frame timing and output validation |
| `scripts/deploy.sh` | Publish the three remote Python files as a release, preflight it and switch |
| `scripts/smoke_gpu.py` | Opt-in integration check with supplied remote media |
| `scripts/privacy_audit.py` | Scan tracked content for identifying values; allowlist in `scripts/privacy_allow.txt` |
| `scripts/beads_scrub.py` | Clear absolute `source_repo_path` values from Beads issues |
| `tests/test_jobs.py` | Job lifecycle, concurrency, path, settings and transfer regressions |
| `tests/test_config.py`, `tests/test_supervisor_config.py` | Remote configuration, identity, GPU lock sharing and health |
| `tests/test_client_config.py`, `tests/test_portability.py` | Client settings, deploy script and non-default end-to-end runs |
| `tests/test_protocol.py` | Actual stdio initialization, tool schemas and discovery |
| `tests/test_server.py` | Stdio tool calls through a local SSH shim: transfers, artifacts, remote errors |
| `config/` | Client, remote configuration and runner examples (placeholders only) |

The client package uses Python 3.11 or newer and the official MCP Python SDK 1.x.
Use `uv` and the committed `uv.lock`. Keep GPU dependencies inside the
workstation Fusion environment. `remote/common.py` and `remote/control.py` must
remain usable with the workstation's system Python and standard library.

## Installation specifics

This repository is public and installation-neutral. The actual SSH alias,
workstation paths, runner and GPU of a given installation come from the client
registration (`VISOMASTER_*`), the deployment's `remote-config.json` and
`health_check`, which are the source of truth. If an untracked `AGENTS.local.md`
exists in the checkout, read it for the local installation's details.

GPU work runs through the configured runner, which supplies the CUDA libraries
and allocator settings of that installation. Recheck GPU use before live tests,
including Fusion GUI activity. Preserve the runner's environment when changing
the adapter. Qualify dependency, provider or GPU changes explicitly against the
intended hardware.

## Privacy rules

- Never commit identifying values: usernames, home or checkout paths, host
  aliases and addresses, workstation mount layouts, hardware model names,
  serials, job IDs or release names. Use the placeholders in `README.md`.
- Run `uv run --locked python scripts/privacy_audit.py` before committing. Keep
  local patterns in `$(git rev-parse --git-common-dir)/info/privacy-patterns`.
- Record verification evidence with descriptive run labels and hardware class
  (architecture, compute capability, VRAM class). Keep raw job IDs in untracked notes.

## Implementation rules

- Reuse Fusion's processing classes and keep the adapter tied to the revision
  declared by `remote/common.py`. A revision upgrade requires inspection of the
  installed pipeline and fresh GPU verification before changing the guard.
- Keep the Mac process free of Fusion and GPU imports. Reserve MCP stdout for
  protocol messages and send diagnostics to stderr. Keep SSH arguments quoted
  with `shlex.join`, and send structured control payloads through stdin.
- Preserve tool names, schemas, structured responses and inline image results
  used by both clients. When the tool set changes, update discovery assertions
  in `tests/test_protocol.py` and the tool list in `README.md`.
- Validate settings against the explicit allowlists and installed metadata.
  Resolve media and artifact paths within their configured roots, including
  symlink checks. Stream large transfers and publish complete files atomically.
- Preserve stable job-ID semantics, conflict rejection, atomic state writes,
  supervisor leases and the shared GPU lock. Test changes against concurrent
  clients and reconnects. Cancellation must target the owned worker process
  group. Keep terminal state consistent when workers or supervisors exit.
- Preserve timestamp, frame-count, duration and audio-presence validation.
  Surface Fusion processing errors and zero-match renders as failed jobs.
  Keep result artifacts isolated from source media.
- Keep generated media, uploaded references, logs, caches, model weights and
  environment files in runtime storage outside tracked source files.

## Verification

Run setup and ordinary checks from the repository root.

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

The ordinary suite runs locally using temporary job backends and stdio clients.
Use targeted tests while iterating, then run the suite and Ruff for Python changes.
For documentation-only changes, verify paths, commands and claims against the
current source and run `git diff --check`.

GPU qualification requires a running workstation and suitable remote fixtures.
Use existing task authorization for live tests and deployment. Start with a short
clip, inspect the returned crops and preview, and check terminal job results.

```bash
uv run --locked python scripts/smoke_gpu.py \
  --media /absolute/workstation/clip.mp4 \
  --source /absolute/workstation/face.png \
  --frames 24
```

This script creates persistent jobs and uses the first detected face as its test
target. Choose fixtures where that selection is intentional. It exercises
preview/render, duplicate submission, reconnect and cancellation. It refuses to queue
anything unless `health_check` reports `code_matches: true`, and its JSON evidence
records the release and hash status plus a separate cancellation result: `passed`
(running worker cancelled), `passed_queued` (cancelled before it started) or
`unexercised` (the render finished first; not a pass, so rerun with more frames). Record the
models, hardware class, checks, observed metrics and remaining gaps in
`VERIFICATION.md` under descriptive run labels when qualifying behavior. Keep client protocol tests and native client UI
verification separately identified in the evidence.

## Deployment and client setup

A Git push publishes source. Workstation deployment uses `scripts/deploy.sh`
with `VISOMASTER_HOST` and `VISOMASTER_REMOTE` exported. The script publishes a
new release directory, checks that it accepts the existing `remote-config.json`,
swaps the `remote` symlink and prunes old releases. It never writes
`remote-config.json`. It refuses while MCP jobs are queued or running, or when it cannot list them;
wait for them rather than using `--force`. The new release rechecks, switches and prunes under
the submit lock, keeping releases that active jobs record.
After deployment, confirm `health_check` reports `code_matches: true` and run the
checks appropriate to the changed behavior.

Read the actual client configuration before changing registration. Preserve
existing server entries and use the setup commands or examples in `README.md`.
Treat its historical registration notes as dated evidence and verify current
configuration when troubleshooting a connection.

## Tracking and delivery

Use Beads Rust (`br`) for project issue tracking. Run `br info --json` and confirm
this checkout's `.beads` workspace before mutations. Follow the user's global
tracking and automatic-commit rules, including selective ownership of issue
records in `.beads/issues.jsonl`. Keep the database and its runtime files ignored.

The project `.beads/config.yaml` sets the neutral actor `maintainer`. If you pass
`--actor`, use a name listed in `scripts/privacy_allow.txt`. `br create` stamps the
absolute checkout path into each new issue, so after creating issues run
`uv run --locked python scripts/beads_scrub.py`; the privacy audit flags any
issue that still carries a path.

Stage task-owned changes, inspect the staged diff and commit coherent completed
work on the selected branch. Push when authorized and verify the branch matches
its upstream. Keep deployment within the task's separate runtime authorization.
Report the commit, verification evidence and any remaining limitations.

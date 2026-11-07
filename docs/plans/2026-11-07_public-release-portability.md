# Plan: public-release privacy cleanup and portable setup (vmcp-5ve)

Status: implemented on a feature branch (see §14.3). The spec is Beads issue `vmcp-5ve` (`br show vmcp-5ve`). It lists
the scope and acceptance criteria. This document says how to meet them. It
deliberately contains no identifying values. Wherever it refers to an
owner-specific value, it names the category ("the owner's SSH alias", "the
workstation mount root") and says how to find it.

Revision 2 (2026-11-07) adds the following after a Codex review that found no
critical issues:
- supervisor startup that survives configuration changes (§3.4)
- canonical GPU lock identity (§3.4)
- a macOS-safe deploy test, with the real switch qualified on Linux (§3.6, §10)
- shared transport validation for `deploy.sh` (§3.3, §3.6)
- a client-safe rollout order (§7.1, §11)
- the pattern-file location for worktrees (§2)
- a corrected process-group description (§1, §3.5)
- the interpreter setting for uploads (§3.3)
- the owner's authorization decisions (§14)

Placeholders used throughout this plan, and later in the docs and examples:

| Placeholder | Meaning |
| --- | --- |
| `<checkout>` | Absolute path of the local clone on the client machine |
| `gpu-host` | Example SSH alias for the GPU workstation |
| `/srv/visomaster` | Example workstation media root |
| `/srv/visomaster/fusion` | Example Fusion checkout |
| `/srv/visomaster/mcp` | Example deployment root (`VISOMASTER_REMOTE`) |
| `/srv/visomaster/runtime/mcp` | Example state directory |
| `/srv/visomaster/bin/fusion-runner` | Example runner |

## 1. Findings that shape the plan

These were checked on 2026-11-07 against the current checkout. Repeat the checks
when implementing.

**Hard-coded owner values in code.** `server.py` defaults `VISOMASTER_HOST` to the
owner's SSH alias and `VISOMASTER_REMOTE` to a path under the workstation mount
root. `remote/common.py` defaults `VM_BASE` to that mount root and hard-codes
`RUNNER = BASE / <installation launcher>`. `scripts/deploy.sh` repeats both
defaults and calls `ssh`/`scp`/`shasum` directly instead of `VISOMASTER_SSH`.

**Remote settings do not travel.** `server.py` runs
`python3 {REMOTE}/remote/control.py` over SSH and sends the JSON payload on stdin.
`VM_*` variables are never forwarded, so in production the remote side always
falls back to its built-in defaults. Supervisors inherit the control process's
environment through `Popen`. Workers get `dict(os.environ, CUDA_*, …)`.
`tests/test_server.py` hides the gap: its SSH shim runs locally and inherits
`VM_BASE`/`VM_STATE` from the test environment. A real SSH session would not
pass them.

**The runner contract is implicit and strict.** `engine.run()` requires
`os.getppid() == VM_SUPERVISOR_PID` and installs `PR_SET_PDEATHSIG`. A runner
therefore has to `exec` the Python interpreter. A runner that forks a child
instead fails with the misleading message "Supervisor disappeared before worker
startup". The worker also inherits the GPU-lock file descriptor (`pass_fds`).
The supervisor starts it with `start_new_session=True`, so the worker gets **its
own** session and process group, whose ID is the worker's PID. Cancellation runs
`os.killpg(child.pid, …)`, which targets the worker's group, not the
supervisor's. The supervisor itself also runs in its own session, which `submit`
creates.

**Supervisor startup depends on mutable configuration.** `submit` launches
`control.py supervise <job_id>`. The supervisor then resolves the job through
`job_path()`, and therefore through `STATE`, opens the lease and reads
`state.json`, all **before** its `try`/`finally`. In the new design `STATE` comes
from a file that can change. If the file is edited, deleted or invalid between
submit and supervisor startup, or while the job waits for the GPU lock, the job
could be looked up in the wrong place. It could stay `queued` until the
lease-based reconciliation in `current()` marks it failed, or never reach a
terminal state at all.

**Hard-coded interpreter.** `command()` and `upload_media()` each build their own
`["python3", f"{REMOTE}/remote/control.py", …]`, and `deploy.sh` hard-codes
`python3` as well.

**`deploy.sh` is Linux-specific on the remote side.** It uses GNU `mv -T`. A shim
test that runs those commands on the macOS client would fail, because BSD `mv` has
no `-T`.

**Production clients run this checkout.** The owner's Claude and Codex
registrations run `<checkout>/.venv/bin/visomaster-mcp`, an editable install. Any
change to `main` in that checkout takes effect for every **new** MCP session, with
no deployment step. Removing the client defaults would break new sessions as soon
as `main` changed, unless the registrations already set the variables.

**Path invariants.** `input_path()` accepts only files inside `BASE`. Uploads and
job artifacts (crops, `normalized.mp4`) live under `STATE` and are reused as
inputs, so `STATE` must resolve inside `BASE`. The GPU lock is
`STATE/gpu-<GPU>.lock`, so two deployments with different state directories do
**not** serialize on the same GPU. The production lock uses the index form
`gpu-<index>.lock`.

**Identifying content in the tracked tree.** Run the audit in §2 to get exact
locations. The categories found were:
- the owner's absolute checkout path and username in `README.md` and both `config/` examples
- the SSH alias in code, `deploy.sh`, `README.md`, `AGENTS.md` and `VERIFICATION.md`
- the mount-root layout in code, docs and examples
- the GPU model and architecture in `AGENTS.md`, `VERIFICATION.md` and one Beads issue
- job IDs, release names and absolute state paths in `VERIFICATION.md`
- the installation launcher's name in docs, tests and one issue
- `.beads/issues.jsonl`: every issue has `source_repo_path` set to the owner's absolute checkout path, and `created_by`, comment `author` and dependency `created_by` hold the OS username on most records

**Beads behaviour (`br` 0.5.10).** These results come from a throwaway workspace
in the session scratchpad.
- `source_repo_path` is stamped once, at `br create`, from the absolute workspace path, and stored per issue. Export does not re-stamp it: a copied workspace kept the old value for old issues and stamped its own path on new ones.
- `br create` has no override flag. `br update <id> --source-repo-path ""` clears the field, so it is omitted from export, and later mutations do not bring it back. `--source-repo-path .` also works.
- A project `.beads/config.yaml` key `actor: <name>` sets `created_by`, comment `author` and dependency `created_by` for new records when no `--actor` is passed. `BEADS_ACTOR` was **not** honoured. `--actor` overrides the config.
- No CLI command rewrites `created_by` on existing records. Editing the JSONL and then running a plain `br sync --import-only` is skipped as "up-to-date", and the next auto-flush **restores the old values**. Editing the JSONL and then running `br sync --import-only --rebuild` does apply the edits. IDs, dependencies, comments and labels survived, `br doctor` was clean, and later mutations kept the new values.
- The "Owner:" line in `br show` is computed for display and is not exported.

**Git/GitHub.** The GitHub repository is **public** (0 forks, 0 stars, no
PRs/issues/releases/tags at check time). All 35 commits use one personal author
and committer address. Owner-specific strings appear in the diffs of several
historical commits. No commit message contains session links. Another local
worktree branch exists and may carry tracker or doc changes that would
reintroduce old values when merged.

**Working tree at planning time.** `.beads/issues.jsonl` has an uncommitted
change that this plan does not own (the in-progress status of this issue).
`.serena/` and `.claude/worktrees/` are untracked and not ignored.

## 2. Repeatable privacy audit

Add `scripts/privacy_audit.py`, standard library only, run with `uv run --locked python`.

**Inputs**
- Built-in generic rules, safe to commit:
  - absolute home paths (`/Users/<name>/`, `/home/<name>/`) other than allowlisted placeholders
  - email addresses other than `example.com`/`example.org`
  - RFC 1918 and link-local IPv4 addresses, and `.local` hostnames
  - private-key headers and common token formats (GitHub, OpenAI-style `sk-`, AWS `AKIA`, Slack, bearer tokens)
  - URLs with embedded credentials
  - tracked files with media, model-weight or key extensions
  - tracked files over 5 MB, except `uv.lock`, which is checked by name
  - Beads export rules: `source_repo_path` absent or `.`, and every actor field in an allowlist
- Owner-specific patterns. These are never committed. The script reads
  `$VISOMASTER_AUDIT_PATTERNS` if set. Otherwise it reads
  `<common dir>/info/privacy-patterns`, where `<common dir>` comes from
  `git rev-parse --path-format=absolute --git-common-dir`. Do not hard-code
  `.git/info`: in a linked worktree `.git` is a file, and the per-worktree
  `--git-dir` would give each worktree its own copy. Using the common directory
  means one file serves the main checkout and every linked worktree of the clone.
  A separate clone needs its own copy, or the environment variable. Nothing under
  the common directory is ever tracked. Each line holds a literal, or a regex
  prefixed with `re:`. `#` starts a comment.
- A tracked allowlist, `scripts/privacy_allow.txt`, holding placeholder values and
  neutral actor names (for example `maintainer`, `assistant`, `codex`). The
  implementer confirms that every allowed actor name is non-identifying.

**Scopes**
- `--tree` (default) scans `git ls-files` content in the working tree.
- `--staged` scans the index. It is suitable for an optional pre-commit hook,
  documented but not installed automatically.
- `--history` runs `git grep` over `git rev-list --all` and reports how many commits
  match each pattern. It also lists distinct author and committer addresses,
  redacted to their domain. This feeds the history assessment in §8.

**Output.** Report `file:line`, the rule name and a pattern *index*, never the
matched owner value, so CI logs and pasted reports leak nothing. Exit non-zero on
findings. `--show-values` is for local use only.

**Populating the local pattern file.** Do this on the owner's machine and keep the
result out of Git. Collect one entry per category:
- the OS username (`whoami`) and the absolute checkout path (`pwd` at the repository root)
- author and committer addresses (`git log --all --format='%ae%n%ce' | sort -u`) and the author name
- the SSH alias, the workstation hostname and any IPs (`grep -iE '^\s*(Host|HostName)\s' ~/.ssh/config` for the alias the client uses)
- the workstation mount root and layout segments, read from the current defaults (`git show HEAD:remote/common.py | grep -n environ`)
- the installation launcher's file name (the `RUNNER` line of `remote/common.py`)
- the GPU model and architecture strings (`ssh <alias> nvidia-smi -L`), excluding UUIDs from any notes
- job-ID and release-name prefixes that appear in `VERIFICATION.md` (``git grep -nE '`[a-z0-9-]+-[0-9a-f]{6,}' -- VERIFICATION.md`` and its tables)
- the GitHub owner name (`git remote -v`)

**Tests** (`tests/test_privacy_audit.py`)
- Each generic rule against a synthetic fixture repository built in `tmp_path`.
- Output never contains the matched value without `--show-values`.
- Pattern-file parsing.
- Pattern-file resolution in both a normal repository and a `git worktree add`
  linked worktree. Both must find the same common-directory file. The environment
  variable overrides both.
- A Beads-rule fixture: an absolute `source_repo_path` and a non-allowlisted actor are both flagged.
- Final commit only: the real tracked tree passes the generic rules. The
  owner-pattern pass runs whenever the local file exists and is skipped in a clean
  clone.

## 3. Configuration contract

### 3.1 Principles

- The client and the workstation each own their own settings. Nothing relies on SSH
  environment forwarding (`SendEnv`/`AcceptEnv`), login-shell dotfiles or the remote
  user's home layout.
- **Client settings** (`VISOMASTER_*`) say *where to connect*. They are read from the
  client registration's environment.
- **Remote settings** say *where things are on the workstation*. They live in one
  file at the deployment root. Every remote process (SSH control command, detached
  supervisor, GPU worker) loads the same file itself, by locating it relative to its
  own resolved code path. Settings are therefore identical across clients,
  reconnects and processes, by construction.
- No owner-specific defaults remain. Required settings that are missing produce an
  actionable error and nothing is guessed.

### 3.2 Why the alternatives were rejected

| Approach | Problem |
| --- | --- |
| Forward `VM_*` inside the JSON payload | Each client could point at different state roots and locks (split brain). It enlarges the trusted input surface, and the upload command (argv, not JSON) and `deploy.sh` would also need it. |
| `env VM_X=… python3 …` prefix | Same split-brain risk and more quoting surface. The supervisor would inherit it, but configuration would depend on whichever client submitted the job. |
| `SendEnv`/`AcceptEnv` | Needs workstation `sshd_config` changes, so it is not portable. |
| Remote shell profile | Noninteractive shells often skip it. That is the failure the issue describes. |

### 3.3 Client settings (`src/visomaster_mcp/transport.py`, used by `server.py` and `deploy.sh`)

Put the client-side validation in one stdlib-only module,
`src/visomaster_mcp/transport.py`. It parses and validates the environment, builds
the SSH argv prefix (`[ssh, -T, -o …, --, host]`) and builds remote control
commands with `shlex.join([python, f"{remote}/remote/control.py", *args])`. It is
the only place that knows the interpreter: `command()`, `upload_media()` and the
`view_artifact`/`download_artifact` `cat` calls all go through it. This replaces
the hard-coded `python3` in `upload_media`.

`deploy.sh` calls it as `python3 src/visomaster_mcp/transport.py check`. That
needs no virtual environment. The check exits non-zero with the same message the
server would give, and on success prints the validated values for the script to
use. The validation rules then exist in one place.

| Variable | Required | Default | Validation |
| --- | --- | --- | --- |
| `VISOMASTER_HOST` | yes | none | Non-empty. Must not start with `-`. No whitespace or control characters. Allowed: `[A-Za-z0-9._@:%\[\]-]`. Also pass `--` before the host in the SSH argv. |
| `VISOMASTER_REMOTE` | yes | none | Same rule as `deploy.sh` today: `^/[A-Za-z0-9_./-]+$`, no `..` component. Implemented once, in `transport.py`. |
| `VISOMASTER_REMOTE_PYTHON` | no | `python3` | Bare command name (`^[A-Za-z0-9._-]+$`, not starting with `-`) or an absolute path in the `REMOTE` character set. Used for every remote Python call: control, upload and the `deploy.sh` checks. |
| `VISOMASTER_SSH` | no | `ssh` | Unchanged. Also used by `deploy.sh`. |
| `VISOMASTER_DOWNLOADS` | no | `~/Downloads/visomaster-mcp` | `expanduser`, then must be absolute. |

- Validate lazily on the first tool call, not at import. The server still starts,
  tool discovery and protocol tests keep working, and every tool call returns one
  clear error, for example: "Set VISOMASTER_HOST (SSH alias of the GPU workstation)
  and VISOMASTER_REMOTE (absolute deployment directory) in the MCP client
  registration; see README › Setup".
- Keep `shlex.join` for the remote command, structured stdin, `BatchMode=yes`, the
  `ControlPath=~/.ssh/visomaster-%C` multiplexing and the timeouts unchanged.
- When an SSH failure looks like a resolution or authentication failure, append a
  hint naming the variable and the `ssh -T -o BatchMode=yes "$VISOMASTER_HOST" true`
  check.

### 3.4 Remote settings file

**Location.** `<deployment root>/remote-config.json`. The deployment root is the
directory that holds `releases/`, and it equals `VISOMASTER_REMOTE`.
`remote/common.py` resolves it from `Path(__file__).resolve().parent`:
- If that directory's parent is named `releases`, the root is two levels up. This
  is the current release layout.
- Otherwise the root is one level up. This covers the legacy flat `remote/`
  directory and a source checkout.

`VM_CONFIG` (an absolute path) overrides the location. It exists for tests and
unusual layouts, and it is the **only** remaining `VM_*` input that changes
behaviour.

`deploy.sh` never writes or overwrites this file. The operator creates it once from
`config/remote-config.example.json`.

**Schema** (version 1; unknown keys are rejected to catch typos):

```json
{
  "version": 1,
  "media_root": "/srv/visomaster",
  "fusion_dir": "/srv/visomaster/fusion",
  "state_dir": "/srv/visomaster/runtime/mcp",
  "gpu": "0",
  "runner": ["/srv/visomaster/bin/fusion-runner"],
  "lock_dir": null
}
```

| Key | Replaces | Rules |
| --- | --- | --- |
| `media_root` | `VM_BASE` | Required. Absolute, existing directory. Resolved, so symlinks are followed once at load. |
| `fusion_dir` | `VM_APP` | Default `<media_root>/tools/visomaster-fusion`, the current layout convention, which is generic. Must be an existing directory containing `.git` and the five `app/ui/widgets/*_layout_data.py` files. |
| `state_dir` | `VM_STATE` | Default `<media_root>/runtime/visomaster-mcp`. Must resolve **inside** `media_root`, because uploads and artifacts are reused as inputs. Must not equal `media_root` and must not lie inside `fusion_dir`. Created with mode 0700 if missing. |
| `gpu` | `VM_GPU` | Default `"0"`. **Index only**, matching `^[0-9]+$`, with leading zeros rejected (so `"00"` cannot alias `"0"`). This one value sets `CUDA_VISIBLE_DEVICES` (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the index is stable across processes on one host) and names the lock `gpu-<index>.lock`. That is the name the running production lock already uses. UUID selectors are rejected, because `0` and `GPU-…` naming the same device would create two different locks. |
| `runner` | hard-coded `RUNNER` | Required. A string or a non-empty list of strings (argv prefix). The first element must be an absolute path to an executable regular file. Never passed through a shell. |
| `lock_dir` | (new, optional) | Default `state_dir`. Directory for `gpu-<index>.lock`. Absolute, existing, writable. **Cooperation rule:** deployments on one host that share a GPU must set the same `lock_dir` and refer to the GPU by the same index, so they serialize on one lock file (§7.3). Document that a deployment with its own `lock_dir` does not coordinate with others. |

**Loading and errors**
- `common.py` exposes `load_config()`. It returns an immutable settings object and
  raises `ConfigError` with messages that name the file, the key and the fix. The
  module-level `BASE`, `APP`, `STATE`, `GPU` and `RUNNER` (now a list) stay as the
  public names, so `control.py`, `engine.py` and the existing monkeypatching tests
  change little.
- Import must not raise. On failure, store the error. `control.py` then answers
  every action except `health` with `{"error": …}`, and `health` reports
  `config_error` along with the path it tried. Supervisors with a configuration
  error mark the job failed with that message instead of dying with a traceback.
- **Legacy guard.** If any of `VM_BASE`, `VM_APP`, `VM_STATE` or `VM_GPU` is present
  in the remote environment and differs from the effective value, fail with a
  message saying these variables are no longer read. A silently ignored override
  would be a silent behaviour change.
- **Configuration identity.** `config_sha256` is the SHA-256 of the canonical JSON
  of the *effective, resolved* settings. At submit it is recorded in `state.json`.
  The release symlink pins code but not this file; the identity covers the file.
  Older jobs that lack the field are left alone.

**Supervisor startup that survives configuration changes.** The supervisor must
never depend on re-resolving the job through mutable configuration. The change is
internal to a release (each release's `submit` starts that same release's
supervisor), so already-running supervisors from older releases are unaffected.
1. `submit` launches
   `control.py supervise <absolute job directory> <config_sha256>` instead of the
   bare job ID.
2. The supervisor first checks the directory *structurally*, without loading the
   configuration:
   - the path is absolute
   - its name is a valid job ID
   - its parent is named `jobs`
   - it holds `state.json`, which is not a symlink

   It then takes the job's `lease`. Everything after that runs inside the
   `try`/`finally` that publishes a terminal state.
3. Inside that block the supervisor loads the configuration, which may fail.
   - It checks that the loaded configuration's identity equals the argument.
   - It checks that `<STATE>/jobs/<id>` resolves to the same directory.
   - Any failure (malformed file, deleted file, invalid values, a different
     identity, a moved state root) sets the job to `failed`. The error says which
     check failed, for example "remote configuration changed after submission
     (state_dir differs); resubmit". The terminal state is written to the original
     directory from the argument, so it is never stranded.
4. While waiting for the GPU lock, the supervisor re-reads the configuration
   identity on each poll iteration (cheap: compare `stat` first, rehash on
   change). It then checks once more immediately before starting the worker. A
   change fails the job the same way and releases nothing it does not hold. The
   lock path comes from the configuration already validated in step 3, so a later
   edit cannot move the lock under a waiting job.
5. The worker receives `VM_CONFIG` (resolved file path) and
   `VM_CONFIG_SHA256`. It also verifies the identity before importing Fusion. A
   mismatch goes to `error.json`, so the supervisor reports it as the job error.

**Worker environment.** The supervisor continues to set `CUDA_DEVICE_ORDER`,
`CUDA_VISIBLE_DEVICES=<gpu>`, `QT_QPA_PLATFORM=offscreen`, `OMP_NUM_THREADS=2`,
`PYTHONUNBUFFERED=1` and `VM_SUPERVISOR_PID`. It adds `VM_CONFIG` and
`VM_CONFIG_SHA256`. These are generated internally and are not operator inputs.
Anything else the GPU stack needs (library paths, allocator settings, the Python
environment) belongs in the runner.

**`health` additions.** All are additive keys, so the tool schema stays the same:
- `config_path`, `config_error` and the effective settings (already partly reported as `app`, `state` and `input_root`)
- `runner`: exists and is executable
- `checks`: availability of `git`, `nvidia-smi`, `ffmpeg` and `ffprobe` on the control process's `PATH`, the Python version, and whether the state and lock directories are writable

Health must still succeed, with `checks` marked failed, when `nvidia-smi` or `git`
is missing. Today it raises.

### 3.5 Runner (launcher) contract

Document in README and `DESIGN.md`, and ship `config/fusion-runner.example.sh`:

```sh
#!/bin/sh
# Invoked as: fusion-runner <engine.py> <job-directory>, with CUDA_VISIBLE_DEVICES set.
# Adapt the three values below to your Fusion installation, then chmod +x.
FUSION_PYTHON=/srv/visomaster/fusion/.venv/bin/python   # Fusion's Python environment
export LD_LIBRARY_PATH="/opt/cuda-libs-for-your-gpu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"  # only if needed
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"      # optional
exec "$FUSION_PYTHON" "$@"   # exec is required: the worker must be the supervisor's direct child
```

**Rules**
- `exec` the interpreter, so the worker stays the supervisor's direct child.
- No `setsid`, `setpgid` or daemonizing. The supervisor already starts the runner
  in a new session and process group, and cancellation signals that whole group.
  Any helper process the runner or the worker spawns must stay in it.
- Keep the inherited file descriptors (the GPU lock) and environment.
- Do not change `CUDA_VISIBLE_DEVICES`.
- The exit status is the worker's.

If Fusion's own environment works with no extra variables, the portable equivalent
is `"runner": ["/srv/visomaster/fusion/.venv/bin/python"]`. No wrapper is needed.

Make failures actionable:
- In `engine.run`, when the parent PID differs from `VM_SUPERVISOR_PID` but that
  process is still alive, raise "the configured runner must exec the Fusion Python
  interpreter (see README › Runner)" instead of "Supervisor disappeared".
- When the runner path cannot be executed, report the `Popen` error together with
  the configured path in the job error. The current code already stores the error.

### 3.6 Deployment (`scripts/deploy.sh`)

- First run `python3 src/visomaster_mcp/transport.py check`. It applies exactly
  the server's validation (§3.3) to `VISOMASTER_HOST`, `VISOMASTER_REMOTE`,
  `VISOMASTER_REMOTE_PYTHON` and `VISOMASTER_SSH`, and exits non-zero **before any
  SSH call** on missing or malformed values.
- Every SSH call uses the same form as the server:
  `"$ssh" -T -o BatchMode=yes -o ConnectTimeout=10 -- "$host" "$remote_command"`.
  Remote paths and the interpreter appear in remote command strings only after
  validation. They are quoted with single quotes, and the character set excludes
  quotes.
- Replace `scp` with a single `tar -cf - … | ssh … "mkdir -p … && tar -xf - -C …"`.
  One transport is easier to configure and lets the shim test cover deployment.
  Compute the release hash locally with `python3 -c` (hashlib), not `shasum`, so the
  script also runs on Linux clients.
- New order:
  1. active-job check (unchanged semantics)
  2. upload into `releases/<release>`
  3. **preflight**: run `releases/<release>/control.py` with `{"action":"health"}` and abort without switching if `config_error` is set (the new release must accept the existing `remote-config.json`); warn but continue if `compatible` is false
  4. symlink swap and pruning (unchanged)
- Keep `--force` semantics and the legacy flat-directory migration.
- The remote switch keeps GNU `mv -T` and `ln -sfn`, because the supported
  workstation is Linux. Document the GNU coreutils requirement. The switch and the
  legacy migration are qualified **on the Linux workstation** (§7.3), not by the
  macOS test suite (§10).

### 3.7 Source-checkout assumption

`code_matches` hashes `<checkout>/remote/*.py`. `uv sync` installs the project in
editable mode, so a clone is the supported installation. Document this. Keep
`null` when the files are not found, which would happen with a non-editable wheel.
Do not package the remote files in this issue.

## 4. Documentation and examples

- **README.md**, rewritten as a setup guide:
  - overview
  - supported platforms: workstation Linux x86-64 with an NVIDIA driver, `nvidia-smi`, system `python3`, `git`, FFmpeg/FFprobe and Fusion's CUDA provider; client tested on macOS, Linux expected but unverified, Windows unsupported
  - prerequisites versus optional features (restoration, expression, tracking)
  - pinned Fusion revision (exact hash from `common.REVISION`) and the qualification process: inspect the pipeline at the new revision, run the adapter tests and GPU smoke on the target hardware, record evidence, then change the guard; never relax the guard otherwise
  - the setup steps:
    1. clone and `uv sync --locked`
    2. SSH alias with key authentication and the BatchMode check
    3. Fusion checkout at the revision, with environment and model assets per Fusion's instructions at that revision (the implementer reads Fusion's README and model-download mechanism at the pinned revision and cites them, without copying weights or URLs that may change)
    4. FFmpeg/FFprobe
    5. runner
    6. `remote-config.json`
    7. `deploy.sh`
    8. client registration with `-e VISOMASTER_HOST=… -e VISOMASTER_REMOTE=…` (Claude Code), an `env` table (Codex) and an `env` object (Claude Desktop)
    9. `health_check`/`get_settings`
    10. upload, detect and `contact.jpg`, preview, then a 12-frame render
  - **Upgrading an existing installation** (§7.1)
  - configuration reference (§3)
  - troubleshooting keyed by error message
  - tools, workflow and storage (kept)
  - licensing (§9)
  - responsible-use note (consent; local law on synthetic media)
  - all example paths use the placeholder table
- **AGENTS.md:** remove the alias, mount layout, launcher name and GPU model. Point to
  `health_check` and the deployment's `remote-config.json` as the source of truth.
  Add rules: never commit identifying values; run `scripts/privacy_audit.py` before
  committing; keep installation specifics in an untracked `AGENTS.local.md`, which
  agents should read if it exists. The owner's current specifics move there, so the
  owner's agent workflow keeps working.
- **DESIGN.md:** replace the launcher name and "GPU 0" with the configured runner and
  GPU. Add a short "Configuration flow" section, and adjust the mermaid diagram to
  show the config file read by control, supervisor and worker.
- **VERIFICATION.md:** keep dates, the Fusion revision, the models, the metrics
  (frame counts, fps, timings, PSNR, jitter percentages) and the coverage limits.
  Replace host names, absolute paths, release names and job IDs with descriptive run
  labels ("restoration run", "multi-frame detection run on a three-person clip").
  Describe hardware by architecture class, compute capability and VRAM class only,
  with no model, serial, UUID or bus ID. State that all evidence comes from a single
  installation. Beads issue IDs (`vmcp-…`) are project identifiers and stay.
  **Decided:** no raw job IDs, job-ID prefixes or release names in tracked docs.
  Hardware is described by architecture, compute capability and VRAM class only.
  This applies to **every** section, including those added after this plan was
  first drafted, which carry many more job IDs. Find them with the job-ID pattern
  from §2 and with
  ``git grep -nE '`[A-Za-z0-9_.-]*[0-9a-f]{6,}[A-Za-z0-9_.-]*`' -- VERIFICATION.md``.
  Replace each with a label that keeps the evidence meaningful: what was run, on
  what kind of fixture, and the observed result and metrics. Raw identifiers may be
  kept only in an untracked local evidence file.
- **config/**
  - `codex.toml` and `claude-desktop.json` use `<checkout>` and placeholder `env` values
  - a header comment (TOML) or a README note (JSON) says that unmodified examples do not run
  - new files: `remote-config.example.json` and `fusion-runner.example.sh`
- **Server instructions string and tool docstrings:** check them for owner specifics.
  None were found, but rerun the audit.

## 5. Ignore rules

Root `.gitignore` additions, each group commented:
- local agent and IDE state: `.serena/`, `.claude/worktrees/`, `.claude/settings.local.json`, `AGENTS.local.md`, `CLAUDE.local.md`
- local configuration and secrets: `.env`, `.env.*` with `!.env.example`, `*.pem`, `*.key`, `id_*`, `/remote-config.json` (accidental copy at the repository root; the example stays tracked)
- runtime and media: `/runtime/`, `/downloads/`, `/media/`, `*.log`, and `*.mp4 *.mkv *.mov *.webm`
- model weights: `*.onnx *.pth *.pt *.ckpt *.safetensors *.engine *.trt *.dfm`
- Do **not** ignore `*.png`/`*.jpg` globally, which would mask future documentation
  images. The audit's extension rule covers tracked images instead, with an
  allowlisted `docs/images/` exception.

Document in README › Contributing that ignore rules do not stop `git add -f`, and
that the audit catches force-added files. Check with
`git check-ignore -v <path>` and `git ls-files -i -c --exclude-standard` (tracked
files that match ignore rules, which should be none).

## 6. Beads export fix

Goal: keep IDs, relations and tracker function; remove local paths and the username
from the export; keep both from coming back.

1. **Durable actor.** Add `actor: maintainer` to `.beads/config.yaml`. The neutral
   name `maintainer` is **decided**. Verify with `br config get actor` in the repository. Note
   in AGENTS.md that agents passing `--actor` must use an allowlisted neutral name.
   The audit enforces it.
2. **Durable path suppression.** Add `scripts/beads_scrub.py` (standard library; it
   only calls `br`). It lists all issues, including closed ones
   (`br list --all --json`; check the flag name), and runs
   `br update <id> --source-repo-path ""` for any issue where the field is set. Run
   it after `br create`. AGENTS.md says: "after creating issues, run
   `uv run python scripts/beads_scrub.py`". Every audit flags omissions, so a missed
   run cannot reach a commit unnoticed. Note: clearing bumps `updated_at` only on
   newly created issues.
3. **One-time migration of existing records.** `created_by` has no CLI setter.
   1. Coordinate first: the uncommitted tracker change and the other worktree branch must be committed or merged, so that nothing is lost.
   2. Back up `.beads/beads.db*` and `issues.jsonl` to the scratchpad.
   3. Rewrite the JSONL with a reviewed script that replaces the username with the neutral actor in issue `created_by`, comment `author` and dependency `created_by`, removes `source_repo_path`, and changes nothing else.
   4. Run `br sync --import-only --rebuild`, then `br sync --flush-only`.
   5. Verify semantically:
      - same ID set, statuses, priorities, labels, dependency edges, comment counts and texts, and timestamps
      - the only differences are the targeted fields
      - `br doctor` is clean
      - a following `br update` on one issue and its auto-flush do not reintroduce the old values
   6. Separately, use `br update --description/--acceptance-criteria` to remove the GPU model, architecture and launcher name from the one affected issue's text.
4. **Tests.** The audit fixtures cover the export rules. Optionally,
   `tests/test_beads_export.py` (skipped when `br` is not on `PATH`) creates a
   temporary workspace, sets `actor`, creates two issues with a dependency and a
   comment, runs `beads_scrub.py` and asserts that the export has no
   `source_repo_path` and only neutral actors.
5. **Optional, owner decision:** an upstream feature request to beads_rust for a
   create-time or config option that suppresses or relativizes `source_repo_path`.
   This would be an outward-facing action.

## 7. Compatibility and migration for the existing installation

### 7.1 Production rollout (authorized; also published generically in README)

The owner has authorized the following as part of finishing this task:
- updating the client registrations, with backups
- writing the production remote configuration with the current values
- deploying, then verifying health, `code_matches` and a short smoke run
- the rollback procedure below, if needed

**Why order matters.** The owner's clients run this checkout's editable install
(§1). Whatever is on `main` in that checkout is what every new MCP session runs.
`main` must therefore never contain code that the current client registrations or
the deployed release cannot work with.

**Constraints**
- Do all implementation in a separate worktree
  (for example `git worktree add`). Never
  commit it directly onto `main` in the production checkout. The production
  checkout changes only at the merge in phase 2.
- **Protocol compatibility in both directions.**
  - The new client keeps the same control actions, JSON payload shape, upload argv
    and `cat` transfers, and must not *require* any new `health` key. Treat
    `config_path`, `checks` and the rest as optional.
  - The new remote release accepts everything the old client sends.
  - New client and old release, and old client and new release, therefore both
    work, so a session that starts between the merge and the deploy is safe.
  - Verify this explicitly (§10, cross-version check).

**Phase 0: prerequisites before anything reaches `main`.** Every step here is a
no-op for the code currently running.
1. **Record a baseline** in an **untracked**, private scratch file: the production
   `health_check` output (release, code hashes, revision, `app`, `state`,
   `input_root`, `gpu`) and the `list_jobs` count.
2. **Check the remote environment.** Confirm that it has no `VM_*` variables:
   `ssh -T -- <alias> 'env | grep "^VM_" || true'`.
3. **Read and back up the client registrations.**
   - Read the actual configuration with `claude mcp get visomaster` (and
     `claude mcp list`), `codex mcp get visomaster`, and the Claude Desktop config
     file if it has a `visomaster` entry. Identify which file each comes from
     (Claude Code user or local scope, the Codex config TOML, the Desktop JSON).
   - Copy those files to a private backup directory outside the repository (mode
     0600) and record their checksums. The backups contain identifying paths and
     must never be committed.
4. **Add explicit env to every registration.** Use the current production values:
   `VISOMASTER_HOST` = the owner's alias and `VISOMASTER_REMOTE` = the current
   deployment root, which are the old code defaults.
   - Claude Code: re-add at the same scope with `-e` for each variable.
   - Codex: add an `env` table to the `[mcp_servers.visomaster]` entry.
   - Desktop: add an `env` object.

   Preserve every other entry and setting. Verify with the same `get` commands.
   Then start a new session and confirm that `health_check` is unchanged against
   the old code. The old code reads these same variables, so behaviour cannot
   change.
5. **Pre-stage the remote config.** Write
   `<production deployment root>/remote-config.json` with exactly the current
   effective values:
   - `media_root` = old `input_root`
   - `fusion_dir` = old `app`
   - `state_dir` = old `state`
   - `gpu` = old `gpu` (an index)
   - `runner` = `[<old base>/<launcher path from the old RUNNER constant>]`
   - `lock_dir` omitted, so the lock stays at `gpu-<index>.lock` in the same place

   The old code ignores this file. Dry-validate it with the branch's loader without
   deploying: copy the branch's `remote/*.py` into a scratch directory on the
   workstation and run its `control.py` health action with `VM_CONFIG` pointing at
   the file.
6. **Set up the deploy shell.** Export `VISOMASTER_HOST` and `VISOMASTER_REMOTE` in
   the shell that will run `deploy.sh`.

**Phase 1: isolated live check (§7.3)** from the worktree or a scratch clone.
Production is untouched.

**Phase 2: merge and deploy together, in one sitting.**
1. Rebase the branch on `main` and fast-forward `main` (linear history).
2. In the production checkout, run `uv sync --locked`.
3. Wait until `list_jobs` shows no queued or running jobs.
4. Run the new `deploy.sh` from the production checkout. Its preflight validates
   `remote-config.json` with the new release before switching.
5. Verify, then record the results generically in VERIFICATION.md:
   - `health_check` shows `code_matches: true`, the same paths and GPU as the
     baseline, `config_path` set and `config_error` empty
   - `list_jobs` still lists the earlier jobs; the job state format is unchanged
     apart from the optional `config_sha256`
   - a short smoke run completes through the production path with the same runner:
     `scripts/smoke_gpu.py` with 12–24 frames, or detect, preview and a short render
   - a new client session from each registered client connects

**Rollback.** Each layer can be rolled back on its own:
- *Remote.* Point `remote` back at the previous release
  (`ln -sfn releases/<prev> remote.next && mv -T remote.next remote` on the
  workstation). The old code ignores `remote-config.json`, which can stay.
- *Client code.* Revert the merged commits on local `main` with `git revert`, which
  keeps history linear. New sessions then run the old code. The pre-staged env
  values equal its defaults, so it behaves as before. Pushing a revert needs push
  authorization.
- *Client registrations.* Restore the backed-up files, or re-add each registration
  exactly as recorded in phase 0 step 3. Verify with the `get` commands and a new
  session.

Document the same sequence generically in README › Upgrading.

### 7.2 Behaviour that must not change

Tool names and input/output schemas must not change. Capture `tools/list` JSON on
`main` and on the branch, then diff them; only docstring changes are acceptable, and
ideally none.

The following must also keep working:
- stable job IDs and conflict rejection
- atomic state writes, leases and the GPU lock path and semantics
- cancellation by process group
- release pinning
- artifact and media root confinement, including symlink checks
- the upload and download protocol
- normalization not taking the GPU lock

### 7.3 Isolated live verification on the workstation

**Authorized** by the owner, who has approved running whatever is needed on the
workstation. It must not disturb production. Run it before phase 2 of §7.1.

1. **Preconditions.** Production is idle (`list_jobs`). The GPU is idle, including
   any Fusion GUI (`nvidia-smi`). Record the production health baseline and the
   production job count.
2. **Non-default SSH alias without editing the owner's `~/.ssh/config`.** Write an
   ssh wrapper in the scratchpad that runs `ssh -F <scratch>/ssh_config "$@"`. That
   config defines a new alias such as `vm-portability`, with `HostName` and `User`
   copied from the existing alias. Set `VISOMASTER_SSH` to the wrapper.
3. **Clean clone.** `git clone <checkout> <scratch>/clone`, check out the branch,
   then `uv sync --locked`.
4. **Isolated deployment.** Use a new deployment root and state directory under the
   media root, both different from production (for example `…/tools/visomaster-mcp-portability`
   and `…/runtime/visomaster-mcp-portability`). Set `lock_dir` to the **production**
   state directory and `gpu` to the **same index** production uses. The isolated
   deployment then opens the same `gpu-<index>.lock` that production already uses,
   so the two serialize.
   - Confirm this before running GPU work: hold the production lock with `flock` in a
     workstation shell, submit an isolated detect job, see it stay `queued`, release
     the lock, and see it run.
   - **Linux-only deploy qualification.** Before the first real deploy, create a
     legacy flat `remote/` directory in the isolated root (copy the current files
     into it). Then run `deploy.sh` three times and verify:
     - the legacy directory is migrated into `releases/legacy-*`
     - the `remote` symlink switches atomically (`mv -T`)
     - each submitted job keeps its release
     - only the five newest releases remain
     - an invalid `remote-config.json` makes the preflight refuse, with the
       symlink unchanged

     The macOS suite does not cover these behaviours (§10).
5. **Non-default runner.** Write a runner from the documented example, using the
   environment that the existing launcher sets (read it on the workstation; do not
   modify it). If that runner fails, fall back to a wrapper that only
   `exec`s the existing launcher, and record that the portable runner remains
   unqualified.
6. **Follow the README** from step 6 onwards (config, deploy, registration). For
   registration, use `claude mcp add --scope project` inside the scratch clone, so
   no user-level configuration changes; codex/desktop entries get a syntax check
   only. Then run health/settings, upload, detection, preview, a 12- to 24-frame
   render, and
   `scripts/smoke_gpu.py` (reconnect, duplicate submission, cancellation).
7. **Negative checks.** A bad `state_dir` (outside the media root), a missing runner
   and an unset `VISOMASTER_HOST` all give the documented errors. `deploy.sh`
   refuses to switch on an invalid config.
8. **Post-checks.** Production health is identical to the baseline. The production
   job list has no new entries. The isolated state holds the test jobs.
9. **Clean up.** Delete the test jobs and the isolated deployment once the evidence
   is recorded (authorized cleanup only). Record the environment class and results
   generically in VERIFICATION.md, and keep job IDs in local notes.
10. If hardware is unavailable, record the live check as outstanding instead of
    claiming end-to-end portability.

## 8. Git history and author metadata (decided 2026-11-07)

**Decision.** The owner chose to squash the whole history into a single commit and
force-push it (a variant of option B below). The orchestrator performs this **after**
the implementation branch is merged and deployed; the implementer does not. The
Git author name and email are public by the owner's decision and are not treated as
an exposure, so no mailmap or identity change is needed.

**Evidence at implementation time (categories only).** The repository is public with
no forks or stars. `privacy_audit.py --history` over all local refs (45 commits,
including the implementation branch) found owner-specific values in almost every
commit before the cleanup: the checkout path and username in 43, the SSH alias,
mount-root layout and launcher name in 42–43, GPU model names in 42, release names
in 37, job IDs in 8–22, and the Beads `source_repo_path`/actor metadata in 40–43.
All author and committer addresses use one public personal domain. Tracked docs and
the tracker reference no commit SHAs other than the pinned Fusion revision (an
upstream commit, unaffected by a rewrite).

**Procedure (orchestrator).**
1. Merge and deploy as in §7.1 phase 2, and confirm the tree passes the audit with the
   owner patterns.
2. Make sure no other branch or worktree holds unmerged work (for example other agent
   worktrees and `codex/*` branches); merge or archive it first, because it will be
   based on the discarded history.
3. Create the squashed commit from the clean tree, for example
   `git checkout --orphan public && git commit -m "…" && git branch -M public main`,
   then re-run the audit in `--history` mode on the new single-commit history.
4. `git push --force-with-lease origin main`, then delete or recreate any other remote
   branches and tags.

**Consequences.** Every commit SHA changes. All other clones, worktrees and branches
(including the production checkout the clients run) must be reset to the new `main`
(`git fetch && git reset --hard origin/main` after checking for local work) or
re-cloned; the Beads JSONL is unaffected because it is part of the tree. A force-push
does not erase the old commits from GitHub immediately: they can stay reachable by SHA
and in caches, the events API or third-party archives until GitHub support purges
unreferenced objects, and any clone or fork made before the push keeps them.

The original options, kept for the record:

| Option | Effect | Consequences |
| --- | --- | --- |
| A. Leave history as is; HEAD clean | No disruption | Earlier commits still show the paths, the alias and the address. Record this as remaining exposure. |
| B. Rewrite with `git filter-repo --replace-text <local patterns> --mailmap <map>`, then force-push | Rewritten history | All SHAs change, and references in docs and the tracker must be checked. Every clone and worktree must be re-cloned or reset. Old commits can stay reachable on GitHub by SHA and in caches or archives until GitHub support purges them. Forks cannot be controlled. |
| C. Publish a fresh repository from a squashed clean snapshot; archive, privatize or delete the old one | Clean public history | History is lost. Old URLs break. Changes visibility of the existing repository. |
| D. Make the repository private while deciding | Limits new exposure | Changes visibility; existing copies remain. |

For future commits, the owner may set a repository-local `user.email` to the GitHub
noreply address and enable GitHub's "block command line pushes that expose my
email". Both change identity or account settings and are owner decisions. The
implementer does none of the above. Record the chosen option and the remaining
exposure on the issue.

## 9. Licensing (decided 2026-11-07)

**Decision.** The owner wants anyone to be able to use the project for any purpose:
the MIT License, copyright holder the Git author name, year 2026. Implemented as
`LICENSE`, `license = "MIT"` in `pyproject.toml` and the README "License" and
"Responsible use" sections. Apache-2.0 was not needed: the project has no patent
grant concerns and MIT is the simplest permissive choice.

**Upstream facts verified on the workstation.** Fusion at the pinned revision ships
the GNU GPL v3.0 (`LICENSE`). Its README carries a responsible-use disclaimer. Its
model list (`app/processors/models_data.py`) records names, URLs and hashes but no
license terms, so the README tells users to check each model's terms and cites
InsightFace's non-commercial research restriction for its models (the Inswapper and
ArcFace recognizers). `engine.py` imports Fusion in-process; MIT is GPL-compatible,
and the README says that distributing the adapter combined with Fusion must follow
the GPL. The project redistributes neither Fusion, its models nor FFmpeg.

Original planning notes:

- The project had no LICENSE. A public repository without one grants no
  usage rights. The owner chooses a license. After that, add `LICENSE`,
  `license = "…"` in `pyproject.toml` and a README section.
- Upstream facts to verify and cite, without restating anything unverified:
  - Fusion's license at the pinned revision (`git -C <fusion_dir> show <revision>:LICENSE`) and its README terms
  - for each qualified model (detector, recognition, Inswapper128, GFPGAN, LivePortrait), the license and redistribution or commercial-use terms, from Fusion's model metadata at that revision
  - InsightFace-derived models are commonly restricted to non-commercial research use; confirm the exact terms
- README must say that this project does not redistribute Fusion or model weights,
  that users obtain them under their own terms, and that the project's license does
  not override upstream terms.
- `engine.py` imports Fusion modules into the same process. If Fusion uses a copyleft
  license, the owner should choose a compatible license, or take advice. Flag this.
  Do not decide it.

## 10. Test plan

Local, part of `uv run --locked pytest -q`:

- **`tests/test_config.py`** (remote settings)
  - a valid file
  - missing file, with an error that names the expected path
  - relative paths
  - nonexistent directories
  - `state_dir` outside or equal to `media_root`
  - `state_dir` inside `fusion_dir`
  - missing layout files
  - bad `gpu` values (`"0;x"`, `"../0"`, empty, `"00"`, `"GPU-…"` UUID form)
  - runner not absolute, not executable, or a list form whose path contains spaces (works, because no shell is involved)
  - unknown keys
  - `media_root` reached through a symlink
  - `VM_CONFIG` override
  - legacy `VM_*` mismatch guard
  - location resolution in the release layout, the legacy flat layout and the symlinked `remote/`
- **Client settings** (`tests/test_server.py` or `test_client_config.py`)
  - missing `HOST`/`REMOTE` gives an actionable tool error while discovery still works
  - rejects a host starting with `-`, a host with whitespace, and a remote that is relative, contains `..` or contains quotes
  - SSH argv puts `--` before the host
  - `shlex.join` quoting with a custom `REMOTE_PYTHON`
  - `REMOTE_PYTHON` rejects values containing spaces, `;`, quotes or a leading `-`
  - a **non-reused** upload (no existing content-addressed file) through the shim
    with a custom interpreter: a wrapper that records its invocation and then
    `exec`s `python3`. Asserts that both the `find_upload` command and the
    streaming `upload` command used it.
  - parity between `deploy.sh` and `server.py`. Both go through
    `transport.py check`. For the same malformed host (leading `-`, whitespace),
    remote and interpreter, both refuse, and `deploy.sh` exits non-zero **before any
    SSH call** (the shim records no invocation).
  - `deploy.sh` SSH invocations carry `--` before the host
  - a new client tolerates a `health` result without the new keys, as returned by
    an old release
- **Cross-version check** (one-time, recorded). Run the new server against the
  remote files of the currently deployed release (`git show <deployed commit>:remote/<file>`
  into a temporary deployment), and the current `main` server against the new
  remote files. Both go through the shim, covering health, submit, status, upload
  and artifact download. This supports the rollout order in §7.1.
- **Propagation and persistence**
  - `config_sha256` recorded at submit
  - supervisor argv carries the absolute job directory and the identity
  - each of the following ends in a **terminal `failed`** state in the original job
    directory, with a specific error, and never stays `queued`:
    - the config is changed so `state_dir` points elsewhere between submit and
      supervisor start
    - the config is made malformed (invalid JSON) at that point
    - the config is deleted at that point
    - the config is changed while the job waits for a GPU lock held by the test
  - the worker refuses a mismatched `VM_CONFIG_SHA256`
  - **two-deployment serialization:** two configs with different `state_dir`s, the
    same `lock_dir` and the same `gpu` index. A job from deployment B stays
    `queued` while deployment A's job holds `gpu-<index>.lock`, and runs after
    release. A control test with *different* `lock_dir`s shows no serialization,
    to document the cooperation rule.
  - supervisor and worker both see the configured `gpu`, through `CUDA_VISIBLE_DEVICES` and the lock name, and `lock_dir`
  - runner without `exec` gives the new actionable message, using a shell runner that runs Python as a child and then exits
  - health with `nvidia-smi` or `git` missing
- **Shim end-to-end** (new `tests/test_portability.py`)
  - The SSH shim records its argv and asserts the custom alias. It runs the command
    under `env -i PATH=… HOME=…`, so no `VM_*` or `VISOMASTER_*` can leak, as with
    real SSH.
  - **Deploy substitute test**, labelled as such in its docstring. `deploy.sh` runs
    against a non-default deployment root in `tmp_path`, through the shim, with a
    custom alias, interpreter and SSH program. Assertions:
    - upload into `releases/<release>`
    - preflight refusing an invalid config before any switch
    - the active-job refusal

    The final `mv -T` switch, the legacy migration and pruning run only when GNU
    `mv` is available (`mv --version` reports GNU coreutils). Otherwise the test
    is skipped with the reason "atomic switch qualified on the Linux workstation
    (plan §7.3)". The test does not imitate `mv -T`.
  - The stdio server runs with a custom alias, root and downloads directory. The
    custom runner records argv, environment and `common.load_config()` as seen from
    the worker, then writes `result.json`.
  - A second, separate stdio session (reconnect) sees the same job and settings.
  - A media path containing quotes, `$()` and spaces travels only in JSON and
    executes nothing.
- **Examples**
  - `config/*.json` parses as JSON; `codex.toml` parses with `tomllib`
  - `remote-config.example.json` passes `load_config()` once its placeholders are mapped to `tmp_path` directories
  - examples contain only allowlisted placeholders (audit)
  - `fusion-runner.example.sh` passes `sh -n` and contains `exec`
- **Audit and Beads:** §2 and §6.
- **Unchanged regressions:** the existing job, protocol and server tests, adapted to
  write a config file instead of setting `VM_*` env. Diff the `tools/list` schemas
  against `main` (§7.2).
- **Gaps to document**
  - Linux-only behaviour (`prctl`, `mv -T` switch, legacy migration) is not
    exercised on a macOS client run. It is qualified on the workstation in §7.3.
  - the real-SSH quoting path is exercised only in the live check
  - there is no CI

Then run `uv run --locked ruff check .`, `uv run --locked ruff format --check .` and
`git diff --check`. Run the audit in tree mode, both generic and with the owner
patterns. Inspect the semantic tracker diff.

## 11. Commit sequence

Each step is a small, coherent commit on a feature branch **in a separate
worktree**, never on `main` in the production checkout that the clients run (§7.1).
Update the tracker alongside, staging only task-owned `.beads/issues.jsonl`
records. Because the clients run `main`, the order in which the branch reaches
`main` matters more than the order of commits inside it:
- phase 0 of §7.1 (client env backups and updates, the pre-staged remote config)
  must be complete before the merge
- the deploy follows the merge in the same sitting
- the cross-version check (§10) guarantees that a session started in between still
  works

1. `chore: add privacy audit script, allowlist and ignore rules` (scanner tests only).
2. `feat(remote): load workstation settings from the deployment config file`
   (`common`/`control`/`engine`, configuration identity, supervisor launched with
   the job directory and identity, index-only GPU, `lock_dir`, runner argv, health
   checks, runner error message, tests moved to config files, config-change and
   two-deployment tests).
3. `feat(client): shared transport settings with explicit host and deployment`
   (`transport.py`, `server.py` including `upload_media`, `REMOTE_PYTHON`, `--`,
   hints, tests).
4. `feat(deploy): validate transport, stream releases over ssh and preflight config`
   (`deploy.sh` using `transport.py check`, deploy substitute test, shim end-to-end
   portability tests, cross-version check).
5. `docs: portable setup guide, configuration reference and examples` (README,
   DESIGN, AGENTS, `config/*`, runner example).
6. `docs: generalize verification evidence` (VERIFICATION.md).
7. `chore(beads): neutral actor and path-free tracker export` (config, scrub script,
   one-time migration, issue text cleanup, Beads test).
8. `test: require the tracked tree to pass the privacy audit`.
9. After the live check (§7.3, authorized): `docs: record portability verification`
   on the branch.
10. Phase 0 of §7.1 (no repository change), then the phase 2 merge and production
    deploy (authorized). Finish with
    `docs: record production migration verification`.
11. Only after the owner decides: `docs: add license and upstream terms`. This is
    not part of finishing this task. Leave the license as an explicitly open item on
    the issue.

Steps 2–4 change remote behaviour. Production changes only at the deploy in step 10,
and the clients change only at the merge in step 10, after their env has been
pre-staged.

## 12. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| New code deployed before `remote-config.json` exists | `deploy.sh` preflight refuses the switch; health reports the missing file; rollback by symlink. |
| Isolated test overlaps a production GPU job | `lock_dir` pointing at the production lock, an idle check beforehand, and checking the GUI too. |
| Config edited, deleted or made invalid while jobs are queued or waiting for the GPU | The supervisor gets the absolute job directory and identity, and re-checks at start, during the lock wait and before the worker starts. Such jobs end `failed` with a specific error and are never stranded. |
| Index and UUID selectors split the lock for one GPU | Index-only `gpu`, `PCI_BUS_ID` ordering, and the same `lock_dir` for cooperating deployments, with a two-deployment test. |
| Merge to `main` breaks new client sessions | Implementation in a worktree; client env pre-staged and backed up (phase 0); cross-version compatibility; deploy in the same sitting; three-layer rollback. |
| macOS test passes but the Linux deploy switch fails | The deploy test is labelled a substitute and skips the switch without GNU `mv`; the real switch, migration and pruning are qualified on the workstation. |
| Beads rebuild loses data | Backups, a rehearsal on a scratch copy of the workspace first, and the semantic diff; coordinate with other worktrees and agents. |
| Future `br create` or a merged branch reintroduces paths or the username | Scrub script plus an audit that fails on any absolute `source_repo_path` or non-allowlisted actor. |
| False confidence from a pattern-based audit | Report it as "no findings for these categories", never "clean"; history stays a separate, open item. |
| Removing defaults breaks the owner's clients | Pre-stage the env in registrations, with backups (§7.1 phase 0); the error names the variables. |
| Runner semantics differ from the existing launcher (CUDA libraries, allocator) | Derive the example from the existing launcher's environment; keep the existing launcher usable as `runner` unchanged. |
| Placeholder values collide with real hosts or paths | Use obviously generic placeholders, and keep them in the allowlist. |

## 13. Acceptance-criteria mapping

| Criterion (vmcp-5ve) | Satisfied by |
| --- | --- |
| Repeatable audit finds no owner identifiers in the tree or tracker export; consistent placeholders; a fresh Beads export does not reintroduce them | §2 audit (tree, staged, owner patterns), §4 placeholder table, §6 actor, scrub, migration and test, commit 8 |
| Current tree separated from historical Git/GitHub metadata; remaining exposure recorded; rewrite is a separate decision | §8 evidence note and options; outcome recorded on the issue |
| One configuration works with non-default paths, alias and runner; settings visible to control, supervisor and worker, including after reconnect; invalid paths fail clearly | §3 contract, §10 shim end-to-end (scrubbed env, custom alias and runner, reconnect), §7.3 live check |
| Upgrade steps for existing users; tools, schemas, jobs, IDs, locking, cancellation, artifacts and deploy safety compatible | §7.1 (client-safe rollout, authorized production migration, three-layer rollback), §7.2 (schema diff, unchanged `gpu-<index>.lock` path), §3.4 (supervisor robustness), §10 cross-version check |
| Automated tests for propagation, quoting, path validation, examples, Beads, regressions; full checks run; gaps documented | §10 |
| New instructions followed from a clean, non-default setup through registration, health and deploy; short detect, preview and render on authorized hardware, or explicitly outstanding | §7.3 |
| Public docs state platforms, Fusion revision, dependencies and model assets, licensing and limits; examples are placeholders, not claims | §4 and §9 |
| No media, weights, secrets or runtime artifacts committed; tracked content scanned; semantic tracker diff inspected | §2 extension and size rules, §5, the final checks in §10 |

## 14. Decisions

### 14.1 Decided by the owner (2026-11-07)

1. **Workstation work is authorized.** The owner has authorized running whatever is
   needed on the workstation.
   - (a) The isolated live check is authorized (§7.3): a separate deployment and
     state directory, sharing the production GPU lock.
   - (b) The production migration is authorized as part of finishing this task
     (§7.1): write the production remote config with the current values, deploy,
     verify health, `code_matches` and a short smoke run. The documented rollback
     is to point the symlink back and restore the client configuration.
   - (c) Updating the client registrations, with backups, is authorized (§7.1
     phase 0).
2. **Beads actor.** The neutral actor name is `maintainer`. The allowlist holds only
   neutral names (§2, §6).
3. **VERIFICATION.md detail.** Hardware is described by architecture, compute
   capability and VRAM class only. Tracked docs carry no raw job IDs, prefixes or
   release names. Every section is generalized, including the recently added job
   IDs, while keeping the evidence meaningful (§4).

4. **License (later on 2026-11-07).** MIT, with a README statement that it covers
   only this repository and that Fusion (GPL-3.0), model weights, FFmpeg and other
   upstream components keep their own terms, plus a responsible-use note (§9).
5. **Git history.** Squash all history into one commit and force-push, performed by
   the orchestrator after the merge and deploy (§8). Repository visibility stays public.
6. **Author identity.** The Git author name and email are public; keep using them.
   They are not audit patterns and are not treated as exposure (§8).

### 14.2 Still open

1. Optional: whether to file an upstream beads_rust feature request about
   `source_repo_path` (§6). This would be an outward-facing action.

### 14.3 Implementation notes (2026-11-07)

- The owner pattern file could not be written to the clone's common Git directory
  from the isolated implementation worktree. It was kept in the private session
  scratchpad and must be installed as `<common dir>/info/privacy-patterns` by the
  orchestrator; the audit reads `VISOMASTER_AUDIT_PATTERNS` meanwhile.
- `br` run inside a linked worktree resolves to the main checkout's `.beads`
  database, so the tracker migration was performed in a scratch copy of the
  workspace and only the resulting `issues.jsonl` was committed. The main checkout's
  database must be rebuilt from the merged JSONL (`br sync --import-only --rebuild`).
- With `br` 0.5.10, new issues created in scratch workspaces did not receive a
  `source_repo_path`; the scrub script and the audit remain as the safeguard.
- A deploy whose current release cannot read `remote-config.json` refuses at the
  active-job check ("Cannot list active jobs"), before the preflight; `--force`
  skips that check, and the preflight still refuses an invalid configuration.

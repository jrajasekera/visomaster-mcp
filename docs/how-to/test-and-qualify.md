# Test, qualify and audit changes

This guide is for people who change this repository. It covers the local checks, the opt-in GPU
smoke test, qualifying a new Fusion revision and the privacy audit.

## Run the local checks

From the repository root:

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/privacy_audit.py
```

The test suite runs locally. It uses temporary job backends, stdio MCP clients and an SSH shim
that clears the environment, as real SSH does. It does not need a workstation. Linux-only
behaviour (the parent-death signal) is exercised only on a workstation. There is no CI.

## Run the GPU smoke test

The smoke test drives a real installation through the MCP server. It needs:

- a deployed workstation whose `health_check` reports `code_matches: true`;
- a short video on the workstation;
- a replacement image on the workstation with exactly one face.

It creates persistent jobs and uses the first detected face as its target, so choose a clip
where that face is the one you want to test.

1. Export the client settings (`VISOMASTER_HOST`, `VISOMASTER_REMOTE` and any optional
   variables). The script passes the whole environment to the server.
2. Run:

   ```bash
   uv run --locked python scripts/smoke_gpu.py \
     --media /srv/visomaster/media/clip.mp4 \
     --source /srv/visomaster/media/face.png \
     --frames 24
   ```

   `--frames` sets the render length (default 12). To exercise face and expression
   restoration, add:

   ```bash
   --parameters '{"FaceRestorerEnableToggle":true,"FaceExpressionEnableBothToggle":true}'
   ```

3. Read the JSON evidence the script prints.

The script runs detection, a preview and a short render, checks that a duplicate submission
returns the same job, reconnects with a new session and cancels a running render. It stops
before queuing anything unless `health_check` reports a compatible installation with
`code_matches: true`.

The evidence records the release and hash status, the reconnect result and the cancellation
outcome:

| Cancellation result | Meaning |
| --- | --- |
| `passed` | A running worker was cancelled |
| `passed_queued` | The job was cancelled before it started, a different and weaker path |
| `unexercised` | The render finished before the cancellation arrived. This is not a pass: run again with more frames |

## Record qualification evidence

When a run qualifies new behaviour or hardware, add a section to
[VERIFICATION.md](../../VERIFICATION.md) with a descriptive run label. Record the models, the
hardware class (architecture, compute capability, VRAM class), the checks, the observed
metrics and the remaining gaps. Keep raw job IDs in untracked notes.

If the run covers a combination that `preflight_swap` reports as `unqualified`, add a record to
`QUALIFICATIONS` in `remote/common.py` that points at the new section. Add a record only after a
GPU qualification run.

## Qualify a new Fusion revision

The adapter is tied to one Fusion commit, `REVISION` in `remote/common.py`. To move to another:

1. Inspect the installed pipeline at the new revision: the processors, worker and layout
   definitions the adapter calls.
2. Run the local test suite.
3. Run the GPU smoke test on the target hardware.
4. Record the evidence in `VERIFICATION.md`.
5. Only then change `REVISION`.

Never relax or bypass the revision guard without that qualification.

## Run the privacy audit

This repository is public. `scripts/privacy_audit.py` scans tracked files for home paths, email
addresses, private network addresses, token formats, credential URLs, media, model weights, key
files, large files and identifying Beads metadata.

1. Put the values of your own installation (host alias, paths, hardware names) in an untracked
   pattern file at `$(git rev-parse --git-common-dir)/info/privacy-patterns`. Write one literal
   per line, or a regular expression prefixed with `re:`; `#` starts a comment. All worktrees of
   the clone share this file. `VISOMASTER_AUDIT_PATTERNS` or `--patterns FILE` overrides its
   location.
2. Run the audit:

   ```bash
   uv run --locked python scripts/privacy_audit.py
   ```

   | Option | Scans |
   | --- | --- |
   | `--tree` (default) | Tracked files in the working tree |
   | `--staged` | The index; suits a pre-commit hook |
   | `--history` | Counts matching commits across history |
   | `--generic-only` | Skips your own patterns |
   | `--show-values` | Prints the matched values; use only locally |

3. Fix every finding. Findings show the file, line and rule, not the matched value. The exit
   status is 1 when anything is found.

A clean result means there are no findings in these categories, not that nothing identifying
remains.

`.gitignore` keeps local configuration, secrets, runtime output, media and model weights out of
commits, but `git add -f` bypasses it; the audit catches such files. This command should print
nothing:

```bash
git ls-files -i -c --exclude-standard
```

# VisoMaster MCP

An MCP server that drives an existing [VisoMaster Fusion](https://github.com/VisoMasterFusion/VisoMaster-Fusion) installation on a Linux GPU workstation. Claude Code, Claude Desktop and Codex run the same Python stdio server on the client machine. The server sends commands to the workstation over SSH, where they run as durable background GPU jobs. It returns progress, face crops, previews and finished media.

See [DESIGN.md](DESIGN.md) for the architecture and [VERIFICATION.md](VERIFICATION.md) for recorded evidence and its limits.

## Supported platforms

| Side | Supported | Notes |
| --- | --- | --- |
| Workstation | Linux x86-64 with an NVIDIA GPU and driver | Needs `nvidia-smi`, system `python3` (3.9 or newer, standard library only), `git`, `tar`, GNU coreutils, FFmpeg and FFprobe on `PATH`, an OpenSSH server, and a Fusion environment with a working CUDA provider |
| Client | macOS (tested), Linux (expected, not verified) | Python 3.11 or newer with [uv](https://docs.astral.sh/uv/), an OpenSSH client with key authentication. Windows is not supported |

All recorded GPU evidence comes from a single installation with Maxwell-generation GPUs (compute capability 5.2, 24 GB class). Other GPUs, drivers and CUDA stacks have not been qualified.

## Prerequisites and optional features

Required: a Fusion checkout at the pinned revision with its Python environment, the detector, recognition and swapper models (`Yolov8`, `Inswapper128` and its ArcFace recognizer by default), FFmpeg/FFprobe, and SSH key access from the client.

Optional: face restoration (`GFPGAN-v1.4` was qualified), expression restoration (LivePortrait models), and Fusion's sequential tracking (ByteTrack and landmark smoothing, off by default). These only need their Fusion models to be present.

## Fusion revision guard

The adapter imports Fusion's own processing classes, so it is tied to one Fusion commit: `f2d6f5ebe190b631ee5c8972332083cf7f8a3e72` (declared as `REVISION` in `remote/common.py`). GPU jobs refuse to run if the checkout is at another commit or has tracked local changes under `app/`, and `health_check` reports `compatible: false`.

To qualify another revision, inspect the installed pipeline at that revision (the processors, worker and layout definitions the adapter calls), run the test suite, run `scripts/smoke_gpu.py` on the target hardware, record the evidence in `VERIFICATION.md`, and only then change `REVISION`. Never relax or bypass the guard without that qualification.

## Setup

The examples below use placeholders. Replace each one with your own value.

| Placeholder | Meaning |
| --- | --- |
| `<checkout>` | Absolute path of your clone on the client machine |
| `gpu-host` | SSH alias of the GPU workstation |
| `/srv/visomaster` | Workstation media root: every media file and job artifact lives below it |
| `/srv/visomaster/fusion` | Fusion checkout |
| `/srv/visomaster/mcp` | Deployment directory (`VISOMASTER_REMOTE`) |
| `/srv/visomaster/runtime/mcp` | Job state, uploads and logs |
| `/srv/visomaster/bin/fusion-runner` | Runner that starts Fusion's Python |

1. **Clone and install the client.**

   ```bash
   git clone https://github.com/<owner>/visomaster-mcp.git <checkout>
   cd <checkout>
   uv sync --locked
   ```

   The server runs from this clone (an editable install). `health_check` compares the deployed workstation files with the clone's `remote/` directory, and reports `code_matches: null` if those files are absent.

2. **Configure SSH.** Add a `Host gpu-host` entry with key authentication to `~/.ssh/config`, then confirm that noninteractive access works without a prompt:

   ```bash
   ssh -T -o BatchMode=yes gpu-host true
   ```

3. **Install Fusion at the pinned revision** on the workstation, following Fusion's README at that commit ("Non-Portable Installation"): clone it, check out the revision, create its Python 3.12 environment, install its requirements file for your GPU generation, and run `python download_models.py`, which downloads the models listed in `app/processors/models_data.py`.

   ```bash
   git clone https://github.com/VisoMasterFusion/VisoMaster-Fusion /srv/visomaster/fusion
   git -C /srv/visomaster/fusion checkout f2d6f5ebe190b631ee5c8972332083cf7f8a3e72
   ```

   Confirm that Fusion itself can use your GPU before continuing.

4. **Install FFmpeg and FFprobe** on the workstation, for example `sudo apt install ffmpeg`. They must be on the `PATH` of noninteractive SSH sessions and of the runner.

5. **Create the runner.** Copy [config/fusion-runner.example.sh](config/fusion-runner.example.sh) to `/srv/visomaster/bin/fusion-runner`, adapt it, and make it executable. See [Runner](#runner).

6. **Create the remote configuration.** Copy [config/remote-config.example.json](config/remote-config.example.json) to `/srv/visomaster/mcp/remote-config.json` and adapt it (see [Remote configuration](#remote-configuration)). Deployment never writes this file.

7. **Deploy the workstation files** from the clone:

   ```bash
   export VISOMASTER_HOST=gpu-host VISOMASTER_REMOTE=/srv/visomaster/mcp
   scripts/deploy.sh
   ```

   The script validates these variables exactly as the server does, uploads the three standard-library files into a new directory under `releases/`, runs the new release's health check against `remote-config.json`, and only then points the `remote` symlink at it. It keeps the five newest releases. Jobs keep the release they were submitted with. It refuses to run while jobs are queued or running; wait for them rather than using `--force`, because pruning can remove a release a queued job still needs.

8. **Register the client.** Every registration must set `VISOMASTER_HOST` and `VISOMASTER_REMOTE`; there are no defaults.

   Claude Code:

   ```bash
   claude mcp add --scope user --transport stdio visomaster \
     -e VISOMASTER_HOST=gpu-host -e VISOMASTER_REMOTE=/srv/visomaster/mcp \
     -- <checkout>/.venv/bin/visomaster-mcp
   claude mcp get visomaster
   ```

   Put the server name before the `-e` options; `-e` accepts several values and would otherwise consume the name.

   Codex: run `codex mcp add visomaster --env VISOMASTER_HOST=gpu-host --env VISOMASTER_REMOTE=/srv/visomaster/mcp -- <checkout>/.venv/bin/visomaster-mcp`, or adapt [config/codex.toml](config/codex.toml). [Codex MCP documentation](https://developers.openai.com/codex/mcp/)

   Claude Desktop: merge the entry from [config/claude-desktop.json](config/claude-desktop.json) into `mcpServers` in its configuration file, keeping your other entries, and restart it. The example files contain placeholders and do not work unmodified. [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp)

9. **Check the installation.** In a new client session, call `health_check` and `get_settings`. Expect `config_error: null`, `compatible: true`, `code_matches: true`, your GPU in `gpus`, and every entry in `checks` true.

10. **Run a first job.** Upload or choose a short clip and a reference image, run `detect_faces`, view `contact.jpg`, submit a `preview`, inspect it, then render about 12 frames (see [Typical workflow](#typical-workflow)). `scripts/smoke_gpu.py` automates this check.

## Upgrading an existing installation

Earlier versions had built-in defaults for the SSH host, deployment directory, media root and runner, and read `VM_BASE`, `VM_APP`, `VM_STATE` and `VM_GPU` on the workstation. Now nothing is assumed. Upgrade in this order, so that new client sessions keep working throughout:

1. Record `health_check` output (`app`, `state`, `input_root`, `gpu`) and back up your client registration files.
2. Add `VISOMASTER_HOST` and `VISOMASTER_REMOTE` with your current values to every client registration. The old version reads the same variables, so nothing changes yet.
3. Write `<VISOMASTER_REMOTE>/remote-config.json` with the values from step 1: `media_root` = `input_root`, `fusion_dir` = `app`, `state_dir` = `state`, `gpu` = `gpu`, and `runner` = your existing launcher. Leave out `lock_dir` so the GPU lock stays where it was. The old release ignores this file.
4. Update the clone, run `uv sync --locked`, wait until `list_jobs` shows nothing queued or running, then run `scripts/deploy.sh`. Its preflight refuses to switch if the new release rejects your configuration.
5. Check that `health_check` shows the same paths and GPU, `config_error: null` and `code_matches: true`, that `list_jobs` still lists earlier jobs, and that a short smoke run completes.

To roll back, point `remote` at the previous release on the workstation (`cd <VISOMASTER_REMOTE> && ln -sfn releases/<previous> remote.next && mv -T remote.next remote`), check out the previous client version, and restore the registration backups if needed. The old release ignores `remote-config.json`, so it can stay.

## Configuration reference

### Client environment

Set these in the MCP client registration. They are validated on the first tool call, so tool discovery works even before they are set.

| Variable | Required | Default | Meaning and validation |
| --- | --- | --- | --- |
| `VISOMASTER_HOST` | yes | none | SSH alias or host name; letters, digits and `._@:%[]-`, no leading `-` |
| `VISOMASTER_REMOTE` | yes | none | Absolute workstation deployment directory; letters, digits and `_./-`, no `..` |
| `VISOMASTER_REMOTE_PYTHON` | no | `python3` | Workstation interpreter for control commands: a command name or an absolute path |
| `VISOMASTER_SSH` | no | `ssh` | Client SSH program |
| `VISOMASTER_DOWNLOADS` | no | `~/Downloads/visomaster-mcp` | Local directory for `download_artifact` |

SSH calls use `BatchMode=yes` and share one multiplexed connection through `~/.ssh/visomaster-%C`, which persists for 120 seconds after the last call. Nothing relies on SSH environment forwarding or remote shell profiles.

### Remote configuration

`<VISOMASTER_REMOTE>/remote-config.json` is read by every workstation process: control commands, detached supervisors and GPU workers. Each process finds it relative to its own code, so all clients and reconnects see the same settings. `VM_CONFIG` (an absolute path) overrides the location; it exists for tests and unusual layouts. Unknown keys are rejected.

| Key | Required | Default | Rules |
| --- | --- | --- | --- |
| `version` | yes | | `1` |
| `media_root` | yes | | Existing absolute directory. Media paths and references must resolve inside it and be self-contained video or image files; playlists and concat lists are refused |
| `fusion_dir` | no | `<media_root>/tools/visomaster-fusion` | Fusion checkout containing `.git` and its layout definition files |
| `state_dir` | no | `<media_root>/runtime/visomaster-mcp` | Inside `media_root` (uploads and artifacts are reused as inputs), not inside `fusion_dir`; created with mode 0700 |
| `gpu` | no | `"0"` | GPU index as a string, without leading zeros. Sets `CUDA_VISIBLE_DEVICES` (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`) and names the lock `gpu-<index>.lock` |
| `runner` | yes | | Path, or list of arguments, whose first element is an absolute executable. Never passed through a shell |
| `lock_dir` | no | `state_dir` | Existing writable directory for `gpu-<index>.lock` |

Several deployments on one host that share a GPU must use the same `lock_dir` and the same `gpu` index, so their jobs serialize on one lock file. A deployment with its own `lock_dir` does not coordinate with others. The legacy `VM_BASE`, `VM_APP`, `VM_STATE` and `VM_GPU` variables are no longer read; if one is set to a different value, the workstation reports an error instead of ignoring it.

Each job records the SHA-256 of the effective configuration. If the file is changed, made invalid or removed while a job is queued, the job fails with an error asking you to resubmit, rather than running with different settings.

### Runner

The supervisor starts each GPU job as `<runner...> <release>/engine.py <job directory>`, with `CUDA_VISIBLE_DEVICES` and the job's configuration identity in the environment. The runner prepares Fusion's environment and must:

- `exec` Fusion's Python interpreter, so the worker remains the supervisor's direct child (otherwise jobs fail with "The configured runner must exec the Fusion Python interpreter")
- not call `setsid`, `setpgid` or daemonize. Cancellation signals the worker's process group, which any helper process must stay in
- keep inherited file descriptors (the GPU lock) and the environment, and leave `CUDA_VISIBLE_DEVICES` unchanged
- exit with the worker's status

[config/fusion-runner.example.sh](config/fusion-runner.example.sh) adds the CUDA libraries installed as wheels inside Fusion's environment to `LD_LIBRARY_PATH`. If Fusion's environment needs nothing extra, the interpreter itself is a valid runner: `"runner": ["/srv/visomaster/fusion/.venv/bin/python"]`.

## Tools

| Tool | Behavior |
| --- | --- |
| `health_check` | Configuration, installed revision, deployed release and code match, GPU inventory, runner and dependency checks |
| `get_settings` | Supported Fusion settings with effective defaults and limits |
| `upload_media` | Stream a local image/video into private workstation storage, reusing identical uploads |
| `inspect_media` | FFprobe metadata for a workstation file |
| `detect_faces` | Queue detection at a selected frame and produce reference crops, or sample up to 64 frames (`sample_frames` or `frames`) and group faces into identities with one reference crop each |
| `submit_swap` | Queue a frame/image preview or a video render with explicit face mappings |
| `normalize_media` | Queue a CPU-only FFmpeg conversion to a render-ready video (CFR, zero start, even dimensions, rotation applied); does not wait for the GPU |
| `job_status` | State, frame progress and result manifest; `wait_seconds` (≤ 50) long-polls for the next change; log tail on request or failure |
| `list_jobs` | Recent job summaries shared across clients (kind, media, frames, timing, status, progress), filterable by `status` and `kind`, `limit` up to 200 |
| `cancel_job` | Request cancellation, then poll for terminal state |
| `delete_job` | Permanently remove a terminal job's directory, logs and artifacts |
| `prune_storage` | Report (dry run, default) or delete old terminal jobs, unused uploads and crash leftovers |
| `view_artifact` | Return a completed job's PNG/JPEG as an inline MCP image, such as a detection's `contact.jpg` grid of labelled crops |
| `download_artifact` | Stream a completed artifact to the client |

Multi-frame detection embeds every detected face with the configured recognition model and groups faces greedily, largest first, into identities. A face joins the identity whose mean embedding scores highest on Fusion's similarity scale (50 + 50 × cosine similarity, the scale `SimilarityThresholdSlider` uses) if that score reaches the slider's default of 60. Two faces from the same frame never share an identity. Each identity reports its appearances, frames, up to five sample boxes and a representative crop: the largest of its five largest faces whose crop re-detects as exactly one face (`reference_verified`).

## Typical workflow

1. Call `health_check` and `get_settings`.
2. Use an absolute workstation media path under `media_root`, or upload a local file with `upload_media`.
3. Call `detect_faces` on a representative frame, or with `sample_frames` (for example 16) to find everyone who appears across a video. Call `job_status` with `wait_seconds: 50` until `completed`, then inspect the returned crops with `view_artifact`; `contact.jpg` shows them all in one labelled grid.
4. Use the selected crop's or identity's `reference_path` as `target_face`. Supply one or more reference images of the replacement identity as `source_faces`. Each reference image must contain exactly one detected face.
5. Submit a `preview`, inspect its image, then submit a short `render`. After reviewing that clip, submit the full frame range.
6. Follow the job with `job_status` and `wait_seconds`, then download `result.mp4` or `result.png` after completion.

Renders reject variable frame rate video, video whose timestamps do not start at zero, odd frame dimensions and rotation metadata. When a render fails for one of these reasons, call `normalize_media` with the same `media_path` (optionally `fps`, such as `25` or `"30000/1001"`), wait for the job, and use its result's `media_path` (the job's `normalized.mp4` artifact) for detection, preview and rendering. Frame indices then refer to the normalized timeline.

Example `submit_swap` arguments:

```json
{
  "job_id": "shot01-preview-v1",
  "media_path": "/srv/visomaster/media/clip.mp4",
  "mode": "preview",
  "frame": 24,
  "mappings": [
    {
      "target_face": "/srv/visomaster/runtime/mcp/jobs/detection01/artifacts/face-0.png",
      "source_faces": ["/srv/visomaster/media/replacement.png"],
      "parameters": {
        "SwapModelSelection": "Inswapper128",
        "SimilarityThresholdSlider": 60,
        "FaceRestorerEnableToggle": false,
        "FaceExpressionEnableBothToggle": false
      }
    }
  ]
}
```

For video, use `mode="render"`, `start_frame` and `max_frames`. A null `max_frames` processes through the end. Frame indices start at zero. Give each intended job a stable `job_id` so a retried submission returns the same job. Reusing an ID with changed settings is rejected. Submit a new ID to retry a failed or cancelled job.

## Storage

Job state, logs and artifacts live in `<state_dir>/jobs/JOB_ID`. Uploaded media lives in `<state_dir>/uploads`, named by SHA-256 so identical uploads share one file. Terminal jobs remain until `delete_job` or `prune_storage` removes them. `prune_storage` deletes terminal jobs finished more than `older_than_days` ago (default 30), uploads untouched for that long that no queued or running job references, staging and trash directories left by crashes, and partial uploads idle for over an hour. It reports byte totals and is a dry run unless `dry_run` is false. Partial files from cancelled or failed jobs remain for diagnosis, and artifact tools expose only completed jobs. Each downloaded artifact has its own job subdirectory. Repeating the same download atomically replaces that local copy.

MCP GPU jobs share a process lock across all client sessions. Fusion's desktop GUI has its own GPU workload, so coordinate large GUI renders with MCP renders.

## Troubleshooting

| Message | Fix |
| --- | --- |
| `VISOMASTER_HOST and VISOMASTER_REMOTE not set` | Add both variables to the client registration (setup step 8) and start a new session |
| `Check VISOMASTER_HOST (...) with: ssh ...` | The alias does not resolve or key authentication failed; run the printed command until it succeeds without a prompt |
| `Remote configuration not found at ...` | Create `remote-config.json` in the deployment directory (setup step 6) |
| `...: <key> ...` from the configuration | Fix the named key in the named file; `health_check` shows `config_path` and `config_error` |
| `VM_... is no longer read` | Remove the variable from the workstation environment and put the value in `remote-config.json` |
| `Fusion revision changed to ...` or `compatible: false` | Check out the pinned revision, or qualify the new one (see [Fusion revision guard](#fusion-revision-guard)) |
| `The configured runner must exec the Fusion Python interpreter` | End the runner with `exec "$FUSION_PYTHON" "$@"` |
| `Cannot start the configured runner ...` | Fix the runner path or its permissions |
| `Remote configuration changed after submission ...` | The file changed while the job was queued; resubmit |
| `Preflight refused release ...` from `deploy.sh` | The new release rejects `remote-config.json`; the error names the problem, and the active release is unchanged |
| `checks` shows `nvidia-smi`, `ffmpeg` or `ffprobe` false | Install it, or put it on the `PATH` of noninteractive SSH sessions |

## Development

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/privacy_audit.py
```

The suite runs locally with temporary job backends, stdio clients and an SSH shim that clears the environment, as real SSH does. Linux-only behavior (the parent-death signal and the `mv -T` release switch) is exercised only on a workstation. There is no CI.

The opt-in GPU check needs a short workstation video and a single-face replacement image. It creates persistent jobs, runs detection, preview and a short render, checks idempotent submission, reconnects with a new session and cancels a running job. It stops before queuing anything unless `health_check` reports `code_matches: true`, and records whether cancellation hit a running job, a queued one, or was unexercised because the render finished first. It uses the first detected face as its target.

```bash
uv run --locked python scripts/smoke_gpu.py \
  --media /srv/visomaster/media/clip.mp4 \
  --source /srv/visomaster/media/face.png \
  --frames 24
```

Pass `--parameters '{"FaceRestorerEnableToggle":true,"FaceExpressionEnableBothToggle":true}'` to exercise restoration.

### Privacy audit

`scripts/privacy_audit.py` scans tracked files for home paths, email addresses, private network addresses, token formats, credential URLs, media, model weights, key files, large files and identifying Beads metadata. Install-specific values (your host alias, paths, hardware names) go in an untracked pattern file at `$(git rev-parse --git-common-dir)/info/privacy-patterns`, one literal or `re:` regex per line, shared by all worktrees of the clone; `VISOMASTER_AUDIT_PATTERNS` overrides the location. Findings show the file, line and rule, never the matched value. `--staged` scans the index and suits a pre-commit hook; `--history` counts matching commits. A clean result means no findings for these categories, not proof that nothing identifying remains.

Ignore rules in `.gitignore` keep local configuration, secrets, runtime output, media and model weights out of commits, but `git add -f` bypasses them; the audit catches such files. `git ls-files -i -c --exclude-standard` should print nothing.

## License

The code in this repository is available under the [MIT License](LICENSE).

That license covers only this repository. It does not cover and does not change the terms of anything this project uses. VisoMaster Fusion is licensed under the GNU General Public License v3.0 (see the `LICENSE` file in its repository). FFmpeg, PyTorch, ONNX Runtime, CUDA and the other components of Fusion's environment have their own licenses. This project does not include or redistribute Fusion, its models or FFmpeg; you obtain them yourself under their own terms.

`remote/engine.py` imports Fusion's modules into the same Python process on the workstation. The MIT License is compatible with the GPL, but if you distribute this adapter together with Fusion as a combined work, review the GPL's requirements for that distribution.

Model weights carry separate terms that Fusion's model list does not record. In particular, [InsightFace states](https://github.com/deepinsight/insightface#license) that models trained on its annotated data are available for non-commercial research purposes only; the Inswapper and ArcFace recognition models used here come from InsightFace. Check the terms of every model you download before using its output.

## Responsible use

Face swapping can be used to deceive or to harm the people depicted. Use this project only with media you have the right to use and with the consent of the people whose likeness you process. Label synthetic media where appropriate, and follow the laws on synthetic media, privacy and likeness that apply to you. Fusion's README includes a fuller disclaimer that also applies here.

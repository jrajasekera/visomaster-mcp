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

The adapter imports Fusion's own processing classes, so it is tied to one Fusion commit: `f2d6f5ebe190b631ee5c8972332083cf7f8a3e72` (declared as `REVISION` in `remote/common.py`). GPU jobs refuse to run if the checkout is at another commit or has tracked local changes under `app/` (staged or not), and `health_check` runs the same check: it reports `compatible: false` with the reason in `compatibility_reason`. Normalization does not use Fusion and is unaffected.

To qualify another revision, inspect the installed pipeline at that revision (the processors, worker and layout definitions the adapter calls), run the test suite, run `scripts/smoke_gpu.py` on the target hardware, record the evidence in `VERIFICATION.md`, and only then change `REVISION`. Never relax or bypass the guard without that qualification.

## Setup

The examples below use placeholders. Replace each one with your own value.

| Placeholder | Meaning |
| --- | --- |
| `<checkout>` | Absolute path of your clone on the client machine |
| `gpu-host` | SSH alias of the GPU workstation |
| `/srv/visomaster` | Workstation media root: every source media file lives below it |
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

   The script validates these variables exactly as the server does, uploads the three standard-library files into a new directory under `releases/`, runs the new release's health check against `remote-config.json`, and only then has the new release select itself. Under the lock that job submission takes, the release checks again for queued or running jobs, points the `remote` symlink at itself and removes all but the five newest releases, keeping any release an active job still uses. Each job runs the release that accepted it. The script refuses while jobs are queued or running, or when it cannot list them. A first install is recognized only when the deployment has no `remote` entry at all. `--force` skips both job checks, and cleanup still keeps the releases active jobs use; while a job submitted by an earlier version is active, no release is removed.

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
| `VISOMASTER_DOWNLOADS` | no | `~/Downloads/visomaster-mcp` | Local directory for `download_artifact` and `export_recipe`; absolute or starting with `~`, relative paths are rejected |

SSH calls use `BatchMode=yes` and share one multiplexed connection through `~/.ssh/visomaster-%C`, which persists for 120 seconds after the last call. Nothing relies on SSH environment forwarding or remote shell profiles.

### Remote configuration

`<VISOMASTER_REMOTE>/remote-config.json` is read by every workstation process: control commands, detached supervisors and GPU workers. Each process finds it relative to its own code, so all clients and reconnects see the same settings. `VM_CONFIG` (an absolute path) overrides the location; it exists for tests and unusual layouts. Unknown keys are rejected.

| Key | Required | Default | Rules |
| --- | --- | --- | --- |
| `version` | yes | | `1` |
| `media_root` | yes | | Existing absolute directory. Media paths and references must resolve inside it (or to an upload or job artifact in `state_dir`) and be self-contained video or image files; playlists and concat lists are refused |
| `fusion_dir` | no | `<media_root>/tools/visomaster-fusion` | Fusion checkout containing `.git`. Detection, preview, render and `get_settings` read its layout definition files; normalization does not |
| `state_dir` | no | `<media_root>/runtime/visomaster-mcp` | Inside `media_root` or on separate storage, but not `media_root` itself, an ancestor of it or inside `fusion_dir`; created with mode 0700. Only `uploads/*` (complete files) and `jobs/<id>/artifacts/**` of completed jobs are accepted as inputs; requests, logs, locks, staging and partial files never are, even when `state_dir` lies inside `media_root`. Paths are resolved before the check, so symlinks and `..` cannot reach other files |
| `gpu` | no | `"0"` | GPU index as a string, without leading zeros. Sets `CUDA_VISIBLE_DEVICES` (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`) and names the lock `gpu-<index>.lock` |
| `runner` | yes | | Path, or list of arguments, whose first element is an absolute executable. Never passed through a shell |
| `lock_dir` | no | `state_dir` | Existing writable directory for `gpu-<index>.lock` and the GPU queue (`gpu-<index>.queue.json`, `gpu-<index>.queue.lock`) |

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
| `detect_faces` | Queue detection at a selected frame and produce reference crops, or sample up to 64 frames (`sample_frames` or `frames`) and group faces into identities (optional `similarity_threshold`) with one reference crop each; only verified crops get a `reference_path`. Queues as `interactive` unless `priority` says otherwise |
| `analyze_references` | Queue a GPU check of up to 20 candidate source images: face count, size, sharpness, duplicates and embedding consistency, with a crop per usable image and a labelled `contact.jpg`. Queues as `interactive` unless `priority` says otherwise |
| `submit_swap` | Queue a frame/image preview (`interactive`) or a video render (`normal`) with explicit face mappings, optional per-source weights and expected fingerprints; results include a per-mapping `swap_report` summary and the `swap-report.json` artifact, and renders write labelled review images (optional `report`: `review_images`, `expected_visible`, `min_coverage`); renders may add `segments` that change mappings, parameters or controls, or pass frames through, by source-frame range; optional `priority` overrides the class |
| `compare_variants` | Queue one job that runs 2–6 named settings variants over the same 1–4 still frames (`interactive`) or the same clip of up to 120 frames (`normal`), with labelled original-versus-variant grids, full-resolution outputs, a `compare.json` manifest and a ready `submit_swap` request per variant; optional `priority` overrides the class |
| `preflight_swap` | Check a `submit_swap` request without starting it: the same arguments, plus `analyze_references` to queue GPU reference analysis. Returns `readiness` and per-check codes, severities, evidence and corrective actions, recorded GPU qualification and an input/configuration `fingerprint`; nothing is reserved |
| `normalize_media` | Queue a CPU-only FFmpeg conversion to a render-ready video (CFR, zero start, even dimensions, rotation and flips applied); does not wait for the GPU |
| `job_status` | State, frame progress and result manifest; `wait_seconds` (≤ 50) long-polls for the next change, including a change of queue position or reason; GPU jobs report `priority` and `sequence`, and queued ones a `queue` standing; log tail on request or failure |
| `list_jobs` | Recent job summaries shared across clients (kind, media, frame selection or sampling, timing, status, progress, and for GPU jobs `priority`, `sequence` and, while queued, `queue`), filterable by `status` and `kind`, `limit` up to 200 |
| `cancel_job` | Request cancellation, then poll for terminal state |
| `resume_render` | Queue a new render (its own `job_id`, linked by `resume_from`) that repeats a failed or cancelled render's request and reuses its verified checkpoint chunks after checking inputs, settings, models and versions; refuses renders with tracking or smoothing |
| `delete_job` | Permanently remove a terminal job's directory, logs and artifacts |
| `prune_storage` | Report (dry run, default) or delete old terminal jobs, unused uploads and crash leftovers, with the bytes each removal frees |
| `view_artifact` | Return a completed job's PNG/JPEG as an inline MCP image, such as a detection's `contact.jpg` grid of labelled crops |
| `download_artifact` | Stream a completed artifact to the client |
| `save_recipe` | Save a new immutable revision of a named recipe from a completed preview or render job, or from an explicit request: pinned settings, mappings, weights, segments, input fingerprints and code/model provenance |
| `get_recipe` | List recipes, or return one revision's document and input retention; `verify` hashes inputs and models now and lists what changed |
| `run_recipe` | Queue a preview or render from a recipe revision, with explicit, recorded media, target, range, mode, report and priority substitutions; refuses changed inputs, defaults, models or versions unless `accept_changes` |
| `export_recipe` | Write a recipe revision as portable JSON (configuration and provenance, no media or weights) to the local downloads directory |
| `import_recipe` | Store an exported recipe file as a new revision after checking its format version, integrity, input paths and bytes, and settings |
| `delete_recipe` | Delete one revision or all revisions of a recipe; its inputs are left to retention |

Multi-frame detection embeds every detected face with the configured recognition model and groups faces greedily, largest first, into identities. A face joins the identity whose mean embedding scores highest on Fusion's similarity scale (50 + 50 × cosine similarity, the scale `SimilarityThresholdSlider` uses) if that score reaches `similarity_threshold`, by default the slider's default of 60. Two faces from the same frame never share an identity. A profile view can score below 50 against the same person's frontal views, about as low as a different person, so it can form its own identity; lowering `similarity_threshold` merges it but can also merge different people who never share a frame, so check the contact sheet. Each identity reports its appearances, frames, up to five sample boxes and a representative crop from the largest of its five largest faces whose crop is verified.

A crop is verified when it is a reference `submit_swap` accepts for that same face: it re-detects as exactly one face under the same detector limit, the detected box overlaps the original by an IoU of at least 0.5, and its embedding scores at least the default `SimilarityThresholdSlider` against the original detection. When the 30% padding takes in a neighbouring face, a crop padded by 10% is tried. Single-frame `faces` and multi-frame `identities` carry `reference_verified`; an unverified crop keeps its `artifact` for inspection but has `reference_path: null`, is marked `*` and framed in red on `contact.jpg`, and a result with no verified crop carries a `warning`.

### Reference analysis

`analyze_references` takes `references` (absolute image paths, 1–20), an optional `swap_model` (default Fusion's default swap model), optional `thresholds` and the same detector `controls` as `submit_swap`. It is a queued GPU job that shares the GPU lock. It reads each image exactly as `submit_swap` does: the same detector with its 20-face limit, and the ArcFace embedding of `swap_model`, which is the embedding the swap averages. Each reference in `result.references` has:

- `id` (`r0`, `r1`, … in request order), `path` and `sha256` of the bytes analyzed;
- `usable`: true when `submit_swap` accepts it (exactly one detected face with an embedding), with `faces`, `box` (or `boxes` when several are found) and `image_size`;
- `face_pixels`: the shorter side of the detector's box in source-image pixels;
- `sharpness`: the variance of the 4-neighbour Laplacian (OpenCV's `Laplacian` with aperture 1) of the box's 8-bit luma resized to 112 × 112, the ArcFace input size, in squared luma levels. Faces smaller than 112 pixels are enlarged and score lower;
- `consistency_score` and `findings` (`code`, `severity`, `message`), and `artifact`, the padded crop `reference-N.png` of a usable reference.

Errors mark images `submit_swap` refuses: `unreadable`, `no_face`, `multiple_faces` and `unrecognized`. Warnings mark images to review: `small_face` (below `min_face_pixels`, default 112), `blurry` (below `min_sharpness`, default 40), `exact_duplicate` (same bytes as an earlier reference), `near_duplicate` (scores at least `duplicate_similarity`, default 95, against an earlier reference), `outlier` and `inconsistent_pair`. `result.duplicates` lists each redundant reference with the earlier one it repeats. Outlier evidence compares embeddings on Fusion's similarity scale (50 + 50 × cosine): each usable reference scores against the normalized mean of the other usable references, excluding byte-identical copies, and scores below `outlier_similarity` (default `SimilarityThresholdSlider`'s default, 60) are flagged. `result.consistency` states the model and basis. It needs three distinct usable references; with two it reports only their `pair_similarity` (flagging both when it is below the threshold), and with one it is unavailable. A low score suggests another person, an unusual view or unusual lighting; it does not establish identity. The defaults come from a small sample of face crops: sharp crops measured about 90–850 and the same crops blurred measured about 10–90, so tune thresholds on your own material.

Analysis never removes, edits or reorders references. `contact.jpg` shows every reference labelled with its id and first finding (`+` when there are more), unusable ones framed in red; `analysis.json` holds the result with the Fusion revision. Raw embeddings are not returned.

### Source weights and fingerprints

A `FaceMapping` may add `source_weights`, one finite, nonnegative number per `source_faces` entry with a positive sum. The adapter merges source embeddings as Fusion's own merge does (`EmbMergeMethodSelection` "Mean", then L2 normalization), weighted: weight 2 counts like two copies of an image, only ratios matter, and weight 0 excludes an image, which is then not read. Omitting `source_weights` gives the equal average. It may also add `source_sha256`, one SHA-256 per source as `analyze_references` reported; submission then fails if any file changed since the analysis. Whether or not they are given, submission records every target and source image's SHA-256 in `resolved.json`, and the worker decodes exactly those bytes or fails the job. Reference images are limited to 256 MiB. Preview and render results list under `references`, per mapping, the target's SHA-256 and each source's path, SHA-256, weight and normalized `share`. Requests that omit the new fields keep their earlier form, so earlier `job_id` retries stay idempotent. Analysis results are not reused by later jobs: each swap recomputes embeddings from the fingerprinted bytes with its own swap model, so rerun the analysis after changing images or `SwapModelSelection`.

### Swap preflight

`preflight_swap` takes `submit_swap`'s arguments, validated by the same client models and, on the workstation, by the same `validate_request` submission runs, and answers whether that request can start now. Without `analyze_references` it loads no Fusion code or models, takes no lock and creates no job: it reads files and runs FFprobe, git and nvidia-smi. A `job_id`, if given, is checked against existing jobs rather than generated.

The result has `readiness`: `blocked` (some check says submission or the job would fail), `pending` (reference analysis still running), `warnings` or `ready`. Each entry of `checks` has `check`, `status` (`passed`, `warning`, `blocked`, `pending` or `not_checked`), `severity` (`error`, `warning` or `info`), a stable `code`, `message`, `evidence` and `actions`, each a description with a `tool` and `arguments` to call, or an operator step without a tool. `not_checked` lists the checks that were skipped, with the reason as their message.

| Check | Codes |
| --- | --- |
| `configuration` | `config_valid`, `config_invalid` (everything else is then `not_checked`) |
| `fusion_revision` | `fusion_compatible`, `fusion_incompatible` (the revision guard `health_check` and GPU workers use) |
| `deployed_code` | `code_matches`, `code_mismatch`, `client_code_unknown` |
| `job_id` | `job_id_generated`, `job_id_free`, `job_exists` (a retry returns that job), `job_id_conflict`, `job_id_invalid` |
| `media_path`, `references` | `path_ok`, `references_ok` (with each image's `sha256` to pin as `source_sha256`), `path_missing`, `path_unreadable`, `path_not_absolute`, `path_outside_roots`, `artifact_job_incomplete`, `reference_too_large`, `reference_changed` |
| `settings` | `settings_ok`, `setting_unsupported`, `setting_invalid`, `swapper_unsupported`, `layouts_unreadable` |
| `model_assets` | `models_present`, `model_missing` (files under the Fusion checkout's `model_assets`, for the base settings and every segment's overrides), `model_unmapped` (a selection this adapter has no file mapping for: not checked), `model_list_unreadable` |
| `media` | `media_ready`; `variable_frame_rate`, `variable_timestamps`, `missing_timestamps`, `nonzero_start`, `odd_dimensions` and `rotated_or_mirrored` suggest `normalize_media`; `empty_range`, `frame_out_of_range`, `expected_visible_beyond_range`, `segment_outside_range`, `all_passthrough`, `hdr_unsupported`, `matrix_unsupported`, `unsupported_container`, `no_video_stream`, `no_video_frames`, `media_unreadable` |
| `storage` | `space_ok`, `low_space` (less than 1 GiB beyond the estimate), `insufficient_space`, `state_not_writable` |
| `gpu` | `gpu_observed` (compute capability, VRAM class, memory use), `gpu_unobserved` |
| `qualification` | `qualified`, `unqualified` (a warning, never a refusal) |
| `request` | `request_valid`, or the refusal code of `submit_swap`'s validation |
| `reference_analysis` | `not_requested`, `analysis_pending`, `references_usable`, `reference_warnings`, `reference_problems`, `analysis_cancelled`, `prerequisite_blocked` |

The media check is the worker's own: a render's checks of constant frame rate, timestamps from zero, a nonempty range, `report.expected_visible` within it, even dimensions and supported colors, and of its `segments` against the frames the video actually has (submission can check them only against the requested range), or a preview's frame and colors. Timestamps are read only through the requested frames, as the render reads them. Media that `normalize_media` can fix carries that action with the same `media_path`; nothing is converted, and the request is not changed. Storage compares free space in `state_dir` with an estimate (per rendered frame, twice the larger of the source's bytes per frame and width × height / 20, plus 64 MiB) and is an observation, as is the GPU's memory use.

Qualification is evidence, not permission. `QUALIFICATIONS` in `remote/common.py` records what was exercised on real hardware: hardware class (architecture, compute capability, VRAM class), Fusion revision, modes, detector, swap model, restorer, expression restoration, masks, the number of mappings and of weighted sources, with the date and the `VERIFICATION.md` section. A request no record covers is `unqualified`, with the nearest record and the fields that differ; it is still allowed. Add a record only after a GPU qualification run.

`fingerprint` has SHA-256 values of the `request` (without `job_id`), the `inputs` (media path, size and modification time; each reference's SHA-256) and the `configuration` (remote settings identity, Fusion revision, worker code, size and modification time of each required model file), and `sha256` over the three; `checked_at` is the check time. Compare fingerprints to see whether anything changed since an earlier preflight. A preflight reserves nothing and is not consulted by submission: `submit_swap` validates again and the worker rechecks Fusion, the media and the reference bytes, so pin references with the reported `sha256` values.

With `analyze_references: true`, and when Fusion and the request pass, the preflight queues one ordinary `analyze` job per distinct set of weighted sources, swap model and controls (`interactive`, shared GPU lock and queue). Its `job_id` is `preflight-` plus a hash of that analysis request, so repeating the preflight returns the same job and its results instead of queuing another, and changed bytes or settings queue a new one. Follow it with `job_status` and preflight again; cancel it with `cancel_job`. A cancelled or failed analysis stays as it is until `delete_job` removes it. Targets are not analyzed; use `detect_faces` crops with `reference_verified`.

## Typical workflow

1. Call `health_check` and `get_settings`.
2. Use an absolute workstation media path under `media_root`, or upload a local file with `upload_media`.
3. Call `detect_faces` on a representative frame, or with `sample_frames` (for example 16) to find everyone who appears across a video. Call `job_status` with `wait_seconds: 50` until `completed`, then inspect the returned crops with `view_artifact`; `contact.jpg` shows them all in one labelled grid.
4. Use the selected crop's or identity's `reference_path` as `target_face` (it is null for unverified crops). Supply one or more reference images of the replacement identity as `source_faces`. Each reference image must contain exactly one detected face. With several, run `analyze_references` on them, view its `contact.jpg`, and pass its `sha256` values as `source_sha256`, plus `source_weights` to exclude (0) or down-weight images it flags.
5. Call `preflight_swap` with the arguments you will submit and resolve what it reports as `blocked`. Submit a `preview`, inspect its image, then submit a short `render`. After reviewing that clip, submit the full frame range. To choose between settings, run `compare_variants` on the same frames or a short clip, view its `compare-f<N>.jpg` grids, and submit the chosen variant's `submit_swap` arguments from the result.
6. Follow the job with `job_status` and `wait_seconds`, then download `result.mp4` or `result.png` after completion. Check the result's `swap_report` for warnings, such as a mapping that never swapped, and view the `review-NN.jpg` images it lists.

Renders reject variable frame rate video, video whose timestamps do not start at zero, odd frame dimensions and rotated or mirrored display metadata. When a render fails for one of these reasons, call `normalize_media` with the same `media_path` (optionally `fps`, such as `25` or `"30000/1001"`), wait for the job, and use its result's `media_path` (the job's `normalized.mp4` artifact) for detection, preview and rendering. Frame indices then refer to the normalized timeline.

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

For video, use `mode="render"`, `start_frame` and `max_frames`. A null `max_frames` processes through the end. Frame indices start at zero. Give each intended job a stable `job_id` (1–64 letters, digits, underscores or hyphens, starting with a letter or digit) so a retried submission returns the same job, even after its inputs have been moved or pruned. Reusing an ID with changed settings is rejected, and an empty ID is an error; only an omitted or null ID is generated. Submit a new ID to retry a failed or cancelled job.

Each mapping swaps only its single best-matching face per frame, because the adapter turns on Fusion's `SwapOnlyBestMatchEnableToggle` by default. Without it, any face that clears the mapping's `SimilarityThresholdSlider` is swapped, which on two-person footage swapped the other person intermittently. Pass `"controls": {"SwapOnlyBestMatchEnableToggle": false}` only when one identity appears several times in a frame and every instance should change. Best match cannot help in frames where the target is absent and another face clears the threshold; if a preview or short render shows that, raise `SimilarityThresholdSlider` (75 removed it on the qualified clip).

### Resuming an interrupted render

Every render is encoded in chunks of 250 source frames, counted from `start_frame` (chunk n holds frames `[start_frame + 250n, start_frame + 250(n + 1))`, the last one clipped to the range). Each chunk is checked (frame count, constant-rate timestamps from zero, dimensions, pixel aspect and codec parameters) before it is published, and `checkpoint/manifest.json` then commits it with its SHA-256 and the render's accounting after its last frame. The manifest also records the complete range, the chunk size, the FFmpeg build and the render's provenance: the SHA-256 of the media and of every reference, the effective settings, the model files those settings load, the Fusion revision, the adapter code and the configuration identity, as recipes record them; the job's supervisor hashes them before the render joins the GPU queue (progress stage `fingerprinting`, queue reason `starting`), so other jobs can use the GPU meanwhile, and the worker reuses those hashes for every file whose identity, size and modification and change times are unchanged, hashing any other file again. The output joins the chunks by stream copy, so no frame is encoded twice, and cuts the audio once over the whole range; it is validated as before, and every frame's timestamp is checked as well.

When a render fails or is cancelled, `resume_render` with its `job_id` as `from_job_id` (and a new `job_id`) queues a new render with the same request. The earlier job keeps its terminal state and is not changed; the new one reports `resume_from` in `job_status` and `list_jobs`. Its submission, and its supervisor again before the job joins the queue, hash the inputs and models, and both the submission and the worker before anything is reused compare everything recorded; any difference, a different FFmpeg build or a request that resolves differently refuses the resume with the list of changes, and the render must be submitted fresh. Renders with `FaceTrackingEnableToggle` or `KPSSmoothingEnableToggle` are refused: their tracking and smoothing state is not checkpointed, and restarting it at the resume point would not match an uninterrupted render. The worker reuses the longest run of chunks from the start whose files still have their recorded bytes, deletes anything else (partial or unlisted writes, a damaged chunk and the chunks after it, whose accounting builds on it) and renders the remaining frames, so no frame is missing or rendered twice. The result's `checkpoint` reports `chunks`, `chunk_frames`, `reused_chunks`, `reused_frames` and `resumed_from`, and its `swap_report`, `plan` and images cover the whole range as an uninterrupted render's would: `before.jpg` and `preview.jpg` show the first processed frame decoded from the source and from `result.mp4`, as review images do. A retry with the same `job_id` and arguments returns the same job, a different `from_job_id` or `priority` under that ID is refused, and a completed, queued or running render cannot be resumed. Cancellation never starts another attempt; only `resume_render` does.

### Swap reports

Every successful preview and render writes `swap-report.json` (version 1) and returns its compact `swap_report` summary in the result. Each requested mapping is reported by its zero-based index in `mappings`, so a mapping that never swaps cannot hide in the aggregate `swapped_faces`. For every processed frame, the report records why each mapping was or was not swapped, using only what Fusion's own matching observed:

| Reason | Meaning |
| --- | --- |
| `swapped` | Fusion's swap step completed for a face of this mapping. This says nothing about how the result looks |
| `no_faces` | The detector found no face. This shows absence, not a failure |
| `unscored` | No scored face cleared the threshold, but some detected faces never reached matching: too small or without valid landmarks, or (with tracking on) filtered out by Fusion's SequentialDetector. Their scores are unknown |
| `below_threshold` | Every detected face was scored, and none reached this mapping's `SimilarityThresholdSlider` |
| `claimed_by_other` | A face cleared this mapping's threshold but matched another mapping better |
| `not_swapped` | A face matched this mapping best, but Fusion swapped no face for it |

Coverage is given against explicit denominators: all requested frames, frames with detected faces, and frames inside `expected_visible` ranges. Scores are Fusion's similarity (50 + 50 × cosine similarity) for faces that reached matching, never invented for others. A frame is ambiguous for a mapping when two or more faces matched it best, or when a face clearing its threshold also clears another mapping's.

Suspicious intervals are zero-based source-frame ranges `[start_frame, end_frame)`: `end_frame` is the first frame after the interval. Times are seconds computed from the exact frame rate: `start_time` and `end_time` on the source clock (`frame / fps`), and `output_start_time` in `result.mp4`, so ranges stay correct when `start_frame` is above zero. The kinds are `expected_missing` (unswapped frames inside an `expected_visible` range, whatever the reason), `dropout` (unswapped frames, at least one with faces, between two swapped frames), `claimed_by_other`, `not_swapped` and `ambiguous`. Frames without faces are reported but never flagged on their own, because they do not prove that the person should have been visible. Each mapping lists at most 100 intervals; `intervals_omitted` counts the rest.

Renders write up to `review_images` (default 6, at most 24) JPEGs named `review-01.jpg` and so on in the job's artifacts. Each image shows the decoded source frame beside the same frame of the published output, under a label with the mapping, source frame, time and reason. They come first from the best-scoring frame of each mapping that never swapped, then from the middle frames of suspicious intervals, by kind and length.

Warnings (`never_swapped`, `expected_missing`, `suspicious_intervals`) are advisory, and a job with warnings still completes. A render still fails when no face swaps at all (outside passthrough segments; see Segments), and it fails when Fusion reports a per-face error, which names the source frame and, when it can be attributed, the mapping. A render passed `min_coverage` fails when any mapping's swapped share falls below it. The share is computed over that mapping's `expected_visible` frames when it has ranges, and over all requested frames otherwise. Failed and cancelled jobs publish no report.

Pass these options to a render as `report`, for example:

```json
{"review_images": 4, "min_coverage": 0.9,
 "expected_visible": [{"mapping": 1, "start_frame": 120, "end_frame": 240}]}
```

`expected_visible` lists at most 100 ranges inside the requested frames; a range that ends past the video's last frame fails the render before processing. Previews accept no `report` options and have no times or review images.

### Segments

A render can apply different intent to different source frames while keeping one timeline. `segments` lists at most 100 ranges in ascending order, each a half-open `[start_frame, end_frame)` of zero-based source frames (`end_frame` is the first frame after it) with a stable `id`, and either `"passthrough": true` or at least one change:

- `mappings`: indices into `mappings` that swap there; the others are not requested in those frames;
- `parameters`: Fusion parameters applied over every active mapping's own, such as a higher `SimilarityThresholdSlider` for a profile shot;
- `controls`: controls over the request's, such as `DetectorScoreSlider`.

```json
{"start_frame": 0, "max_frames": 300, "segments": [
  {"id": "profile", "start_frame": 0, "end_frame": 120,
   "parameters": {"SimilarityThresholdSlider": 75}},
  {"id": "titles", "start_frame": 120, "end_frame": 150, "passthrough": true},
  {"id": "two-shot", "start_frame": 150, "end_frame": 240, "mappings": [1]}]}
```

Frames outside every segment use the base mappings, parameters and controls. Segments that overlap, are out of order, change nothing, combine passthrough with a change, name unknown mappings or settings, or set `SwapModelSelection`, `DetectorModelSelection`, `FaceTrackingEnableToggle` or `KPSSmoothingEnableToggle` (fixed for the whole render) are refused at submission. A segment that starts before `start_frame` or ends after the rendered range is clipped to it; one with no rendered frame, or a plan whose rendered frames are all passthrough, is refused at submission when the requested range shows it, and otherwise (`max_frames` null or past the video's end) before models load. `preflight_swap` reports both as blocked media (`segment_outside_range`, `all_passthrough`). Passthrough frames are encoded exactly as decoded, without detection. Frame count, order, dimensions, frame rate, duration and audio are those of a uniform render of the same range. Tracking and keypoint smoothing state is reset at every segment boundary and gap edge.

The render writes `plan.json`: each segment as given with its `applied` frames and whether it was `clipped`, and every run (segment or gap) with the effective controls and per-mapping parameters it used. The result and the swap report carry `plan.runs`, each run with its `frames`, `swaps` and `swapped_frames`. Passthrough frames count as `not_requested` in the swap report, so they never lower coverage. The render fails when no face swaps outside passthrough segments; a segment that swapped nothing gives the warning `segment_never_swapped`. Previews and `compare_variants` accept no segments.

### Recipes

A recipe is a named, reusable swap configuration. `save_recipe` takes either a completed preview or render `job_id` or an explicit request in `submit_swap`'s form, and stores a new revision `<state_dir>/recipes/<name>/<revision>.json` (format `visomaster-recipe`, version 1). A revision holds:

- `request`: the explicit swap request a new job runs: resolved media path and its `media_sha256`, the range (`frame`, or `start_frame` and `max_frames`), every adjustable control and per-mapping parameter at its effective value, mappings with their `target_sha256`, `source_sha256` and any `source_weights`, and `report`, `segments` and `priority` when given;
- `provenance`: the full effective settings (Fusion and adapter defaults included), the Fusion revision, the adapter release and the SHA-256 of its three files, the configuration identity, and per model its path, the SHA-256 Fusion declares and the SHA-256 of its file. Models are those the settings load, as `preflight_swap` determines them (detector, recognizer, swapper with its ArcFace models, and an enabled restorer, occluder, XSeg or expression restorer, including one a segment enables), plus, for a recipe saved from a job, those the job had loaded. A revision saved before restorer, mask and expression models were recorded reports such a model as not recorded when the revision was saved; save a new revision to record it;
- `frame_convention`, `origin` (`request`, `job` or `import`), `note`, `retained`, `created_at`, and `sha256`, the digest of everything else.

Revisions are immutable: saving again under a name adds the next revision, and a deleted revision's number is never reused. Face references are copied into the content-addressed upload store (`uploads/<sha256><suffix>`, PNG, JPEG or WebP) and the request points at the copies, so deleting the detection job that produced a crop does not break a recipe. Retention keeps every file a recipe revision names, including a job whose artifact (for example `normalized.mp4`) is the media. Media in `media_root` and references of other types are unretained: `get_recipe` reports them as such, and `missing` when they are gone. Deleting a recipe deletes no inputs; `prune_storage` may remove them later once no recipe or queued or running job names them.

`run_recipe` hashes every input and model file, compares the result with the recipe and submits a new job. Changed bytes at the same path, changed settings or defaults, changed or missing model files, different adapter code, Fusion revision or configuration make it fail with a readable list, unless `accept_changes` is true; `get_recipe` with `verify` shows the same list without submitting. Unavailable inputs always fail. Substitutions are explicit: `media_path`, `targets` (one `target_face` per mapping), `frame_range`, `mode`, `report` and `priority`. Media with different content is a new clip, which needs explicit `targets` from a detection on that clip and an explicit `frame_range` (and `report` when the recipe has `expected_visible` ranges); a mode change needs a `frame_range`, and a preview drops the recipe's `report` and `segments`. A recipe with `segments` cannot run on media with different content, because its segments name frames of the recorded clip; save a recipe for the new clip. The job's request and result carry `recipe`: name, revision, the document's `sha256`, the invocation, its substitutions and any accepted changes. The job hashes the media again before it takes the GPU, and the worker fails it if the media changed. A retry with the same `job_id` and arguments returns the existing job.

`export_recipe` writes `<downloads>/recipes/<name>-r<revision>.json`. It contains configuration and provenance, never media or model weights. `import_recipe` refuses other format versions, a `sha256` that does not match the content, unknown or malformed fields, inputs outside the allowed roots, inputs whose bytes differ and settings the installed Fusion rejects. An input missing at its recorded path is found by digest in the upload store, so uploading the same file with `upload_media` first is enough; the relocation is recorded in `origin`.

Saving, verifying and running a recipe hash the media, references and model files in a streamed pass on the workstation, so a long video takes a while (these calls allow 15 minutes); status polls never hash. A recipe reproduces configuration, inputs and versions, not bits: the GPU, its driver and CUDA libraries are not recorded, and outputs are expected to be equivalent rather than bit-identical.

### Variant comparisons

`compare_variants` takes the arguments of `submit_swap` that every variant shares (`media_path`, `mappings`, `controls`) and a list of `variants`, each with a `name` and optional `parameters` and `controls`. A variant's `parameters` are merged over every mapping's `parameters`, and its `controls` over the common `controls`; nothing else differs between variants. It compares either explicit still `frames` or a contiguous clip (`start_frame` and `max_frames`) of a constant-frame-rate video, as renders require. Nothing picks a best variant; the choice stays with you.

| Limit | Value |
| --- | --- |
| Variants | 2–6, names of 1–20 letters, digits, `_` or `-`, unique ignoring case, with distinct overrides |
| Still frames | 1–4 distinct frames (images: `[0]`) |
| Clip | 1–120 frames per variant |
| Source frame size | At most 3840 × 2160 pixels (8,294,400); every output keeps the source dimensions |
| Grids | Cells of 480 × 480, 4 per row: at most 1952 pixels wide and 2 rows |

Submission validates each variant's merged request as `submit_swap` would, so an unknown setting, an invalid value, a duplicate name or an over-limit selection is refused before anything is queued, naming the variant. The job is one durable job like any other: a stable `job_id` makes retries return it, a changed request is a conflict, `job_status` and `list_jobs` show its progress (`stage: comparing`, with `variant`, `variants` and `name`; a long poll returns when the variant changes), and `cancel_job` stops every variant. Variants run one after another, in request order, in a single worker that holds the GPU for the whole comparison. Still comparisons queue as `interactive` and clip comparisons, which render up to six clips, as `normal`.

Every variant gets the same decoded frames. Stills are decoded once and passed to each variant, which processes each frame as a separate preview. A clip is decoded once per variant with the render's frame-accurate path, and the job fails the variant if its frames differ from the others' (each variant's `input_sha256` covers every frame it received) or from the separately decoded original stills. Each clip covers the same range, frame count, rate and duration, and its frames sit at the same output times. Clips have no composite: each `v<index>-<name>.mp4` carries the source's first audio stream over the compared interval, cut and encoded as renders do, so all clips share identical audio and nothing is duplicated.

Artifacts, in the order the result lists them:

- `compare-f<N>.jpg`: one grid per compared frame (for clips, the first, middle and last frame): the original, then each variant labelled `<index> <name>`. Each tile is scaled to fit a 480-pixel cell at the source's display aspect ratio (pixel aspect applied), so tiles are directly comparable. View these first.
- `compare.json`: the manifest, identical to the job result plus the Fusion revision.
- `original-f<N>.png`: the decoded source frames at full resolution.
- `v<index>-<name>-f<N>.png`: each variant's full-resolution output of those frames, before any encoding.
- `v<index>-<name>.mp4` (clips): the variant's clip, validated like a render's `result.mp4`.
- `v<index>-<name>-swap-report.json`: the variant's swap report (for stills, a list of one preview report per frame).

The manifest records `source` (dimensions, `sample_aspect_ratio`, compared `frames` and the SHA-256 of each original frame, and for clips the rational `fps`, `start_frame`, `end_frame` (exclusive), `frame_count`, `start_time` on the source clock and `duration`), the ordering and audio policy, and per variant: `index`, `name`, `status`, `error`, `overrides`, `settings` (the effective value of every adjustable control and, per mapping, every adjustable parameter after Fusion's and the adapter's defaults), `references` (fingerprints and weights used), `input_sha256`, `artifacts`, swap report summaries, `swapped_faces` and `submit_swap`. `submit_swap` holds arguments that repeat that variant alone with exactly its merged settings: a preview of the first compared frame, or a render of the compared range, with each mapping's `source_sha256` pinned to the bytes the comparison used. Add a `job_id`, change `frame` or the range as needed, and submit it.

A variant that fails, for example on a Fusion error or because it swapped no face in any compared frame, keeps its `error`, loses its artifacts and appears as a red `FAILED` tile. The job still completes when at least one variant succeeded, but then `complete` is false, `failed_variants` names the failures and `warning` explains them; only a result with `complete: true` is a full comparison. A frame where a variant swapped nothing is reported in its swap report, not as a failure. When every variant fails, the job fails with each variant's reason, and as for every failed job its artifacts are withheld.

## Storage

Job state, logs and artifacts live in `<state_dir>/jobs/JOB_ID`, and recipe revisions in `<state_dir>/recipes/NAME`. Uploaded media lives in `<state_dir>/uploads`, named by SHA-256 so identical uploads share one file. Terminal jobs remain until `delete_job` or `prune_storage` removes them. `prune_storage` deletes terminal jobs finished more than `older_than_days` ago (default 30) and uploads untouched for that long, unless a saved recipe, a queued or running job, or a failed or cancelled render that keeps a checkpoint (and so can still be resumed) uses their files as input, as well as staging and trash directories left by crashes and partial uploads idle for over an hour. It reports byte totals and is a dry run unless `dry_run` is false; the dry run applies the same checks. Partial files from cancelled or failed jobs remain for diagnosis, and artifact tools expose only completed jobs. A render's checkpoint (`checkpoint/` in its job directory) is removed when the render completes; a failed or cancelled render keeps it, and so stays resumable, until `delete_job` or `prune_storage` removes the job. A resumed render takes its own hard links to the chunks it may reuse when it is submitted, so deleting or pruning the earlier attempt afterwards cannot remove them; its links go with it. Each downloaded artifact has its own job subdirectory. Repeating the same download atomically replaces that local copy.

MCP GPU jobs share a process lock across all client sessions. Fusion's desktop GUI has its own GPU workload, so coordinate large GUI renders with MCP renders.

## GPU scheduling

GPU jobs run one at a time, in an order every client and every deployment sharing the `lock_dir` agrees on. Each job gets a class and a sequence number at submission, both reported by `job_status` and `list_jobs` and kept for the job's lifetime:

| Class | Default for | Order |
| --- | --- | --- |
| `interactive` | `detect_faces`, `submit_swap` previews, `analyze_references`, still `compare_variants` | Starts before waiting normal jobs, oldest first |
| `normal` | `submit_swap` renders, clip `compare_variants`, and any GPU job kind not listed | Oldest first; becomes due under the fairness rule |

Pass `priority` to `detect_faces`, `analyze_references`, `submit_swap` or `compare_variants` to choose the class, for example `"normal"` for a batch of detections that should not delay someone else's previews. A retried submission with the same `job_id` keeps its sequence, so reconnecting never moves a job forward or back. Normalization does not use the GPU and is not queued.

Fairness rule: when both classes are waiting, an interactive job starts next unless the waiting normal work is due. It becomes due after 3 interactive jobs have started in a row while it waited, or 300 seconds after the first of them started. Starting a normal job resets both, so a steady stream of previews delays a waiting render by at most three previews (and no preview starts more than five minutes after the first of them), and previews then go first again.

Scheduling only happens between jobs. A running render keeps the GPU until it finishes, fails or is cancelled; a preview never interrupts it, the queue never stops other processes, and Fusion's GUI does not take part. Yielding to previews at chunk boundaries of long renders is separate future work.

While a GPU job is queued, its `queue` object reports:

- `position`: its place in the order the rule would produce now, with 1 next. It is advisory and can change: new interactive jobs move normal jobs back, a due render moves forward, and jobs that finish, fail or are cancelled move the rest forward.
- `reason`: `next` (first and the GPU is free), `gpu_busy` (first, waiting for the running job), `behind_earlier` (older jobs of its own class are ahead), `behind_interactive` (a normal job waiting for interactive jobs), `behind_due_normal` (an interactive job waiting for a due normal job), `cancel_requested` (no longer eligible) or `starting` (no queue entry yet: being admitted, or a render or recipe job whose inputs are being hashed before it joins, shown by progress stage `fingerprinting`; later jobs may start meanwhile).
- `interactive_streak`: interactive jobs started in a row ahead of waiting normal work (the rule allows 3).
- `waiting_seconds`: time since submission.

There is no start-time estimate. `job_status` with `wait_seconds` returns early when the position or reason changes. Cancelling a queued job takes it out of the order at once; a cancellation that arrives just as the job starts stops it like any running job, and one that arrives after the job succeeded leaves it completed.

## Troubleshooting

| Message | Fix |
| --- | --- |
| `VISOMASTER_HOST and VISOMASTER_REMOTE not set` | Add both variables to the client registration (setup step 8) and start a new session |
| `Check VISOMASTER_HOST (...) with: ssh ...` | The alias does not resolve or key authentication failed; run the printed command until it succeeds without a prompt |
| `Remote configuration not found at ...` | Create `remote-config.json` in the deployment directory (setup step 6) |
| `...: <key> ...` from the configuration | Fix the named key in the named file; `health_check` shows `config_path` and `config_error` |
| `VM_... is no longer read` | Remove the variable from the workstation environment and put the value in `remote-config.json` |
| `Fusion revision changed to ...`, `Fusion app has tracked local changes` or `compatible: false` | Read `compatibility_reason`. Check out the pinned revision and discard tracked changes under `app/`, or qualify them (see [Fusion revision guard](#fusion-revision-guard)) |
| `The configured runner must exec the Fusion Python interpreter` | End the runner with `exec "$FUSION_PYTHON" "$@"` |
| `Cannot start the configured runner ...` | Fix the runner path or its permissions |
| `Remote configuration changed after submission ...` | The file changed while the job was queued; resubmit |
| `Cannot list active jobs (... query): ...` from `deploy.sh` | The installed release could not report its jobs; fix the named error, or use `--force` if no jobs can be running |
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

The suite runs locally with temporary job backends, stdio clients and an SSH shim that clears the environment, as real SSH does. Linux-only behavior (the parent-death signal) is exercised only on a workstation. There is no CI.

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

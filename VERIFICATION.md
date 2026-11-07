# Verification

All evidence below comes from a single installation: one Linux workstation with Maxwell-generation NVIDIA GPUs (compute capability 5.2, 24 GB VRAM class), Fusion commit `f2d6f5ebe190b631ee5c8972332083cf7f8a3e72`, and a runner that adds the CUDA libraries from Fusion's environment. Job identifiers, release names, host names and paths are kept in the installation's private notes, not here.

# Verification on 2026-11-07

The adapter was exercised against the installed Fusion commit on GPU 0 of that workstation, through its existing runner.

## Automated checks

Nine tests passed. They cover actual MCP stdio initialization and tool discovery, structured output schemas, rejection of malformed mappings, shared GPU serialization, idempotent submission and conflict detection, status from a separate process, queued and running cancellation, worker failure, orphaned-job reconciliation, the completion/status race, path traversal and symlink rejection, setting types and bounds, interrupted uploads, and VFR timing detection. Ruff lint and formatting checks passed.

## Real GPU and protocol checks

The integration fixture was a 960×540, 24000/1001 fps music-video excerpt with a synthetic replacement-face reference. Requests went through an MCP SDK client, the client-side server, SSH job control and the deployed adapter.

| Check | Observed result |
| --- | --- |
| Tool discovery | All 11 tools exposed through stdio |
| Face detection | CUDA YOLO detection returned reference crops and inline images |
| Default preview | Inswapper128 swapped the selected target and returned PNG/JPEG artifacts |
| Default short video | 12 frames, 0.5005 seconds, 24000/1001 fps, H.264 video and AAC audio |
| Restored preview | Inswapper128, GFPGAN-v1.4 and LivePortrait expression restoration completed |
| Restored short video | 24 frames, 1.001 seconds, 24 successful face swaps, H.264 and AAC |
| Duplicate submission | The same job ID and request returned the existing job |
| MCP reconnection | A new session found the detached render job |
| Running cancellation | The reconnected session cancelled the render while loading models |
| Artifact transfer | Video and image artifacts streamed to the client |
| Upload round trip | The downloaded video uploaded successfully, retained 24 frames and audio, and passed another detection job |
| GPU release | The GPU returned to its initial idle allocation (about 155 MiB) after testing |

The restoration run and a final transfer-and-detection run kept their manifests, logs and artifacts in the installation's job state directory.

The default and restored preview images were visually inspected. Automated checks verified output duration, frame count, dimensions and audio presence. Audio listening and a full-length video quality review remain outside this verification.

## Coverage limits

The qualified combination is Yolov8, Inswapper128, optional GFPGAN-v1.4 and optional LivePortrait expression restoration. Other exposed embedding-based swapper choices and mask settings follow Fusion's pipeline but still need GPU qualification. Multiple target mappings and multiple source-reference averaging are implemented and remain untested with real multi-person footage. Long renders, cancellation during a blocked CUDA kernel, and a real workstation reboot were not exercised.

Claude and Codex configuration examples were prepared. The real MCP protocol was tested with the official SDK client. Native Claude/Codex UI sessions were not launched as part of verification, and their global configuration files were not modified at that time.

# Improvement release verification on 2026-11-07

The improvement backlog (epic `vmcp-harden-pqj`) was deployed as a new release with `scripts/deploy.sh`, which migrated the previous flat `remote/` directory into a release. The same Fusion commit, GPU and runner were used. Each feature was first qualified on a separate development deployment, then the merged release was checked through the production path.

## Automated checks

39 tests passed, and Ruff lint and formatting checks passed. New coverage includes end-to-end stdio tools through a local SSH shim (upload, download, inline images, remote errors), long-poll status, list filters, deletion and pruning with active-input protection, content-addressed uploads, release pinning across a deploy, engine error surfacing, the cancel-after-success race, packet-timestamp validation, multi-frame identity clustering, normalization command construction (including bit depth and color tags) and a real FFmpeg normalization when FFmpeg is installed.

## Production checks

| Check | Observed result |
| --- | --- |
| Deployment | `health_check` reported the new release, compatible revision and `code_matches: true` |
| Smoke script | Detect, preview, 24-frame render (24 swaps, AAC, 5.8 fps) and reconnect cancellation |
| Tool discovery | 14 tools |
| Multi-frame detection | 8 sampled frames: main subject as identity 0 in 8/8 with a verified crop; tiny background faces marked unverified; contact sheet returned inline |
| Long-poll status | Three `job_status(wait_seconds=50)` calls covered the detect job; no log tail on success |
| VFR rejection | A render of a VFR clip failed with "Variable frame timestamps detected; run normalize_media first" |
| Normalization | A VFR clip normalized to CFR media; a render from frame 12 decoded exactly frame 12 and failed correctly with zero matches because the fixture has no face |
| 10-bit normalization | A 10-bit BT.709 clip normalized to yuv420p10le, High 10, BT.709 tags and limited range kept |
| Upload dedup | Second upload of the same file reused the existing path (0.16 s, then 0.08 s) |
| Deletion and pruning | `delete_job` removed a terminal job; `prune_storage` dry run reported nothing older than 365 days |

## Development qualification evidence

| Feature | Run | Result |
| --- | --- | --- |
| Engine error surfacing | Detection of an out-of-range frame | `state.error` carried "Cannot read frame 999999 …" |
| SSH multiplexing | — | `health` call 0.44 s → 0.12 s; bare SSH 0.33 s → 0.015 s |
| Timing validation | — | 200 s 720p: 39.5 s → 1.06 s; 600 s file: 12.4 s → 0.41 s |
| Exact decoding | Render from frame 30 | Bit-identical to sequential decode at 25 indices; render from frame 30 had 0.0 ms A/V lag |
| Atomic deploy | Detection during a deploy | Migration, refusal with active jobs, release pinning of a running job and retention of five releases |
| Long-poll and listing | One detection | Two status calls covered a full detect job |
| Cleanup and dedup | Scratch state only | Prune and delete removed exactly the expected entries; 20 MB re-upload 0.94 s → 0.09 s |
| Multi-frame detection | A two-person clip, a three-person clip, and a preview from an identity crop | Two and three people separated; a profile view formed a separate identity; preview with an identity crop swapped only the target |
| Normalization | VFR, offset, odd-size, rotated and audio-skewed clips | VFR, start offset, odd size, rotation and early/late audio normalized; renders of rotated and odd-size outputs completed |
| Color and bit depth | Renders of a 10-bit BT.709 clip and an 8-bit BT.601 limited-range clip | Identity round trip +4 dB PSNR and the −2 level RGB shift removed; 10-bit output High 10 with source tags |
| Sequential tracking | Toggle combinations on one- and two-person clips | Toggles off: byte-identical output, 16–25% slower; ByteTrack: no change on these clips; landmark smoothing: about 3% lower face-box jitter |

## Remaining limits

Tracking remains opt-in. The observed flicker on two-person footage was a non-target face crossing the default similarity threshold (51/96 frames at 60, 0/96 at 75); `vmcp-best-match-dnq` tracks exposing best-match swapping. Clustering uses Fusion's default threshold and can split profile views (`vmcp-detect-threshold-ljb`). HDR input is rejected, not tone-mapped; HDR rejection, 4:2:2/4:4:4 output and ByteTrack recovery of missed detections were not exercised live. Long renders, a workstation reboot and native Claude/Codex UI sessions remain unverified.

# Portability verification on 2026-11-07

Issue `vmcp-5ve` removed the built-in host, path and runner defaults, moved workstation settings into `remote-config.json` and added the privacy audit. The checks below cover that change. Production was not switched to it as part of this run.

## Automated checks

On the macOS client, 138 tests passed and 2 were skipped: the GNU-only release switch (qualified on the workstation below) and the owner-pattern audit pass, which runs only where the local pattern file exists (it passed when supplied). Ruff lint and formatting passed. New coverage: configuration validation and location, the legacy-variable guard, configuration identity through submit, supervisor and worker (changed, invalid, deleted and moved configurations all end `failed`), GPU lock sharing between deployments, health without `git` or `nvidia-smi`, client settings validation and quoting, a custom remote interpreter for uploads, `deploy.sh` parity with the server, a deploy substitute through an SSH shim, and a non-default end-to-end run through a shim that clears the environment, with a reconnect. The MCP tool list (names, schemas and annotations) is byte-identical to the previous version. A cross-version check through the shim covered upload, submit, status, list, download and inline view with the new client against the previous release's remote files, and with the previous client against the new ones. The new client's `health_check` against the running previous release succeeded.

## Isolated live check

A clean clone of the branch was set up following the README from the runner step onwards, with every setting non-default: a separate SSH alias defined in a scratch `ssh -F` configuration and selected through `VISOMASTER_SSH`, a separate deployment directory and state directory, a runner written from `config/fusion-runner.example.sh`, and a project-scope Claude Code registration. The runner reproduced the existing launcher's library path exactly. The isolated configuration set `lock_dir` to the production state directory and used the same GPU index, so both deployments shared one GPU lock. Fixtures were the same music-video excerpt and synthetic reference face as above.

| Check | Observed result |
| --- | --- |
| Deploy into a legacy flat layout | Migrated `remote/` into a `legacy-*` release and switched the symlink; preflight accepted the configuration |
| Health and settings | `config_error: null`, `compatible: true`, `code_matches: true`, every dependency check true; settings returned |
| Shared GPU lock | With the production lock held by `flock`, an isolated detection stayed queued for about 28 seconds, then ran when the lock was released |
| Detection, preview, render | 3 faces and a contact sheet; preview swapped 1 face (inspected visually); 12-frame render with 12 swaps, 24000/1001 fps, 0.5005 s, AAC, 4.4 fps |
| Transfers | Content-addressed upload, downloads and an inline image through the custom alias |
| Smoke script | Detect, preview, 24-frame render (24 swaps, AAC, 1.001 s), duplicate submission, reconnect and cancellation |
| Release pinning | A plain deploy refused while a job was queued; a forced deploy switched releases while the queued job's supervisor and worker ran from the release it was submitted with, with the runner exec-ing Fusion's interpreter |
| Pruning | After further deploys exactly the five newest releases remained; releases contained only the three files and Python's bytecode cache |
| Invalid configuration | A `state_dir` outside the media root made the preflight refuse and remove the new release with the symlink unchanged; `health_check` reported the error and other tools refused |
| Missing runner and unset host | The configuration error named the runner path; a client without `VISOMASTER_HOST` returned the setup message |
| Hostile media path | A path with quotes, `$()` and backticks reached the workstation only as data; nothing executed |
| Production unaffected | Production health and job list were identical before and after; the isolated deployment and its jobs were then deleted |

The project-scope registration was confirmed with `claude mcp get`; the tool calls used the MCP SDK client with the same command and environment, because a native client session was not opened. Codex and Claude Desktop examples received syntax checks only.

## Remaining limits

Only one installation, one GPU generation and a macOS client were exercised. A Linux client, other GPUs and CUDA stacks, and a fully portable runner on a fresh Fusion installation remain unverified. The production migration is recorded below.

## Production migration

The existing installation was upgraded in place. Client registrations received explicit `VISOMASTER_HOST` and `VISOMASTER_REMOTE` values before the client defaults were removed, and the production `remote-config.json` was written with the previous effective settings before deployment. `scripts/deploy.sh` preflighted the new release against that file and switched releases with no jobs active.

| Check | Observed result |
| --- | --- |
| Health | New release active, `code_matches: true`, compatible revision, no configuration error, settings hash identical to the pre-migration baseline |
| Durable jobs | Every job recorded before the migration was still listed afterwards |
| Smoke script | Detect, preview (1 swap), 24-frame render (24 swaps) and reconnect cancellation |
| Clients | The Claude Code registration connected; the Codex registration carries the same settings |

Rollback is pointing the `remote` symlink back at the previous release and reverting the client commit; the registrations work with both versions.

# Detection reference and best-match qualification on 2026-11-07

Issues `vmcp-review-remediation-qpn.11`, `vmcp-detect-threshold-ljb` and `vmcp-best-match-dnq` were qualified on the same workstation class (Maxwell, compute capability 5.2, 24 GB VRAM class) and Fusion commit, through a separate deployment root with its own state directory that shared the production GPU lock. Jobs were driven through the MCP SDK client and the client-side server; production was not changed. The GPU and the Fusion GUI were idle before the runs.

## Automated checks

`tests/test_detect.py` drives detection with a stubbed detector and recognizer over synthetic scenes: a neighbour inside the padding, a lone neighbour re-detection, an overlapping face of another identity, clean crops in both modes accepted by `configure_faces`, all-unverified results, clustering thresholds, threshold validation, list summaries, the client schema and the best-match control and default.

## Detection references

| Run | Observed result |
| --- | --- |
| Two-person clip, single frame | Both faces verified; each crop contained only its own face (contact sheet inspected) |
| Close-up clip, 8 sampled frames | The main subject verified in 8/8; four small, blurred background faces unverified, framed in red and marked `*`, with null `reference_path` |
| Three-person clip, single frame and 16 sampled frames | Single frame: all three faces verified. Sampled: three people verified; the mother's strong profile view (2 frames) formed a fourth identity and was unverified |
| Reference check diagnostics on the three-person clip (frames 102, 114, 126) | Every crop re-detected one face. Frontal and three-quarter faces re-recognized at 62–99 against their full-frame embeddings with IoU 0.69–0.94; the profile views scored 47–50 (IoU about 0.6), and their 10% crops detected no face |
| Two-person clip, verified crop as target | Used as the render target below; matched on 96/96 frames |

The profile view's own crop scores below the default threshold against the face it came from, so it would not reliably match that face at swap time; the previous count-only check had marked it verified.

## Clustering threshold

On the three-person clip with 16 sampled frames, thresholds 60 (default), 55 and 50 gave four identities, the mother's profile view separate. Thresholds 45 and 40 gave three identities, each with 16 appearances and a verified crop. Pairwise scores between different people in these frames were 45–53, between the profile and the mother's three-quarter view 60–64 and between the profile and her frontal views lower; a threshold low enough to merge the profile is therefore close to the scores of different people, and the contact sheet should confirm a merge. `list_jobs` showed `sample_frames` and `similarity_threshold` for multi-frame detection and `frame` for single-frame detection.

## Best-match swapping

96-frame renders of the two-person clip with a synthetic source face, target the man. Swapped frames were counted per face region by pixel change against the source (a region counts when more than 1% of its pixels change by a mean level above 25); this measure reproduced the earlier tracking baselines exactly (51/96 and 0/96).

| Best match | `SimilarityThresholdSlider` | Non-target frames swapped | Target frames swapped | Swaps reported | Speed |
| --- | --- | --- | --- | --- | --- |
| off | 60 | 50/96 | 96/96 | 146 | 5.2 fps |
| on (new default) | 60 | 0/96 | 96/96 | 96 | 6.8 fps |
| off | 75 | 0/96 | 96/96 | 96 | 6.6 fps |
| on | 75 | 0/96 | 96/96 | 96 | 6.7 fps |

Frames 0, 32, 64 and 95 of the default render were inspected: the woman's face was unchanged and the man's swapped. `get_settings` reported `SwapOnlyBestMatchEnableToggle` with default `true`, and `health_check` reported `code_matches: true`.

## Remaining limits

Best match was qualified on one two-person clip with one mapping. Multiple mappings, crowds, mirrors or a target absent while a look-alike remains, and tracking combined with best match were not exercised. Native Claude/Codex UI sessions were not opened.

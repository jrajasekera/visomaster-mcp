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

# Review remediation production deployment on 2026-11-07

All 29 findings from the whole-tree code review (`vmcp-review-remediation-qpn`) plus `vmcp-best-match-dnq` and `vmcp-detect-threshold-ljb` were deployed together. The hardware class and Fusion revision are the same as in the runs above.

## Automated checks

`uv run --locked pytest -q` gave 380 passed on macOS (FFmpeg 8.1). A copy of the same tree on the Linux workstation (FFmpeg 6.1, system Python 3.12) gave 360 passed and 1 skipped, with the 19 privacy-audit tests deselected because that copy had no Git checkout. Ruff check, Ruff format and the privacy audit were clean. No test processes remained after either run.

## Production checks

| Check | Observed result |
| --- | --- |
| Deployment | `scripts/deploy.sh` checked the queued and running queries separately, preflighted the candidate and promoted it through `control.py promote` under the submit lock |
| Health | `code_matches: true`, `compatible: true`, no `compatibility_reason` or `config_error`; the configuration hash matched the previous release |
| Existing jobs | Earlier jobs still listed after the switch |
| GPU smoke test, close-up clip with a synthetic source face | Preflight recorded `hash_status: match`. Single-frame detection verified the subject's crop and left four small background faces unverified with null `reference_path`. Preview swapped 1 face. A 24-frame render completed with 24 swaps, 24 frames, 1.001 s and AAC audio. The reconnected session found the disconnect render running, and cancellation was recorded as `passed` on the running path |
| Client | `claude mcp list` reported the server connected |

## Remaining limits

The supervisor-loss, mirrored-matrix, RGB, 10-bit, anamorphic, VFR and audio-overlap behaviors were verified with synthetic fixtures and stub Fusion workers on both FFmpeg versions, not with live GPU renders. Follow-ups: `vmcp-n88` (artifacts of non-completed jobs are accepted as inputs), `vmcp-y0h` (non-standard source matrices and limited-range gray under FFmpeg 8) and `vmcp-tsb` (descendants survive if the cancellation guard is killed on its own).

# Follow-up fixes on 2026-11-07

Resolves the three follow-ups listed above.

## Changes

- `vmcp-n88`: job artifacts are accepted as inputs only once the producing job's state is `completed`. This applies at submission, in the worker's revalidation and in `inspect_media`, with the state directory inside or outside the media root.
- `vmcp-y0h`: sources tagged with matrices the scaler cannot apply (ycgco, bt2020c, ictcp, smpte2085, chroma-derived and reserved) are refused before any artifact, with transcoding guidance. FFmpeg 8 rejected them, and 6.1 had silently decoded them with the wrong coefficients. Limited-range gray is now converted at the correct levels on both versions.
- `vmcp-tsb`: if the process-group guard dies, the supervisor stops the group at once and fails the job. If the supervisor dies as well, reconciliation stops only processes in the recorded group that carry the job's random token, rechecking each one through a pidfd before signalling.

## Automated checks

macOS (FFmpeg 8.1): `uv run --locked pytest -q` gave 471 passed, 2 skipped (two Linux-only guard tests); Ruff check and format and the privacy audit were clean. Linux workstation copy (FFmpeg 6.1, system Python 3.12): 453 passed, 1 skipped, with the 19 privacy-audit tests deselected because the copy had no Git checkout. No test processes remained on either machine. The new tests fail against the previous code: 32 of 36 for artifact inputs, 41 of 53 on FFmpeg 6.1 for matrices and gray, and all of the guard-loss cases.

## Production checks

The release deployed while no jobs were active. Health reported `compatible: true`, with no compatibility reason or configuration error. The GPU smoke test on the close-up clip recorded `hash_status: match`. Detection completed; the preview swapped 1 face; a 24-frame render completed with 24 swaps and AAC audio. The reconnected session found the disconnect render running, and cancellation passed on the running path. No worker, FFmpeg or supervisor processes remained afterwards, and `claude mcp list` reported the server connected.

# Feature qualification on 2026-11-08

Qualifies the eight features of `vmcp-harden-pqj`: reference analysis and source weights, swap reports, GPU scheduling, recipes, variant comparisons, swap preflight, render segments, and render checkpoints with resume. It also covers the fixes from the cross-feature review. Hardware class, Fusion revision, detector (`Yolov8`, model `YoloFace8n`) and recognizer (`Inswapper128ArcFace`) are the same as in the runs above. All fixtures were synthetic: the close-up clip and synthetic source faces used above, a 150-frame two-person clip, a 150-frame 25 fps multi-shot clip with a title card and audio beeps at known times, a 774-frame looped clip, a 72-frame clip in which the face turns to profile, a 4497-frame clip, a 10-minute clip, a 10-photo set of one synthetic person, and derived blurred, shrunken, duplicated, two-face, faceless and non-image references. Every request went through a stdio MCP client started from the checkout.

## Changes during qualification

- Reference analysis failed the whole job when any reference had no face or several: Fusion returns detections as a NumPy array, and the engine tested its truth value. The engine now tests its length, and the test doubles behave like the array. After redeployment, a 10-reference analysis that included a two-face image, a faceless image and a text file completed with `multiple_faces`, `no_face` and `unreadable` findings and full results for the other seven.
- The checkpoint refusal for one temporal control now reads "carries"; three `QUALIFICATIONS` records cover the combinations below.

## Automated checks

macOS (FFmpeg 8.1): `uv run --locked pytest -q` gave 771 passed, 2 skipped (two Linux-only guard tests). Ruff check, Ruff format and the privacy audit were clean. Linux workstation copy (FFmpeg 6.1, system Python 3.12): 753 passed, 1 skipped, with the 19 privacy-audit tests deselected because the copy had no Git checkout. No test processes remained.

## Deployment and baselines

Before the first deployment, the previous release rendered and previewed fixed requests as baselines. Afterwards health reported `code_matches: true` and `compatible: true` after each deployment, and the GPU smoke test passed with `hash_status: match`. Its reconnect cancellation now meets the job while queued (`passed_queued`), because a render spends about 2 s fingerprinting before it joins the queue; a separate cancellation of a running render reached `cancelled` in 1.3 s with no processes left and GPU memory back to idle.

| Comparison with the previous release | Observed result |
| --- | --- |
| Inswapper preview, one and two equally weighted sources | Pixel-identical |
| Inswapper render, 72 frames | Every video frame identical. Worker time 20.5–20.7 s against 19.9 s (+3–4%); frames per second unchanged |
| GhostFace-v1 preview, one and two sources | Changed as intended: the merged embedding is now L2-normalized, as in Fusion. Face-region PSNR 26–30 dB against the old output; identity transfer is visibly weaker (the result stays closer to the target) |

## Feature checks

| Feature | Observed result |
| --- | --- |
| Reference analysis | Eight references reported in order with stable ids: clear photos 528–576 px faces, sharpness 385–468, consistency 92–95; a byte copy `exact_duplicate`; a 1/6-scale copy `small_face` (87 px); a text file `unreadable`, framed red on the contact sheet. A single reference reports consistency as unavailable with the reason. With the preflight's `analyze_references`, results matched a direct analysis exactly |
| Source weights | Shares 0.5/0.5, 0.75/0.25 and 1/0 reported. Weights `[3, 1]` gave a result byte-identical to listing the first source three times. A weight-0 source was never read: a missing file at weight 0 gave the same pixels as the one-source preview. A pinned SHA-256 that no longer matched, including after editing the file, was refused at submission |
| Swap report | Two-person clip, three mappings (the third a person absent from the clip): coverage 110/110, 93/110 with a 7-frame `expected_missing` interval, and 0 with a `never_swapped` warning at best score 57.1. Per-mapping counts summed to `swapped_faces`; source and output times matched the start offset. Six review images labelled correctly and matched the output at their times (47–50 dB). `min_coverage: 0.9` failed the job naming both short mappings, with no artifacts. Tracking runs reported scored, unscored and below-threshold faces. Six review images cost about 3.3 s |
| GPU scheduling | During a 600-frame render, four previews, a second render and a fifth preview queued with reasons `gpu_busy`, `behind_earlier`, `behind_interactive` and `behind_due_normal`; they started in the order render, three previews, render, preview, so the queued render ran after three interactive jobs. The running render was never interrupted. A long poll returned within 0.2 s of a position change. Cancelled queued previews never started. An identical resubmission from a new session kept its sequence number; resubmitting an older job's request returned that job. A process monitor sampling every 0.5 s for 13.5 minutes never saw two Fusion workers, and the queue file ended empty |
| Recipes | A saved recipe recorded model hashes equal to the files on disk, the deployed code hashes and the Fusion revision; `verify` reported it unchanged. A replay was byte-identical to the original preview with the same swap report. Running it on other media without targets was refused; with a new target and range it completed and recorded the substitutions, and a render-mode run produced 12 frames with audio at `normal` priority. A dry-run prune at 0 days kept the recipes' retained inputs and a job whose output a recipe uses. Export and import preserved the request and provenance |
| Variant comparisons | Three variants on three frames ran as one interactive job; grids kept the aspect ratio, put the original first and labelled each variant; inputs had one hash, and each variant differed from the base only in its own override. A variant rerun alone with `submit_swap` was byte-identical. Three 48-frame clip variants (base, `GFPGAN-v1.4`, tracking) had identical frame counts, durations and audio, aligned to the source at 12/fps. A variant that matched nothing produced a red `FAILED` tile and `complete: false`; cancelling a six-variant job stopped its worker within 0.2 s and the next job started |
| Swap preflight | Checks for compatibility, code, models, GPU (compute capability 5.2, 24 GB class), qualification and 24 media frames took under 1 s, on a 10-minute clip too, and started no job or GPU process. It reported `variable_timestamps` with a `normalize_media` action, an unqualified swap model with the nearest record, a changed reference and an empty range. A queued analysis waited behind a running render, ran only after it and was reported cancelled, with a `delete_job` action, after cancellation |
| Segments | A 120-frame render of the multi-shot clip with four segments (a threshold override, one mapping only, a passthrough title card, a detector-score override) gave 120 frames of 4.8 s with the plan clipped to the range. Title-card frames matched the source at 48 dB, and the frame before the passthrough was swapped while the first frame of it was not. Audio onsets moved by exactly the 0.8 s start offset. Tracking with landmark smoothing gave the same results with no worker errors. Render-wide settings in a segment, all-passthrough ranges, overlaps and segments outside the range were refused |
| Checkpoints and resume | A 760-frame render was cancelled at frame 300 and resumed, reusing 250 frames through a hard link; that resume was cancelled in its third chunk and resumed again, reusing 500 frames. The result was byte-identical to an uninterrupted render, with identical swap reports; a segmented render with a passthrough crossing a chunk boundary also resumed to identical output, plan and report. Resuming a tracking render, or one whose source changed, was refused. Inputs of resumable renders were protected from pruning and deletion |
| Fingerprinting before admission | Renders, resumed renders and recipe runs showed the `fingerprinting` stage with queue reason `starting` before joining the queue: 1.8–2.0 s, almost all of it model hashing, so it does not grow with clip length. A preview submitted meanwhile started first. Hashing after admission took 0.05–0.2 s. Results were byte-identical to the same requests run earlier |

## Remaining limits

- Qualified combinations are those recorded in `QUALIFICATIONS`: `Yolov8` with Inswapper128 in previews and renders with up to three mappings and two sources per mapping, optionally with `GFPGAN-v1.4`; Inswapper128 previews with up to four sources; GhostFace-v1 previews with up to two sources. Other swap models, expression restoration with several mappings, occluder and XSeg masks were not exercised.
- The default outlier threshold of 60 flagged one unrelated synthetic face (55.3) but not two others (62.0 and 64.0), and a moderately blurred copy (sharpness 57.9) stayed above the `blurry` threshold of 40.
- Swap-report frames with no faces or dropouts were not exercised on the GPU; every frame of the two-person clip had a face.
- The swap report shows each mapping's base threshold even where a segment overrides it, and a resumed render's progress briefly reports a low rate because reused frames count in its total. A resumed render whose source changed before it started advises re-analysis instead of a fresh render.
- Native Claude and Codex UI sessions were not opened.

# Reference-analysis calibration on 2026-11-08

Calibrates the `analyze_references` defaults for `outlier_similarity` and `min_sharpness` (`vmcp-d84`), after the feature qualification above found that the outlier default of 60 missed unrelated faces at 62.0 and 64.0 and that a blurred copy at sharpness 57.9 was not flagged. Same hardware class (Maxwell, compute capability 5.2, 24 GB VRAM class), Fusion commit and detector (`Yolov8`, default controls) as above. Every measurement ran as an `analyze_references` job on the deployed release that preceded this change, through a stdio MCP client started from the checkout. The change alters only defaults, so the measured scores carry over.

## Method

All material was synthetic:

- **Photos**: eight photos of one generated person (angles, expressions, lighting) and single photos of three other generated people, 1024 × 1024 with faces of 497–576 pixels.
- **Video crops**: 42 face crops of 14 people (three frames each) cut with FFmpeg from boxes found by multi-frame `detect_faces` in seven generated 4K scenes and the two-person clip. Faces were 75–282 pixels; four of the people were children.
- **Altered copies**: 46 copies with Gaussian blur (sigma 1–32 on full-size photos, 1–8 on video crops), downscales to 1/2–1/8, downscale then blur, downscale then re-enlarge, darkening, gamma, halved contrast and JPEG quality 31.

Pairwise similarities came from analyses with `duplicate_similarity` 50: each reference then reports the first earlier reference scoring at least 50, which yields the first reference's row of the similarity matrix. 106 analyses of the 53 photos and crops in a circulant order covered all 1,378 pairs. 157 pairs scored below 50 and were clamped to 50; all were different people (photos against video crops), and clamping can only raise an intruder's score. Leave-one-out scores for every collection below were computed from the matrix with the engine's formula; on 112 reported scores that involve no clamped pair, the reconstruction matched within 0.08. Copies were scored against the eight photos or against their own person's other crops. Generated "same character" crops from different scenes scored only 59–72 against each other, so the generator does not keep identity across scenes. Those pairs, and two girls who may be one person, were left out of both classes.

## Outlier scores (`Inswapper128ArcFace`)

| Class | Count | Min | Median | 95th percentile | Max |
| --- | --- | --- | --- | --- | --- |
| Clean collections of 3–6 photos of one person, each reference | 896 | 84.6 | 92.5 | 94.4 | 95.1 |
| Clean collections of 3 video crops of one person, each reference | 42 | 68.1 | 85.3 | 92.3 | 93.4 |
| Different adult or photo added to 2–5 references of one person | 4,672 | 50.0 | 53.7 | 61.2 | 66.2 |
| The same, photos added to the person's photos | 144 | 54.6 | 62.5 | 64.8 | 66.0 |
| A child added to 2–3 crops of another child | 120 | 55.4 | 68.1 | 74.7 | 76.6 |
| Pairs: same person, photos / video crops | 28 / 42 | 82.0 / 61.5 | 89.2 / 81.6 | 91.8 / 91.6 | 92.6 / 92.8 |
| Pairs: different people, except child with child | 1,043 | 50.0 | 54.3 | 60.9 | 65.9 |

Three of the person's photos with one other person's photo added reproduced the earlier observation: medians 55.9, 62.3 and 63.8 for the three other people (ranges 54.9–65.8). The video crops of one person below 70 were a 102–115 pixel face (68.8) and a child turning away (68.1). An intruder also lowers the other references' scores, because their mean includes it; with two references of one adult plus an intruder, 14% of those references also fell below 70. The intruder was the lowest-scoring reference in 99.4% of collections with two references and in 99.5–100% with three to five.

| Default | Adult and photo intruders flagged | Clean references flagged (photos / video crops) |
| --- | --- | --- |
| 60 (former) | 93.4% (photos among photos: 33%) | 0 / 0 |
| 65 | 99.8% | 0 / 0 |
| 70 (new) | 100% | 0 / 2 of 42 |
| 75 | 100% | 0 / 6 of 42 |

## Other recognizers

20 of the references (five of the person's photos, the three other photos and 12 video crops) were analyzed with `SimSwap512`, `GhostFace-v1` and `CSCS`, 20 analyses each, covering all 190 pairs.

| Recognizer | Clean 3 photos, each | Photo intruder vs 3 photos | Video intruder vs 3 photos | Same-person pairs | Different-person pairs (max) |
| --- | --- | --- | --- | --- | --- |
| `SimSwapArcFace` | 87.3–96.6 | 57.7–64.5 | 50.0–60.3 | 67.0–94.7 | 63.8 |
| `GhostArcFace` | 87.3–95.5 | 55.2–65.4 | 50.0–57.3 | 78.0–93.8 | 65.3 |
| `CSCSArcFace` | 92.7–97.9 | 75.0–82.8 | 64.2–75.1 | 85.8–96.9 | 91.4 (children) |

Pairs of generated "same character" crops from different scenes are excluded here too. SimSwap and Ghost recognizers share Inswapper's scale. CSCS scores everything higher, so 70 would flag no one.

## Sharpness

Unaltered references measured 328.9–554.5 (photos) and 73.8–381.8 (video crops); the lowest was a 95-pixel crop enlarged to the 112-pixel grid.

| Copy (face size) | Sharpness | Score against the original | Leave-one-out against the other photos |
| --- | --- | --- | --- |
| Sigma 1 / 4 / 6 / 8 (576 px) | 389.8 / 123.2 / 63.4 / 37.1 | 97.0 / 96.9 / 96.5 / 95.9 | 94.8 / 94.1 / 93.5 / 93.1 |
| Sigma 12 / 16 / 24 / 32 (576 px) | 16.3 / 9.0 / 4.5 / 3.3 | 95.2 / 94.0 / 90.3 / 84.7 | 92.4 / 91.6 / 88.5 / 83.9 |
| Sigma 6 / 8 / 12 / 24 (another photo, 490 px) | 40.7 / 23.9 / 10.7 / 3.2 | 96.3 / 96.0 / 95.3 / 88.6 | 94.4 / 94.1 / 93.4 / 87.4 |
| 1/4 scale, then sigma 1 / 2 / 3 / 6 (144 px) | 110.3 / 37.1 / 16.7 / 5.0 | 97.1 / 96.0 / 95.2 / 90.1 | 94.2 / 93.1 / 92.5 / 88.3 |
| Video crops, sigma 1–6 (178–286 px) | 7.1–192.5 | 91.8–99.5 | |
| 1/2, 1/4, 1/6, 1/8 scale (288, 143, 96, 71 px) | 430.2, 328.5, 360.3, 215.6 | 97.0–98.3 | 94.5–94.8 |
| Darker, gamma 0.5 / 1.8, half contrast, JPEG q31 | 251.6, 264.2 / 381.2, 122.3, 391.6 | 93.5–98.3 | 92.2–94.5 |
| The original photo | 468.0 | | 94.5 |

The recognizer's embedding barely moves until little detail is left at its 112-pixel input: copies scored at least 95 against their original down to a sharpness of about 16, and about 90, the score of two different photos of the person, near 5. Visible full-size blur such as sigma 6 on a 576-pixel face keeps the identity. Sharpness separates blurred from clear faces only at recognition scale: the sigma-6 photo (63.4) measured lower than the softest unaltered video crop (73.8), so no threshold flags it without also flagging clear small faces. Four blurred copies of three video crops were not usable: detection found no face or two.

## Decisions

- `outlier_similarity` defaults per recognizer: 70 for `Inswapper128ArcFace`, `SimSwapArcFace` and `GhostArcFace` (3.8 above the highest adult or photo intruder; flags the 62.0 and 64.0 cases; below every clean photo collection and 40 of 42 clean video crops), and 85 for `CSCSArcFace` (2.2 above its highest intruder, 7.7 below its lowest clean photo). Fusion's matching default of 60 no longer applies.
- `min_sharpness` stays 40: below every unaltered reference, and above the point where the embedding starts to drift. The how-to now explains that `blurry` measures recognition-scale detail and that 70 would also flag visibly soft full-size photos, together with clear small faces.
- The sharpness metric is unchanged: it is already normalized to the recognition input, which is what the swap uses. Measuring at source resolution would flag blur the embedding ignores.

## Automated checks

macOS (FFmpeg 8.1): `uv run --locked pytest -q` gave 775 passed, 2 skipped (two Linux-only guard tests). Ruff check, Ruff format, `git diff --check` and the privacy audit were clean. A new test fails on the former default: a reference scoring 64.2 against two references of another person is flagged at the default, and the defaults follow the recognizer.

## Remaining limits

- All people were synthetic; real photos, other ethnicities and ages may score differently. Different generated people scored fairly high against each other (photos 55–66); if unrelated real people score lower, as is commonly reported for ArcFace models, the margins on real material are wider.
- Children are not separated by any default: a child scored up to 76.6 against another child's crops, above clean crops of one child (68.1).
- The other recognizers rest on 20 references, and CSCS on few video pairs. CSCS also rated three of ten pairs of distinct photos of one person at 95 or more, so its default `duplicate_similarity` of 95 can mark distinct photos `near_duplicate`.
- Halving contrast lowered sharpness from 468 to 122; Laplacian variance scales with contrast, so a very flat, sharp face can measure low.
- The new defaults were not yet run on the GPU after deployment.

# Tools reference

The server provides 24 tools. Paths are absolute workstation paths unless a parameter is named
`local_path`, which is a path on the client. Tools that queue work return the job's state at
once; follow it with `job_status`.

Annotations: **read** tools change nothing, **write** tools create jobs or files, and
**destructive** tools delete data.

| Group | Tools |
| --- | --- |
| Installation | [`health_check`](#health_check), [`get_settings`](#get_settings) |
| Media | [`upload_media`](#upload_media), [`inspect_media`](#inspect_media), [`normalize_media`](#normalize_media) |
| Faces | [`detect_faces`](#detect_faces), [`analyze_references`](#analyze_references) |
| Swapping | [`preflight_swap`](#preflight_swap), [`submit_swap`](#submit_swap), [`compare_variants`](#compare_variants), [`resume_render`](#resume_render) |
| Jobs | [`job_status`](#job_status), [`list_jobs`](#list_jobs), [`cancel_job`](#cancel_job), [`delete_job`](#delete_job), [`prune_storage`](#prune_storage) |
| Artifacts | [`view_artifact`](#view_artifact), [`download_artifact`](#download_artifact) |
| Recipes | [`save_recipe`](#save_recipe), [`get_recipe`](#get_recipe), [`run_recipe`](#run_recipe), [`export_recipe`](#export_recipe), [`import_recipe`](#import_recipe), [`delete_recipe`](#delete_recipe) |

## Common arguments

### job_id

1–64 letters, digits, underscores or hyphens, starting with a letter or digit. Omit it, or pass
`null`, to generate one. Resubmitting an ID with the same arguments returns the existing job; with
different arguments it is rejected. See [Job IDs](jobs.md#job-ids).

### priority

`"interactive"` or `"normal"`: the GPU queue class. The default depends on the tool; see
[GPU queue](jobs.md#gpu-queue).

### controls and parameters

`controls` apply to the whole request; `parameters` apply to one mapping. Only these names are
accepted. `get_settings` lists their defaults, choices and limits on your installation.

| Kind | Names |
| --- | --- |
| Controls | `DetectorModelSelection`, `DetectorScoreSlider`, `MaxFacesToDetectSlider`, `FaceTrackingEnableToggle`, `ByteTrackTrackThreshSlider`, `ByteTrackMatchThreshSlider`, `ByteTrackTrackBufferSlider`, `KPSSmoothingEnableToggle`, `KPSEmaAlphaSlider`, `SwapOnlyBestMatchEnableToggle` |
| Parameters | `SwapModelSelection`, `SwapperResSelection`, `SimilarityThresholdSlider`, `FaceRestorerEnableToggle`, `FaceRestorerTypeSelection`, `FaceRestorerBlendSlider`, `FaceRestorerDetTypeSelection`, `FaceFidelityWeightDecimalSlider`, `FaceExpressionEnableBothToggle`, `BorderTopSlider`, `BorderBottomSlider`, `BorderLeftSlider`, `BorderRightSlider`, `OverallMaskBlendAmountSlider`, `OccluderEnableToggle`, `DFLXSegEnableToggle`, `StrengthEnableToggle`, `StrengthAmountSlider` |

The adapter changes three of Fusion's defaults: `DetectorModelSelection` is `Yolov8`,
`MaxFacesToDetectSlider` is 20 and `SwapOnlyBestMatchEnableToggle` is true. The swap models
`AlphaFace` and `DeepFaceLive (DFM)` are not supported.

### FaceMapping

One entry of `mappings`: which face to replace, and with whom.

| Field | Type | Required | Rules |
| --- | --- | --- | --- |
| `target_face` | string | yes | Image of the face to replace, containing exactly one face. Use a verified `reference_path` from `detect_faces` |
| `source_faces` | list of strings | yes | 1–20 images of the replacement person, each with exactly one face |
| `source_weights` | list of numbers | no | One per source, in order: finite, nonnegative, with a positive sum. 0 excludes an image, which is then not read. Omit for equal weights |
| `source_sha256` | list of strings or nulls | no | One SHA-256 (64 lowercase hex digits) per source, as `analyze_references` reported. Submission fails if a file changed. `null` pins nothing; a weight-0 source is never checked |
| `parameters` | object | no | Parameters for this mapping |

A request has 1–20 mappings. Each face image is limited to 256 MiB.

Source embeddings are averaged with the weights, then L2-normalized, as Fusion's own "Mean"
merge does. Only the ratios of the weights matter. Requests that omit `source_weights` and
`source_sha256` keep their earlier form, so earlier `job_id` retries stay idempotent.

### Report options

`report` applies to renders only. Previews accept no `report` options.

| Field | Type | Default | Rules |
| --- | --- | --- | --- |
| `review_images` | integer | 6 | 0–24: the most labelled before-and-after images to write |
| `expected_visible` | list | `[]` | At most 100 ranges `{"mapping", "start_frame", "end_frame"}` where a mapping's person is known to be visible. `mapping` is 0–19, `end_frame` is exclusive. Each range must lie inside the rendered frames; one that ends past the video's last frame fails the render before processing |
| `min_coverage` | number | none | Greater than 0, at most 1. The render fails if any mapping's swapped share falls below it |

```json
{"review_images": 4, "min_coverage": 0.9,
 "expected_visible": [{"mapping": 1, "start_frame": 120, "end_frame": 240}]}
```

### Segments

`segments` applies to renders only. It is a list of at most 100 segments in ascending order.

| Field | Type | Rules |
| --- | --- | --- |
| `id` | string | 1–64 letters, digits, `_` or `-`, starting with a letter or digit; unique in the request |
| `start_frame` | integer | First zero-based source frame, at least 0 |
| `end_frame` | integer | First source frame after the segment; greater than `start_frame` |
| `passthrough` | `true` | Leave these frames exactly as decoded. Excludes the other changes |
| `mappings` | list of integers | 1–20 distinct indices into `mappings` that swap here; the others are not requested. Default: all |
| `parameters` | object | Parameters applied over every active mapping's own |
| `controls` | object | Controls applied over the request's |

Submission refuses segments that overlap, are out of order, change nothing, combine passthrough
with a change, name unknown mappings or settings, or set `SwapModelSelection`,
`DetectorModelSelection`, `FaceTrackingEnableToggle` or `KPSSmoothingEnableToggle`.

A segment that starts before `start_frame` or ends after the rendered range is clipped to it. A
segment with no rendered frame, or a plan whose rendered frames are all passthrough, is refused:
at submission when the requested range shows it, otherwise (`max_frames` null or past the end of
the video) by the worker before models load.

## Installation

### health_check

**Read.** No arguments. Reports the configuration, the Fusion revision, the deployed release,
the GPUs and the tools the workstation needs.

| Field | Meaning |
| --- | --- |
| `release` | Name of the deployed release |
| `code` | SHA-256 of the deployed `common.py`, `control.py` and `engine.py` |
| `code_matches` | Whether `code` equals your clone's `remote/` files; `null` if they are missing |
| `supported_revision` | The Fusion revision this release supports |
| `revision` | The Fusion checkout's revision |
| `compatible`, `compatibility_reason` | Whether GPU jobs may run, and why not |
| `config_path`, `config_error` | The configuration file read and its error, if any |
| `checks` | `git`, `nvidia-smi`, `ffmpeg`, `ffprobe`: whether each is on the `PATH`; `python`: the workstation Python version; `state_writable`, `lock_dir_writable` |
| `gpu`, `gpus` | The configured index, and `nvidia-smi` output: index, name, memory used and total, utilization |
| `app`, `state`, `input_root`, `lock_dir` | The effective `fusion_dir`, `state_dir`, `media_root` and `lock_dir` |
| `runner` | `argv`, and whether its first element is an `executable` file |
| `config_sha256` | Identity of the effective configuration |

When the configuration is invalid, only `release`, `code`, `code_matches`,
`supported_revision`, `config_path`, `config_error`, `checks`, `revision` (null), `compatible`
(false) and `compatibility_reason` are returned.

### get_settings

**Read.** No arguments. Returns `controls` and `parameters`: every accepted name with its
`default` and, where Fusion defines them, `options`, `min_value`, `max_value` and `help`. Also
returns `excluded_swappers`.

## Media

### upload_media

**Write.** Streams a local image or video to the workstation's upload store.

| Argument | Type | Rules |
| --- | --- | --- |
| `local_path` | string | A client file: `.png`, `.jpg`, `.jpeg`, `.webp`, `.mp4`, `.mkv` or `.mov`, from 1 byte to 20 GiB. `~` is expanded |

Returns `path` (the workstation path), `bytes`, `sha256` and `reused` (true when identical
content was already uploaded).

### inspect_media

**Read.** Returns FFprobe's JSON (`streams` and `format`) for a workstation file.

| Argument | Type | Rules |
| --- | --- | --- |
| `media_path` | string | An allowed input path ([Input paths](media.md#input-paths)) |

### normalize_media

**Write.** Queues a CPU-only conversion into a render-ready video. It does not wait for the
GPU. See [Normalization output](media.md#normalization-output).

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `media_path` | string | | The video to convert |
| `fps` | string or number | source rate | Output rate, such as `25`, `29.97` or `"30000/1001"`. Without it, the source's average rate snapped to a standard rate within 2% |
| `job_id` | string | generated | |

## Faces

### detect_faces

**Write.** Queues face detection. Single-frame detection returns `faces`; multi-frame detection
groups faces into `identities`. Both write `contact.jpg` when any face is found. See
[Face detection](results.md#face-detection).

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `media_path` | string | | Image or video |
| `frame` | integer | 0 | Single-frame detection: the zero-based frame |
| `sample_frames` | integer | none | Multi-frame: 1–64 evenly spaced frames |
| `frames` | list of integers | none | Multi-frame: 1–64 explicit frames |
| `similarity_threshold` | integer | 60 | Multi-frame only: 1–100 on Fusion's similarity scale. Faces join an identity at or above it. Default: Fusion's `SimilarityThresholdSlider` default |
| `controls` | object | `{}` | Detector controls |
| `job_id` | string | generated | |
| `priority` | string | `interactive` | |

### analyze_references

**Write.** Queues a GPU check of candidate source images. See
[Reference analysis](results.md#reference-analysis).

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `references` | list of strings | | 1–20 image paths |
| `swap_model` | string | Fusion's default swap model | The `SwapModelSelection` whose ArcFace embedding is compared |
| `thresholds` | object | | Optional `min_face_pixels` (1–4096, default 112), `min_sharpness` (0–1,000,000, default 40), `duplicate_similarity` (50–100, default 95, or 96, 95.5 and 98 when the swap model uses the SimSwap, Ghost and CSCS recognizers), `outlier_similarity` (0–100, default 70, or 85 when the swap model uses the CSCS recognizer) |
| `controls` | object | | The same detector controls as `submit_swap` |
| `job_id` | string | generated | |
| `priority` | string | `interactive` | |

## Swapping

### preflight_swap

**Write** (it can queue analysis jobs). Checks whether a `submit_swap` request can start, without
starting it. Takes every `submit_swap` argument, plus:

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `analyze_references` | boolean | false | Also queue GPU reference analysis of each mapping's weighted sources |

A given `job_id` is checked against existing jobs, not used. See
[Preflight checks](preflight-checks.md).

### submit_swap

**Write.** Queues a preview of one frame or image, or a render of a video range.

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `media_path` | string | | Image or video |
| `mappings` | list of [FaceMapping](#facemapping) | | 1–20 mappings |
| `mode` | string | `preview` | `preview` or `render` |
| `frame` | integer | 0 | Preview: the zero-based frame |
| `start_frame` | integer | 0 | Render: the first frame |
| `max_frames` | integer or null | null | Render: at most this many frames; null renders through the end |
| `controls` | object | `{}` | See [controls and parameters](#controls-and-parameters) |
| `report` | object | defaults | Renders only; see [Report options](#report-options) |
| `segments` | list | none | Renders only; see [Segments](#segments) |
| `job_id` | string | generated | |
| `priority` | string | `interactive` for previews, `normal` for renders | |

Renders reject variable frame rate, nonzero start times and odd dimensions, and every mode
rejects rotated or mirrored video ([Render requirements](media.md#render-requirements)). Results are described in
[Previews and renders](results.md#previews-and-renders).

### compare_variants

**Write.** Queues one job that runs 2–6 named settings variants over the same frames or clip.
See [Variant comparisons](results.md#variant-comparisons).

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `media_path` | string | | Image or video |
| `mappings` | list of [FaceMapping](#facemapping) | | Shared by every variant |
| `variants` | list | | 2–6 entries of `{"name", "parameters", "controls"}`. `name`: 1–20 letters, digits, `_` or `-`, starting with a letter or digit, unique ignoring case |
| `frames` | list of integers | none | Still comparison: 1–4 distinct frames (images: `[0]`) |
| `start_frame` | integer | 0 | Clip comparison: first frame |
| `max_frames` | integer | none | Clip comparison: 1–120 frames per variant |
| `controls` | object | `{}` | Shared controls |
| `job_id` | string | generated | |
| `priority` | string | `interactive` for stills, `normal` for clips | |

A variant's `parameters` are merged over every mapping's, and its `controls` over the shared
ones. Every merged request is validated like `submit_swap` before anything is queued.

### resume_render

**Write.** Queues a new render that reuses the verified chunks of a failed or cancelled render.
See [Resume a failed or cancelled render](../how-to/resume-a-render.md).

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `from_job_id` | string | | A failed or cancelled render. It is not changed |
| `job_id` | string | generated | The new job's ID |
| `priority` | string | the earlier render's explicit priority, otherwise `normal` | |

Refused when the earlier render is completed, queued or running, is not a render, has no
checkpoint manifest, used `FaceTrackingEnableToggle` or `KPSSmoothingEnableToggle`, or when
anything it depends on changed. A retry with the same `job_id` and arguments returns the same
job; a different `from_job_id` or `priority` under that ID is refused.

## Jobs

### job_status

**Read.** Returns a job's state, progress and, once completed, its `result`. See
[States](jobs.md#states).

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `job_id` | string | | |
| `wait_seconds` | integer | 0 | 0–50. Wait for a change before returning ([Long polling](jobs.md#long-polling)) |
| `include_log` | boolean | false | Return the last 6000 bytes of the worker log. Failed jobs always include it |

### list_jobs

**Read.** Lists the most recently updated jobs from every client.

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `status` | string | all | `queued`, `running`, `completed`, `failed` or `cancelled` |
| `kind` | string | all | `detect`, `preview`, `render`, `normalize`, `analyze` or `compare` |
| `limit` | integer | 50 | 1–200 |

Each summary shows the job's frame selection by kind:

| Kind | Fields |
| --- | --- |
| `detect` | `frame`, or `sample_frames` / `frames` and any `similarity_threshold` |
| `preview` | `frame` |
| `render` | `start_frame`, `max_frames` |
| `normalize` | `fps` |
| `analyze` | `references` |
| `compare` | `frames` or `start_frame` / `max_frames`, and the variant names as `variants` |

### cancel_job

**Write.** Requests cancellation of a queued or running job. Returns `job_id`,
`cancel_requested` and `status`. Poll `job_status` until the state is terminal.

| Argument | Type |
| --- | --- |
| `job_id` | string |

### delete_job

**Destructive.** Permanently deletes a completed, failed or cancelled job with its logs and
artifacts. Returns `job_id`, `status`, `deleted` and the `bytes` freed.

| Argument | Type |
| --- | --- |
| `job_id` | string |

Refused for queued and running jobs, and for a job whose artifacts are an input of an active
job, a resumable render or a saved recipe. See [Retention](jobs.md#retention).

### prune_storage

**Destructive.** Reports or deletes old terminal jobs, old uploads and files left by crashes.
See [Retention](jobs.md#retention).

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `older_than_days` | number | 30 | 0–36500 |
| `dry_run` | boolean | true | Report only. The dry run applies the same checks |

Returns `dry_run`, `jobs`, `uploads` and `stale` (each entry with its `bytes`), and
`total_bytes`.

## Artifacts

### view_artifact

**Read.** Returns a completed job's PNG or JPEG artifact as an inline image.

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `job_id` | string | | A completed job |
| `name` | string | `preview.jpg` | A plain file name ending in `.png`, `.jpg` or `.jpeg`, smaller than 10 MB (10,000,000 bytes) |

### download_artifact

**Write.** Streams a completed artifact to the client.

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `job_id` | string | | A completed job |
| `name` | string | `result.mp4` | A plain file name |

Writes `<VISOMASTER_DOWNLOADS>/<job_id>/<name>` and returns its `path` and `bytes`. Repeating
the download atomically replaces the local copy.

## Recipes

Recipe names are 1–64 letters, digits, underscores or hyphens, starting with a letter or digit.
`revision` is an integer of at least 1; when omitted, the latest revision is used. See
[Recipes](results.md#recipes) for the document format.

### save_recipe

**Write.** Saves a new immutable revision of a named recipe, from a completed job or from an
explicit request. Hashes the media and model files (up to 15 minutes).

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `name` | string | | Recipe name |
| `job_id` | string | | A completed preview or render to save. Give this, or `media_path` and `mappings` |
| `media_path`, `mappings`, `mode`, `frame`, `start_frame`, `max_frames`, `controls`, `report`, `segments` | | as `submit_swap` | An explicit request |
| `priority` | string | none stored | Queue class stored for the jobs `run_recipe` queues. Applies to explicit requests only; a recipe saved from a job keeps that job's explicit priority |
| `note` | string | none | At most 2000 characters |

### get_recipe

**Read.** Without `name`, lists recipes and their revisions. With `name`, returns a revision's
document and its `retention`.

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `name` | string | none | |
| `revision` | integer | latest | |
| `verify` | boolean | false | Hash the inputs and models now and list every change in `verification.changes` |

### run_recipe

**Write.** Queues a preview or render from a recipe revision. Hashes every input and model file
first and fails with a list of changes unless `accept_changes` is true.

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `name` | string | | |
| `revision` | integer | latest | |
| `mode` | string | the recipe's | `preview` or `render`. A change needs `frame_range` |
| `media_path` | string | the recipe's | Media with different content needs `targets` and `frame_range` |
| `targets` | list of strings | the recipe's | One `target_face` per recipe mapping, in order |
| `frame_range` | object | the recipe's | `frame` for a preview; `start_frame` and `max_frames` (null: through the end) for a render |
| `report` | object | the recipe's | Required with new media when the recipe has `expected_visible` ranges |
| `priority` | string | the recipe's stored priority, otherwise by mode | |
| `accept_changes` | boolean | false | Run despite changed inputs, defaults, models, adapter code, Fusion revision or configuration; the changes are recorded |
| `job_id` | string | generated | |

A retry with the same `job_id` and arguments returns the existing job.

### export_recipe

**Write.** Writes a recipe revision as JSON to
`<VISOMASTER_DOWNLOADS>/recipes/<name>-r<revision>.json` on the client. Returns `path`, `name`,
`revision` and `sha256`.

| Argument | Type | Default |
| --- | --- | --- |
| `name` | string | |
| `revision` | integer | latest |

### import_recipe

**Write.** Stores an exported recipe file as a new revision on this workstation.

| Argument | Type | Default | Rules |
| --- | --- | --- | --- |
| `local_path` | string | | A client JSON file of at most 4 MiB |
| `name` | string | the recipe's own | Store under this name |

Every input must exist with its recorded bytes, at its recorded path or as an upload with those
bytes.

### delete_recipe

**Destructive.** Deletes one revision of a recipe, or all of them. A deleted revision's number
is never reused. Deletes no media or uploads.

| Argument | Type | Default |
| --- | --- | --- |
| `name` | string | |
| `revision` | integer | all revisions |

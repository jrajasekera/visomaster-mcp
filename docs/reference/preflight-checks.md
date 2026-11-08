# Preflight checks

`preflight_swap` takes `submit_swap`'s arguments, plus `analyze_references`, and reports whether
that request can start now. The client validates the arguments with the same models as
`submit_swap`, and the workstation runs the same request validation that submission runs.

Without `analyze_references`, a preflight loads no Fusion code or models, takes no lock and
creates no job. It reads files and runs FFprobe, git and nvidia-smi. A `job_id`, if given, is
checked against existing jobs rather than generated. Nothing is reserved.

## Result

| Field | Meaning |
| --- | --- |
| `readiness` | `blocked`, `pending`, `warnings` or `ready` |
| `checks` | One entry per check (see below) |
| `not_checked` | Names of the skipped checks; each one's reason is its `message` in `checks` |
| `fingerprint` | Hashes that identify the request, inputs and configuration |
| `checked_at` | The check time |
| `notice` | A reminder that a preflight reserves nothing |

| Readiness | Meaning |
| --- | --- |
| `blocked` | Some check says that submission or the job would fail |
| `pending` | Reference analysis is still running |
| `warnings` | It can start, but some checks raised warnings |
| `ready` | No check is blocked, pending or a warning (skipped checks do not count) |

Each entry of `checks` has:

| Field | Values |
| --- | --- |
| `check` | The check's name |
| `status` | `passed`, `warning`, `blocked`, `pending` or `not_checked` |
| `severity` | `error`, `warning` or `info` |
| `code` | A stable code from the table below |
| `message` | A readable description |
| `evidence` | The observations behind the verdict |
| `actions` | What to do: each a description with a `tool` and `arguments` to call, or an operator step without a tool |

## Checks and codes

| Check | Codes |
| --- | --- |
| `configuration` | `config_valid`, `config_invalid` (every other check is then `not_checked`) |
| `fusion_revision` | `fusion_compatible`, `fusion_incompatible` (the revision guard that `health_check` and GPU workers use) |
| `deployed_code` | `code_matches`, `code_mismatch`, `client_code_unknown` |
| `job_id` | `job_id_generated`, `job_id_free`, `job_exists` (a retry returns that job), `job_id_conflict`, `job_id_invalid` |
| `media_path`, `references` | `path_ok`, `references_ok` (with each image's `sha256` to pin as `source_sha256`), `path_missing`, `path_unreadable`, `path_not_absolute`, `path_outside_roots`, `artifact_job_incomplete`, `reference_too_large`, `reference_changed` |
| `settings` | `settings_ok`, `setting_unsupported`, `setting_invalid`, `swapper_unsupported`, `layouts_unreadable` |
| `model_assets` | `models_present`; `model_missing` (files under the Fusion checkout's `model_assets`, for the base settings and every segment's overrides); `model_unmapped` (a selection this adapter has no file mapping for: not checked); `model_list_unreadable` |
| `media` | `media_ready`; `variable_frame_rate`, `variable_timestamps`, `missing_timestamps`, `nonzero_start`, `odd_dimensions` and `rotated_or_mirrored`, which suggest `normalize_media`; `empty_range`, `frame_out_of_range`, `expected_visible_beyond_range`, `segment_outside_range`, `all_passthrough`, `hdr_unsupported`, `matrix_unsupported`, `unsupported_container`, `no_video_stream`, `no_video_frames`, `media_unreadable` |
| `storage` | `space_ok`, `low_space` (less than 1 GiB beyond the estimate), `insufficient_space`, `state_not_writable` |
| `gpu` | `gpu_observed` (compute capability, VRAM class, memory use), `gpu_unobserved` |
| `qualification` | `qualified`, `unqualified` (a warning, never a refusal) |
| `request` | `request_valid`, or the refusal code of `submit_swap`'s validation (`invalid` for a validation error without a specific code) |
| `reference_analysis` | `not_requested`, `analysis_pending`, `references_usable`, `reference_warnings`, `reference_problems`, `analysis_cancelled` |

Any check skipped because an earlier one failed (for example every check after
`config_invalid`, or `media` after a refused path) has status `not_checked` and the code
`prerequisite_blocked`.

### Media

The media check is the worker's own.

- For a render: constant frame rate, timestamps from zero, a nonempty range,
  `report.expected_visible` within it, even dimensions and supported colors, and `segments`
  against the frames the video actually has. Submission can check segments only against the
  requested range.
- For a preview: the frame exists, the video is not rotated or mirrored, and its colors
  are supported.

Timestamps are read only through the requested frames, as the render reads them. Media that
`normalize_media` can fix carries that action, with the same `media_path`. Nothing is converted,
and the request is not changed.

### Storage

Storage compares free space in `state_dir` with an estimate: per rendered frame, twice the
larger of the source's bytes per frame and width × height / 20, plus 64 MiB. It is an
observation, as is the GPU's memory use.

### Qualification

`QUALIFICATIONS` in `remote/common.py` records what was exercised on real hardware: hardware
class (architecture, compute capability, VRAM class), Fusion revision, modes, detector, swap
model, restorer, expression restoration, masks (occluder, XSeg), and the number of mappings and
of weighted sources, with the date and the `VERIFICATION.md` section.

A request that no record covers is `unqualified`, with the nearest record and the fields that
differ. It is still allowed. Qualification is evidence, not permission.

## Fingerprint

| Field | SHA-256 over |
| --- | --- |
| `request` | The request without `job_id` |
| `inputs` | The media path, size and modification time, and each reference's SHA-256 |
| `configuration` | The remote settings identity, Fusion revision, worker code, and the size and modification time of each required model file |
| `sha256` | The three values above |

Compare fingerprints to see whether anything changed since an earlier preflight. Submission does
not consult a preflight: `submit_swap` validates again, and the worker rechecks Fusion, the media
and the reference bytes. Pin references with the reported `sha256` values.

## Reference analysis from a preflight

With `analyze_references: true`, and when Fusion and the request pass, the preflight queues one
ordinary `analyze` job per distinct set of weighted sources, swap model and controls. Only
sources with a positive weight are analyzed. The jobs queue as `interactive` and share the GPU
lock and queue.

- The job ID is `preflight-` plus a hash of the analysis request, so repeating the preflight
  returns the same job and its results instead of queuing another. Changed bytes or settings
  queue a new one.
- Follow it with `job_status`, then run the preflight again. Cancel it with `cancel_job`.
- A cancelled or failed analysis stays as it is until `delete_job` removes it.
- Targets are not analyzed. Use `detect_faces` crops with `reference_verified: true`.

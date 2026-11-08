# Job results and artifacts

A completed job's `job_status` carries its `result`. Artifact files are listed in
`result.artifacts`; read them with `view_artifact` (PNG and JPEG, under 10 MB) or
`download_artifact`. Failed and cancelled jobs expose no artifacts.

## Face detection

### Single frame

`detect_faces` with `frame` returns:

| Field | Meaning |
| --- | --- |
| `faces[]` | One entry per detected face: `index`, `box`, `reference_path`, `reference_verified`, `artifact` (`face-<index>.png`) |
| `artifacts` | `frame.jpg` (the decoded frame), `contact.jpg` (when any face is found), and every `face-<index>.png` |
| `warning` | Present when faces were found but none has a verified crop |

### Several frames

`detect_faces` with `sample_frames` or `frames` returns:

| Field | Meaning |
| --- | --- |
| `frames` | The frames examined |
| `detections` | Number of faces detected |
| `similarity_threshold` | The threshold used to group faces |
| `identities[]` | One entry per person, most appearances first: `identity`, `appearances`, `frames`, `frame` and `box` of the representative crop, up to five `samples` (`frame`, `box`), `reference_path`, `reference_verified`, `artifact` (`identity-<n>.png`) |
| `artifacts` | `contact.jpg` (when any face is found) and every `identity-<n>.png` |
| `warning` | Present when no identity has a verified crop |

### Crops and the contact sheet

Crops are padded by 30% of the face box's longer side. A crop is **verified** when it is a
reference that `submit_swap` accepts for that same face:

- it re-detects as exactly one face under the same detector limit;
- the detected box overlaps the original by an IoU of at least 0.5;
- its embedding scores at least the default `SimilarityThresholdSlider` against the original
  detection.

When the 30% padding takes in a neighbouring face, a crop padded by 10% is tried. For an
identity, up to five of its largest faces are tried in turn.

An unverified crop keeps its `artifact` for inspection but has `reference_path: null`. On
`contact.jpg` it is marked `*` and framed in red. Tiles are labelled `face N` or
`id N: <appearances>x`.

## Previews and renders

| Field | Mode | Meaning |
| --- | --- | --- |
| `swapped_faces` | both | Faces swapped in total |
| `swap_report` | both | Summary of the swap report (see below) |
| `references` | both | Per mapping: the target's SHA-256 and each source's path, SHA-256, weight and normalized `share` |
| `models` | both | Fusion models loaded when the job finished |
| `revision` | both | Fusion revision |
| `frames`, `fps`, `duration` | render | Output frame count, exact rate and duration |
| `audio` | render | `AAC`, or `none` when the output has no audio |
| `checkpoint` | render | Chunk summary (see [Checkpoints](#checkpoints)) |
| `plan` | render with segments | Runs with their counts (see [Segment plan](#segment-plan)) |
| `recipe` | recipe jobs | Recipe name, revision, document `sha256`, invocation, substitutions and accepted changes |

| Artifact | Mode | Contents |
| --- | --- | --- |
| `result.png` | preview | The swapped frame |
| `preview.jpg` | both | The swapped frame (render: the first processed frame, decoded from `result.mp4`) |
| `before.jpg` | both | The same frame before the swap |
| `result.mp4` | render | The rendered video |
| `swap-report.json` | both | The full swap report |
| `plan.json` | render with segments | The resolved segment plan |
| `review-NN.jpg` | render | Labelled before-and-after review images |

A preview in which no face swaps fails with `No faces matched the target references`.

## Swap reports

Every successful preview and render writes `swap-report.json` (version 1). The result's
`swap_report` summary holds `artifact`, `version`, `warnings` (messages), `review_images`, and per
mapping: `mapping`, `swapped_frames`, `requested_frames`, `coverage`, `ambiguous_frames`,
`suspicious_intervals` and, when it has expected-visible ranges, `expected_visible_coverage`.

Each requested mapping is reported by its zero-based index in `mappings`. For every processed
frame, the report records why each mapping was or was not swapped, using only what Fusion's own
matching observed:

| Reason | Meaning |
| --- | --- |
| `swapped` | Fusion's swap step completed for a face of this mapping. This says nothing about how the result looks |
| `no_faces` | The detector found no face. This shows absence, not a failure |
| `unscored` | No scored face cleared the threshold, but some detected faces never reached matching: too small, without valid landmarks, or (with tracking on) filtered out by Fusion's `SequentialDetector`. Their scores are unknown |
| `below_threshold` | Every detected face was scored, and none reached this mapping's `SimilarityThresholdSlider` |
| `claimed_by_other` | A face cleared this mapping's threshold but matched another mapping better |
| `not_swapped` | A face matched this mapping best, but Fusion swapped no face for it |
| `not_requested` | The mapping was not requested in this frame (passthrough segments, or segments that leave it out). Excluded from every denominator |

Each mapping's `threshold` is its base `SimilarityThresholdSlider`, the one used outside
segments. Every frame is classified with the threshold applied in that frame. In a render with
segments, `segment_thresholds` lists each run that matched the mapping with a different
threshold, as `segment`, `start_frame`, `end_frame` (half-open) and `threshold`. The list is
empty when no segment changed the mapping's threshold, and in previews. The `never_swapped`
warning compares the best unswapped score with the threshold applied in that frame.

```json
"threshold": 60.0,
"segment_thresholds": [
  {"segment": "profile", "start_frame": 120, "end_frame": 180, "threshold": 75.0}
]
```

### Coverage and scores

- Coverage is given against explicit denominators: all requested frames, frames with detected
  faces, and frames inside `expected_visible` ranges.
- Scores are Fusion's similarity, 50 + 50 × cosine similarity, for faces that reached
  matching. No score is invented for the others.
- A frame is **ambiguous** for a mapping when two or more faces matched it best, or when a face
  that clears its threshold also clears another mapping's.

### Suspicious intervals

Intervals are zero-based source-frame ranges `[start_frame, end_frame)`. Times are in seconds,
computed from the exact frame rate: `start_time` and `end_time` on the source clock
(`frame / fps`), and `output_start_time` in `result.mp4`.

| Kind | Meaning |
| --- | --- |
| `expected_missing` | Unswapped frames inside an `expected_visible` range, whatever the reason |
| `dropout` | Unswapped frames, at least one with faces, between two swapped frames |
| `claimed_by_other` | Consecutive frames with the reason `claimed_by_other` |
| `not_swapped` | Consecutive frames with the reason `not_swapped` |
| `ambiguous` | Consecutive ambiguous frames |

Frames without faces are reported but never flagged on their own, because they do not prove
that the person should have been visible. Each mapping lists at most 100 intervals;
`intervals_omitted` counts the rest. Previews have no times and no review images.

### Review images

Renders write up to `report.review_images` (default 6, at most 24) JPEGs, `review-01.jpg` and so
on. Each shows the decoded source frame beside the same frame of the published output, under a
label with the mapping, source frame, time and reason. They are chosen first from the
best-scoring frame of each mapping that never swapped, then from the middle frames of suspicious
intervals, by kind and length.

### Warnings and failures

| Outcome | Effect |
| --- | --- |
| Warnings `never_swapped`, `expected_missing`, `suspicious_intervals`, `segment_never_swapped` | Advisory; the job still completes |
| No face swapped at all (outside passthrough segments) | The job fails |
| Fusion reports a per-face error | The job fails; the error names the source frame and, when it can be attributed, the mapping |
| A mapping's swapped share is below `report.min_coverage` | The render fails. The share is over that mapping's `expected_visible` frames when it has ranges, otherwise over all requested frames |

## Segment plan

For a render with `segments`:

- `plan.json` lists each segment as given, with its `applied` frames and whether it was
  `clipped`, and every run (segment or gap) with the effective controls and per-mapping
  parameters it used.
- The result and the swap report carry `plan.runs`: each run with its `frames`, `swaps` and
  `swapped_frames`.
- Each swap-report mapping lists in `segment_thresholds` the runs whose
  `SimilarityThresholdSlider` differed from its base `threshold` (see
  [Swap reports](#swap-reports)).
- Progress records name the current `segment`.

## Checkpoints

A render's `checkpoint` result field reports `chunks`, `chunk_frames`, `reused_chunks`,
`reused_frames` and `resumed_from`.

- Chunk *n* holds source frames `[start_frame + 250n, start_frame + 250(n + 1))`; the last one is
  clipped to the range.
- Each chunk is checked (frame count, constant-rate timestamps from zero, dimensions, pixel
  aspect and codec parameters) before `checkpoint/manifest.json` commits it with its SHA-256 and
  the render's accounting after its last frame.
- The manifest also records the complete range, the chunk size, the FFmpeg build and the
  render's provenance: the SHA-256 of the media and of every reference, the effective settings,
  the model files those settings load, the Fusion revision, the adapter code and the
  configuration identity.
- The checkpoint directory is removed when the render completes. A failed or cancelled render
  keeps it until the job is deleted or pruned.

## Reference analysis

`analyze_references` returns:

| Field | Meaning |
| --- | --- |
| `swap_model` | The swap model analyzed for |
| `model` | The ArcFace recognizer that swap model uses, whose embeddings were compared |
| `usable`, `errors`, `warnings` | Counts |
| `references[]` | One entry per image (see below) |
| `duplicates` | Each redundant reference with the earlier one it repeats |
| `consistency` | The model and basis of the consistency scores, `available` and `reason`, and `pair_similarity` when there are exactly two distinct (by bytes) usable references |
| `thresholds` | The thresholds applied |
| `metrics` | Definitions of the measurements |
| `artifacts` | `contact.jpg`, `analysis.json` and a `reference-N.png` crop per usable image |

Each entry in `references` has:

| Field | Meaning |
| --- | --- |
| `id` | `r0`, `r1`, … in request order |
| `path`, `sha256` | The image and the SHA-256 of the bytes analyzed |
| `usable` | True when `submit_swap` accepts it: exactly one detected face with an embedding |
| `faces`, `box` (or `boxes`), `image_size` | What the detector found |
| `face_pixels` | The shorter side of the detector's box, in source-image pixels |
| `sharpness` | Variance of the 4-neighbour Laplacian (OpenCV `Laplacian`, aperture 1) of the box's 8-bit luma resized to 112 × 112, in squared luma levels. Faces smaller than 112 pixels are enlarged and score lower |
| `consistency_score` | Leave-one-out similarity against the other usable references |
| `findings` | `code`, `severity`, `message` |
| `artifact` | `reference-N.png`, the padded crop of a usable image |

Detection uses the same detector, with its 20-face limit, that `submit_swap` uses.

| Finding | Severity | Meaning |
| --- | --- | --- |
| `unreadable` | error | The image cannot be read |
| `no_face` | error | No face detected |
| `multiple_faces` | error | More than one face detected |
| `unrecognized` | error | The face has no embedding |
| `small_face` | warning | `face_pixels` below `min_face_pixels` (default 112) |
| `blurry` | warning | `sharpness` below `min_sharpness` (default 40) |
| `exact_duplicate` | warning | Same bytes as an earlier reference |
| `near_duplicate` | warning | Scores at least `duplicate_similarity` (default 95) against an earlier reference |
| `outlier` | warning | Scores below `outlier_similarity` (default 60) against the normalized mean of the other usable references, excluding byte-identical copies |
| `inconsistent_pair` | warning | Exactly two usable references whose `pair_similarity` is below `outlier_similarity`; both are flagged |

Outliers need three distinct usable references. With one, consistency is unavailable. A low
score suggests another person, an unusual view or unusual lighting; it does not establish
identity. `contact.jpg` shows every image labelled with its ID and first finding (`+` when there
are more), unusable ones framed in red. `analysis.json` holds the result with the Fusion
revision. Raw embeddings are not returned.

## Variant comparisons

### Limits

| Limit | Value |
| --- | --- |
| Variants | 2–6; names of 1–20 letters, digits, `_` or `-`, unique ignoring case, with distinct overrides |
| Still frames | 1–4 distinct frames (images: `[0]`) |
| Clip | 1–120 frames per variant |
| Source frame size | At most 3840 × 2160 pixels (8,294,400); every output keeps the source dimensions |
| Grids | Cells of 480 × 480, 4 per row: at most 1952 pixels wide and 2 rows |

### Artifacts


| Artifact | Contents |
| --- | --- |
| `compare-f<N>.jpg` | One grid per compared frame (for clips: the first, middle and last frame). The original, then each variant labelled `<index> <name>`, each tile scaled to a 480-pixel cell at the source's display aspect ratio |
| `compare.json` | The manifest: the job result plus the Fusion revision |
| `original-f<N>.png` | The decoded source frames at full resolution |
| `v<index>-<name>-f<N>.png` | Each variant's full-resolution output of those frames, before any encoding |
| `v<index>-<name>.mp4` | Clips only: the variant's clip, validated like a render's `result.mp4`, with the source's first audio stream over the compared interval |
| `v<index>-<name>-swap-report.json` | The variant's swap report (for stills, a list of one preview report per frame) |

Clips have no composite. All clips share the same range, frame count, rate, duration and audio.

### Manifest

| Field | Meaning |
| --- | --- |
| `source` | Dimensions, `sample_aspect_ratio`, compared `frames` and each original frame's SHA-256 (`frame_sha256`); for clips also the rational `fps`, `start_frame`, `end_frame` (exclusive), `frame_count`, `start_time` on the source clock and `duration` |
| `order`, `audio_policy` | How variants were run and how clip audio was handled |
| `media_path`, `original`, `grids` | The compared media, the original-frame artifacts and the grid artifacts |
| `variants[]` | Per variant: `index`, `name`, `status`, `error`, `overrides`, `settings` (every adjustable control and per-mapping parameter at its effective value), `references` (fingerprints and weights), `input_sha256` (over every frame it received), `artifacts`, swap report summaries, `swapped_faces` and `submit_swap` |
| `complete` | True only when every variant succeeded |
| `failed_variants` | Names of failed variants; empty when all succeeded |
| `warning` | Present when some variants failed |

A variant's `submit_swap` holds arguments that repeat it alone with exactly its merged settings:
a preview of the first compared frame, or a render of the compared range, with each mapping's
`source_sha256` pinned to the bytes the comparison used.

A variant that fails (for example a Fusion error, or no swap in any compared frame) keeps its
`error`, loses its artifacts and appears as a red `FAILED` tile. The job completes when at least
one variant succeeded and fails when every variant fails. A frame where a variant swapped
nothing is reported in its swap report, not as a failure.

## Recipes

A recipe revision is stored as `<state_dir>/recipes/<name>/<revision>.json` with format
`visomaster-recipe`, version 1.

| Field | Contents |
| --- | --- |
| `request` | The explicit swap request a new job runs: resolved media path and `media_sha256`, the range (`frame`, or `start_frame` and `max_frames`), every adjustable control and per-mapping parameter at its effective value, mappings with `target_sha256`, `source_sha256` and any `source_weights`, and `report`, `segments` and `priority` when given |
| `provenance` | The full effective settings (Fusion and adapter defaults included), the Fusion revision, the adapter release and the SHA-256 of its three files, the configuration identity, and per model its path, the SHA-256 Fusion declares and the SHA-256 of its file |
| `frame_convention` | How frames are numbered: zero-based indices in decoded order |
| `origin` | `request`, `job` or `import`, with any input relocations from an import |
| `note` | Your note, up to 2000 characters |
| `retained` | Original paths of face images copied into the upload store |
| `created_at` | Save time |
| `sha256` | Digest of everything else in the document |

The recorded models are those the settings load: the detector, the recognizer, the swapper with
its ArcFace models, and an enabled restorer, occluder, XSeg or expression restorer, including
one a segment enables. A recipe saved from a job also records the models that job had loaded. A
revision saved before restorer, mask and expression models were recorded reports such a model as
not recorded; save a new revision to record it.

### Retention

| Input | Status in `get_recipe` | Kept by retention |
| --- | --- | --- |
| Face images, copied to `uploads/<sha256><suffix>` (PNG, JPEG or WebP) | retained | yes, while a revision names them |
| Job artifacts named as inputs, such as `normalized.mp4` | retained | yes |
| Media in `media_root`, and references of other types | unretained | never removed, never copied |
| Any input that no longer exists | `, missing` added to its status | |

### Changes that `verify` and `run_recipe` report

Changed bytes at the same path, changed settings or defaults, changed or missing model files,
different adapter code, a different Fusion revision, or a different configuration. Unavailable
inputs always fail `run_recipe`.

### Export file

`export_recipe` writes `<VISOMASTER_DOWNLOADS>/recipes/<name>-r<revision>.json`: the document
unchanged, without media or model weights. `import_recipe` refuses:

- other format versions;
- a `sha256` that does not match the content;
- unknown or malformed fields;
- inputs outside the allowed roots;
- inputs whose bytes differ;
- settings the installed Fusion rejects.

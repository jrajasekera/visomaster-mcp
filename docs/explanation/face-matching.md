# Face matching

This page explains how the adapter uses Fusion to find faces, decide which face is which, and
combine several source images. The arguments and result fields are in the
[Tools reference](../reference/tools.md) and [Job results](../reference/results.md).

## Reusing Fusion's pipeline

The adapter does not reimplement face swapping. It builds a small state object with the
controls, parameters and face references Fusion expects. It reads literal default settings from
the installed layout definitions, then calls Fusion's `ModelsProcessor`, `FunctionWorker` and
`FrameWorker`.

Before importing the pipeline, it validates the Fusion commit and tracked `app` changes (staged
or not), with the same standard-library check whose result and reason `health_check` reports.
GPU calls use the CUDA provider.

User-adjustable settings are explicitly allowlisted, and checked against the installed choices
and numeric limits. Anything outside the allowlist cannot be passed in a request.

## Matching faces to targets

Reference matching uses Fusion's recognition embeddings and similarity thresholds. Fusion scores
a match as 50 + 50 × the cosine similarity of two embeddings, the scale its
`SimilarityThresholdSlider` uses. A face is a
candidate for a target when its score reaches the mapping's `SimilarityThresholdSlider`.

Several target mappings are supported by the same pipeline. By default the adapter detects and
matches each video frame independently.

### Sequential tracking

A render that enables Fusion's `FaceTrackingEnableToggle` (ByteTrack) or
`KPSSmoothingEnableToggle` (landmark EMA) uses Fusion's playback path instead. One
`SequentialDetector` sees the frames in order, keeps ByteTrack, smoothing and identity state
between them, and feeds its detections to a non-single-frame `FrameWorker`. Previews and
detection stay single-frame.

At the pinned revision this path is opt-in, because qualification showed no default benefit:

- With both toggles off, it decoded to frames identical to the per-frame path.
- ByteTrack alone changed no output on the qualified clips.
- Smoothing lowered face-region frame-to-frame residual change by about 3%.
- The path rendered 16–25% slower, because Fusion recognizes each face twice.

Fusion's temporal identity rescue does not reach the output either. The worker's threshold
bypass applies only to plain dictionaries, and Fusion passes `ParametersDict` objects, in its
own playback too.

### Swapping only the best match

Tracking does not stop a non-target face that sometimes clears the similarity threshold from
being swapped intermittently. Fusion's `SwapOnlyBestMatchEnableToggle` does: for each target it
swaps only the highest-scoring face among those whose best match is that target and that clear
its threshold.

The adapter turns it on by default; callers may turn it off. On the qualified two-person clip it
removed every non-target swap at the default threshold of 60 while keeping the target on every
frame. Raising the threshold to 75 achieved the same, but best match does it without raising the
score the target itself must reach. Without it, any face that clears the threshold is swapped,
which on two-person footage swapped the other person intermittently.

Best match has two limits:

- It cannot stop a look-alike from being swapped in frames where the target is absent. Raising
  `SimilarityThresholdSlider` is the remedy there.
- A target seen twice in one frame is swapped once.

## Verifying detection crops

`detect_faces` publishes a crop of each face so that it can be used as a `target_face`. A crop
pads the face's box by 30% of its longer side. That crop is only useful if `submit_swap`, which
re-detects the reference image, will find the same face in it. So each crop is verified as the
reference `submit_swap` will read:

1. It is re-detected with the same detector limit that `configure_faces` uses, and must contain
   exactly one face.
2. That face's box must overlap the original by an IoU of at least 0.5.
3. Its recognition embedding must score at least the default `SimilarityThresholdSlider` against
   the original detection's.

A neighbour inside the padding fails the count, and a lone neighbour fails the overlap and the
score. In either case a crop padded by 10% is tried next. Multi-frame detection tries up to five
of an identity's largest faces in turn.

An unverified crop is still published for inspection, but its `reference_path` is null, it is
framed in red and marked `*` on the contact sheet, and a result with no verified crop carries a
warning.

The score check also rejects crops whose own face would not match itself at swap time.
Qualification found strong profile views re-recognizing at 47–50 against their full-frame
embedding, while frontal and three-quarter faces scored 62–99.

## Grouping faces into identities

Multi-frame detection embeds every detected face with the configured recognition model and
groups faces greedily, largest first:

- A face joins the identity whose mean embedding scores highest, if that score reaches
  `similarity_threshold`. The default is the slider's default of 60, the scale that
  `SimilarityThresholdSlider` uses.
- Two faces from the same frame never share an identity.
- Otherwise the face starts a new identity.

Each identity reports its appearances, frames, up to five sample boxes, and a representative crop
from the largest of its five largest faces whose crop is verified.

A profile view can score below 50 against the same person's frontal views, about as low as a
different person scores. It can therefore form its own identity. Lowering `similarity_threshold`
merges it, but can also merge different people who never share a frame. Check the contact sheet.

## Combining source images

Several source images are averaged per identity, then L2-normalized, as Fusion's own merge in
`calculate_assigned_input_embedding` does (`EmbMergeMethodSelection` "Mean", then L2
normalization). Earlier releases applied the mean without the normalization:

- Inswapper-family, InStyleSwapper and SimSwap latents normalize their input, so their swaps are
  unchanged.
- GhostFace and CSCS take the embedding as given, and now receive the unit vector that Fusion's
  GUI gives them.

With explicit weights the mean is weighted, and only the ratios matter. A weight of 0 excludes
an image without reading it: submission, the worker, preflight and recipe provenance check only
that its path is absolute. They never hash it or require it to exist, and record its fingerprint
as null. It is not one of the request's inputs (`recorded_inputs`).

Submission validates the weights (finite, nonnegative, one per source, with a positive and
finite sum) and records the request unchanged. Requests that omit weights keep their earlier
form and their idempotent retries.

### Fingerprinting references

Every reference a job reads is fingerprinted:

1. Submission stores the SHA-256 of each target, source and analyzed image in `resolved.json`.
   If the caller gave `source_sha256`, it checks those first; a null entry pins nothing.
2. Worker validation compares the digests again.
3. The engine hashes the bytes it actually decodes, with `cv2.imdecode`, which applies EXIF
   orientation as `imread` does.

A file edited after submission therefore fails the job rather than being used. Preview and
render results list the fingerprints, weights and shares each mapping used.

Reference files are limited to 256 MiB, so a misplaced video cannot hold the submit lock while
it is hashed.

## Reference analysis

Reference analysis is a queued GPU job kind, `analyze`, with its own request keys
(`references`, `swap_model`, `thresholds`, `controls`). It has no `media_path`, and other keys
are refused. It shares the GPU lock, retention protection (its references count as inputs) and
the validation path with other jobs. It needs nothing beyond Fusion's detector and recognizer.

Each image is decoded from its fingerprinted bytes and detected with `detect_reference`,
exactly as `configure_faces` reads it. It is then embedded with the ArcFace model of the
requested swap model, which is the embedding the swap merge averages.

**Sharpness** is the Laplacian variance of the face box's luma at the 112-pixel ArcFace input
scale, so it combines blur and small size. It is computed in plain Python on that 112 × 112 grid.

**Collection checks** run on the usable references only:

- byte-identical files are exact duplicates;
- embedding similarity at or above `duplicate_similarity` marks near duplicates;
- a leave-one-out score against the normalized mean of the others marks outliers.

The outlier default is set per recognizer: the recognizers place the same and different people
on different parts of the similarity scale, so Fusion's matching default does not fit them.

With fewer than three distinct usable references no outlier is named, because a disagreement
between two cannot be attributed to either.

Findings report and never alter the collection. Swaps do not consume analysis results: a swap
recomputes embeddings from its own fingerprinted bytes, so changed images or a different swap
model cannot reuse a stale analysis.

# Video processing

This page explains how frames get from the source file into Fusion and back into a video, and
how swap reports, segments, comparisons and checkpoints build on that path. The rules and
output properties are listed in [Media requirements](../reference/media.md).

## Selecting frames

Detection and previews select a zero-based frame by decoded order. The video stream's packet
timestamps, read without decoding, give that frame's presentation time. Variable-frame-rate
input therefore yields the exact frame, not one estimated from the average rate.

- An index at or beyond the frame count is rejected as out of range.
- Still images have only frame 0.
- Sampling counts the video stream's own frames, because the container duration can describe a
  longer audio track, or be missing.
- Frames after the first of a video whose packets lack timestamps (as in some AVI files) require
  normalization.

## Rendering

### Decoding

Video rendering first checks packet presentation timestamps over the requested range. It then
decodes frames at exact indices through an FFmpeg pipe and writes H.264 through FFmpeg.

Decoding and the source-audio trim seek on the video stream's own timestamps
(`-seek_timestamp`). A container that starts earlier, for example with leading audio, shifts
neither the frames nor the audio interval.

Rendering rejects variable frame rate, nonzero video start timestamps, odd frame dimensions and
any display matrix that rotates or mirrors the picture, because frames are decoded as stored. A
flip can report rotation 0, so the matrix itself is checked.

### Keeping the picture intact

Output keeps the input frame rate, dimensions and sample aspect ratio. The raw frame pipe loses
these, so the encoder sets them again and output validation checks them.

Decoding to BGR and encoding back use the source's YUV matrix and range, with accurate swscale
rounding, and the output carries the source's color tags. Unswapped pixels therefore keep their
color. Some cases need special handling:

- **RGB sources** (FFprobe's `gbr` matrix, which a YUV output cannot carry) are encoded as BT.709
  limited range and tagged so.
- **Gray sources** without a range tag are full range, as swscale treats them. swscale treats
  gray as full range whatever its tag, so limited-range gray is first copied code for code into
  YUV 4:4:4 of the same depth, and then tagged limited.
- **Unsupported matrices.** Only the matrices swscale implements are accepted: BT.709, BT.601
  (`smpte170m`, `bt470bg`), FCC, SMPTE 240M, BT.2020 non-constant luminance, and untagged video,
  which swscale decodes as BT.601. Other matrices, such as YCgCo, BT.2020 constant luminance,
  ICtCp, SMPTE 2085 and the chromaticity-derived ones, are refused by renders, detection and
  normalization alike, before anything is written, with advice to transcode the source to BT.709
  elsewhere. FFmpeg 8 cannot convert them, and 6.1 would silently apply other coefficients.

Primaries and transfer are set on the frames, because FFmpeg 8 ignores encoder options for them.
The audio mux copies the video stream, so they and the sample aspect ratio reach the result.

### Bit depth and HDR

Sources above 8 bits are encoded as 10-bit H.264, although Fusion itself processes 8-bit BGR.
Depth comes from the stream's raw sample depth. When FFprobe reports none, it comes from the
deepest component of the decoded pixel format's FFmpeg descriptor, so gray, packed RGB and
semiplanar formats count as well as planar YUV.

HDR input (PQ or HLG) is rejected rather than tone-mapped, because the face models would see
8-bit nonlinear HDR values.

### Audio

Audio comes from the first audio stream alone.

- When it has samples in the rendered interval, as one decoded frame through the mux's own input
  window shows, that interval is encoded as AAC. It is padded with silence where the stream
  starts later, so it keeps its timing.
- When it has none, because it ends before the interval or starts after it, the output has no
  audio and the result reports `audio: none`.

### Output validation and failures

Every output is checked for video frame count and duration, for the audio that the overlap
requires, and for audio ending no more than one AAC frame after the video. The result's `audio`
describes the written file.

- A render with zero successful face swaps fails.
- Fusion's caught per-face errors also fail the job when they are reported on stdout.
- A worker exception's message becomes the job error, while its traceback stays in
  `worker.log`.

## Normalization

Normalization jobs convert inputs that renders reject, with FFmpeg alone. They are kept away from
the GPU path entirely:

- The worker never imports Fusion.
- Submission and worker validation never read Fusion's layout files.
- A normalization request may carry only `media_path` and `fps`; GPU fields are refused rather
  than ignored.
- The supervisor neither waits for nor holds the GPU lock, so normalization runs alongside a GPU
  job and alongside other normalizations.

FFmpeg applies the display matrix, rotations and flips alike, and the output carries no display
matrix or container metadata. The result's `rotation_applied` is FFprobe's rotation angle, and
`reflection_applied` reports a mirrored matrix.

Video timestamps are rebased to zero and resampled to a constant rate: the requested `fps`, or
the source average rate snapped to a standard rate within 2%. The output has the source duration
times the rate, rounded to the nearest frame as FFmpeg's `fps` filter does, but at least one
frame. A clip shorter than half a target frame period keeps its first frame for one period.

Odd dimensions are padded by one pixel on the right or bottom rather than cropped, so no face
pixels are lost. Video is encoded as H.264 at CRF 18 with the render's depth and color policy,
through an explicit swscale conversion so that FFmpeg versions do not negotiate a different
matrix or range. The sample aspect ratio is kept. HDR input is rejected as for renders.

The video is encoded first. The first audio stream is then added as AAC over exactly the encoded
video's interval. Resampling can make that interval shorter or longer than the source video, and
it needs no source duration tag. Earlier audio is trimmed and later audio padded with silence to
stay aligned with the first video frame. Audio ends with the video or, if shorter, before it,
without shortening the video. As for renders, a stream without samples in that interval yields
no audio.

The output passes the render timestamp check. Its frame count, duration, required audio and
audio end (within one AAC frame of the video's) are validated before `normalized.mp4` is
published. The result's `audio`, `audio_start_time` and `audio_duration` describe the published
track.

The expected frame count comes from the video stream's own duration, or from a scan of its packet
timestamps when that is missing, never from the container duration. Output much shorter than
that fails. It is only an estimate: progress reports it as `estimated_total_frames`, with
`eta_seconds`, both null when the extent is unknown, and the result's `frames` is the validated
count. Detection and render progress report exact `total_frames`.

The job's artifact path is accepted as an input wherever the state directory lies, so it is
directly usable as a render input.

## Swap reports

Swap reports instrument Fusion's matching instead of repeating it. No model runs again.

- The adapter's `FrameWorker` subclass wraps `_find_best_target_match`, which both of Fusion's
  swap loops call once per target for each face. For each face embedding seen in the frame, the
  wrapper records Fusion's chosen target and computes the face's similarity to every mapping's
  target embedding once, with Fusion's own `findCosineDistance`.
- It wraps `swap_core` and attributes each swap to the mapping whose target embedding array
  Fusion passes as `t_e`, or records a raised error against it.
- It counts each frame's detections by wrapping the window's `FunctionWorker.run_detect`, which
  both the per-frame path and the `SequentialDetector` call.

Faces that never reach matching are counted as unscored, and their scores are unknown. In the
per-frame path these are faces below Fusion's minimum size or with invalid landmarks. In the
sequential path they are the faces its identity filter dropped.

Recorded outcomes go into per-mapping counters and online run-length intervals: one record per
mapping and reason, a bounded interval list, and constant state per open run. Long renders
therefore keep their telemetry bounded. A `not_requested` state, which passthrough segments and
segments that leave a mapping out record, is excluded from every denominator, and ends a dropout
without flagging it.

The report is written only after the output passes validation, before `result.json`. A failed or
cancelled job therefore never exposes it, and its artifacts are withheld anyway.

Review images decode the source and `result.mp4` at the same frame index, with the same
constant-frame-rate seek that renders rely on, so each pair shows the actual published output.

## Segments

A render may carry `segments`: an ordered list of at most 100 source-frame ranges. Each is
`[start_frame, end_frame)`, half-open, zero-based, in decoded order (the convention swap reports
use), with a stable `id` and one intent.

- A **passthrough** segment leaves its frames exactly as decoded.
- Any other segment changes at least one of `mappings` (indices into the request's mappings;
  the others are not requested there), `parameters` (applied over every active mapping's own)
  and `controls` (over the request's).

Frames outside every segment are gaps and use the base request. Previews and comparisons accept
no segments, and a request without them is unchanged, so earlier `job_id` retries stay
idempotent.

### Validation

`common.validate_segments` runs at submission and again in the worker before Fusion loads. It
refuses:

- unknown fields, duplicate or malformed IDs;
- bounds that are not integers with `0 <= start_frame < end_frame`;
- segments out of order or overlapping (no winner is chosen);
- passthrough combined with a change, and a segment that changes nothing;
- mapping indices that do not exist or repeat;
- settings outside the allowlists or their limits;
- the render-wide selections `SwapModelSelection`, `DetectorModelSelection`,
  `FaceTrackingEnableToggle` and `KPSSmoothingEnableToggle`.

Those selections stay fixed for a reason. Embeddings are bound per swap model when faces are
configured, model provenance follows the base selection, and the tracking toggles choose the
detection path for the whole render.

Clipping is explicit. A segment may begin before `start_frame` or end after the rendered range,
and is clipped to it. A segment with no rendered frame is refused, as is a plan whose every
rendered frame is passthrough. Validation knows only the requested range. With `max_frames`
null, or beyond the video's end, the rendered frames are known only from the video, so the
worker repeats the check with the frame count before Fusion loads, and preflight's media check
runs it too (`segment_outside_range`, `all_passthrough`).

### Applying the plan

`common.segment_plan` resolves the plan into runs that cover the rendered frames in order: each
clipped segment and each gap, with its active mappings and overrides. The engine writes it as
`plan.json` before processing: every segment as requested, with its `applied` frames and
`clipped`, and every run with the effective allowlisted controls and per-mapping parameters it
swaps with.

A frame's run is found by bisecting the run starts with the frame index alone, so a chunked or
resumed render can start anywhere. Entering a run refills `window.target_faces`,
`window.parameters` and `window.control` in place (Fusion reads them live), and the telemetry's
thresholds. The frame worker and its matching are unchanged.

Passthrough frames skip Fusion. The decoded pixels go to the encoder, and telemetry records the
frame with no mapping requested (`not_requested`). Passthrough therefore never counts against
coverage, and ends a dropout without flagging it.

Every frame is still decoded once and encoded once, in order. Frame count, timestamps, duration
and the audio cut are those of a uniform render of the same range. `before.jpg` and
`preview.jpg` show the first processed frame.

### Temporal state

Temporal state is reset at every run boundary, including the edges of passthrough segments and
of gaps: a boundary declares a shot change. On the sequential path the `SequentialDetector` is
reset. That clears ByteTrack (its tracker is re-created from the run's controls on the next
frame, so ByteTrack thresholds may change per segment), its track history, the identity memory
that rescues unmatched faces by overlap, and the keypoint EMA. The per-frame path keeps no state
between frames.

A later tracking implementation must honour the same contract: no track, identity or smoothing
state crosses a run boundary.

### Failure rules

A render with segments still fails when no face swaps in any frame outside passthrough segments,
so passthrough cannot disable the failure check for requested swaps. `min_coverage` and
expected-visible ranges count only requested frames. A segment that requested swaps and swapped
none adds the advisory warning `segment_never_swapped`. The result and the swap report carry
`plan`: every run with its frames, faces swapped and frames with a swap.

Recipes store `segments` with the request. A preview of a segmented recipe drops them as a
recorded substitution, and running the recipe on media with different content is refused,
because segments name frames of the recorded clip.

## Variant comparisons

### One job, not many

A comparison is one job of kind `compare`, not a parent with child jobs. Its request holds:

- the shared `media_path`, `mappings` and `controls`;
- either still `frames` or a clip (`start_frame`, `max_frames`);
- the named `variants`, each overriding only `parameters` (merged over every mapping's) and
  `controls` (merged over the common ones).

`variant_request` in `common.py` builds each variant's single-variant preview or render request.
Submission validates every merged request with the same `validate_fusion_request` as
`submit_swap`, plus the limits (2–6 variants, 1–4 stills, 1–120 clip frames), case-insensitively
unique names and distinct overrides. Nothing invalid is queued, and the error names the variant.
Every variant reads the same references, so pinned fingerprints are checked once, with the first
variant, rather than hashed again per variant under the submit lock.

Because the comparison is one job, everything that holds for a job holds for it unchanged: the
submit lock, stable-ID retries and conflicts on the canonical request, `resolved.json`
fingerprints and retention protection of its inputs, the supervisor lease, release pinning, one
GPU admission and the GPU lock for the whole comparison, and cancellation of the owned process
group. Cancellation stops whichever variant is running and leaves no other job behind.

A still comparison is `interactive`. A clip comparison renders up to six clips, so
`job_priority` makes it `normal` unless `priority` says otherwise. Progress names the running
variant, and a long poll also returns when the variant changes.

### Running the variants

The worker loads Fusion once and runs the variants in request order on that window, as Fusion's
GUI does when settings change. For each variant it refills `window.control` and
`window.parameters` in place (Fusion reads them live), configures the faces again and builds new
frame workers. Detection, tracking and telemetry state therefore never carry from one variant to
the next, and each still frame gets its own preview worker.

Inputs are fixed before any variant runs:

- **Stills** are decoded once, on the packet timeline so that variable-frame-rate sources select
  exact frames, and each variant receives the same arrays.
- **A clip variant** goes through the render path (`encode_range`, shared with `render_video`),
  which decodes the range again with the same frame-accurate constant-rate seek. It is encoded in
  one piece without checkpoints, so a comparison cannot be resumed.

The job hashes every frame each variant receives. It fails a variant whose frames differ from the
earlier variants', or, at the sampled stills, from the separately decoded originals. Every clip
therefore has the same range, frame count, rate and duration, and carries the source's first
audio over the same interval as a render would. No composite combines clips, so no audio is
duplicated or offset.

### Partial failure

A variant failure (a Fusion error, a changed reference, or no swap in any compared frame) is
caught, its reason recorded and its partial files deleted, and the next variant runs. The job
completes when at least one variant succeeded. The result then says `complete: false`, lists
`failed_variants` and carries a `warning`, and the grids show the failures as red `FAILED`
tiles, so a partial comparison never reads as a full one. When every variant fails the job
fails, and its artifacts are withheld as for every failed job.

Grids reuse the contact-sheet layout with display-aspect thumbnails (sample aspect applied).
Full-resolution PNGs, clips and per-variant swap reports are separate artifacts.

The manifest (`compare.json` and the job result) records each variant's overrides, effective
adjustable settings, reference fingerprints, input digest, outcome and artifacts. It also holds a
`submit_swap` argument set built from the request rather than from labels, with source
fingerprints pinned, so the chosen variant can be submitted alone.

Model reuse across variants is the GUI's ordinary in-process path. GPU qualification checks
that a variant's still matches the preview that its follow-up request produces.

## Checkpoints

Every render is encoded in chunks, so an interruption costs at most the chunk in progress.

### Chunks

Chunk *n* holds source frames `[start_frame + n * CHUNK_FRAMES, start_frame + (n + 1) *
CHUNK_FRAMES)`, clipped to the range. `CHUNK_FRAMES` is 250, libx264's default keyframe
interval, so the forced keyframe at each chunk start costs little.

Frames are still decoded by one decoder and processed in order by one worker, so an
uninterrupted render's frames are the same as without chunks; only the encoder restarts per
chunk. Each chunk goes through these steps:

1. The render's ordinary encoder command writes `checkpoint/chunks/<n>.partial.mp4`.
2. The chunk is checked: frame count, constant-rate timestamps from zero (`validate_timing`),
   dimensions, pixel aspect and codec parameters. The codec check includes a hash of the decoder
   configuration (SPS and PPS), which must equal the first chunk's, so that the chunks join
   without re-encoding.
3. It is fsynced, hashed and renamed to `<n>.mp4`.
4. The render's accounting after its last frame is written atomically as `<n>.json`: the swap
   count, the swap telemetry's counters, intervals and open runs, and the segment plan's per-run
   counts.
5. Only then does an atomic rewrite of `checkpoint/manifest.json` list both files with their
   SHA-256.

A crash between these steps leaves a file that the manifest does not list, and such a file is
never trusted.

### The manifest and provenance

The manifest (`visomaster-checkpoint`, version 1, with an exact field set) records:

- the job, and the attempt it resumed;
- the SHA-256 of the request's intent (the resolved request without `resume` and `priority`);
- the complete range with its rational rate, the chunk size and rule;
- whether temporal toggles were on;
- the FFmpeg version line and the codec parameters;
- the provenance;
- the committed chunks, each with its range, frame count, file names, sizes, digests and the job
  that encoded it.

The provenance is `common.provenance` of the resolved request, computed by the worker before
Fusion loads (progress stage `fingerprinting`). The media hash also serves a recipe job's
`media_sha256` check. It is the recipes' fingerprint, so `provenance_changes` compares it, and
the model set it hashes is the one recipes use.

### Hashing before the GPU is taken

The expensive part, hashing the media, references, model files and inherited chunks, happens
earlier, outside the GPU lock. The job's supervisor runs the same functions after it takes the
lease and before the job joins the queue. It runs them in a thread, so it notices a cancellation
or configuration change meanwhile. It records each digest in `digests.json`, keyed by the file's
device, inode, size, and modification and change times (`common.DIGESTS`).

The worker loads that table. Its own hashing, the same code, returns a recorded digest when the
file it opens still has that identity, and hashes any other file. The checks under the lock are
therefore one `fstat` per file.

Changed bytes are still detected: a rewrite changes at least the change time, which cannot be
set back. Two windows remain, as before: a change after the worker's check and before FFmpeg or
Fusion reads the bytes, and a change that keeps the identity, which only a filesystem whose
timestamps are coarser than the time between the supervisor's hash and the next write could
allow. Hashing failures are left to the worker, which hashes what is not recorded and fails as
it would have.

A recipe job that is not a render (a preview with `media_sha256`) has its media hashed the same
way.

### Joining the chunks

The output joins the chunks with FFmpeg's concat demuxer and stream copy, opened through
`common.concat_input`. The list names chunk files by safe relative paths, and only the `file`
protocol and the concat and MP4 demuxers are allowed; the concat demuxer passes that restriction
on to each chunk.

Each listed chunk has an explicit duration, chosen so that the running total is the exact chunk
boundary rounded to the microsecond. Without it, FFmpeg adds up each chunk's own
microsecond-rounded duration, and at 30000/1001 the error reaches a whole tick of the 1/30000
time base after about 50 chunks.

The same command adds the source audio once over the whole range, so audio is cut and encoded
once. The output is validated as usual (frame count, duration, required audio and its end), and
its timestamps are checked frame by frame from zero, so a gap or duplicate where chunks meet
fails the render.

`before.jpg` and `preview.jpg` are decoded from the source and from `result.mp4` at the first
processed frame (after a leading passthrough segment), as review images are, so a resumed render
shows the same images. When the supervisor publishes `completed` it removes the checkpoint,
because the output is final.

### Resuming

Resumption is explicit. `resume_render` queues a new render with its own job ID. Its request is
the earlier attempt's resolved request plus `resume` (`from_job_id`, `attempt` and the
invocation), and its state carries `resume_from`. The earlier job is never changed and keeps its
terminal state.

A retry with the same ID is answered from the persisted invocation before the earlier attempt is
consulted, and a different invocation under that ID is a conflict, as for recipes. Only a failed
or cancelled render with a readable manifest can be resumed. A completed, active or non-render
job, or one without a manifest, is refused with the reason. Cancellation only stops a job: no
path in the supervisor or the worker queues another attempt.

### Checking compatibility

Compatibility is checked twice:

1. `resume_render` hashes the inputs and models of the new request and lists every difference
   from the manifest before it submits. `common.resume_problems` compares the intent, the
   provenance changes, the FFmpeg version and the temporal toggles.
2. The new job's worker repeats the check from its own provenance before Fusion loads, using the
   digests its supervisor recorded before admission, and fails with the same list.

Renders with `FaceTrackingEnableToggle` or `KPSSmoothingEnableToggle` are refused. Their
`SequentialDetector` carries ByteTrack tracks, identity memory and keypoint smoothing from frame
to frame, which no chunk records. Restarting it at the resume frame, as a segment boundary does
by design, would change frames after the boundary while claiming an equivalent recovery. Their
chunks are still written, and the manifest records `temporal: true`. Restoring that state would
need the detector's internals checkpointed, or a warm-up over earlier frames qualified to
reproduce it. Until then such a render is resubmitted from the start.

### Linking instead of sharing

Under the submit lock, after validation, `submit` rereads the earlier attempt (terminal, a
render, the same intent). It hard-links each chunk and accounting file its manifest lists into
the new job's staging `checkpoint/chunks`, and copies the manifest as `inherited.json`, before
the job directory is published.

Chunk files are never rewritten in place: a worker replaces one only by renaming a new file over
its own link. Concurrent resumes of one attempt therefore share inodes safely, and each owns its
directory entries.

From then on the new job never reads the earlier attempt's directory. Deleting or pruning that
attempt, which the ordinary retention rules allow once it is old enough, removes only its links.
The new job's links keep the data while it is queued or running, when retention cannot touch it.
A filesystem without hard links gets copies.

### Verifying reused chunks

The worker trusts nothing it did not verify. It walks the inherited chunks in order and keeps
the longest prefix whose entries have the expected index, range and file names, and whose files
still have their recorded sizes and SHA-256. Then it:

1. deletes everything else in its `chunks` directory;
2. writes its own manifest with that prefix;
3. restores the swap count, telemetry and segment counts from the last kept chunk's accounting;
4. decodes from the first frame after it.

A damaged chunk invalidates the chunks after it too, because their accounting includes its
frames. In practice an interruption leaves exactly a prefix.

`Segments.at` finds the run from the frame index alone, and a fresh plan enters it at the resume
frame, binding the same settings an uninterrupted render had there. The per-frame path keeps no
state between frames, so the resumed frames match.

If every chunk survives, nothing is decoded or processed, and the render only joins and
validates. The swap report, the all-zero-swap rule, `min_coverage` and the plan summary then see
the whole range, as they would have without the interruption.

Resumed chunks are encoded by the same command. A new chunk whose codec parameters differ from
the reused ones (for example after an FFmpeg upgrade that the version check missed) fails the
render rather than producing a stream that needs re-encoding.

### Retention of checkpoints

A checkpoint lives in its job's directory, so it follows the job:

- A queued or running render's chunks are protected with the job.
- A failed or cancelled render's chunks stay resumable until `delete_job` or `prune_storage`
  removes the job, by its finish time like any terminal job. While it is kept, its inputs are
  protected like those of a queued or running job, so pruning cannot leave it unresumable.
- A completed render's chunks are removed at completion or, if that removal is interrupted,
  with the job.

Releases need no extra pinning. The resume runs the release selected when it is submitted, and
resuming requires its adapter code to equal the recorded code, whichever release directory held
it.

Disk use while rendering is the chunks plus the final output, as it was the video-only
intermediate plus the output before chunking. Deletion and pruning report the bytes a removal
frees. A file is counted once per report, for the first job or path listed with it, and only
when every hard link to it is removed. Chunks that a resumed render links are therefore counted
with whichever removal takes their last link.

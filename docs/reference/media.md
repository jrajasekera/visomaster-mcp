# Media requirements

This page lists which files the tools accept and what they produce.
[Video processing](../explanation/video-processing.md) explains the reasons.

## Input paths

Media paths and face image paths are absolute workstation paths. After symlinks and `..` are
resolved, each must be one of:

- a file inside `media_root`;
- a complete upload, `<state_dir>/uploads/<name>`;
- a file in `<state_dir>/jobs/<job_id>/artifacts/` of a **completed** job.

Artifacts of queued, running, failed or cancelled jobs are refused, with the job's status.
Nothing else in `state_dir` is ever an input: requests, logs, locks, staging, trash and partial
uploads are refused even when `state_dir` lies inside `media_root`.

## Containers

Every FFmpeg and FFprobe input is opened with only the `file` protocol and these demuxers:

| Type | Formats |
| --- | --- |
| Video | MP4/MOV, Matroska/WebM, AVI, MPEG-TS |
| Image | Single PNG, JPEG, WebP or BMP files |

Any other detected format is refused before its header is read, whatever the file extension.
That includes HLS playlists and concat lists. The probe reports it as `unsupported_container`.
`%d` patterns in file names are never expanded.

## Uploads

`upload_media` accepts local files with these extensions, from 1 byte to 20 GiB:
`.png`, `.jpg`, `.jpeg`, `.webp`, `.mp4`, `.mkv`, `.mov`.

Uploads are named by the SHA-256 the client computes. If a file with that name and size exists,
it is returned without a transfer (`reused: true`). A new file is published only after the
workstation verifies its size and hash.

## Face images

- Target and source face images must contain exactly one detectable face.
- Each face image may be at most 256 MiB.
- Images are decoded with EXIF orientation applied.
- Every face image is fingerprinted with SHA-256 at submission. If the file changes before the
  worker reads it, the job fails.

## Frame numbering

- Frames are numbered from zero, in decoded order.
- Ranges are half-open, `[start_frame, end_frame)`: `end_frame` is the first frame after the
  range.
- A still image has only frame 0.
- Detection and previews find a frame's presentation time from the video stream's packet
  timestamps, so even variable-frame-rate input gives the exact frame. An index at or beyond the
  frame count is refused as out of range.
- Frames after the first of a video whose packets lack timestamps (as in some AVI files) need
  normalization.
- Sampling for multi-frame detection counts the video stream's own frames, not the container
  duration.

## Render requirements

Renders and clip comparisons refuse video with the problems below. Rotated or mirrored video
is refused by every tool that reads video frames, including detection and previews.

| Problem | Preflight code | Fixed by `normalize_media` |
| --- | --- | --- |
| Variable frame rate | `variable_frame_rate` | yes |
| Variable or missing timestamps | `variable_timestamps`, `missing_timestamps` | yes |
| Video timestamps that do not start at zero | `nonzero_start` | yes |
| Odd width or height | `odd_dimensions` | yes |
| A display matrix that rotates or mirrors the picture | `rotated_or_mirrored` | yes |
| HDR transfer (PQ or HLG) | `hdr_unsupported` | no |
| An unsupported color matrix | `matrix_unsupported` | no |

Timestamps are checked only through the requested frames.

### Color matrices

Accepted: BT.709, BT.601 (`smpte170m`, `bt470bg`), FCC, SMPTE 240M, BT.2020 non-constant
luminance, RGB (`gbr`) and untagged video (decoded as BT.601).

Refused by renders, detection and normalization alike, before anything is written: YCgCo,
BT.2020 constant luminance, ICtCp, SMPTE 2085 and the chromaticity-derived matrices. Transcode
such sources to BT.709 with another tool first.

## Render output

| Property | Value |
| --- | --- |
| Video codec | H.264 (libx264, CRF 18), encoded in chunks of 250 source frames and joined without re-encoding |
| Bit depth | 8-bit for 8-bit sources; 10-bit for deeper sources (Fusion itself processes 8-bit BGR) |
| Color | The source's matrix, range, primaries and transfer tags. RGB sources are encoded as BT.709 limited range |
| Frame rate, dimensions, sample aspect ratio | Same as the source |
| Audio | The first audio stream, cut to the rendered interval and encoded as AAC. Padded with silence where the stream starts later. No audio (`audio: none`) when the stream has no samples in the interval |

Every output is checked for frame count, duration, timestamps from zero, required audio, and
audio ending no more than one AAC frame after the video. A render in which no face swaps fails.

## Normalization output

`normalize_media` writes `normalized.mp4`:

| Property | Value |
| --- | --- |
| Video | H.264, CRF 18, with the render's depth and color policy |
| Frame rate | Constant: `fps` if given (such as `25`, `29.97` or `"30000/1001"`), otherwise the source's average rate snapped to a standard rate within 2% |
| Timestamps | Start at zero |
| Orientation | Rotation and flips applied; no display matrix or container metadata |
| Dimensions | Odd dimensions padded by one pixel on the right or bottom, never cropped |
| Sample aspect ratio | Kept |
| Length | Source duration times the rate, rounded to the nearest frame; at least one frame |
| Audio | The first audio stream as AAC over exactly the video's interval: trimmed or padded with silence to align with the first frame, and ending with the video or before it |

The result reports `media_path` (the `normalized.mp4` artifact, usable as input wherever
`state_dir` lies), `frames` (the validated count), `fps`, `duration`, `width`, `height`,
`rotation_applied` (FFprobe's rotation angle), `reflection_applied` (whether the matrix
mirrored the picture), `source` (the source's dimensions, rates and start time), `audio` (`AAC`
or `none`), `audio_start_time` and `audio_duration`. HDR input is refused. Normalization does not use
Fusion or the GPU.

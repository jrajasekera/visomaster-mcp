# Swap faces in a video

This guide takes a video from detection to a reviewed render: find the people in it, choose
targets and source images, check the request, preview, render and review. For a guided first
run, see the tutorial [Your first face swap](../tutorials/first-face-swap.md).

Paths in every tool call are absolute workstation paths. Each tool's arguments are listed in
[Tools](../reference/tools.md).

## 1. Check the installation

Call `health_check` and `get_settings`. `get_settings` lists the parameters and controls you can
change, with their defaults, choices and limits.

## 2. Make the media available

Use an absolute workstation path below `media_root`, or upload a local file with
`upload_media`. Uploads are named by their SHA-256, so uploading the same file again returns the
existing path.

## 3. Find the people to replace

- **One frame:** call `detect_faces` with a `frame` where the person is clearly visible.
- **The whole video:** call `detect_faces` with `sample_frames` (for example 16) or an explicit
  list of `frames` (at most 64). Faces are grouped into `identities`, one per person.

Follow the job with `job_status` and `wait_seconds: 50` until it completes. Then call
`view_artifact` with `name: "contact.jpg"`: one grid shows every crop, labelled `face N` or
`id N: <appearances>x`.

Use the `reference_path` of the face or identity you want as its `target_face`. A crop whose
`reference_verified` is false has `reference_path: null`, is marked `*` and framed in red. Try
another frame for that person.

If one person appears as two identities, typically a profile and a frontal view, run detection
again with a lower `similarity_threshold`, such as 50. A lower threshold can also merge two
different people who never share a frame, so check the contact sheet.
[Face matching](../explanation/face-matching.md#grouping-faces-into-identities) explains why.

## 4. Choose the source images

Give one or more images of the replacement person as `source_faces`. Each image must contain
exactly one detectable face.

With several images, check them first with `analyze_references`, and pin and weight them as
described in [Check and weight source images](check-reference-images.md).

## 5. Check the request

Call `preflight_swap` with exactly the arguments you plan to give `submit_swap`. It starts
nothing. Fix every check reported as `blocked`; each one lists `actions` to take. See
[Preflight checks](../reference/preflight-checks.md) for the codes.

## 6. Preview, then render

1. Submit a `preview` of one frame and view its `preview.jpg`:

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

2. Submit a short `render` with `start_frame` and `max_frames`, and review it.
3. Submit the full range. A `max_frames` of `null` renders through the end of the video.

Frame indices start at zero. To choose between settings, run
[`compare_variants`](compare-settings.md) on the same frames before the full render.

### Give each job a stable ID

Pass a `job_id` for each job you intend to run: 1–64 letters, digits, underscores or hyphens,
starting with a letter or digit. Submitting the same ID with the same arguments returns the
existing job, even after its inputs have been moved or pruned, so a retry after a dropped
connection never starts a second job. The same ID with different arguments is rejected, and an
empty ID is an error. Only an omitted or `null` ID is generated.

To retry a failed or cancelled job, use a new ID. A failed or cancelled render can also be
[resumed](resume-a-render.md).

### Replace several people

Add one entry to `mappings` per person, each with its own `target_face`, `source_faces` and
`parameters`. Each entry is reported separately in the swap report by its zero-based index.

## 7. Follow the job

Call `job_status` with `wait_seconds: 50`. It returns early when the status, progress stage or
queue position changes. Jobs keep running if the client disconnects; `list_jobs` shows jobs from
every client. To stop a job, call `cancel_job` and poll until its state is terminal.

## 8. Review the result

1. Read the result's `swap_report`. Its `warnings` (for example that a mapping swapped in no
   frame) point at frames to check. They do not fail the job. The warning codes, such as
   `never_swapped` and `suspicious_intervals`, are in the `swap-report.json` artifact.
2. View each `review-NN.jpg` the report lists. Each shows a source frame beside the same output
   frame.
3. Download `result.mp4` (render) or `result.png` (preview) with `download_artifact`.

The fields are described in [Swap reports](../reference/results.md#swap-reports). To make a
render fail when a person is missed, pass `report` options such as `min_coverage`; see
[Report options](../reference/tools.md#report-options).

## Fix media that renders reject

Renders reject video with a variable frame rate, timestamps that do not start at zero, odd frame
dimensions, or display metadata that rotates or mirrors the picture. Detection and previews also
reject rotated or mirrored video. `preflight_swap` reports
these with an action that suggests `normalize_media`.

1. Call `normalize_media` with the same `media_path`. Pass `fps` (such as `25` or
   `"30000/1001"`) to choose the output rate; by default the source's average rate is used,
   snapped to a standard rate within 2%.
2. Wait for the job to complete. It runs on the CPU and does not wait for the GPU.
3. Use the result's `media_path` (the job's `normalized.mp4` artifact) for detection, preview
   and rendering.

Frame indices then refer to the normalized timeline, so detect faces again on the normalized
video. See [Media requirements](../reference/media.md) for the full rules.

## Stop a look-alike from being swapped

By default each mapping swaps only the single face that matches its target best in each frame
(the control `SwapOnlyBestMatchEnableToggle` is on). This keeps a second person from being
swapped while the target is in the frame.

- If a preview or short render shows another face swapped in frames where the target is
  absent, raise that mapping's `SimilarityThresholdSlider`. On the qualified clip, 75 removed
  such swaps.
- If one person appears several times in a frame and every instance should change, pass
  `"controls": {"SwapOnlyBestMatchEnableToggle": false}`.

## Keep batch work from delaying others

GPU jobs run one at a time. Detection and previews queue as `interactive` and start before
waiting renders. Pass `"priority": "normal"` for batch work, such as many detections, that
should not delay someone else's previews. See [Jobs and scheduling](../reference/jobs.md#gpu-queue).

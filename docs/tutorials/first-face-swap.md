# Your first face swap

In this tutorial we replace one person's face in a short video clip. We will check the
installation, upload two files, find the face to replace, look at a single swapped frame, and
then render and download a short clip. Most of the time goes into waiting for GPU jobs.

## Before you start

You need:

- an installation that passes its checks (see [Install VisoMaster MCP](../how-to/install.md));
- an MCP client session (Claude Code, Codex or Claude Desktop) with the `visomaster` server;
- on your computer, a short MP4 clip (a few seconds) in which one person's face is clearly
  visible in the first frame;
- on your computer, a sharp photo of the replacement face, with exactly one face in it.

You can ask your assistant in plain words to call each tool, or call the tools directly. The
arguments are shown as JSON. Replace the paths with your own.

## Step 1: Check the installation

Call `health_check`.

Look for `compatible: true`, `code_matches: true` and `config_error: null`. If any of them is
different, stop and fix it with [Troubleshooting](../reference/troubleshooting.md).

Then call `get_settings`. It lists every setting you can change, with its default. We will use
the defaults in this tutorial.

## Step 2: Upload the clip and the photo

Call `upload_media` for the clip:

```json
{"local_path": "~/Movies/clip.mp4"}
```

The result has a `path` on the workstation, ending in `.mp4`. Write it down; we will call it
`CLIP` below.

Call `upload_media` again for the photo:

```json
{"local_path": "~/Pictures/replacement.png"}
```

Write down its `path` as `FACE`.

## Step 3: Find the face to replace

Call `detect_faces` on the first frame of the clip:

```json
{"media_path": "CLIP", "frame": 0, "job_id": "tutorial-detect"}
```

This queues a GPU job and returns straight away with `status: "queued"`. Follow it with
`job_status`:

```json
{"job_id": "tutorial-detect", "wait_seconds": 50}
```

Each call waits up to 50 seconds for something to change. Repeat it until `status` is
`completed`.

Now look at the faces it found. Call `view_artifact`:

```json
{"job_id": "tutorial-detect", "name": "contact.jpg"}
```

You see a grid of face crops labelled `face 0`, `face 1` and so on. In the job's `result`, find
the entry in `faces` whose `index` matches the face you want to replace. Its `reference_path` is
the target. Write it down as `TARGET`.

If `reference_path` is `null`, that crop failed verification: it is marked `*` and framed in
red on the grid. Run detection again on a frame where the face is larger and seen from the
front.

## Step 4: Preview one frame

Call `submit_swap` with a preview of frame 0:

```json
{
  "job_id": "tutorial-preview",
  "media_path": "CLIP",
  "mode": "preview",
  "frame": 0,
  "mappings": [{"target_face": "TARGET", "source_faces": ["FACE"]}]
}
```

Follow it with `job_status` and `wait_seconds: 50` until it completes. Then call
`view_artifact` with `job_id: "tutorial-preview"`. The default artifact, `preview.jpg`, shows
the swapped frame. `before.jpg` shows the same frame before the swap.

The result's `swap_report` should show the mapping as swapped. If the preview job failed with
`No faces matched the target references`, check that `TARGET` comes from the same clip and
that the face is visible in frame 0.

## Step 5: Render a short clip

Submit a render of the first 48 frames:

```json
{
  "job_id": "tutorial-render",
  "media_path": "CLIP",
  "mode": "render",
  "start_frame": 0,
  "max_frames": 48,
  "mappings": [{"target_face": "TARGET", "source_faces": ["FACE"]}]
}
```

Follow it with `job_status` and `wait_seconds: 50`. While it runs, `progress` counts the frames
done. Render jobs wait behind previews and detections in the GPU queue, so it may stay `queued`
for a while if others are using the workstation.

If the render fails with an error about variable frame rate, timestamps, odd dimensions or
rotation, the clip needs converting first. [Swap faces in a video](../how-to/swap-faces.md#fix-media-that-renders-reject)
shows how.

## Step 6: Review and download the result

When the render completes, look at its `swap_report` in the result. Any `warnings` there, such
as a mapping that never swapped, point at frames to check. Call `view_artifact` for any
`review-NN.jpg` images the result lists; each shows a source frame beside the rendered one.

Download the video:

```json
{"job_id": "tutorial-render", "name": "result.mp4"}
```

`download_artifact` returns the local path of the file, by default under
`~/Downloads/visomaster-mcp/tutorial-render/`. Open it and watch the swap.

## What we did

We checked the installation, uploaded media, detected a face and used its verified crop as the
target, previewed one frame and rendered a short clip. Each step was a durable job with a stable
`job_id`, so calling the same tool again with the same arguments would have returned the same
job instead of starting a new one.

## Next steps

- [Swap faces in a video](../how-to/swap-faces.md) covers longer videos, several people and
  media that needs converting.
- [Check and weight source images](../how-to/check-reference-images.md) helps when you have
  several photos of the replacement face.
- [Architecture](../explanation/architecture.md) explains what happens on the workstation.

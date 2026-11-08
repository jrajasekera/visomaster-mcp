# Compare settings side by side

This guide runs several settings variants on the same frames in one job, so you can choose
between them by eye, then submit the chosen one.

## Run a comparison

1. Prepare the `mappings` and `controls` you would give `submit_swap`. Every variant shares
   them.

2. Define 2–6 `variants`. Each has a `name` (1–20 letters, digits, `_` or `-`, unique ignoring
   case) and optional `parameters` and `controls`. A variant's `parameters` are merged over every
   mapping's `parameters`, and its `controls` over the common `controls`. Nothing else differs
   between variants.

3. Choose what to compare:

   - **Stills:** `frames`, a list of 1–4 distinct frame indices (`[0]` for an image). The job
     queues as `interactive`.
   - **A clip:** `start_frame` (default 0) and `max_frames` (1–120). The video must have a
     constant frame rate, as for renders. The job queues as `normal`.

4. Call `compare_variants`:

   ```json
   {
     "job_id": "shot01-compare-v1",
     "media_path": "/srv/visomaster/media/clip.mp4",
     "mappings": [
       {
         "target_face": "/srv/visomaster/runtime/mcp/jobs/detection01/artifacts/face-0.png",
         "source_faces": ["/srv/visomaster/media/replacement.png"]
       }
     ],
     "frames": [24, 180],
     "variants": [
       {"name": "plain"},
       {"name": "restored", "parameters": {"FaceRestorerEnableToggle": true}}
     ]
   }
   ```

   Every variant's merged request is validated like a `submit_swap` request before anything is
   queued. An error names the variant at fault.

5. Follow the job with `job_status`. Progress shows `stage: comparing` with the running
   `variant`. `cancel_job` stops every variant.

## Review the variants

1. View each `compare-f<N>.jpg` grid with `view_artifact`. Each shows the original frame, then
   every variant labelled `<index> <name>`, scaled to the same cell. For clips there is one grid
   each for the first, middle and last frame.
2. For detail, download the full-resolution `original-f<N>.png` and `v<index>-<name>-f<N>.png`
   files, or the clips `v<index>-<name>.mp4`.
3. Check `complete` in the result. If it is false, some variants failed: `failed_variants`
   names them and `warning` explains them, and the grids show them as red `FAILED` tiles. Only a
   result with `complete: true` compares every variant.

Nothing picks a best variant for you.

## Submit the chosen variant

Each variant in the result has a `submit_swap` object: arguments that repeat that variant alone
with exactly its merged settings, and with each mapping's `source_sha256` pinned to the bytes
the comparison used. For stills it is a preview of the first compared frame; for a clip, a
render of the compared range.

1. Copy the chosen variant's `submit_swap` arguments.
2. Add a `job_id`, and change `frame` or the range as needed.
3. Call `submit_swap` with them.

A failed or cancelled comparison cannot be resumed. Submit it again.

The limits and the manifest fields are listed in [Variant comparisons](../reference/results.md#variant-comparisons).

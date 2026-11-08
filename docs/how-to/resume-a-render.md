# Resume a failed or cancelled render

Renders are written in checkpointed chunks of 250 source frames. When a long render fails or is
cancelled, you can continue it from the chunks it finished instead of starting again.

## Resume

1. Find the render's `job_id`. Its status must be `failed` or `cancelled`. Completed, queued and
   running renders cannot be resumed, and neither can other job kinds.

2. Call `resume_render` with that ID as `from_job_id` and a new `job_id`:

   ```json
   {"from_job_id": "shot01-render-v1", "job_id": "shot01-render-v1-resume1"}
   ```

   Optionally pass `priority`. By default the new job uses the earlier render's priority if that
   was given explicitly, otherwise `normal`.

3. Follow the new job with `job_status`. `job_status` and `list_jobs` show the earlier job as
   `resume_from`. Its progress counts `frames` from the start of the range, reused ones
   included, and reports them as `reused_frames`; `frames_per_second` and `eta_seconds` cover
   only the frames it renders.

The earlier job keeps its terminal state and is not changed. The new job repeats the earlier
request, reuses the longest run of chunks from the start whose files still verify, and renders
the remaining frames. Its result and swap report cover the whole range, as an uninterrupted
render's would. The result's `checkpoint` reports `chunks`, `chunk_frames`, `reused_chunks`,
`reused_frames` and `resumed_from`.

Calling `resume_render` again with the same `job_id` and arguments returns the same job. A
different `from_job_id` or `priority` under that ID is refused.

## When a resume is refused

`resume_render` hashes the inputs and models and compares everything the earlier render
recorded. It refuses, with a list of the differences, when any of these changed:

- the bytes of the media or any reference image;
- the effective settings, or the request resolves differently;
- the model files the settings load;
- the Fusion revision, the adapter code or the remote configuration;
- the FFmpeg build.

Submit a fresh render in that case. The new job's worker repeats the comparison before it reuses
anything: a reference image that changes after `resume_render` accepted the job fails it with
`reference_changed`, which also advises a fresh render.

Renders with `FaceTrackingEnableToggle` or `KPSSmoothingEnableToggle` are always refused: their
tracking and smoothing state is not checkpointed. Submit them again from the start.

## Keep a render resumable

A failed or cancelled render keeps its checkpoint, and stays resumable, until `delete_job` or
`prune_storage` removes the job. While it is kept, pruning also keeps its inputs. Deleting the
earlier attempt after you resumed it is safe: the new job holds its own links to the chunks it
may reuse.

Cancelling a job never starts another attempt; only `resume_render` does.

[Checkpoints](../explanation/video-processing.md#checkpoints) explains how chunks are verified
and joined.

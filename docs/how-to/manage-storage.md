# Free workstation storage

Jobs, their logs and artifacts, and uploads stay on the workstation until you remove them. This
guide removes them safely.

## Delete one job

1. Make sure the job is terminal (`completed`, `failed` or `cancelled`). Cancel a queued or
   running job with `cancel_job` and wait for it to stop.
2. Call `delete_job` with its `job_id`.

The job's directory, logs and artifacts are removed permanently. The result reports the bytes
freed. Deletion is refused while a queued or running job, a saved recipe or a resumable render
uses one of the job's artifacts as input.

Deleting a failed or cancelled render also removes its checkpoint, so it can no longer be
resumed.

## Prune old files

1. Preview what would be removed. `prune_storage` is a dry run by default:

   ```json
   {"older_than_days": 30}
   ```

   The result lists `jobs`, `uploads` and `stale` entries, each with the bytes it frees, and
   `total_bytes`.

2. Check the list. Then run the same call with `dry_run: false`.

Pruning removes:

- terminal jobs that finished more than `older_than_days` ago (default 30);
- uploads older than that (uploading identical media again resets an upload's age);
- staging and trash directories left by crashes;
- partial uploads idle for longer than both `older_than_days` and one hour.

It keeps every file that a saved recipe, a queued or running job, or a failed or cancelled render
that can still be resumed uses as input. The dry run applies the same checks.

Partial files of failed and cancelled jobs stay for diagnosis until the job is removed.
[Storage](../reference/jobs.md#storage) describes the layout and retention rules.

## Clear old downloads

Downloaded artifacts are on the client, under the downloads directory (default
`~/Downloads/visomaster-mcp`), one subdirectory per job. Delete them with your file manager;
nothing on the workstation depends on them.

# Jobs and scheduling reference

Every tool that processes media creates a job on the workstation. Jobs are durable: they
survive client disconnection, and every client sees the same jobs.

## Job kinds

| Kind | Created by | Uses the GPU | Default class |
| --- | --- | --- | --- |
| `detect` | `detect_faces` | yes | `interactive` |
| `analyze` | `analyze_references`, `preflight_swap` with `analyze_references` | yes | `interactive` |
| `preview` | `submit_swap` with `mode: "preview"`, `run_recipe` | yes | `interactive` |
| `render` | `submit_swap` with `mode: "render"`, `resume_render`, `run_recipe` | yes | `normal` |
| `compare` | `compare_variants` | yes | `interactive` for stills, `normal` for clips |
| `normalize` | `normalize_media` | no | none: never queued |

## Job IDs

- 1–64 letters, digits, underscores or hyphens, starting with a letter or digit.
- Omitted or `null`: the client generates a random ID. An empty string is an error.
- Submitting an existing ID with the same request returns that job, even after its inputs were
  moved or pruned. Requests are compared as canonical JSON.
- Submitting an existing ID with a different request is rejected: `job_id already exists with
  a different request`.
- To retry a failed or cancelled job, submit it under a new ID.

## States

| Status | Terminal | Meaning |
| --- | --- | --- |
| `queued` | no | Submitted; waiting to start, for the GPU or for input hashing |
| `running` | no | The worker is processing |
| `completed` | yes | Finished; artifacts are available |
| `failed` | yes | Stopped with an `error`; the log tail is returned |
| `cancelled` | yes | Stopped by `cancel_job` |

A job whose supervisor disappeared becomes `failed` with the error `Supervisor exited or host
restarted`: at once if it had recorded its process group, otherwise after a 15-second startup
grace period. Terminal states record `finished_at`.

### State fields

`job_status` returns the job's state, with:

| Field | Present | Meaning |
| --- | --- | --- |
| `job_id`, `status` | always | |
| `created_at`, `started_at`, `finished_at` | when reached | Unix timestamps |
| `error` | failed jobs | The failure message |
| `progress` | once started | Current `stage` and counts (see below) |
| `priority`, `sequence` | GPU jobs | Queue class and FIFO order within it |
| `queue` | queued GPU jobs | Advisory standing (see [GPU queue](#gpu-queue)) |
| `resume_from` | resumed renders | The render it continues |
| `result` | completed jobs | The result manifest (see [Job results](results.md)) |
| `log_tail` | on request or failure | The last 6000 bytes of `worker.log` |

`list_jobs` returns summaries of the most recently updated jobs: `job_id`, `kind`, `status`,
`media_path`, the frame selection (see [`list_jobs`](tools.md#list_jobs)), `created_at`,
`started_at`, `finished_at`, `elapsed_seconds`, `error`, `progress`, and for GPU jobs `priority`,
`sequence`, `queue` and `resume_from` where they apply.

### Progress stages

| Stage | Jobs | Other fields |
| --- | --- | --- |
| `fingerprinting` | Renders and recipe jobs, while inputs and models are hashed | |
| `verifying_media` | Recipe previews, while the media is hashed | |
| `loading_models` | GPU jobs | |
| `detecting` | `detect` | `frames`, `total_frames` |
| `analyzing` | `analyze` | `references`, `total_references` |
| `rendering` | `render` | `frames`, `total_frames`, `swapped_faces`, `frames_per_second`, `eta_seconds`, `reused_frames`, and `segment` when the render has segments |
| `comparing` | `compare` | `variant`, `variants`, `name`, `frames`, `total_frames`; clip comparisons also `swapped_faces`, `frames_per_second`, `eta_seconds` |
| `normalizing` | `normalize` | `frames`, `estimated_total_frames`, `frames_per_second`, `eta_seconds` (the estimate and ETA are null when the extent is unknown) |

### Long polling

`job_status` with `wait_seconds` (at most 50) blocks on the workstation until one of these
differs from its value at call time, the job ends, or the wait elapses:

- the status;
- the progress stage, or the variant a comparison is running;
- the queue `position` or `reason`.

Frame counts advancing within a stage do not end the wait.

### Cancellation

`cancel_job` returns `cancel_requested` and the job's `status`. Poll `job_status` until the state
is terminal.

- A queued job leaves the queue at once.
- A running worker gets SIGTERM on its process group, then 10 seconds to exit, after which the
  remaining group members are killed.
- A cancellation that arrives just as a job starts stops it like any running job. One that
  arrives after the job succeeded leaves it `completed`.
- Cancellation never starts another attempt. See [`resume_render`](tools.md#resume_render).

## GPU queue

GPU jobs run one at a time, in an order that every client and every deployment sharing the
`lock_dir` agrees on. Each GPU job gets a class and a sequence number at submission. Both are
kept for the job's lifetime, and a retried submission with the same `job_id` keeps its sequence.

### Classes

| Class | Default for | Order |
| --- | --- | --- |
| `interactive` | `detect_faces`, `submit_swap` previews, `analyze_references`, still `compare_variants` | Starts before waiting normal jobs, oldest first |
| `normal` | `submit_swap` renders, clip `compare_variants`, any other GPU job kind | Oldest first; becomes due under the fairness rule |

`priority` sets the class on these tools:

| Tool | Default when omitted |
| --- | --- |
| `detect_faces`, `analyze_references`, `submit_swap`, `compare_variants` | The class in the table above |
| `resume_render` | The earlier render's explicit priority, otherwise `normal` |
| `run_recipe` | The recipe's stored priority, otherwise the class of its mode |
| `save_recipe` | From an explicit request: stores no priority. From a job: keeps the job's explicit priority, if it had one; the `priority` argument is ignored |

### Fairness rule

When only one class is waiting, its oldest job starts next. When both are waiting, the oldest
interactive job starts next unless waiting normal work is due. Normal work becomes due when
either:

- 3 interactive jobs have started in a row while it waited (`MAX_STREAK`), or
- 300 seconds have passed since the first of those started (`AGE_LIMIT`).

Starting a normal job resets both counters. So a steady stream of previews delays a waiting
render by at most three previews, and no more than five minutes after the first of them.

Scheduling happens only between jobs. A running job keeps the GPU until it finishes, fails or
is cancelled. The queue never stops other processes, and Fusion's GUI does not take part:
coordinate large GUI renders with MCP renders yourself.

### Queue fields

While a GPU job is queued, `job_status` and `list_jobs` include a `queue` object:

| Field | Meaning |
| --- | --- |
| `position` | Place in the order the rule would produce now; 1 is next. Advisory: it changes as jobs arrive, end, or become due |
| `reason` | Why the job is waiting (see below) |
| `interactive_streak` | Interactive jobs started in a row ahead of waiting normal work |
| `waiting_seconds` | Time since submission |

| Reason | Meaning |
| --- | --- |
| `next` | First, and the GPU is free |
| `gpu_busy` | First, waiting for the running job |
| `behind_earlier` | Older jobs of its own class are ahead |
| `behind_interactive` | A normal job waiting for interactive jobs |
| `behind_due_normal` | An interactive job waiting for a due normal job |
| `cancel_requested` | No longer eligible |
| `starting` | No queue entry yet: being admitted, or a render or recipe job whose inputs are being hashed before it joins (progress stage `fingerprinting`). Later jobs may start meanwhile |

There is no start-time estimate.

## Storage

All job data lives under `state_dir` on the workstation.

| Path | Contents |
| --- | --- |
| `jobs/<job_id>/` | One directory per job |
| `jobs/<job_id>/artifacts/` | Output files; available only after the job completes |
| `jobs/<job_id>/checkpoint/` | A render's chunks and manifest; removed when the render completes |
| `jobs/<job_id>/request.json` | The request as submitted |
| `jobs/<job_id>/resolved.json` | The request with every input path resolved and fingerprinted |
| `jobs/<job_id>/state.json`, `progress.json`, `result.json` | State, progress and result |
| `jobs/<job_id>/worker.log` | Supervisor and worker log |
| `uploads/<sha256><suffix>` | Uploaded media and recipe face images, named by content |
| `recipes/<name>/<revision>.json` | Recipe revisions |
| `submit.lock` | Lock shared by submission, cancellation and retention |

Downloads go to the client: `<VISOMASTER_DOWNLOADS>/<job_id>/<name>`, and exported recipes to
`<VISOMASTER_DOWNLOADS>/recipes/`. Downloading the same artifact again atomically replaces the
local copy.

### Retention

Nothing is removed automatically. `delete_job` removes one terminal job, and `prune_storage`
removes old files. Both refuse to remove anything that is an input of:

- a queued or running job;
- a failed or cancelled render that still keeps a checkpoint, and so can be resumed;
- any saved recipe revision.

`prune_storage` removes, after the same checks:

| Category | Removed when |
| --- | --- |
| `jobs` | Terminal and finished more than `older_than_days` ago (default 30) |
| `uploads` | Last written or re-uploaded more than `older_than_days` ago |
| `stale` | Staging (`prepare-*`) and trash (`trash-*`) directories left by crashes; partial uploads idle longer than both `older_than_days` and one hour |

Its report lists each entry with the bytes it frees, and `total_bytes`. A file hard-linked into
several removed entries is counted once. While any active job lacks `resolved.json` (submitted
by an older release), or a recipe revision cannot be read, nothing that could be an input is
removed.

Partial files of failed and cancelled jobs stay until the job is removed, for diagnosis. The
artifact tools serve only completed jobs.

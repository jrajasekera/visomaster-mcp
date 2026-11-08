# GPU scheduling

One GPU can run one job at a time. Several people, or several agents, may share it: someone
iterating on previews wants quick turnarounds, while a long render wants to finish eventually.
This page explains how the shared queue balances the two. The values and fields are listed in
[Jobs and scheduling](../reference/jobs.md#gpu-queue).

## Lock and queue

The GPU lock still provides mutual exclusion. A queue decides which waiting job may take it
next.

Each GPU's queue is `gpu-<index>.queue.json` beside `gpu-<index>.lock` in the lock directory,
guarded by `gpu-<index>.queue.lock`. Deployments that share a lock directory therefore share one
queue, one sequence counter and one decision.

## Classes and sequence numbers

Every GPU job gets a class when it is submitted: the request's optional `priority`, otherwise
`interactive` for detection, previews, reference analysis and still comparisons, and `normal`
for renders and clip comparisons. The mapping is `PRIORITIES` in `common.py`; a GPU kind missing
from it is `normal` (`DEFAULT_PRIORITY`). Normalization has no class and never queues.

Under the submit lock and then the scheduling lock, submission takes the next sequence number,
adds the job's queue entry and publishes the job directory in one step. The class and sequence
stay in `state.json`.

Some jobs hash their inputs before admission: a render, or a recipe job that pins its media's
SHA-256 (see [Checkpoints](video-processing.md#checkpoints)). Such a job takes its sequence the
same way, but its supervisor adds the queue entry, at that sequence, only after the hashing. The
job is not eligible meanwhile, and jobs that are ready may start first.

An exact retry returns the persisted job, so a retry or reconnect never moves a job in the
queue. Within a class, jobs start in sequence order (FIFO). Ties, which only a rebuilt queue can
produce, fall back to enqueue time and then to job path.

## The decision

Each waiting supervisor makes one scheduling decision every 0.2 seconds under the scheduling
lock:

1. It reads the queue.
2. It drops the entries of jobs that are terminal, missing or replaced under the same ID.
3. It skips the entries of jobs with a pending cancellation.
4. If the decision picks its own job, it takes the GPU lock without blocking and removes its
   entry before it releases the scheduling lock.

Choosing a job and admitting it are therefore one atomic step for every supervisor and client
sharing the lock directory. Status and listing take the scheduling lock too, so their probe of
the GPU lock cannot delay an admission.

The worker still inherits the GPU lock, so a supervisor crash after admission cannot release the
GPU while the job's processes run.

Supervisors from releases before this queue take the GPU lock directly. They still exclude other
jobs but can start ahead of the queue. That matters only when a deployment is forced while such
jobs are active.

## The fairness rule

The rule is evaluated at each decision.

- When only one class is waiting, its oldest job starts next.
- When both are waiting, the oldest interactive job starts next, unless waiting normal work is
  due.
- Normal work becomes due after `MAX_STREAK` (3) interactive admissions in a row while normal
  work waited, or once `AGE_LIMIT` (300) seconds have passed since the first of those
  admissions.
- Admitting a normal job resets the streak and the clock. So does admitting an interactive job
  while no normal job waits.

While normal work waits, at most three interactive jobs start between two normal admissions,
none of them more than five minutes after the first. After each normal admission, interactive
work goes first again, so interactive jobs never wait behind more than one normal admission in a
row.

## No preemption

Scheduling is non-preemptive. A decision happens only when the GPU lock is free, so a running
render keeps the GPU until it ends. Renders are checkpointed in chunks but do not yield at chunk
boundaries; yielding would be a separate change.

The queue never signals processes, and it cannot stop GPU work that does not take the lock, such
as Fusion's GUI.

## The queue view

`job_status` and `list_jobs` add a `queue` object to queued GPU jobs.

Its `position` is the job's place in the order the rule would produce if no job arrived or ended
and no time passed. Positions are advisory:

- new interactive jobs move normal jobs back;
- a streak or the clock moves them forward;
- jobs that end, fail or are cancelled move everything behind them forward.

`reason` says why the job waits. `starting` covers a job that is queued but has no entry: being
admitted, hashing its inputs before it joins (progress stage `fingerprinting`), or rejoining a
lost queue file. `interactive_streak` is the current streak and `waiting_seconds` the time since
submission.

No start time is estimated. A long poll ends when the position or reason changes, as it does for
status and progress stage changes. The queue object has a fixed size.

## Cancellation and recovery

Cancelling a queued job writes its marker under the submit lock. From then on the scheduler
skips the job's entry, and its supervisor exits without running it and removes the entry. A
cancellation that races with admission reaches a running job, which is stopped as usual.

Crash recovery reuses the lease rules. An entry whose supervisor died keeps its place during the
15-second startup grace period. After that, status reconciliation fails the job and the next
decision drops it. A job is admitted only by its own supervisor, which holds the job's lease, so
recovery cannot run a job twice.

If the queue file is lost or damaged, each live supervisor adds its job back at the sequence
recorded in its state, and new sequences continue above the ones that rejoined.

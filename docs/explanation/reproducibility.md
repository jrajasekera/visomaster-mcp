# Recipes, preflight and qualification

Three features answer three related questions:

- **Recipes:** can I run exactly this again later, and what changed if not?
- **Preflight:** will this request start, and if not, what do I fix?
- **Qualification:** has this combination of hardware and settings been tested on a real GPU?

All three rely on fingerprints of inputs, settings, code and models. The formats are listed in
[Recipes](../reference/results.md#recipes) and [Preflight checks](../reference/preflight-checks.md).

## Recipes

### What a revision stores

A recipe revision is one JSON document, `recipes/<name>/<revision>.json` in the state directory,
with a format name, a format version (1), and a `sha256` over its canonical JSON without that
field.

It stores the explicit request a new job runs, not the caller's partial overrides: resolved
paths, the range, every allowlisted control and per-mapping parameter at its effective value,
mappings with source weights, and the SHA-256 of the media, each target and each source.

Pinning the allowlisted settings means a later change of a default cannot alter them. Settings
outside the allowlist cannot be passed in a request, so their effective values are recorded in
the provenance instead, where a change shows up as a difference.

Readers check the version, the digest, the exact field set, and that every input is an absolute
path with a SHA-256. An edited, damaged or newer document is refused with a reason rather than
read with defaults. `priority` and `report` travel with the request. Priority is scheduling only
and is not part of the provenance.

### Provenance

The provenance is computed by `common.provenance`, a standard-library function. It takes a swap
request and returns what its output depends on, apart from the GPU, its driver and the CUDA
stack:

- the bytes of every input, keyed by its place in the request (`media_path`,
  `mappings[0].target_face`, `mappings[0].source_faces[1]`, ...);
- the effective settings;
- the Fusion checkout's revision;
- the adapter release, with the SHA-256 of its three files;
- the configuration identity;
- per model: its path, the hash Fusion's `models_data.py` declares, and the hash of the file.

The model list is read from `models_data.py` with `ast`, as layouts are, without importing
Fusion. The selected models come from `model_names`, which uses the same selection tables that
preflight checks (`model_choices` and `required_models` over `DETECTOR_MODELS`,
`SWAPPER_MODELS`, `RESTORER_MODELS` and the mask and expression models), so the two cannot
drift. They are:

- the detector and the recognizer;
- the swapper with its ArcFace models (CSCS adds its ID adapter);
- an enabled restorer, occluder, XSeg or expression restorer, in the base settings or in any
  segment's overrides. `setting_variants` merges segments as the engine's `Segments` does;
  preflight checks the same files and qualifies each segment's choices too.

Models that a setting does not enable are neither recorded nor hashed. A job's result lists the
models Fusion had loaded when it finished, and a recipe saved from that job adds them.

Revisions saved before restorer, mask, expression and CSCS adapter models were recorded keep
format version 1, and verify as before for the models they recorded. A model they lack, but that
their settings now select, is reported as a `model` change that was not recorded when the
revision was saved. Its bytes at that time are unknown, so `run_recipe` needs `accept_changes`
or a new revision.

Every call hashes every file in a streamed pass, and nothing caches hashes by file metadata, so a
file rewritten with the same size and modification time is still detected. The one exception is
a job's worker, which reuses the hashes its own supervisor took before admission for files whose
device, inode, size, and modification and change times are unchanged (see
[Checkpoints](video-processing.md#checkpoints)). A rewrite changes the change time, which cannot
be set back.

`common.provenance_changes` compares two records and returns readable entries (input, setting,
fusion, adapter, config, model). An input at a different path is a substitution, not a change.
Release names alone are not compared, because a new release with identical code runs the same
adapter. Checkpointed work reuses both functions: it records the provenance with the work and
compares before reusing it.

### When hashing happens

Hashing happens only at deliberate boundaries:

- saving a recipe;
- `get_recipe` with `verify`;
- `run_recipe`, before it submits;
- a recipe job: its supervisor hashes the media before the job joins the GPU queue, and its
  worker checks the media against `media_sha256` before Fusion loads, failing the job if it
  changed;
- the checkpoint boundaries: a render's supervisor before the job joins the queue, and
  `resume_render` before it submits.

Status polls and listings never hash. Submission itself still hashes only face references under
the submit lock, so a long video never holds the lock while it is hashed.

A recipe saved from a job takes the reference fingerprints the job recorded at submission, and
refuses to save if the files no longer have those bytes. Its media and model files are hashed
when the recipe is saved, because ordinary jobs do not hash them. Its Fusion revision must equal
the current one, because otherwise its effective defaults could not be reconstructed.

### Immutable revisions

Revisions are immutable. A writer computes the next number from the stored revisions and
tombstones (`<n>.deleted`, left by deletion so that a number is never reused). It writes a
complete temporary file, fsyncs it and hard-links it to `<n>.json`. The link fails rather than
replace an existing revision, so a writer that lost a race takes the next number, and no reader
can see partial JSON. Writes and deletions also hold the submit lock, which keeps them consistent
with retention decisions.

### Recipes and retention

Saving copies each face reference whose type has an upload name into the content-addressed
upload store, and points the request at the copy. The document's `retained` maps the copy to the
original path. The copy is published under the submit lock, so a concurrent prune either sees
the copy or removed an old one first.

Deletion and pruning treat every input named by any stored revision like an input of a queued or
running job, or of a resumable render. A referenced upload is kept, and a job whose artifact is
referenced (such as normalized media) is not deleted. A revision that cannot be parsed makes the
protected set unknown, so nothing that could be an input is removed until that revision is
deleted.

Deleting a recipe removes only its revision files. Media in the media root is never removed by
retention and is not copied. It is reported as unretained, and its change or removal is detected
when the recipe is verified or run.

### Running a recipe

Instantiation applies explicit substitutions to the stored request, and records each with the
recorded and the used value.

- Media at another path with the same bytes needs nothing else.
- Media with different bytes is a different clip. The caller must give targets for every mapping
  and a range, and a report when the recipe's report has expected-visible ranges, because a
  target or range from another clip is not assumed to be valid.
- Missing inputs always fail, with what to restore.
- Other differences fail with the readable list, unless the caller accepts them. Accepted
  differences are recorded in the job.

The job's `request.json`, `resolved.json` and `result.json` carry `recipe`: the parent's name,
revision and document digest, the invocation, the substitutions and the accepted changes. A
retry with the same job ID is answered from the persisted request's recorded invocation before
anything is hashed, so it works after the inputs are gone.

### Export and import

Export writes the document unchanged to the client's downloads directory. It never contains
media or model weights.

Import checks the document as a reader does, then requires every input to exist inside the
allowed roots with its recorded bytes. An input missing at its recorded path is looked up by
digest in the upload store, where `upload_media` stores identical bytes, and the relocation is
recorded in the new revision's origin. The recorded provenance is kept, so running an imported
recipe on another installation reports its version and model differences.

### What a recipe does not promise

Recipes do not claim bit-for-bit reproduction. The GPU, its driver and the CUDA libraries are not
recorded, and GPU kernels need not be deterministic. Outputs are expected to be equivalent, not
identical.

## Preflight

A preflight answers whether one specific swap request could start now, so its input is the
request itself.

The client builds the request with the function that `submit_swap` uses, from the same
parameters and pydantic models; a protocol test compares the two input schemas. The workstation
runs the same `validate_request` that submission runs, as its `request` check.

The other checks break that verdict into actionable parts, and add what validation does not see:
Fusion compatibility, deployed code, model files, the media checks a worker makes, storage, the
GPU and qualification evidence. Refusals that preflight reports carry a stable code through
`common.Refusal`, a `ValueError` subclass. Validation and the engine keep their messages and
exception types, and preflight does not parse messages.

### Cheap checks

Cheap checks run inside the short-lived control command with the standard library, FFprobe, git
and nvidia-smi. The media check imports `engine`, whose module level uses only the standard
library.

- For a render, it calls `clip_range`, the function that renders and comparison clips call
  before decoding, and then `common.segment_plan` with the frame count, as the render's worker
  does before Fusion loads. Segments the video lacks are therefore blocked rather than reported
  ready.
- For a preview, it calls `preview_source`. That repeats the checks a preview's `load_frame` and
  decoder make (an image has only frame 0, the frame exists on the packet timeline, and `color`
  accepts the stream) with the same helpers, though it recognizes an image by its demuxer where
  `load_frame` tries OpenCV first.

Both read timestamps only through the requested frames.

Model files come from `common.model_catalog` (Fusion's `models_data.models_list`, the catalog
recipes use), and from tables in `common.py` that map each selection (detector, swap model with
its ArcFace models, restorer, occluder, XSeg, expression restoration) to the models it loads at
the pinned revision. A selection missing from the tables is reported as not checked rather than
guessed. Recipe provenance hashes the files of the same selections (`model_names`).

### Why preflight does not hash

Preflight checks that model files are present and their sizes, not their hashes. Optimized
installations replace the published files, and hashing every model would make a cheap check
expensive.

For the same reason the preflight fingerprint is not `common.provenance`, which hashes the media
and every model file to decide whether recorded work can be reused. The fingerprint identifies:

- the media by path, size and modification time;
- references by their SHA-256, already computed for the reference check;
- the configuration by settings identity, revision, adapter code hashes (`code_identity`) and
  model file sizes and times.

Storage and GPU memory are observations against an estimate. Submission neither consults a
preflight nor treats it as a reservation, so submission and the worker check every mutable
condition again.

### The optional GPU stage

The optional reference-analysis stage adds no job kind. It queues ordinary `analyze` jobs
through `submit`, so they get the submit lock, validation, `resolved.json`, retention
protection, the interactive class, the shared GPU queue and lock, supervisor leases,
cancellation and reconnect like any job.

Their IDs are `preflight-` followed by a hash of the analysis request, which includes the
references' SHA-256 values. Repeating a preflight is an exact retry that returns the existing
job, and changed bytes, controls or swap model produce a new job. Only sources with a positive
weight are analyzed, one job per distinct set. The stage runs only when Fusion and the request
pass, so a blocked request never occupies the GPU.

## Qualification

Qualification is data: `QUALIFICATIONS` in `common.py`. Each record names the hardware class
(never a model name), the Fusion revision, the exercised choices and counts, the date and the
`VERIFICATION.md` section. Matching uses the observed compute capability and VRAM class.

An allowed combination that no record covers is `unqualified`, which is a warning. It means the
evidence is incomplete, not that the request is wrong. Qualification is evidence, not
permission: records are added only after a GPU qualification run (see
[Record qualification evidence](../how-to/test-and-qualify.md#record-qualification-evidence)).

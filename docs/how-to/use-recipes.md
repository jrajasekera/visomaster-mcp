# Save and reuse settings with recipes

A recipe is a named, reusable swap configuration. It pins every setting, the input
fingerprints and the versions it ran with, so you can run it again later, on the same clip or a
new one, and see exactly what changed in between.

## Save a recipe

Save from a completed job, or from an explicit request.

- **From a job:** call `save_recipe` with a `name` and the `job_id` of a completed preview or
  render.

  ```json
  {"name": "person1-closeups", "job_id": "shot01-preview-v1", "note": "approved look"}
  ```

- **From a request:** call `save_recipe` with a `name` and the arguments you would give
  `submit_swap` (`media_path`, `mappings`, `mode` and its range, `controls`, and optionally
  `report` and `segments`).

When saving from a request, pass `priority` to store a queue class for the jobs the recipe
runs. A recipe saved from a job keeps that job's explicit priority, if it had one.

Saving hashes the media, the face images and the model files on the workstation, so a long
video takes a while (the call allows 15 minutes). Saving again under the same name adds a new
revision; earlier revisions never change.

Face images are copied into the upload store, so deleting the detection job that produced a
target crop does not break the recipe.

## Check a recipe

- `get_recipe` with no arguments lists recipes and their revisions.
- `get_recipe` with a `name` (and optionally `revision`) returns the document and its
  `retention`: whether each input is retained or unretained, with `, missing` added when it no
  longer exists.
- `get_recipe` with `verify: true` hashes the inputs and models now and lists every change
  since the recipe was saved, without running anything.

## Run a recipe

1. Call `run_recipe` with the `name`, an optional `revision` (default: the latest) and a
   `job_id`:

   ```json
   {"name": "person1-closeups", "job_id": "shot02-preview-v1"}
   ```

2. If the call fails with a list of changes, decide whether they are acceptable. Pass
   `accept_changes: true` to run anyway; the accepted changes are recorded in the job. Missing
   inputs always fail, with the list of inputs to restore.

3. Follow the job with `job_status` as usual.

### Run on another clip or range

Substitutions are explicit:

| Argument | Substitutes |
| --- | --- |
| `media_path` | The media |
| `targets` | One `target_face` per recipe mapping, in order |
| `frame_range` | `frame`, or `start_frame` and `max_frames` |
| `mode` | `preview` or `render` |
| `report` | Report options |
| `priority` | Queue class |

- Media at another path with the same bytes needs nothing else.
- Media with different content is a new clip. It needs `targets` from a detection on that clip,
  a `frame_range`, and a `report` if the recipe has `expected_visible` ranges.
- A mode change needs a `frame_range`. A preview drops the recipe's `report` and `segments`.
- A recipe with `segments` cannot run on media with different content, because its segments
  name frames of the recorded clip. Save a new recipe for the new clip.

## Move a recipe to another installation

1. Call `export_recipe` with the `name` (and optionally `revision`). It writes
   `<downloads>/recipes/<name>-r<revision>.json` on the client and returns the path. The file
   holds configuration and provenance, never media or model weights.
2. On the other installation, upload each input with `upload_media` if it is not already at its
   recorded path.
3. Call `import_recipe` with the file's `local_path` (at most 4 MiB), and optionally a new
   `name`. Inputs missing at their recorded paths are found by digest in the upload store.

Running an imported recipe reports the version and model differences from the original
installation.

## Delete a recipe

Call `delete_recipe` with the `name`, and a `revision` to delete only that one. A deleted
revision's number is never reused. Deleting a recipe deletes no inputs; `prune_storage` may
remove them later once nothing else uses them.

The document format is listed in [Recipes](../reference/results.md#recipes).
[Recipes, preflight and qualification](../explanation/reproducibility.md#recipes) explains the
design.

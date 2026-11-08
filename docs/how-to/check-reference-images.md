# Check and weight source images

When you have several photos of the replacement person, the swap averages their face
embeddings. One bad photo (blurred, tiny, a different person) pulls the average away. This guide
checks the photos before a swap, then pins and weights them in the request.

## Analyze the candidates

1. Call `analyze_references` with up to 20 absolute image paths:

   ```json
   {
     "references": [
       "/srv/visomaster/media/ref-a.png",
       "/srv/visomaster/media/ref-b.jpg",
       "/srv/visomaster/media/ref-c.png"
     ],
     "job_id": "refs-person1-v1"
   }
   ```

   If you will swap with a model other than Fusion's default, pass it as `swap_model`, so the
   analysis compares the same embedding the swap uses. Pass the same detector `controls` you
   will give `submit_swap`.

2. Follow the job with `job_status` until it completes.

3. View `contact.jpg`. Every image is labelled with its ID (`r0`, `r1`, … in request order) and
   its first finding; `+` means it has more. Unusable images are framed in red.

4. Read `result.references`. For each image check `usable` and its `findings`.

   - **Errors** mean `submit_swap` would refuse the image: `unreadable`, `no_face`,
     `multiple_faces`, `unrecognized`. Replace or drop it.
   - **Warnings** mark images to review: `small_face`, `blurry`, `exact_duplicate`,
     `near_duplicate`, `outlier`, `inconsistent_pair`.

   The full list of fields and codes is in [Reference analysis](../reference/results.md#reference-analysis).

Analysis never removes, edits or reorders images. Deciding what to drop is up to you.

## Tune the thresholds

The default thresholds come from a small sample of face crops. Sharp crops measured about
90–850 and the same crops blurred about 10–90. If the findings do not match what you see on the
contact sheet, pass `thresholds` with your own values:

```json
{"thresholds": {"min_face_pixels": 96, "min_sharpness": 25}}
```

| Threshold | Default | Flags |
| --- | --- | --- |
| `min_face_pixels` | 112 | `small_face` |
| `min_sharpness` | 40 | `blurry` |
| `duplicate_similarity` | 95 | `near_duplicate` |
| `outlier_similarity` | 60 | `outlier`, `inconsistent_pair` |

## Use the results in the swap

In each mapping:

- Set `source_sha256` to the `sha256` values the analysis reported, in `source_faces` order.
  The submission then fails if any file changed since the analysis. Use `null` for an image you
  do not want to pin.
- Set `source_weights` to exclude (`0`) or down-weight images the analysis flagged. Only the
  ratios matter: weight 2 counts like two copies of the image.

```json
{
  "target_face": "/srv/visomaster/runtime/mcp/jobs/detection01/artifacts/face-0.png",
  "source_faces": [
    "/srv/visomaster/media/ref-a.png",
    "/srv/visomaster/media/ref-b.jpg",
    "/srv/visomaster/media/ref-c.png"
  ],
  "source_weights": [1, 1, 0],
  "source_sha256": ["<sha256 of r0>", "<sha256 of r1>", null]
}
```

An image with weight 0 is never read: only its path's form is checked, it need not exist, and
its `source_sha256` entry is not checked.

The swap does not reuse the analysis. It computes embeddings again from the pinned bytes with
its own swap model, so run the analysis again after you change the images or
`SwapModelSelection`.

## Analyze as part of a preflight

`preflight_swap` with `analyze_references: true` queues the same analysis for each mapping's
weighted sources and reports the outcome under the `reference_analysis` check. Repeat the
preflight to collect the results; it finds the same analysis job instead of queuing another.
See [Preflight checks](../reference/preflight-checks.md#reference-analysis-from-a-preflight).

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

If the findings do not match what you see on the contact sheet, pass `thresholds` with your own
values. `result.thresholds` reports the values applied.

```json
{"thresholds": {"min_face_pixels": 96, "min_sharpness": 25}}
```

| Threshold | Default | Flags |
| --- | --- | --- |
| `min_face_pixels` | 112 | `small_face` |
| `min_sharpness` | 40 | `blurry` |
| `duplicate_similarity` | 95; 96 for `SimSwapArcFace`, 95.5 for `GhostArcFace`, 98 for `CSCSArcFace` | `near_duplicate` |
| `outlier_similarity` | 70; 85 for the `CSCSArcFace` recognizer (swap model `CSCS`) | `outlier`, `inconsistent_pair` |

The defaults were calibrated on synthetic people: studio photos, face crops from generated
video, and blurred, resized and relit copies. Expect real photos to differ somewhat.

### Outlier scores

With `Inswapper128ArcFace`, the recognizer of Fusion's default swap model:

- Clear photos of one person scored 85–95 against the mean of the others. Video crops of one
  person scored 68–93; the lowest were faces of about 100 pixels and children.
- A different adult scored at most 66 against another person's references, and usually 50–61.
  Photos of other people scored 55–66 against a person's photos, so the former default of 60
  missed two in three of them.
- Children are not separated: a child scored up to 77 against crops of another child, while
  crops of one child scored as low as 68. Compare children's crops by eye.

The `SimSwapArcFace` and `GhostArcFace` recognizers scored on the same scale. `CSCSArcFace`
scores everyone higher: different people scored 64–83 against one person's photos, the same
person 86–98, so its default is 85.

In a small collection an intruder also pulls the others' scores down, because their mean
includes it. When several references are flagged, the lowest score is the likeliest intruder;
it was the lowest in over 99% of the measured sets. Adding more clear photos of the person makes
the mean, and the scores, steadier.

### Duplicate scores

A near duplicate is a copy of an earlier reference that adds weight but no new view. Re-encoded,
resized or re-lit copies scored above distinct photos of one person, so each recognizer's default
sits just above the highest distinct photo it saw:

| Recognizer (swap model) | Distinct photos of one person | Re-encoded, resized or re-lit copies | Default |
| --- | --- | --- | --- |
| `Inswapper128ArcFace` (`Inswapper128`) | 82.0–94.4 | 95.9–100 | 95 |
| `SimSwapArcFace` (`SimSwap512`) | 84.1–95.5 | 95.7–100 | 96 |
| `GhostArcFace` (`GhostFace-v*`) | 83.5–94.9 | 95.4–100 | 95.5 |
| `CSCSArcFace` (`CSCS`) | 91.2–97.2 | 98.8–100, except resized small faces | 98 |

The lowest copies were JPEG quality 40; only that copy of a 94-pixel video face fell below
the SimSwap and Ghost defaults. Photos edited from one source photo that keep its pose and expression
scored up to 95.4–97.5 against it, so some of those are flagged. Copies that shift, tighten or
rotate the crop move the face alignment and scored lower, down to 91, so a few are not flagged.
Mirror images scored 87–98, like distinct photos of a similar view; 10 of 24 were flagged.
Video frames a few frames apart scored 97–100 and are flagged. With the first three
recognizers, frames seconds apart scored 68–90.

`CSCSArcFace` is unreliable on small faces. Two different people with faces of about 90 pixels
scored 98.9, so frames of one small face are flagged whatever they show, and copies of a
203-pixel face resized to 102–181 pixels scored only 67–87 against it. Compare small CSCS
references by eye, or analyze them with another swap model.

To flag only nearly untouched copies, raise `duplicate_similarity` to about 99. To catch more
cropped copies, lower it towards the distinct-photo maximum and expect similar views of the
person to be flagged too.

### Sharpness

Sharpness measures the detail left at the 112-pixel scale the recognizer works at. Unaltered
photos measured 329–555 and unaltered video crops (75–282 pixel faces) 74–382.

Blur that is visible at full size can leave the recognizer's view intact. A 576-pixel face
blurred with a Gaussian sigma of 6 measured 63 and still scored 96.5 against the original,
like a re-saved copy. Copies kept a score of 95 or more against the original down to a
sharpness of about 16 (sigma 12 at that size); at about 5 they scored 90, the score of another
photo of the same person. The default of 40 lies below every unaltered face measured and flags
faces that have lost detail at recognition scale.

No threshold separates a softly blurred large photo from a clear small face: the sigma-6 photo
(63) scored lower than a clear 95-pixel video crop (74). To flag visibly soft full-size photos
too, raise `min_sharpness` to about 70 and expect small faces near it to be flagged as well.

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

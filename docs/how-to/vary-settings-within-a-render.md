# Change settings within one render

A render can treat different parts of a video differently while keeping one continuous output:
a stricter match threshold for a profile shot, only one of two people in a two-shot, or title
cards left untouched. This guide does that with `segments`.

## Define the segments

Add `segments` to a render request (`mode: "render"`). Each segment covers the zero-based source
frames `[start_frame, end_frame)`: `end_frame` is the first frame after it. Give each segment a
stable `id` and exactly one kind of intent:

- `"passthrough": true` leaves the frames exactly as decoded, without detection or swapping; or
- one or more changes:
  - `mappings`: indices into `mappings` that swap here. The others are not requested in these
    frames.
  - `parameters`: Fusion parameters applied over every active mapping's own.
  - `controls`: controls applied over the request's.

```json
{
  "mode": "render",
  "start_frame": 0,
  "max_frames": 300,
  "segments": [
    {"id": "profile", "start_frame": 0, "end_frame": 120,
     "parameters": {"SimilarityThresholdSlider": 75}},
    {"id": "titles", "start_frame": 120, "end_frame": 150, "passthrough": true},
    {"id": "two-shot", "start_frame": 150, "end_frame": 240, "mappings": [1]}
  ]
}
```

Frames outside every segment use the base mappings, parameters and controls.

Rules:

- List at most 100 segments, in ascending order, without overlaps.
- A segment cannot set `SwapModelSelection`, `DetectorModelSelection`,
  `FaceTrackingEnableToggle` or `KPSSmoothingEnableToggle`. These hold for the whole render.
- A segment that extends past the rendered range is clipped to it. A segment with no rendered
  frame is refused, and so is a render whose rendered frames are all passthrough.

The full validation rules are in [Segments](../reference/tools.md#segments). Previews and
`compare_variants` do not accept segments.

## Check and run the render

1. Call `preflight_swap` with the request. When `max_frames` is `null` or runs past the end of
   the video, only the video itself shows which segments have frames, and preflight reports
   problems as `segment_outside_range` or `all_passthrough`.
2. Submit the render with `submit_swap`.

## Read the outcome

- The result's `plan.runs` lists every run (segment or gap) with its `frames`, `swaps` and
  `swapped_frames`.
- The `plan.json` artifact records each segment as given, the frames it `applied` to, whether
  it was `clipped`, and the effective settings of every run.
- A segment that requested swaps and swapped none gives the warning `segment_never_swapped`.
- The render fails if no face swaps in any frame outside passthrough segments.

Passthrough frames count as `not_requested` in the swap report, so they never lower coverage.
Tracking and landmark smoothing restart at every segment boundary and at the edges of gaps.

"""Segmented renders round-trip through recipes (vmcp-harden-pqj.6 with pqj.5)."""

from pathlib import Path

import common
import control
import pytest
import test_recipes
from test_recipes import job_request, run, saved

make_site, site, library = test_recipes.make_site, test_recipes.site, test_recipes.library


def segment(name, low, high, **changes):
    return {"id": name, "start_frame": low, "end_frame": high, **changes}


SEGMENTS = [
    segment("close", 1, 3, parameters={"SimilarityThresholdSlider": 75}),
    segment("title", 3, 4, passthrough=True),
]


def test_segmented_requests_round_trip_through_recipes(site):
    document = saved(site, kind="render", start_frame=0, max_frames=6, segments=SEGMENTS)
    assert document["request"]["segments"] == SEGMENTS
    assert control.get_recipe({"name": "shot", "verify": True})["verification"]["unchanged"]
    run(job_id="again")
    assert job_request("again")["segments"] == SEGMENTS
    # Export and import keep them, and the imported revision runs alike.
    imported = control.import_recipe({"document": document, "name": "copy"})["recipe"]
    assert imported["request"]["segments"] == SEGMENTS
    # A preview drops them, recorded as a substitution.
    run(job_id="still", mode="preview", range={"frame": 2})
    request = job_request("still")
    assert "segments" not in request
    assert request["recipe"]["substitutions"]["segments"] == {"recorded": SEGMENTS, "used": None}
    # A range that misses a segment is refused like any request.
    with pytest.raises(ValueError, match=r"close \[1, 3\) ends before the rendered frames"):
        run(job_id="late", range={"start_frame": 4, "max_frames": 2})


def test_a_new_clip_cannot_reuse_segments_of_the_recipes_clip(site):
    saved(site, kind="render", start_frame=0, max_frames=6, segments=SEGMENTS)
    other = Path(site.media) / "other.mp4"
    test_recipes.test_submit.ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=0.3", "-c:v", "libx264", other
    )
    with pytest.raises(ValueError, match="segments name frames of that clip"):
        run(
            media_path=str(other),
            targets=[site.source],
            range={"start_frame": 0, "max_frames": 4},
        )


def test_recipe_from_a_segmented_render_job_keeps_its_segments(site):
    request = test_recipes.preview(site, job_id="cut", kind="render", segments=SEGMENTS)
    for key in ("frame",):
        request.pop(key)
    control.submit(request)
    test_recipes.complete("cut")
    common.atomic_json(common.job_path("cut") / "result.json", {"revision": None, "models": []})
    document = control.save_recipe({"name": "from-cut", "job_id": "cut"})["recipe"]
    assert document["request"]["segments"] == SEGMENTS

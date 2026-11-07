"""Frames are selected on the video stream's own timeline, whatever the container or audio says."""

import os
import subprocess
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace

import common
import engine
import pytest
from conftest import apply_config, write_config

IMAGE = SimpleNamespace(shape=(16, 32, 3))
# Solid frames whose luma encodes the source frame number, so every frame differs.
NUMBERED = "format=yuv420p,geq=lum='16+8*N':cb=128:cr=128"
TONE = (0.64, 0.72)  # A 1 kHz burst during frames 16 and 17 of the 25 fps clips.


def ffmpeg(*arguments):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *map(str, arguments)], check=True)


def numbered(output, rate, frames, *options):
    source = f"color=black:s=32x16:r={rate},{NUMBERED}"
    ffmpeg("-f", "lavfi", "-i", source, "-frames:v", frames, *options, "-c:v", "libx264", output)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("timing")
    work = root / "work"
    work.mkdir()
    numbered(work / "video.mkv", 25, 25, "-bf", "2")
    start, end = (t + 0.423 for t in TONE)
    burst = f"aevalsrc='if(between(t,{start},{end}),0.5*sin(6283.2*t),0)':s=48000:d=2"
    ffmpeg("-f", "lavfi", "-i", burst, work / "audio.wav")
    # Video starts at PTS 0 and audio at -0.423 s, so the container starts before the video.
    ffmpeg(
        *("-i", work / "video.mkv", "-itsoffset", "-0.423", "-i", work / "audio.wav"),
        *("-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "pcm_s16le"),
        *("-avoid_negative_ts", "disabled", root / "early.mkv"),
    )
    plain = ("-i", work / "video.mkv", "-f", "lavfi", "-i", "sine=d=1", "-c:v", "copy")
    ffmpeg(*plain, root / "plain.mp4")
    numbered(root / "ntsc.mp4", "30000/1001", 20, "-bf", "2")
    # Dropping source frames 3-12 while keeping timestamps leaves 15 unevenly spaced frames.
    drop = ["-vf", "select='not(between(n,3,12))'", "-fps_mode", "passthrough"]
    ffmpeg("-i", work / "video.mkv", *drop, "-c:v", "libx264", root / "vfr.mp4")
    # One second of video with three of audio: no video duration, container duration from audio.
    long = ("-i", work / "video.mkv", "-f", "lavfi", "-i", "sine=d=3", "-c:v", "copy")
    ffmpeg(*long, "-c:a", "pcm_s16le", root / "long.mkv")
    (root / "face.png").write_bytes(b"png")
    return root


@pytest.fixture
def media(library, tmp_path, monkeypatch):
    apply_config(monkeypatch, write_config(tmp_path / "remote-config.json", library))

    class Frame(bytes):
        """Raw BGR bytes standing in for a numpy frame (numpy is a workstation dependency)."""

        def reshape(self, *shape):
            frame = Frame(self)
            frame.shape = shape
            return frame

        def tobytes(self):
            return bytes(self)

    numpy = SimpleNamespace(uint8="uint8", frombuffer=lambda data, dtype: Frame(data))
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    cv2 = SimpleNamespace(imread=lambda path: IMAGE if path.endswith(".png") else None)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return library


def stream_of(path):
    return engine.video_stream(common.probe(path))


def sequential(path, distinct=True):
    """Every frame in decoded order, read without seeking."""
    frames = list(engine.decode(path, stream_of(path), 0, 10**6))
    assert len(set(frames)) == len(frames) or not distinct
    return frames


def run_job(directory, monkeypatch, request):
    """Run engine.run with Fusion replaced by recorders; returns the frames seen and the result."""
    seen = []
    monkeypatch.setenv("VM_SUPERVISOR_PID", str(os.getppid()))
    monkeypatch.setenv("VM_CONFIG_SHA256", "identity")
    monkeypatch.setattr(engine, "SETTINGS", SimpleNamespace(sha256="identity"))
    monkeypatch.setattr(engine, "CONFIG_ERROR", None)
    prctl = SimpleNamespace(prctl=lambda *a: 0)
    monkeypatch.setattr(engine, "ctypes", SimpleNamespace(CDLL=lambda name: prctl))
    control = {"RecognitionModelSelection": "model", "FaceTrackingEnableToggle": False}
    window = SimpleNamespace(control={**control, "KPSSmoothingEnableToggle": False})
    monkeypatch.setattr(engine, "initialize", lambda controls: window)
    monkeypatch.setattr(
        engine, "layouts", lambda: ({}, {"SimilarityThresholdSlider": {"default": 60}})
    )
    detect = lambda window, image, max_faces=None: seen.append(image) or (None, [], [])  # noqa: E731
    monkeypatch.setattr(engine, "detect", detect)
    monkeypatch.setattr(engine, "configure_faces", lambda window, mappings: None)
    monkeypatch.setattr(
        engine, "make_worker", lambda window, sequential: SimpleNamespace(swap_count=1)
    )
    monkeypatch.setattr(engine, "process", lambda worker, bgr, index: seen.append(bgr) or bgr)
    monkeypatch.setattr(engine, "save_image", lambda path, image: None)
    (directory / "artifacts").mkdir()
    common.atomic_json(directory / "request.json", request)
    engine.run(directory)
    return seen, common.read_json(directory / "result.json")


def render(source, directory, monkeypatch, start, count):
    """Render with a pass-through Fusion stub through the real decoder, encoder and audio mux."""
    processed = []

    def process(worker, bgr, index):
        processed.append((index, bgr))
        return bgr

    monkeypatch.setattr(engine, "process", process)
    monkeypatch.setattr(engine, "save_image", lambda path, image: None)
    (directory / "artifacts").mkdir(parents=True)
    request = {"media_path": str(source), "start_frame": start, "max_frames": count}
    result = engine.render_video(None, request, directory, SimpleNamespace(swap_count=1))
    return processed, result


def audible(path):
    """First and last second at which the output's audio is loud, or None if it is silent."""
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0"]
    pcm = subprocess.run(
        command + ["-ac", "1", "-ar", "48000", "-f", "s16le", "-"], check=True, capture_output=True
    )
    loud = [i for i, value in enumerate(array("h", pcm.stdout)) if abs(value) > 3000]
    return (loud[0] / 48000, loud[-1] / 48000) if loud else None


def test_fixture_container_starts_before_its_video(media):
    metadata = common.probe(media / "early.mkv")
    video, audio = metadata["streams"]
    assert float(video["start_time"]) == 0 and float(audio["start_time"]) == -0.423
    assert float(metadata["format"]["start_time"]) == -0.423


@pytest.mark.parametrize(
    ("name", "total"), [("early.mkv", 25), ("plain.mp4", 25), ("ntsc.mp4", 20)]
)
def test_frames_match_a_sequential_decode(media, name, total):
    path = media / name
    frames = sequential(path)
    assert len(frames) == total
    for number in (0, 1, 5, 14, total - 1):
        assert engine.load_frame(path, number) == frames[number]
        # Renders seek on the validated CFR timeline without a scan.
        assert list(engine.decode(path, stream_of(path), number, 2)) == frames[number : number + 2]
    with pytest.raises(ValueError, match=f"Frame {total} is out of range; the video has {total}"):
        engine.load_frame(path, total)


@pytest.mark.parametrize(("start", "count"), [(14, 5), (0, 20)])
def test_render_selects_frames_and_audio_on_the_video_clock(
    media, tmp_path, monkeypatch, start, count
):
    frames = sequential(media / "early.mkv")
    processed, result = render(media / "early.mkv", tmp_path, monkeypatch, start, count)
    assert [index for index, _ in processed] == list(range(start, start + count))
    assert [frame for _, frame in processed] == frames[start : start + count]
    assert result["frames"] == count and result["audio"] == "AAC"
    span = audible(tmp_path / "artifacts/result.mp4")
    assert span is not None, "the source audio interval was not selected"
    first, last = span
    assert abs(first - (TONE[0] - start / 25)) < 0.025
    assert abs(last - (TONE[1] - start / 25)) < 0.025


def test_fractional_rate_render_without_audio_keeps_its_range(media, tmp_path, monkeypatch):
    frames = sequential(media / "ntsc.mp4")
    processed, result = render(media / "ntsc.mp4", tmp_path, monkeypatch, 7, 5)
    assert [frame for _, frame in processed] == frames[7:12]
    assert (result["frames"], result["fps"], result["audio"]) == (5, "30000/1001", "none")
    with pytest.raises(ValueError, match="empty"):
        render(media / "ntsc.mp4", tmp_path / "late", monkeypatch, 20, 1)


def test_variable_rate_frames_select_decoded_ordinals(media):
    path = media / "vfr.mp4"
    frames = sequential(path)
    assert len(frames) == 15
    with pytest.raises(ValueError, match="Variable"):
        engine.validate_timing(path, engine.Fraction(25), stream_of(path))
    for number in range(15):
        assert engine.load_frame(path, number) == frames[number]
    with pytest.raises(ValueError, match="Frame 15 is out of range; the video has 15 frames"):
        engine.load_frame(path, 15)


def vfr_request(media, fields):
    request = {"media_path": str(media / "vfr.mp4"), **fields}
    if request["kind"] == "preview":
        face = str(media / "face.png")
        request["mappings"] = [{"target_face": face, "source_faces": [face]}]
    return request


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"kind": "detect", "frame": 8}, [8]),
        ({"kind": "detect", "frames": [8, 2, 5]}, [2, 5, 8]),
        ({"kind": "detect", "sample_frames": 5}, [1, 4, 7, 10, 13]),
        ({"kind": "preview", "frame": 8}, [8]),
    ],
)
def test_detection_and_preview_follow_variable_rate_ordinals(
    media, tmp_path, monkeypatch, fields, expected
):
    frames = sequential(media / "vfr.mp4")
    seen, result = run_job(tmp_path, monkeypatch, vfr_request(media, fields))
    assert seen == [frames[number] for number in expected]
    assert result.get("frames", expected) == expected


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "detect", "frame": 15},
        {"kind": "detect", "frames": [2, 15]},
        {"kind": "preview", "frame": 15},
    ],
)
def test_detection_and_preview_reject_the_frame_after_the_last(
    media, tmp_path, monkeypatch, fields
):
    with pytest.raises(ValueError, match="Frame 15 is out of range; the video has 15 frames"):
        run_job(tmp_path, monkeypatch, vfr_request(media, fields))


def test_normalized_variable_rate_video_indexes_consistently(media, tmp_path):
    (tmp_path / "artifacts").mkdir()
    request = {"media_path": str(media / "vfr.mp4"), "fps": 25}
    result = engine.normalize_video(request, tmp_path)
    output = Path(result["media_path"])
    stream = stream_of(output)
    assert engine.validate_timing(output, engine.Fraction(25), stream) == result["frames"] == 25
    # Resampling repeats frame 2 across the dropped span; frame 13 is the next new one.
    frames = sequential(output, distinct=False)
    assert len(set(frames)) == 15 and frames[12] == frames[2] != frames[13]
    for number in (1, 2, 3, 12, 13, 14, 24):
        assert engine.load_frame(output, number) == frames[number]
        assert list(engine.decode(output, stream, number, 1)) == [frames[number]]


def test_still_images_keep_frame_zero(media, tmp_path, monkeypatch):
    assert engine.load_frame(media / "face.png") is IMAGE
    with pytest.raises(ValueError, match="only have frame 0"):
        engine.load_frame(media / "face.png", 1)
    request = {"kind": "detect", "media_path": str(media / "face.png"), "sample_frames": 4}
    seen, result = run_job(tmp_path, monkeypatch, request)
    assert seen == [IMAGE] and result["frames"] == [0]


def test_sampling_spans_the_video_stream_not_the_longer_audio(media, tmp_path, monkeypatch):
    metadata = common.probe(media / "long.mkv")
    video = engine.video_stream(metadata)
    assert "nb_frames" not in video and "duration" not in video
    assert abs(float(metadata["format"]["duration"]) - 3) < 0.05
    frames = sequential(media / "long.mkv")
    request = {"kind": "detect", "media_path": str(media / "long.mkv"), "sample_frames": 5}
    seen, result = run_job(tmp_path, monkeypatch, request)
    assert result["frames"] == [2, 7, 12, 17, 22]
    assert seen == [frames[number] for number in result["frames"]]


def strip_durations(metadata):
    """Drop every duration and frame count, as some readable sources report none."""
    metadata["format"].pop("duration", None)
    for stream in metadata["streams"]:
        stream.pop("duration", None)
        stream.pop("nb_frames", None)
    return metadata


def normalize(source, directory, monkeypatch):
    progress, write = [], engine.atomic_json

    def record(path, value):
        if path.name == "progress.json":
            progress.append(value)
        write(path, value)

    monkeypatch.setattr(engine, "atomic_json", record)
    (directory / "artifacts").mkdir()
    return engine.normalize_video({"media_path": str(source)}, directory), progress


@pytest.mark.parametrize("case", ["exact metadata", "no video duration", "no container duration"])
def test_normalization_validates_the_video_extent(media, tmp_path, monkeypatch, case):
    source = media / ("plain.mp4" if case == "exact metadata" else "long.mkv")
    if case == "exact metadata":
        assert engine.video_stream(common.probe(source))["nb_frames"] == "25"
    real, stripped = engine.probe, []
    if case == "no container duration":

        def strip(path):
            metadata = real(path)
            if Path(path).name != "long.mkv":
                return metadata
            stripped.append(path)
            return strip_durations(metadata)

        monkeypatch.setattr(engine, "probe", strip)
    result, progress = normalize(source, tmp_path, monkeypatch)
    assert result["frames"] == 25 and abs(result["duration"] - 1) < 0.001
    assert progress and all(p["estimated_total_frames"] == 25 for p in progress)
    assert not any("total_frames" in p for p in progress)
    assert bool(stripped) == (case == "no container duration")


def test_unknown_extent_completes_on_output_validation(media, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "video_extent", lambda source, stream: None)
    result, progress = normalize(media / "long.mkv", tmp_path, monkeypatch)
    assert result["frames"] == 25
    assert progress and all(p["estimated_total_frames"] is None for p in progress)
    assert all(p["eta_seconds"] is None for p in progress)


# Depending on the version, FFmpeg fails an encode without video frames or writes no video.
EMPTY = "normalization failed|no video frames"


@pytest.mark.parametrize(("frames", "error"), [(10, "expected about 25"), (0, EMPTY)])
def test_truncated_or_empty_normalization_fails(media, tmp_path, monkeypatch, frames, error):
    command = engine.normalize_command

    def truncated(*args):
        arguments = command(*args)
        return arguments[:-1] + ["-frames:v", str(frames), arguments[-1]]

    monkeypatch.setattr(engine, "normalize_command", truncated)
    with pytest.raises((RuntimeError, ValueError), match=error):
        normalize(media / "long.mkv", tmp_path, monkeypatch)
    assert not (tmp_path / "artifacts/normalized.mp4").exists()

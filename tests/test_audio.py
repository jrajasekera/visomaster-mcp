"""Output audio follows the first audio stream's overlap with the video interval actually written."""

import subprocess
from array import array
from pathlib import Path

import common
import engine
import pytest
import test_timing
from test_timing import ffmpeg, numbered, render

media = test_timing.media
TONE = (0.7, 0.8)  # A 1 kHz burst, on the video clock, in delayed.mp4.


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("audio")
    work = root / "work"
    work.mkdir()
    numbered(root / "video.mp4", 25, 25, "-bf", "2")
    video = ("-i", root / "video.mp4")
    ffmpeg("-f", "lavfi", "-i", "sine=d=0.3:sample_rate=48000", work / "short.wav")
    ffmpeg("-f", "lavfi", "-i", "sine=d=3:sample_rate=48000", work / "long.wav")
    burst = "aevalsrc='if(between(t,0.2,0.3),0.5*sin(6283.2*t),0)':s=48000:d=1"
    ffmpeg("-f", "lavfi", "-i", burst, work / "burst.wav")
    aac = ("-c:v", "copy", "-c:a", "aac")
    # The first audio stream ends at 0.3 s, before frame 15.
    ffmpeg(*video, "-i", work / "short.wav", *aac, root / "short-audio.mp4")
    # Audio starts 1.48 s in, after the one-second video ends.
    late = ("-itsoffset", "1.48", "-i", work / "short.wav", "-map", "0:v", "-map", "1:a")
    ffmpeg(*video, *late, *aac, root / "late.mp4")
    # Audio starts 0.5 s in, partway through the video, and carries a burst at TONE.
    delayed = ("-itsoffset", "0.5", "-i", work / "burst.wav", "-map", "0:v", "-map", "1:a")
    ffmpeg(*video, *delayed, *aac, root / "delayed.mp4")
    # Two audio streams: renders and normalization take the first.
    for name, first, second in (("short-first", "short", "long"), ("long-first", "long", "short")):
        streams = ("-i", work / f"{first}.wav", "-i", work / f"{second}.wav")
        maps = ("-map", "0:v", "-map", "1:a", "-map", "2:a")
        ffmpeg(*video, *streams, *maps, *aac, "-t", 1, root / f"{name}.mp4")
    # Audio from -0.423 s, before the video, with a burst at test_timing.TONE on the video clock.
    first, last = (t + 0.423 for t in test_timing.TONE)
    burst = f"aevalsrc='if(between(t,{first},{last}),0.5*sin(6283.2*t),0)':s=48000:d=2"
    ffmpeg("-f", "lavfi", "-i", burst, work / "early.wav")
    early = ("-itsoffset", "-0.423", "-i", work / "early.wav", "-map", "0:v", "-map", "1:a")
    pcm = ("-c:v", "copy", "-c:a", "pcm_s16le")
    ffmpeg(*video, *early, *pcm, "-avoid_negative_ts", "disabled", root / "early.mkv")
    # Video without a duration tag, and audio running two seconds past it.
    ffmpeg(*video, "-i", work / "long.wav", *pcm, root / "long.mkv")
    # 1.44 s of video with matching audio.
    numbered(work / "v36.mp4", 25, 36)
    sine = ("-f", "lavfi", "-i", "sine=d=1.44:sample_rate=48000")
    ffmpeg("-i", work / "v36.mp4", *sine, *aac, root / "matched.mp4")
    # Clips on both sides of half and one and a half frame periods at 1 fps.
    for frames in (12, 13, 37, 38):
        numbered(root / f"frames-{frames}.mp4", 25, frames)
    numbered(work / "v5.mp4", 25, 5)
    ffmpeg("-i", work / "v5.mp4", "-i", work / "short.wav", *aac, "-t", 0.2, root / "tiny.mp4")
    ffmpeg("-i", work / "long.wav", "-c:a", "aac", "-t", 1, root / "sound.m4a")
    return root


def audible(path):
    """First and last second at which the audio is loud on the video clock, or None if silent.

    Decoding pads from zero, so audio that starts late keeps its offset.
    """
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0"]
    command += ["-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", "48000"]
    pcm = subprocess.run(command + ["-f", "s16le", "-"], check=True, capture_output=True)
    loud = [i for i, value in enumerate(array("h", pcm.stdout)) if abs(value) > 3000]
    return (loud[0] / 48000, loud[-1] / 48000) if loud else None


def streams(path, kind):
    return [s for s in common.probe(path)["streams"] if s["codec_type"] == kind]


def normalize(source, directory, fps=None):
    (directory / "artifacts").mkdir(parents=True)
    return engine.normalize_video({"media_path": str(source), "fps": fps}, directory)


def published(result):
    """Check the normalization manifest against FFprobe; returns the audio stream or None."""
    (video,) = streams(result["media_path"], "video")
    assert int(video["nb_frames"]) == result["frames"]
    assert float(video["duration"]) == result["duration"]
    audio = streams(result["media_path"], "audio")
    if result["audio"] == "none":
        assert not audio and result["audio_start_time"] is result["audio_duration"] is None
        return None
    (audio,) = audio
    assert result["audio"] == "AAC" and audio["codec_name"] == "aac"
    assert float(audio["start_time"]) == result["audio_start_time"]
    assert float(audio["duration"]) == result["audio_duration"]
    return audio


def end(audio):
    return float(audio["start_time"]) + float(audio["duration"])


def frame(audio):
    """One AAC frame in seconds: the encoder pads the last one."""
    return 1024 / int(audio["sample_rate"])


def drop_audio(monkeypatch):
    """Simulate a mux or encode that loses the expected audio."""
    real = engine.mux
    monkeypatch.setattr(
        engine,
        "mux",
        lambda video, output, audio=None, options=(): real(video, output, None, options),
    )


# vmcp-review-remediation-qpn.5


def test_render_after_the_first_audio_ends_completes_without_audio(media, tmp_path, monkeypatch):
    processed, result = render(media / "short-audio.mp4", tmp_path, monkeypatch, 15, 5)
    assert [index for index, _ in processed] == list(range(15, 20))
    assert (result["frames"], result["audio"]) == (5, "none")
    output = tmp_path / "artifacts/result.mp4"
    assert not streams(output, "audio") and streams(output, "video")[0]["nb_frames"] == "5"


def test_render_inside_the_first_audio_keeps_it(media, tmp_path, monkeypatch):
    _, result = render(media / "short-audio.mp4", tmp_path, monkeypatch, 0, 10)
    assert result["audio"] == "AAC"
    (audio,) = streams(tmp_path / "artifacts/result.mp4", "audio")
    assert abs(end(audio) - 0.3) <= frame(audio)


def test_normalization_before_late_audio_completes_without_audio(media, tmp_path):
    result = normalize(media / "late.mp4", tmp_path)
    assert (result["frames"], result["duration"]) == (25, 1.0)
    assert result["audio"] == "none" and published(result) is None


@pytest.mark.parametrize(("start", "count"), [(10, 10), (0, 25)])
def test_render_keeps_partially_overlapping_delayed_audio_in_time(
    media, tmp_path, monkeypatch, start, count
):
    _, result = render(media / "delayed.mp4", tmp_path, monkeypatch, start, count)
    assert result["audio"] == "AAC"
    output = tmp_path / "artifacts/result.mp4"
    first, last = audible(output)
    assert abs(first - (TONE[0] - start / 25)) < 0.025
    assert abs(last - (TONE[1] - start / 25)) < 0.025
    (audio,) = streams(output, "audio")
    assert abs(end(audio) - count / 25) <= frame(audio)


def test_normalization_keeps_partially_overlapping_delayed_audio_in_time(media, tmp_path):
    result = normalize(media / "delayed.mp4", tmp_path)
    audio = published(result)
    assert result["frames"] == 25 and result["audio_start_time"] == 0
    first, last = audible(result["media_path"])
    assert abs(first - TONE[0]) < 0.025 and abs(last - TONE[1]) < 0.025
    assert abs(end(audio) - 1) <= frame(audio)


def test_lost_expected_audio_fails_the_render(media, tmp_path, monkeypatch):
    drop_audio(monkeypatch)
    with pytest.raises(RuntimeError, match="Audio missing from output"):
        render(media / "delayed.mp4", tmp_path, monkeypatch, 10, 10)


def test_lost_expected_audio_fails_normalization(media, tmp_path, monkeypatch):
    drop_audio(monkeypatch)
    with pytest.raises(RuntimeError, match="Audio missing from normalized output"):
        normalize(media / "delayed.mp4", tmp_path)
    assert not (tmp_path / "artifacts/normalized.mp4").exists()


@pytest.mark.parametrize(("name", "audio"), [("short-first", "none"), ("long-first", "AAC")])
def test_multiple_audio_streams_follow_the_first(media, tmp_path, monkeypatch, name, audio):
    assert len(streams(media / f"{name}.mp4", "audio")) == 2
    _, result = render(media / f"{name}.mp4", tmp_path / "render", monkeypatch, 15, 5)
    assert result["audio"] == audio
    assert len(streams(tmp_path / "render/artifacts/result.mp4", "audio")) == (audio == "AAC")
    normalized = normalize(media / f"{name}.mp4", tmp_path / "normalize")
    output = published(normalized)
    # The first stream is 0.3 s long; the second runs past the video.
    assert abs(end(output) - (0.3 if name == "short-first" else 1)) <= frame(output)


def test_sources_without_audio_still_complete(media, tmp_path, monkeypatch):
    _, result = render(media / "video.mp4", tmp_path / "render", monkeypatch, 3, 5)
    assert result["audio"] == "none"
    normalized = normalize(media / "video.mp4", tmp_path / "normalize")
    assert normalized["frames"] == 25 and published(normalized) is None


# vmcp-review-remediation-qpn.6


def test_rate_change_bounds_audio_by_the_resampled_video(media, tmp_path):
    result = normalize(media / "matched.mp4", tmp_path, fps=1)
    assert (result["frames"], result["duration"]) == (1, 1.0)
    audio = published(result)
    assert audio["start_time"] == "0.000000"
    assert abs(end(audio) - 1) <= frame(audio)


@pytest.mark.parametrize("fps", [None, "10"])
def test_video_without_duration_bounds_longer_audio(media, tmp_path, fps):
    source = streams(media / "long.mkv", "video")[0]
    assert "duration" not in source and "nb_frames" not in source
    result = normalize(media / "long.mkv", tmp_path, fps)
    assert result["frames"] == (25 if fps is None else 10) and result["duration"] == 1.0
    audio = published(result)
    assert abs(end(audio) - 1) <= frame(audio)


def test_shorter_audio_keeps_the_whole_video(media, tmp_path):
    result = normalize(media / "short-audio.mp4", tmp_path)
    assert (result["frames"], result["duration"]) == (25, 1.0)
    audio = published(result)
    assert result["audio_start_time"] == 0 and abs(end(audio) - 0.3) <= frame(audio)


def test_earlier_audio_starts_with_the_video(media, tmp_path):
    result = normalize(media / "early.mkv", tmp_path)
    audio = published(result)
    assert result["frames"] == 25 and result["audio_start_time"] == 0
    first, last = audible(result["media_path"])
    assert abs(first - test_timing.TONE[0]) < 0.025 and abs(last - test_timing.TONE[1]) < 0.025
    assert abs(end(audio) - 1) <= frame(audio)


def test_audio_past_the_video_end_fails(media, tmp_path, monkeypatch):
    real = engine.audio_window
    monkeypatch.setattr(engine, "audio_window", lambda *window: real(*window[:2], window[2] + 0.5))
    with pytest.raises(RuntimeError, match="after the video's end at 1.000 s"):
        normalize(media / "long.mkv", tmp_path)
    assert not (tmp_path / "artifacts/normalized.mp4").exists()


# vmcp-review-remediation-qpn.10


@pytest.mark.parametrize(
    ("name", "fps", "frames"),
    [
        ("tiny.mp4", "1", 1),  # 0.2 s: under half a period, one frame is kept.
        ("frames-12.mp4", "1", 1),  # 0.48 s
        ("frames-13.mp4", "1", 1),  # 0.52 s: rounds to one frame.
        ("frames-37.mp4", "1", 1),  # 1.48 s
        ("frames-38.mp4", "1", 2),  # 1.52 s
        ("video.mp4", "10", 10),  # An ordinary downsample.
    ],
)
def test_short_clips_keep_at_least_one_frame(media, tmp_path, name, fps, frames):
    result = normalize(media / name, tmp_path, fps)
    assert result["frames"] == frames and result["duration"] == frames / int(fps)
    output = Path(result["media_path"])
    stream = engine.video_stream(common.probe(output))
    assert engine.validate_timing(output, engine.Fraction(fps), stream) == frames
    audio = published(result)
    if name == "tiny.mp4":
        # The 0.2 s of audio does not shorten the one-second frame.
        assert abs(end(audio) - 0.2) <= frame(audio)


def test_input_without_video_keeps_its_own_error(media, tmp_path):
    with pytest.raises(ValueError, match="Media has no video stream"):
        normalize(media / "sound.m4a", tmp_path, "1")
    assert not (tmp_path / "artifacts/normalized.mp4").exists()

"""FFmpeg and FFprobe open only self-contained media; manifests cannot reach outside the root."""

import io
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import common
import control
import engine
import pytest
from conftest import apply_config, write_config

ATTACKS = ("playlist.m3u8", "concat.mp4")
VIDEOS = ("clip.mp4", "clip.mov", "clip.mkv")


def ffmpeg(*arguments):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *map(str, arguments)], check=True)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """A media root holding ordinary media and two manifests that name an outside video."""
    top = tmp_path_factory.mktemp("confine")
    root, outside = top / "media", top / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "secret.mp4"
    lavfi = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=0.4"]
    lavfi += ["-f", "lavfi", "-i", "sine=duration=0.4"]
    ffmpeg(*lavfi, "-c:v", "libx264", "-bf", "0", "-c:a", "aac", "-shortest", secret)
    ffmpeg("-i", secret, "-c", "copy", "-f", "mpegts", outside / "secret.ts")
    for name in VIDEOS:
        # PCM keeps Matroska video timestamps at zero (AAC priming would shift them).
        audio = "pcm_s16le" if name.endswith(".mkv") else "copy"
        ffmpeg("-i", secret, "-c:v", "copy", "-c:a", audio, root / name)
    ffmpeg("-i", secret, "-frames:v", "1", root / "face.png", "-frames:v", "1", root / "face.jpg")
    # image2 would expand %d into frame1.jpg, a symlink to a larger outside image.
    ffmpeg("-i", secret, "-frames:v", "1", "-s", "32x24", outside / "large.jpg")
    shutil.copy(root / "face.jpg", root / "frame%d.jpg")
    (root / "frame1.jpg").symlink_to(Path("../outside/large.jpg"))
    # An HLS playlist naming an absolute out-of-root segment.
    (root / "playlist.m3u8").write_text(
        f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:0.4,\n{outside / 'secret.ts'}\n#EXT-X-ENDLIST\n"
    )
    # An FFconcat list with a video extension, naming an in-root symlink that escapes the root.
    (root / "linked.mp4").symlink_to(Path("../outside/secret.mp4"))
    (root / "concat.mp4").write_text("ffconcat version 1.0\nfile linked.mp4\n")
    return SimpleNamespace(root=root, outside=outside, secret=secret)


@pytest.fixture
def media(library, tmp_path, monkeypatch):
    apply_config(monkeypatch, write_config(tmp_path / "remote-config.json", library.root))
    return library


@pytest.mark.parametrize("name", ATTACKS)
def test_probe_refuses_manifests_with_named_container(media, name):
    container = {"playlist.m3u8": "hls", "concat.mp4": "concat"}[name]
    path = common.input_path(str(media.root / name))
    with pytest.raises(ValueError, match=f"Unsupported container '{container}'"):
        common.probe(path)
    with pytest.raises(ValueError, match="Unsupported container"):
        control.dispatch("inspect", {"media_path": str(path)})


def test_playlist_is_refused_before_its_segment_is_opened(media):
    playlist = media.root / "playlist.m3u8"
    command = ["ffprobe", "-v", "debug", *common.ffmpeg_input(playlist, ffprobe=True)]
    log = subprocess.run(command, capture_output=True, text=True, timeout=30).stderr
    assert f"Opening '{playlist}'" in log and "not on whitelist" in log
    assert "secret" not in log


@pytest.mark.parametrize("name", ATTACKS)
def test_every_ffmpeg_entry_point_refuses_manifests(media, name, monkeypatch):
    """Later opens enforce the policy too, so a file swapped after probing stays confined."""
    source = media.root / name
    stream = engine.video_stream(common.probe(media.root / "clip.mp4"))
    frame = SimpleNamespace(reshape=lambda *shape: shape)
    numpy = SimpleNamespace(uint8="uint8", frombuffer=lambda data, dtype: frame)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    assert list(engine.decode(source, stream, 0, 1)) == []
    assert list(engine.decode(source, stream, 2, 1)) == []
    with pytest.raises(subprocess.CalledProcessError):
        engine.validate_timing(source, Fraction(25), stream)
    output = media.outside / f"{name}.normalized.mp4"
    command = engine.normalize_command(source, output, stream, Fraction(25))
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert not output.exists()


@pytest.mark.parametrize("name", ATTACKS)
def test_normalize_job_refuses_manifests(media, name, tmp_path):
    (tmp_path / "artifacts").mkdir()
    with pytest.raises(ValueError, match="Unsupported container"):
        engine.normalize_video({"media_path": str(media.root / name)}, tmp_path)
    assert not (tmp_path / "artifacts/normalized.mp4").exists()


def render(media, source, directory, monkeypatch):
    """Run render_video with a pass-through worker; the source metadata comes from clip.mp4."""
    metadata, real = common.probe(media.root / "clip.mp4"), engine.probe
    frame = SimpleNamespace(shape=(48, 64, 3), tobytes=lambda: b"")

    class Encoder:
        def __init__(self, command, stdin):
            shutil.copy(media.root / "clip.mp4", command[-1])
            self.stdin = io.BytesIO()

        def wait(self, timeout):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(engine, "probe", lambda path: metadata if path == source else real(path))
    monkeypatch.setattr(engine, "validate_timing", lambda *a: 10)
    monkeypatch.setattr(engine, "decode", lambda *a: (f for f in [frame] * 10))
    monkeypatch.setattr(engine, "process", lambda worker, bgr, index: frame)
    monkeypatch.setattr(engine, "save_image", lambda *a: None)
    # Only the encoder is faked; the audio mux runs the real FFmpeg.
    run = SimpleNamespace(Popen=Encoder, PIPE=subprocess.PIPE, run=subprocess.run)
    monkeypatch.setattr(engine, "subprocess", run)
    (directory / "artifacts").mkdir()
    request = {"media_path": str(source), "kind": "render"}
    request["mappings"] = [{"target_face": "face.png", "source_faces": []}]
    worker = SimpleNamespace(swap_count=1, telemetry=engine.Telemetry({"0": 60.0}))
    return engine.render_video(None, request, directory, worker)


@pytest.mark.parametrize("name", ATTACKS)
def test_render_audio_input_refuses_manifests(media, name, tmp_path, monkeypatch):
    with pytest.raises(subprocess.CalledProcessError):
        render(media, media.root / name, tmp_path, monkeypatch)
    assert not (tmp_path / "artifacts/result.mp4").exists()


def test_render_audio_input_accepts_ordinary_video(media, tmp_path, monkeypatch):
    result = render(media, media.root / "clip.mov", tmp_path, monkeypatch)
    assert result["frames"] == 10 and result["audio"] == "AAC"


@pytest.mark.parametrize("name", VIDEOS)
def test_ordinary_video_passes_every_ffmpeg_entry_point(media, name, tmp_path, monkeypatch):
    source = common.input_path(str(media.root / name))
    assert control.dispatch("inspect", {"media_path": str(source)})["streams"]
    stream = engine.video_stream(common.probe(source))
    assert engine.validate_timing(source, Fraction(25), stream) == 10
    sizes, frame = [], SimpleNamespace(reshape=lambda *shape: shape)
    frombuffer = lambda data, dtype: sizes.append(len(data)) or frame  # noqa: E731
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(uint8="uint8", frombuffer=frombuffer))
    list(engine.decode(source, stream, 3, 2))
    assert sizes == [64 * 48 * 3] * 2
    (tmp_path / "artifacts").mkdir()
    result = engine.normalize_video({"media_path": str(source)}, tmp_path)
    assert result["frames"] == 10 and result["audio"] == "AAC"


@pytest.mark.parametrize("name", ("face.png", "face.jpg"))
def test_ordinary_images_still_probe(media, name):
    metadata = control.dispatch("inspect", {"media_path": str(media.root / name)})
    # FFmpeg detects some JPEGs as image2 by extension; FFprobe reads them with patterns off.
    assert metadata["format"]["format_name"] in {"png_pipe", "jpeg_pipe", "image2"}
    assert metadata["streams"][0]["width"] == 64


def test_image_patterns_do_not_expand_to_other_files(media, monkeypatch):
    pattern = common.input_path(str(media.root / "frame%d.jpg"))
    metadata = common.probe(pattern)
    assert metadata["format"]["format_name"] == "image2"
    assert (metadata["streams"][0]["width"], metadata["streams"][0]["height"]) == (64, 48)
    frame = SimpleNamespace(reshape=lambda *shape: shape)
    numpy = SimpleNamespace(uint8="uint8", frombuffer=lambda data, dtype: frame)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    # The FFmpeg CLI cannot turn pattern expansion off, so it refuses image2 altogether.
    stream = {"width": 32, "height": 24, "avg_frame_rate": "25/1", "pix_fmt": "yuvj420p"}
    assert list(engine.decode(pattern, stream, 0, 1)) == []


def test_outside_paths_and_escaping_symlinks_stay_rejected(media):
    for path in (media.secret, media.root / "linked.mp4", media.root / "../outside/secret.mp4"):
        with pytest.raises(ValueError, match="inside"):
            common.input_path(str(path))

"""Renders and normalization keep color, bit depth, pixel aspect and display orientation."""

import subprocess
import sys
from types import SimpleNamespace

import engine
import pytest
from conftest import apply_config, write_config
from test_timing import normalize, render, stream_of

WIDTH, HEIGHT = 96, 64
# Quadrant colors (RGB) of the patch fixtures: top-left, top-right, bottom-left, bottom-right.
PATCHES = [(200, 60, 40), (40, 180, 90), (50, 70, 200), (128, 128, 128)]
# The H.264 decoder reports full-range 8-bit 4:2:0 as yuvj420p.
DECODED = {"tv": "yuv420p", "pc": "yuvj420p"}
GRADIENT = 128  # Width of the 10-bit gray ramp, which steps by one code every two columns.


def ffmpeg(*arguments, data=None):
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", *map(str, arguments)]
    subprocess.run(command, input=data, check=True)


def patches(frames=10):
    """Raw rgb24 frames split into the four PATCHES quadrants."""
    rows = []
    for y in range(HEIGHT):
        top, bottom = PATCHES[:2] if y < HEIGHT // 2 else PATCHES[2:]
        rows.append(bytes(top) * (WIDTH // 2) + bytes(bottom) * (WIDTH // 2))
    return b"".join(rows) * frames


def raw(pixels, size=f"{WIDTH}x{HEIGHT}"):
    return ("-f", "rawvideo", "-pix_fmt", pixels, "-s", size, "-r", 25, "-i", "-")


def tagged(matrix_range, sar="1"):
    """Options converting RGB to BT.709 YUV of the given range, with matching tags."""
    scale = f"scale=out_color_matrix=bt709:out_range={matrix_range},format=yuv420p"
    scale += f",setparams=color_primaries=bt709:color_trc=bt709,setsar={sar}"
    tags = ("-colorspace", "bt709", "-color_range", matrix_range)
    return ("-vf", scale, *tags, "-c:v", "libx264", "-qp", 0)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("color")
    ffmpeg(*raw("rgb24"), "-c:v", "libx264rgb", "-qp", 0, root / "rgb.mp4", data=patches())
    ffmpeg(*raw("rgb24"), "-pix_fmt", "rgb48be", "-c:v", "png", root / "rgb48.mov", data=patches())
    for value in ("tv", "pc"):
        ffmpeg(*raw("rgb24"), *tagged(value), root / f"bt709-{value}.mp4", data=patches())
    # Anamorphic tagged video with audio: the audio mux copies the video stream.
    audio = ("-f", "lavfi", "-i", "sine=d=0.4", "-c:a", "aac", "-shortest")
    output = root / "tagged-audio.mp4"
    ffmpeg(*raw("rgb24"), *audio, *tagged("tv", "4/3"), output, data=patches())
    ramp = b"".join((384 + x // 2).to_bytes(2, "little") for x in range(GRADIENT)) * 16
    ffmpeg(*raw("gray10le", f"{GRADIENT}x16"), "-c:v", "ffv1", root / "gray10.mkv", data=ramp * 10)
    ffmpeg(
        *raw("rgb24"),
        "-pix_fmt",
        "yuv420p10le",
        "-c:v",
        "libx264",
        root / "y10.mp4",
        data=patches(),
    )
    # A white marker in the top-left corner of black frames shows the stored orientation.
    marker = "color=black:s=96x64:r=25:d=0.4,drawbox=x=8:y=8:w=24:h=16:color=white:t=fill"
    for name, sar in (("square", "1"), ("anamorphic", "4/3")):
        ffmpeg(
            "-f", "lavfi", "-i", f"{marker},setsar={sar}", "-c:v", "libx264", root / f"{name}.mp4"
        )
    for name, display in (
        ("vflip", ["-display_vflip"]),
        ("hflip", ["-display_hflip"]),
        ("rotate", ["-display_rotation", "90"]),
    ):
        ffmpeg(*display, "-i", root / "anamorphic.mp4", "-c", "copy", root / f"{name}.mp4")
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
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imread=lambda path: None))
    return library


def first_frame(path):
    return next(engine.decode(path, stream_of(path), 0, 1))


def mean_rgb(frame, x, y, size=8):
    """Mean RGB of the size x size block centred on (x, y) of a decoded BGR frame."""
    width = frame.shape[1]
    total = [0, 0, 0]
    for row in range(y - size // 2, y + size // 2):
        for column in range(x - size // 2, x + size // 2):
            offset = (row * width + column) * 3
            for channel in range(3):
                total[channel] += frame[offset + 2 - channel]
    return [value / size**2 for value in total]


def quadrants(frame):
    return [mean_rgb(frame, x, y) for y in (16, 48) for x in (24, 72)]


def assert_close(actual, expected, tolerance):
    for got, want in zip(actual, expected):
        assert max(abs(a - b) for a, b in zip(got, want)) <= tolerance, (actual, expected)


def marker(frame):
    """The corner of the frame that holds the centre of the bright marker."""
    height, width = frame.shape[:2]
    bright = [i // 3 for i in range(0, len(frame), 3) if frame[i] > 128]
    x = sum(i % width for i in bright) / len(bright) / width
    y = sum(i // width for i in bright) / len(bright) / height
    return "bottom" if y > 0.5 else "top", "right" if x > 0.5 else "left"


def outputs(source, tmp_path, monkeypatch, start=0, count=3):
    """Normalize and render `source`; returns the probed output streams and the results."""
    (tmp_path / "normalize").mkdir()
    normalized, _ = normalize(source, tmp_path / "normalize", monkeypatch)
    _, rendered = render(source, tmp_path / "render", monkeypatch, start, count)
    paths = [normalized["media_path"], tmp_path / "render/artifacts/result.mp4"]
    return [stream_of(path) for path in paths], paths, (normalized, rendered)


def test_fixtures_carry_the_formats_under_test(media):
    rgb = stream_of(media / "rgb.mp4")
    assert (rgb["pix_fmt"], rgb["color_space"], rgb["color_range"]) == ("gbrp", "gbr", "pc")
    assert stream_of(media / "rgb48.mov")["pix_fmt"] == "rgb48be"
    assert "bits_per_raw_sample" not in stream_of(media / "rgb48.mov")
    gray = stream_of(media / "gray10.mkv")
    assert (gray["pix_fmt"], gray["bits_per_raw_sample"]) == ("gray10le", "10")
    for value in ("tv", "pc"):
        stream = stream_of(media / f"bt709-{value}.mp4")
        assert (stream["pix_fmt"], stream["color_space"], stream["color_range"]) == (
            DECODED[value],
            "bt709",
            value,
        )
    flip = stream_of(media / "anamorphic.mp4")
    assert (flip["sample_aspect_ratio"], flip["display_aspect_ratio"]) == ("4:3", "2:1")


@pytest.mark.parametrize(("name", "pixels"), [("rgb.mp4", "yuv420p"), ("rgb48.mov", "yuv420p10le")])
def test_rgb_sources_encode_as_bt709_yuv_with_their_colors(
    media, tmp_path, monkeypatch, name, pixels
):
    streams, paths, _ = outputs(media / name, tmp_path, monkeypatch)
    for stream, path in zip(streams, paths):
        assert stream["pix_fmt"] == pixels
        assert (stream["color_space"], stream["color_range"]) == ("bt709", "tv")
        # Decoding by the output's own tags recovers the source colors.
        assert_close(quadrants(first_frame(path)), PATCHES, 4)


@pytest.mark.parametrize("value", ["tv", "pc"])
def test_tagged_bt709_yuv_keeps_matrix_range_and_tags(media, tmp_path, monkeypatch, value):
    source = media / f"bt709-{value}.mp4"
    stream = stream_of(source)
    command = engine.encode_command(stream, 25, "o.mp4")
    out = f"out_color_matrix=bt709:out_range={value}:"
    assert command[command.index("-vf") + 1].startswith(f"scale={out}")
    convert = engine.normalize_command("in.mp4", "o.mp4", stream, 25)
    assert f"in_color_matrix=bt709:in_range={value}:{out}" in convert[convert.index("-vf") + 1]
    expected = quadrants(first_frame(source))
    assert_close(expected, PATCHES, 3)
    streams, paths, _ = outputs(source, tmp_path, monkeypatch)
    for output, path in zip(streams, paths):
        tags = ("pix_fmt", "color_space", "color_range", "color_primaries", "color_transfer")
        assert [output[key] for key in tags] == [DECODED[value], "bt709", value, "bt709", "bt709"]
        assert_close(quadrants(first_frame(path)), expected, 3)


@pytest.mark.parametrize("transfer", ["smpte2084", "arib-std-b67"])
@pytest.mark.parametrize("pixels", ["gbrp", "yuv420p10le"])
def test_hdr_is_rejected_for_rgb_and_yuv(transfer, pixels):
    stream = {"width": 4, "height": 2, "pix_fmt": pixels, "color_transfer": transfer}
    with pytest.raises(ValueError, match="HDR"):
        engine.encode_command(stream, 25, "o.mp4")
    with pytest.raises(ValueError, match="HDR"):
        engine.normalize_command("in.mov", "o.mp4", stream, 25)


@pytest.mark.parametrize(
    ("stream", "tags"),
    [
        ({"pix_fmt": "gbrp", "color_space": "gbr", "color_range": "pc"}, ["bt709", "tv"]),
        ({"pix_fmt": "bgr0", "color_transfer": "iec61966-2-1"}, ["bt709", "tv", "iec61966-2-1"]),
        ({"pix_fmt": "yuv420p", "color_transfer": "bt470bg"}, ["unknown", "tv", "bt470bg"]),
        ({"pix_fmt": "yuv420p", "color_transfer": "bt470m"}, ["unknown", "tv", "bt470m"]),
        (
            {"color_space": "smpte170m", "color_primaries": "bt470bg"},
            ["smpte170m", "tv", "unknown", "bt470bg"],
        ),
        ({"pix_fmt": "yuvj420p", "color_primaries": "reserved"}, ["unknown", "pc"]),
    ],
)
def test_encoded_tags_follow_the_conversion(tmp_path, stream, tags):
    """Every tag FFprobe names is accepted and lands in the output on this FFmpeg version."""
    output = tmp_path / "o.mp4"
    command = engine.encode_command({"width": 16, "height": 16, **stream}, 25, output)
    subprocess.run(command, input=bytes(16 * 16 * 3 * 2), check=True)
    probed = stream_of(output)
    keys = ("color_space", "color_range", "color_transfer", "color_primaries")
    assert [probed.get(key, "unknown") for key in keys] == tags + ["unknown"] * (4 - len(tags))


@pytest.mark.parametrize(
    ("stream", "deep"),
    [
        ({"pix_fmt": "gray10le"}, True),
        ({"pix_fmt": "p010le"}, True),
        ({"pix_fmt": "rgb48be"}, True),
        ({"pix_fmt": "yuv420p10le"}, True),
        ({"pix_fmt": "y210le"}, True),
        ({"pix_fmt": "yuv420p"}, False),
        ({"pix_fmt": "yuvj422p"}, False),
        ({"pix_fmt": "nv12"}, False),
        ({"pix_fmt": "gbrp"}, False),
        # Absent or zero raw depth is unknown and resolved from the decoded format.
        ({"pix_fmt": "p010le", "bits_per_raw_sample": "0"}, True),
        ({"pix_fmt": "gray10le", "bits_per_raw_sample": "N/A"}, True),
        ({"pix_fmt": "yuv420p", "bits_per_raw_sample": "8"}, False),
        ({"bits_per_raw_sample": "10"}, True),
        ({}, False),
    ],
)
def test_depth_comes_from_component_bit_depth(stream, deep):
    assert engine.deep_color(stream) is deep
    pixels = "yuv420p10le" if deep else "yuv420p"
    assert engine.output_pixels(stream) == pixels
    command = engine.encode_command({"width": 4, "height": 2, **stream}, 25, "o.mp4")
    assert command[command.index("-crf") + 2 :][:2] == ["-pix_fmt", pixels]
    command = engine.normalize_command("in.mkv", "o.mp4", stream, 25)
    assert command[command.index("-pix_fmt") + 1] == pixels


@pytest.mark.parametrize(
    ("name", "pixels"),
    [("gray10.mkv", "yuv420p10le"), ("y10.mp4", "yuv420p10le"), ("square.mp4", "yuv420p")],
)
def test_render_and_normalization_share_the_depth_policy(
    media, tmp_path, monkeypatch, name, pixels
):
    streams, _, _ = outputs(media / name, tmp_path, monkeypatch)
    assert [stream["pix_fmt"] for stream in streams] == [pixels, pixels]


def luma(path, width, height):
    """The first frame's 10-bit luma samples, read without any conversion."""
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo"]
    data = subprocess.run(
        command + ["-pix_fmt", "yuv420p10le", "-"], capture_output=True, check=True
    ).stdout
    return [int.from_bytes(data[i : i + 2], "little") for i in range(0, width * height * 2, 2)]


@pytest.mark.parametrize(
    ("stream", "value"),
    [
        ({"pix_fmt": "gray10le"}, "pc"),  # FFmpeg 6 reports no range for gray FFV1.
        ({"pix_fmt": "gray10le", "color_range": "pc"}, "pc"),
        ({"pix_fmt": "gray", "color_range": "tv"}, "tv"),
        ({"pix_fmt": "yuv420p10le"}, "tv"),
    ],
)
def test_untagged_gray_is_full_range(stream, value):
    assert engine.color(stream) == (None, value)


def test_ten_bit_gray_normalizes_with_its_range_and_fine_steps(media, tmp_path, monkeypatch):
    result, _ = normalize(media / "gray10.mkv", tmp_path, monkeypatch)
    stream = stream_of(result["media_path"])
    assert (stream["pix_fmt"], stream["color_range"]) == ("yuv420p10le", "pc")
    samples = luma(result["media_path"], GRADIENT, 16)
    row = samples[8 * GRADIENT : 9 * GRADIENT]
    source = [384 + x // 2 for x in range(GRADIENT)]
    # Full range stays full range: the ramp keeps its codes rather than being rescaled.
    assert sum(abs(a - b) for a, b in zip(row, source)) / GRADIENT < 1
    # 8-bit quantization would leave at most 17 levels of the 64-code ramp.
    assert len(set(row)) > 32


@pytest.mark.parametrize(("start", "count"), [(3, 4), (0, None)])
@pytest.mark.parametrize(
    ("name", "sar", "dar"), [("anamorphic", "4:3", "2:1"), ("square", "1:1", "3:2")]
)
def test_render_keeps_the_source_display_aspect(
    media, tmp_path, monkeypatch, start, count, name, sar, dar
):
    _, result = render(media / f"{name}.mp4", tmp_path, monkeypatch, start, count)
    assert result["frames"] == (count or 10 - start)
    output = stream_of(tmp_path / "artifacts/result.mp4")
    assert (output["sample_aspect_ratio"], output["display_aspect_ratio"]) == (sar, dar)


def test_muxed_audio_keeps_the_video_tags_and_aspect(media, tmp_path, monkeypatch):
    streams, paths, results = outputs(media / "tagged-audio.mp4", tmp_path, monkeypatch)
    assert [result["audio"] for result in results] == ["AAC", "AAC"]
    keys = ("color_space", "color_range", "color_primaries", "color_transfer")
    for stream, path in zip(streams, paths):
        assert [stream[key] for key in keys] == ["bt709", "tv", "bt709", "bt709"]
        assert (stream["sample_aspect_ratio"], stream["display_aspect_ratio"]) == ("4:3", "2:1")
        assert any(s["codec_type"] == "audio" for s in engine.probe(path)["streams"])
        assert_close(quadrants(first_frame(path)), PATCHES, 4)


def test_render_validates_the_output_pixel_aspect(media, tmp_path, monkeypatch):
    command = engine.encode_command
    square = lambda *args: [a.replace("setsar=4/3", "setsar=1/1") for a in command(*args)]  # noqa: E731
    monkeypatch.setattr(engine, "encode_command", square)
    with pytest.raises(RuntimeError, match="pixel aspect"):
        render(media / "anamorphic.mp4", tmp_path, monkeypatch, 0, 2)


@pytest.mark.parametrize("name", ["vflip", "hflip"])
def test_mirrored_video_is_refused_before_artifacts(media, tmp_path, monkeypatch, name):
    stream = engine.probe(media / f"{name}.mp4")["streams"][0]
    # A vertical flip has rotation 0; only the matrix shows the reflection.
    assert engine.display_transform(stream)[1] is True
    assert (stream["side_data_list"][0]["rotation"] == 0) is (name == "vflip")
    with pytest.raises(ValueError, match="normalize_media"):
        render(media / f"{name}.mp4", tmp_path, monkeypatch, 0, 2)
    assert list((tmp_path / "artifacts").iterdir()) == []
    with pytest.raises(ValueError, match="normalize_media"):
        engine.load_frame(media / f"{name}.mp4")


@pytest.mark.parametrize(
    ("name", "position", "sar", "rotation", "reflection"),
    [
        ("anamorphic", ("top", "left"), "4:3", 0, False),
        ("vflip", ("bottom", "left"), "4:3", 0, True),
        ("hflip", ("top", "right"), "4:3", -180, True),
        ("rotate", ("bottom", "left"), "3:4", 90, False),
    ],
)
def test_normalization_applies_the_display_matrix(
    media, tmp_path, monkeypatch, name, position, sar, rotation, reflection
):
    assert marker(first_frame(media / "anamorphic.mp4")) == ("top", "left")
    result, _ = normalize(media / f"{name}.mp4", tmp_path, monkeypatch)
    assert (result["rotation_applied"], result["reflection_applied"]) == (rotation, reflection)
    output = stream_of(result["media_path"])  # No display matrix remains.
    assert "side_data_list" not in output and output["sample_aspect_ratio"] == sar
    assert marker(first_frame(result["media_path"])) == position

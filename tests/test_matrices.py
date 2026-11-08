"""YUV matrices and limited-range gray convert alike on FFmpeg 6.1 and 8, or are refused."""

import subprocess

import engine
import pytest
from test_color import (
    HEIGHT,
    PATCHES,
    WIDTH,
    assert_close,
    ffmpeg,
    first_frame,
    media,  # noqa: F401
    outputs,
    quadrants,
    raw,
)
from test_timing import normalize, render, stream_of

# Luma and blue-difference weights (Kr, Kb) from ITU-T H.273, by FFprobe matrix name.
WEIGHTS = {
    "bt709": (0.2126, 0.0722),
    "smpte170m": (0.299, 0.114),
    "bt470bg": (0.299, 0.114),
    "fcc": (0.30, 0.11),
    "smpte240m": (0.212, 0.087),
    "bt2020nc": (0.2627, 0.0593),
}
# Matrices swscale cannot apply: FFmpeg 8 fails to convert them and 6.1 uses other coefficients.
REFUSED = ["ycgco", "bt2020c", "smpte2085", "chroma-derived-nc", "chroma-derived-c", "ictcp"]
RAMP = 96  # Width of the gray ramps, which run from limited-range black to white.


def limited_yuv(rgb, weights):
    """8-bit limited-range Y, Cb and Cr of an RGB color, computed without FFmpeg."""
    kr, kb = weights
    r, g, b = (value / 255 for value in rgb)
    y = kr * r + (1 - kr - kb) * g + kb * b
    cb, cr = (b - y) / (2 - 2 * kb), (r - y) / (2 - 2 * kr)
    return round(16 + 219 * y), round(128 + 224 * cb), round(128 + 224 * cr)


def yuv444(weights):
    """One planar yuv444p frame of the PATCHES quadrants."""
    planes = [bytearray(), bytearray(), bytearray()]
    for y in range(HEIGHT):
        for x in range(WIDTH):
            color = PATCHES[(y >= HEIGHT // 2) * 2 + (x >= WIDTH // 2)]
            for plane, value in zip(planes, limited_yuv(color, weights)):
                plane.append(value)
    return b"".join(map(bytes, planes))


def ramp(depth):
    """Limited-range gray codes from black to white across RAMP columns."""
    black, white = 16 << (depth - 8), 235 << (depth - 8)
    return [black + (white - black) * x // (RAMP - 1) for x in range(RAMP)]


def gray(depth):
    width = 1 if depth == 8 else 2
    return b"".join(value.to_bytes(width, "little") for value in ramp(depth)) * 16


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("matrices")
    # Lossless 4:4:4 sources whose samples follow each matrix exactly; tagged on the frames,
    # which FFmpeg 8 encoders read. "unknown" carries no matrix tag and swscale's BT.601 default.
    for name, weights in [*WEIGHTS.items(), ("unknown", WEIGHTS["smpte170m"])]:
        tags = ("" if name == "unknown" else f"colorspace={name}:") + "range=tv"
        data = yuv444(weights) * 10
        ffmpeg(
            *raw("yuv444p"),
            "-vf",
            f"setparams={tags}",
            "-c:v",
            "ffv1",
            root / f"{name}.mkv",
            data=data,
        )
    for name in REFUSED:
        data = yuv444(WEIGHTS["bt709"]) * 10
        tags = f"setparams=colorspace={name}:range=tv"
        ffmpeg(*raw("yuv444p"), "-vf", tags, "-c:v", "ffv1", root / f"{name}.mkv", data=data)
    for depth, pixels in ((8, "gray"), (10, "gray10le")):
        for value in ("tv", "pc"):
            output = root / f"gray{depth}-{value}.mkv"
            options = ("-vf", f"setparams=range={value}", "-c:v", "ffv1")
            ffmpeg(*raw(pixels, f"{RAMP}x16"), *options, output, data=gray(depth) * 10)
    return root


def test_fixtures_carry_the_matrices_under_test(media):  # noqa: F811
    for name in [*WEIGHTS, *REFUSED]:
        stream = stream_of(media / f"{name}.mkv")
        assert (stream["pix_fmt"], stream["color_space"], stream["color_range"]) == (
            "yuv444p",
            name,
            "tv",
        )
    assert "color_space" not in stream_of(media / "unknown.mkv")
    for depth, pixels in ((8, "gray"), (10, "gray10le")):
        for value in ("tv", "pc"):
            stream = stream_of(media / f"gray{depth}-{value}.mkv")
            assert (stream["pix_fmt"], stream["color_range"]) == (pixels, value)


@pytest.mark.parametrize("name", [*WEIGHTS, "unknown"])
def test_supported_matrices_decode_and_encode_with_their_coefficients(
    media,  # noqa: F811
    tmp_path,
    monkeypatch,
    name,
):
    source = media / f"{name}.mkv"
    # Samples computed from the matrix's own weights decode to the colors they encode.
    assert_close(quadrants(first_frame(source)), PATCHES, 3)
    streams, paths, _ = outputs(source, tmp_path, monkeypatch)
    for stream, path in zip(streams, paths):
        # H.264 without any color tag omits the range too, which then means limited.
        tags = stream.get("color_space", "unknown"), stream.get("color_range", "tv")
        assert tags == (name, "tv")
        assert_close(quadrants(first_frame(path)), PATCHES, 4)


@pytest.mark.parametrize(
    "name", [*REFUSED, "ipt-c2", "ycgco-re", "ycgco-ro", "reserved", "unrecognized"]
)
@pytest.mark.parametrize("pixels", ["yuv420p", "yuv444p10le", "gray"])
def test_other_matrices_are_refused_with_guidance(name, pixels):
    stream = {"width": 4, "height": 2, "pix_fmt": pixels, "color_space": name}
    for call in (
        lambda: engine.encode_command(stream, 25, "o.mp4"),
        lambda: engine.normalize_command("in.mkv", "o.mp4", stream, 25),
    ):
        with pytest.raises(ValueError, match=f"{name} matrix .* transcode it to BT.709"):
            call()


@pytest.mark.parametrize("name", REFUSED)
def test_refused_sources_publish_nothing(media, tmp_path, monkeypatch, name):  # noqa: F811
    source = media / f"{name}.mkv"
    with pytest.raises(ValueError, match="matrix"):
        render(source, tmp_path / "render", monkeypatch, 0, 2)
    assert list((tmp_path / "render/artifacts").iterdir()) == []
    with pytest.raises(ValueError, match="matrix"):
        normalize(source, tmp_path, monkeypatch)
    assert list((tmp_path / "artifacts").iterdir()) == []
    assert not (tmp_path / "normalized.video.mp4").exists()
    with pytest.raises(ValueError, match="matrix"):
        engine.load_frame(source)


def rgb_row(frame, row=8):
    """R, G and B of every column of one row of a decoded BGR frame."""
    start = row * RAMP * 3
    return [tuple(frame[start + 3 * x + c] for c in (2, 1, 0)) for x in range(RAMP)]


def luma_row(path, depth, row=8):
    """One row of the first frame's stored luma codes, read without conversion."""
    pixels = "yuv420p" if depth == 8 else "yuv420p10le"
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo"]
    data = subprocess.run(command + ["-pix_fmt", pixels, "-"], capture_output=True, check=True)
    width = 1 if depth == 8 else 2
    start = row * RAMP * width
    line = data.stdout[start : start + RAMP * width]
    return [int.from_bytes(line[i : i + width], "little") for i in range(0, len(line), width)]


def mean_error(actual, expected):
    return sum(abs(a - b) for a, b in zip(actual, expected)) / len(expected)


@pytest.mark.parametrize("depth", [8, 10])
@pytest.mark.parametrize("value", ["tv", "pc"])
def test_gray_decodes_to_its_tagged_levels(media, depth, value):  # noqa: F811
    codes = ramp(depth)
    if value == "tv":
        black, white = 16 << (depth - 8), 235 << (depth - 8)
        expected = [(code - black) * 255 / (white - black) for code in codes]
    else:
        expected = [code * 255 / ((1 << depth) - 1) for code in codes]
    row = rgb_row(first_frame(media / f"gray{depth}-{value}.mkv"))
    assert all(r == g == b for r, g, b in row)
    assert max(abs(r - want) for (r, _, _), want in zip(row, expected)) <= 1.5, (row, expected)


@pytest.mark.parametrize("depth", [8, 10])
def test_limited_gray_keeps_its_levels_through_render_and_normalization(
    media,  # noqa: F811
    tmp_path,
    monkeypatch,
    depth,
):
    source = media / f"gray{depth}-tv.mkv"
    streams, paths, _ = outputs(source, tmp_path, monkeypatch)
    pixels = "yuv420p" if depth == 8 else "yuv420p10le"
    source_row = rgb_row(first_frame(source))
    # Allows for CRF 18 coding; levels treated as full range are off by 8 or 32 codes on average.
    tolerance = 1 if depth == 8 else 4
    for stream, path in zip(streams, paths):
        assert (stream["pix_fmt"], stream.get("color_range", "tv")) == (pixels, "tv")
        # Limited-range codes stay as they were rather than being treated as full range.
        assert mean_error(luma_row(path, depth), ramp(depth)) < tolerance
        row = rgb_row(first_frame(path))
        assert mean_error([r for r, _, _ in row], [r for r, _, _ in source_row]) < 1

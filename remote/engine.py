"""Version-pinned adapter to Fusion's FrameWorker and FunctionWorker."""

import bisect
import contextlib
import copy
import ctypes
import functools
import hashlib
import io
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from common import (
    APP,
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    CONFIG_ERROR,
    CONTROLS,
    MAX_COMPARE_PIXELS,
    PARAMETERS,
    REVISION,
    SETTINGS,
    Refusal,
    atomic_json,
    concat_input,
    ffmpeg_input,
    ffmpeg_version,
    file_sha256,
    fusion_compatibility,
    initial_controls,
    input_path,
    intent_digest,
    layouts,
    probe,
    provenance,
    read_checkpoint,
    read_json,
    resume_problems,
    segment_plan,
    temporal,
    validate_request,
    variant_request,
)


def initialize(controls):
    _, reason = fusion_compatibility(APP)  # The same check health reports.
    if reason:
        raise RuntimeError(reason)
    sys.path.insert(0, str(APP))
    os.chdir(APP)
    import torch
    from app.helpers.miscellaneous import ParametersDict
    from app.processors.models_processor import ModelsProcessor
    from app.processors.workers.function_worker import FunctionWorker

    torch.set_num_threads(2)
    control_specs, parameter_specs = layouts()
    control = initial_controls(control_specs, controls)
    defaults = {k: v["default"] for k, v in parameter_specs.items()}
    window = SimpleNamespace(
        gpu_id=0,
        control=control,
        default_parameters=ParametersDict(defaults, defaults),
        parameters={},
        target_faces={},
        video_processor=SimpleNamespace(file_type="video"),
        swapfacesButton=SimpleNamespace(isChecked=lambda: True),
        editFacesButton=SimpleNamespace(isChecked=lambda: False),
        model_loading_signal=SimpleNamespace(emit=lambda *a: None),
        model_loaded_signal=SimpleNamespace(emit=lambda *a: None),
    )
    window.models_processor = ModelsProcessor(window, device="cuda")
    window.models_processor.update_provider_configuration("CUDA")
    window.function_worker = FunctionWorker(window.models_processor)
    return window


def display_transform(stream):
    """FFprobe's display rotation and whether the display matrix mirrors the picture."""
    for data in stream.get("side_data_list", []):
        lines = data.get("displaymatrix", "").splitlines()
        m = [int(v) for line in lines for v in line.partition(":")[2].split()]
        if data.get("rotation") or len(m) == 9:
            return data.get("rotation") or 0, len(m) == 9 and m[0] * m[4] < m[1] * m[3]
    return 0, False


def video_stream(metadata):
    stream = next((s for s in metadata["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise Refusal("no_video_stream", "Media has no video stream")
    # Frames are decoded as stored; a flip can have rotation 0, so the matrix itself is checked.
    if display_transform(stream) != (0, False):
        raise Refusal(
            "rotated_or_mirrored",
            "Rotated or mirrored video is not supported; run normalize_media first",
            normalize=True,
        )
    return stream


HDR = {"smpte2084": "PQ", "arib-std-b67": "HLG"}
# FFprobe colorspace names mapped to swscale matrix names. Other matrices are rejected: swscale
# has none, so FFmpeg 8 cannot convert them and 6.1 silently applies other coefficients.
MATRICES = {
    "bt709": "bt709",
    "smpte170m": "smpte170m",
    "bt470bg": "bt470",
    "fcc": "fcc",
    "smpte240m": "smpte240m",
    "bt2020nc": "bt2020",
}
SWS_FLAGS = "accurate_rnd+full_chroma_int+full_chroma_inp"


@functools.cache
def pixel_formats():
    """FFmpeg's pixel format descriptors by name: flags and per-component bit depths."""
    command = ["ffprobe", "-v", "error", "-show_pixel_formats", "-of", "json"]
    output = subprocess.check_output(command, timeout=30)
    return {f["name"]: f for f in json.loads(output)["pixel_formats"]}


def pixel_format(stream):
    name = stream.get("pix_fmt")
    return pixel_formats().get(name, {}) if name else {}


def rgb(stream):
    """Whether the source is RGB (or paletted RGB) rather than YUV or gray."""
    flags = pixel_format(stream).get("flags", {})
    return bool(flags.get("rgb") or flags.get("palette")) or stream.get("color_space") == "gbr"


def color(stream):
    """Source YUV matrix and range for decoding; RGB has no matrix and is full range."""
    transfer = stream.get("color_transfer")
    if transfer in HDR:
        raise Refusal(
            "hdr_unsupported",
            f"HDR ({HDR[transfer]}) video is not supported because faces are processed as 8-bit "
            "SDR; tone-map the input to SDR first",
        )
    if rgb(stream):
        return None, "pc"
    space = stream.get("color_space", "unknown")
    if space != "unknown" and space not in MATRICES:
        raise Refusal(
            "matrix_unsupported",
            f"Video with the {space} matrix is not supported because FFmpeg's scaler cannot "
            "convert it; transcode it to BT.709 with a tool that supports that matrix, such as "
            "FFmpeg's zscale filter, and use the result",
        )
    tagged = stream.get("color_range")
    # Untagged gray is full range, as swscale treats it and FFmpeg 8 reports it.
    gray = pixel_format(stream).get("nb_components", 3) < 3 and tagged != "tv"
    full = tagged == "pc" or stream.get("pix_fmt", "").startswith("yuvj") or gray
    return MATRICES.get(space), "pc" if full else "tv"


def output_color(stream):
    """YUV matrix and range to encode: the source's, so unswapped pixels keep their color, or
    BT.709 limited range for RGB, whose identity matrix a YUV output cannot carry."""
    source = color(stream)  # Also rejects HDR.
    return ("bt709", "tv") if rgb(stream) else source


def limited_gray(stream):
    """Filters presenting limited-range gray as limited-range YUV of the same depth. swscale
    treats gray as full range whatever its tag (FFmpeg 8 at every depth, 6.1 at 8 bits), so its
    codes are copied unchanged and only then tagged limited."""
    format_ = pixel_format(stream)
    if rgb(stream) or format_.get("nb_components", 3) > 2 or color(stream)[1] != "tv":
        return []
    depth = max((c["bit_depth"] for c in format_.get("components", [])), default=8)
    name = "yuv444p" if depth == 8 else f"yuv444p{depth}le"
    name = name if name in pixel_formats() else "yuv444p16le"
    return [
        f"scale=in_range=pc:out_range=pc:flags={SWS_FLAGS}",
        f"format={name}",
        "setparams=range=tv",
    ]


def scale_filter(stream, *sides):
    """swscale conversion from the source's colors ("in") and to the output's ("out")."""
    options = []
    for side in sides:
        matrix, value = color(stream) if side == "in" else output_color(stream)
        options += [f"{side}_color_matrix={matrix}"] if matrix else []
        options.append(f"{side}_range={value}")
    convert = "scale=" + ":".join(options + [f"flags={SWS_FLAGS}"])
    return ",".join((limited_gray(stream) if "in" in sides else []) + [convert])


def deep_color(stream):
    """Whether the source has more than 8 bits per component. The decoder's raw sample depth
    is authoritative; when FFprobe reports none, the decoded pixel format's deepest component."""
    try:
        depth = int(stream.get("bits_per_raw_sample") or 0)
    except ValueError:
        depth = 0
    components = pixel_format(stream).get("components", [])
    return (depth or max((c["bit_depth"] for c in components), default=8)) > 8


def output_pixels(stream):
    """H.264 pixel format for renders and normalization alike: 10-bit for deeper sources."""
    return "yuv420p10le" if deep_color(stream) else "yuv420p"


def sample_aspect(stream):
    """The stream's sample aspect ratio; unknown (absent or 0:1) means square pixels."""
    numerator, _, denominator = stream.get("sample_aspect_ratio", "").partition(":")
    try:
        return Fraction(int(numerator), int(denominator)) or Fraction(1)
    except (ValueError, ZeroDivisionError):
        return Fraction(1)


def output_filters(stream, *sides):
    """Convert to the output's matrix, range and depth, and tag the source's primaries and
    transfer on the frames: FFmpeg 8 takes those from frames and ignores encoder options."""
    filters = [scale_filter(stream, *sides), f"format={output_pixels(stream)}"]
    tags = [
        f"{option}={stream[key]}"
        for option, key in (("color_primaries", "color_primaries"), ("color_trc", "color_transfer"))
        if stream.get(key, "unknown") not in ("unknown", "reserved")
    ]
    return ",".join(filters + (["setparams=" + ":".join(tags)] if tags else []))


def color_tags(stream):
    """Encoder options tagging the output's range and matrix."""
    tags = ["-color_range", output_color(stream)[1]]
    space = "bt709" if rgb(stream) else stream.get("color_space", "unknown")
    return tags + (["-colorspace", space] if space not in ("unknown", "reserved") else [])


def encode_command(stream, fps, output):
    """Encode bgr24 frames with the source's matrix, range, tags, depth and pixel aspect."""
    width, height = int(stream["width"]), int(stream["height"])
    pixels, sar = output_pixels(stream), sample_aspect(stream)
    # The raw pipe carries no pixel aspect, so the source's is set again.
    filters = f"{output_filters(stream, 'out')},setsar={sar.numerator}/{sar.denominator}"
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", "2", "-f", "rawvideo"]
    command += ["-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
    command += ["-an", "-vf", filters, "-c:v", "libx264"]
    command += ["-threads", "2", "-preset", "medium", "-crf", "18", "-pix_fmt", pixels]
    return command + color_tags(stream) + [str(output)]


def decode(path, stream, start, count, times=None):
    """Yield BGR frames start..start+count-1 in decoded order.

    `times` are presentation times of frames 0..start from a Timeline; without them the stream
    must be CFR from zero, as validate_timing checks. Seeks use the video stream's own clock:
    without -seek_timestamp FFmpeg offsets -ss by the container start, which an earlier audio
    stream can move.
    """
    import numpy as np

    width, height = int(stream["width"]), int(stream["height"])
    command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-noautorotate"]
    if start:
        # Accurate seeking drops frames before a point midway between the previous frame and
        # this one, which absorbs timestamp rounding.
        if times:
            point = (times[start - 1] + times[start]) / 2
        else:
            point = float((start - Fraction(1, 2)) / Fraction(stream["avg_frame_rate"]))
        command += ["-seek_timestamp", "1", "-ss", str(point)]
    command += [*ffmpeg_input(path), "-map", "0:v:0", "-frames:v", str(count), "-fps_mode"]
    command += ["passthrough", "-vf", scale_filter(stream, "in") + ",format=bgr24"]
    command += ["-f", "rawvideo", "pipe:1"]
    decoder = subprocess.Popen(command, stdout=subprocess.PIPE)
    size = width * height * 3
    try:
        for _ in range(count):
            data = decoder.stdout.read(size)
            if len(data) != size:
                return
            yield np.frombuffer(data, np.uint8).reshape(height, width, 3)
    finally:
        if decoder.poll() is None:
            decoder.kill()
        decoder.wait()
        decoder.stdout.close()


def load_frame(path, frame_number=0, timing=None):
    """Read zero-based decoded frame `frame_number`, selected on the video stream's timeline.

    `timing` is a Timeline reaching that frame; without it a bounded scan is made here. Packet
    timestamps give each ordinal's actual time, so variable frame rates select exact frames.
    """
    import cv2

    image = cv2.imread(str(path))
    if image is not None:
        if frame_number:
            raise ValueError("Images only have frame 0")
        return image
    stream, times = video_stream(probe(path)), None
    if frame_number:
        times = (timing or timeline(path, frame_number + 1)).times
        if frame_number >= len(times):
            raise ValueError(
                f"Frame {frame_number} is out of range; the video has {len(times)} frames"
            )
    return read_frame(path, stream, frame_number, times)


def read_frame(path, stream, number, times=None):
    """Decode the single frame `number`; see decode."""
    decoded = decode(path, stream, number, 1, times)
    try:
        frame = next(decoded, None)
    finally:
        decoded.close()
    if frame is None:
        raise ValueError(f"Cannot read frame {number} from {path}")
    return frame


def detect(window, bgr, max_faces=None):
    import numpy as np
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1)
    tensor = tensor.to(window.models_processor.device)
    boxes, points, _ = window.function_worker.run_detect(
        tensor,
        window.control["DetectorModelSelection"],
        max_num=max_faces or window.control["MaxFacesToDetectSlider"],
        score=window.control["DetectorScoreSlider"] / 100,
        use_landmark_detection=False,
        bypass_bytetrack=True,
    )
    return tensor, boxes, points


def detect_reference(window, image):
    """Detections deciding whether an image is a single-face reference, as submit_swap checks."""
    return detect(window, image, max_faces=20)


def recognize(window, tensor, point):
    """A detected face's embedding from the recognition model Fusion matches with, or None."""
    model = window.control["RecognitionModelSelection"]
    embedding = window.function_worker.run_recognize_direct(tensor, point, "Auto", model)[0]
    return None if embedding is None else [float(v) for v in embedding.ravel()]


def unit(vector):
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def similarity(a, b):
    """Fusion's match score: 100 - 50 * cosine distance (FunctionWorker.findCosineDistance)."""
    return 50 + 50 * sum(x * y for x, y in zip(unit(a), unit(b)))


# Crops pad a face's box by a share of its longer side, enough to re-detect; when the padding
# takes in a neighbouring face, the tighter crop is tried.
PADDING = (0.3, 0.1)


def crop_bounds(box, ratio, limits):
    """`box` padded by `ratio` of its longer side and clamped to `limits`, both x1, y1, x2, y2."""
    x1, y1, x2, y2 = box
    pad = int(max(x2 - x1, y2 - y1) * ratio)
    return (
        max(limits[0], x1 - pad),
        max(limits[1], y1 - pad),
        min(limits[2], x2 + pad),
        min(limits[3], y2 + pad),
    )


def region(image, bounds):
    x1, y1, x2, y2 = bounds
    return image[y1:y2, x1:x2]


def find_faces(window, image, frame):
    """Faces detected in `image`, each with its box, embedding (None if unrecognized) and
    padded crop, whose top-left corner in the image is `origin`."""
    tensor, boxes, points = detect(window, image)
    faces = []
    for box, point in zip(boxes, points):
        box = [int(v) for v in box[:4]]
        bounds = crop_bounds(box, PADDING[0], (0, 0, image.shape[1], image.shape[0]))
        faces.append(
            {
                "frame": frame,
                "box": box,
                "area": (box[2] - box[0]) * (box[3] - box[1]),
                "embedding": recognize(window, tensor, point),
                "crop": region(image, bounds).copy(),
                "origin": bounds[:2],
            }
        )
    return faces


def overlap(a, b):
    """Intersection over union of two boxes."""
    width, height = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    return shared / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared)


def reference_crop(window, face, threshold):
    """The first crop of `face` that submit_swap accepts as a reference to this face, or None.

    submit_swap requires a target_face image to re-detect as exactly one face. Here that face
    must also be this one: its box overlaps the original by an IoU of at least 0.5, and its
    embedding scores at least `threshold` against the original's. A neighbour inside the
    padding fails the count and a lone neighbour fails the match; the tighter crop is next.
    """
    if face["embedding"] is None:
        return None
    image, (left, top) = face["crop"], face["origin"]
    x1, y1, x2, y2 = face["box"]
    box = [x1 - left, y1 - top, x2 - left, y2 - top]
    for ratio in PADDING:
        bounds = crop_bounds(box, ratio, (0, 0, image.shape[1], image.shape[0]))
        candidate = region(image, bounds)
        tensor, boxes, points = detect_reference(window, candidate)
        if len(boxes) != 1:
            continue
        found = [float(v) + shift for v, shift in zip(boxes[0][:4], bounds[:2] * 2)]
        if overlap(found, box) < 0.5:
            continue
        embedding = recognize(window, tensor, points[0])
        if embedding is not None and similarity(embedding, face["embedding"]) >= threshold:
            return candidate
    return None


def save_reference(window, candidates, threshold, path):
    """Save the first verified reference crop of `candidates`, else the first padded crop.

    Returns the face it shows, the image and whether it is verified."""
    for face in candidates:
        image = reference_crop(window, face, threshold)
        if image is not None:
            save_image(path, image)
            return face, image, True
    save_image(path, candidates[0]["crop"])
    return candidates[0], candidates[0]["crop"], False


UNVERIFIED = (
    "No crop re-detected as exactly its own face, so none is a ready target_face; the crops are "
    "for inspection only. Detect other frames, or pass a reference image showing one face."
)


def references(result, entries):
    """`result` with a warning when it has faces but no verified reference."""
    if entries and not any(entry["reference_verified"] for entry in entries):
        result["warning"] = UNVERIFIED
    return result


def cluster_faces(embeddings, frames, threshold):
    """Greedily group faces into identities on Fusion's similarity scale.

    Fusion scores a match as 100 - 50 * cosine distance (50 + 50 * cosine similarity) and swaps
    at or above SimilarityThresholdSlider. Faces should arrive best first. Each joins the most
    similar identity whose normalized mean embedding scores at least `threshold` and that has
    no face from the same frame; otherwise it starts a new identity. Returns one label per face.
    """
    identities, labels = [], []
    for embedding, frame in zip(embeddings, frames):
        vector = unit(embedding)
        best, best_score = None, threshold
        for number, identity in enumerate(identities):
            if frame in identity["frames"]:
                continue
            score = 50 + 50 * sum(a * b for a, b in zip(identity["mean"], vector))
            if score >= best_score:
                best, best_score = number, score
        if best is None:
            best = len(identities)
            identities.append({"sum": [0.0] * len(vector), "frames": set()})
        identity = identities[best]
        identity["sum"] = [a + b for a, b in zip(identity["sum"], vector)]
        identity["mean"] = unit(identity["sum"])
        identity["frames"].add(frame)
        labels.append(best)
    return labels


def sample_frames(total, count):
    """Pick `count` frame indices at the centres of equal spans of `total` frames."""
    return sorted({(2 * index + 1) * total // (2 * count) for index in range(count)})


def detect_frame(window, request, directory):
    """Every face of one frame, with a reference crop each."""
    number = request.get("frame", 0)
    image = load_frame(input_path(request["media_path"]), number)
    threshold = layouts()[1]["SimilarityThresholdSlider"]["default"]
    artifacts = directory / "artifacts"
    faces, tiles = [], []
    for index, face in enumerate(find_faces(window, image, number)):
        name = f"face-{index}.png"
        _, shown, verified = save_reference(window, [face], threshold, artifacts / name)
        tiles.append((f"face {index}" + ("" if verified else " *"), shown, verified))
        faces.append(
            {
                "index": index,
                "box": face["box"],
                "reference_path": str(artifacts / name) if verified else None,
                "reference_verified": verified,
                "artifact": name,
            }
        )
    save_image(artifacts / "frame.jpg", image)
    names = ["frame.jpg"] + [f["artifact"] for f in faces]
    if tiles:
        contact_sheet(artifacts / "contact.jpg", tiles)
        names.insert(1, "contact.jpg")
    return references({"faces": faces, "artifacts": names}, faces)


def detect_identities(window, request, directory):
    import cv2

    path = input_path(request["media_path"])
    frames, timing = sorted(request.get("frames") or []), None
    if cv2.imread(str(path)) is not None:
        frames = frames or [0]
    elif not frames:
        # The scan counts the selected video stream's frames; container metadata can describe
        # a longer audio track or be missing.
        timing = timeline(path)
        frames = sample_frames(len(timing.times), request["sample_frames"])
    elif frames[-1]:
        timing = timeline(path, frames[-1] + 1)
    faces = []
    for done, number in enumerate(frames):
        atomic_json(
            directory / "progress.json",
            {"stage": "detecting", "frames": done, "total_frames": len(frames)},
        )
        faces += find_faces(window, load_frame(path, number, timing), number)
    faces = sorted((f for f in faces if f["embedding"] is not None), key=lambda f: -f["area"])
    # References are verified at the threshold submit_swap matches with by default.
    reference = layouts()[1]["SimilarityThresholdSlider"]["default"]
    threshold = request.get("similarity_threshold", reference)
    labels = cluster_faces([f["embedding"] for f in faces], [f["frame"] for f in faces], threshold)
    groups = {}
    for face, label in zip(faces, labels):
        groups.setdefault(label, []).append(face)
    artifacts = directory / "artifacts"
    identities, tiles = [], []
    for number, members in enumerate(sorted(groups.values(), key=lambda m: -len(m))):
        # Members are largest first; the first of the five largest with a verified crop wins.
        name = f"identity-{number}.png"
        best, shown, verified = save_reference(window, members[:5], reference, artifacts / name)
        tiles.append(
            (f"id {number}: {len(members)}x" + ("" if verified else " *"), shown, verified)
        )
        identities.append(
            {
                "identity": number,
                "appearances": len(members),
                "frames": sorted({m["frame"] for m in members}),
                "frame": best["frame"],
                "box": best["box"],
                "reference_path": str(artifacts / name) if verified else None,
                "reference_verified": verified,
                "artifact": name,
                "samples": [
                    {"frame": m["frame"], "box": m["box"]}
                    for m in sorted(members, key=lambda m: m["frame"])[:5]
                ],
            }
        )
    names = [i["artifact"] for i in identities]
    if tiles:
        contact_sheet(artifacts / "contact.jpg", tiles)
        names.insert(0, "contact.jpg")
    result = {
        "frames": frames,
        "detections": len(faces),
        "similarity_threshold": threshold,
        "identities": identities,
        "artifacts": names,
    }
    return references(result, identities)


# Reference analysis measures faces as submit_swap reads them: the same detector and limit, and
# the swap model's ArcFace embedding, which the source merge averages.
RECOGNITION_SIDE = 112  # Fusion's ArcFace input size; sharpness is measured at this scale.
ANALYSIS_DEFAULTS = {"min_face_pixels": 112, "min_sharpness": 40.0, "duplicate_similarity": 95.0}
METRICS = {
    "face_pixels": "Shorter side of the detector's face box, in source-image pixels",
    "sharpness": (
        "Variance of the 4-neighbour Laplacian (cv2.Laplacian, ksize 1) of the face box's 8-bit "
        "luma resized to 112x112, the recognition input size, in squared luma levels. Lower is "
        "blurrier; faces smaller than 112 pixels are enlarged and score lower"
    ),
    "similarity": "Fusion's match scale: 50 + 50 x cosine similarity of the embeddings (0-100)",
}
CONSISTENCY_BASIS = (
    "Leave-one-out: each usable reference's embedding scored on Fusion's similarity scale "
    "against the normalized mean of the other usable references' unit embeddings, excluding "
    "byte-identical copies of it"
)
LABELS = {
    "unreadable": "unreadable",
    "no_face": "no face",
    "unrecognized": "no embedding",
    "small_face": "small",
    "blurry": "blurry",
    "exact_duplicate": "copy",
    "near_duplicate": "duplicate",
    "outlier": "outlier",
    "inconsistent_pair": "differs",
}


def laplacian_variance(rows):
    """Variance of the 4-neighbour Laplacian over the interior of `rows` (lists of luma)."""
    values = [
        rows[y - 1][x] + rows[y + 1][x] + rows[y][x - 1] + rows[y][x + 1] - 4 * rows[y][x]
        for y in range(1, len(rows) - 1)
        for x in range(1, len(rows[y]) - 1)
    ]
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def face_sharpness(image, box):
    import cv2

    gray = cv2.cvtColor(region(image, box), cv2.COLOR_BGR2GRAY)
    enlarge = min(gray.shape) < RECOGNITION_SIDE
    gray = cv2.resize(
        gray,
        (RECOGNITION_SIDE, RECOGNITION_SIDE),
        interpolation=cv2.INTER_CUBIC if enlarge else cv2.INTER_AREA,
    )
    return round(laplacian_variance(gray.tolist()), 1)


def finding(entry, code, severity, message):
    entry["findings"].append({"code": code, "severity": severity, "message": message})


def inspect_reference(window, entry, image, model, thresholds):
    """Detect and measure one decoded reference as submit_swap would; returns its embedding,
    or None when submit_swap would refuse it."""
    tensor, boxes, points = detect_reference(window, image)
    entry["faces"] = len(boxes)
    if len(boxes) != 1:
        entry["boxes"] = [[int(v) for v in box[:4]] for box in boxes]
        if boxes:
            message = f"{len(boxes)} faces detected; crop it to one face (detect_faces does)"
            finding(entry, "multiple_faces", "error", message)
        else:
            finding(entry, "no_face", "error", "No face detected; submit_swap requires one")
        return None
    height, width = image.shape[:2]
    box = crop_bounds([int(v) for v in boxes[0][:4]], 0, (0, 0, width, height))
    entry["box"] = list(box)
    entry["face_pixels"] = pixels = min(box[2] - box[0], box[3] - box[1])
    if pixels < thresholds["min_face_pixels"]:
        message = (
            f"The face is {pixels} px on its shorter side, below {thresholds['min_face_pixels']}; "
            "recognition enlarges it, so its identity detail is limited"
        )
        finding(entry, "small_face", "warning", message)
    if pixels > 2:
        entry["sharpness"] = face_sharpness(image, box)
        if entry["sharpness"] < thresholds["min_sharpness"]:
            message = (
                f"Sharpness {entry['sharpness']} is below {thresholds['min_sharpness']}; the "
                "face looks blurred at recognition scale"
            )
            finding(entry, "blurry", "warning", message)
    embedding = window.function_worker.run_recognize_direct(tensor, points[0], "Auto", model)[0]
    if embedding is None:
        finding(entry, "unrecognized", "error", f"{model} produced no embedding for the face")
        return None
    entry["usable"] = True
    return [float(v) for v in embedding.ravel()]


def assess_collection(entries, embeddings, thresholds, model):
    """Duplicates and embedding consistency within one collection, as findings on `entries`.

    `embeddings` maps the index of each usable entry to its embedding. Returns the duplicates
    and the consistency summary. Consistency needs at least two distinct usable references, and
    outlier scores three: with two, a disagreement cannot be attributed to either.
    """
    duplicates = []
    for later, entry in enumerate(entries):
        for earlier, other in enumerate(entries[:later]):
            score = None
            if other["sha256"] == entry["sha256"]:
                kind, code = "identical_bytes", "exact_duplicate"
                message = f"Same bytes as {other['id']}, so an average counts that image twice"
            elif earlier in embeddings and later in embeddings:
                score = round(similarity(embeddings[earlier], embeddings[later]), 1)
                if score < thresholds["duplicate_similarity"]:
                    continue
                kind, code = "near_identical", "near_duplicate"
                message = (
                    f"Scores {score} against {other['id']}, at least "
                    f"{thresholds['duplicate_similarity']}: nearly the same image, so it adds "
                    "weight rather than new information"
                )
            else:
                continue
            duplicates.append(
                {
                    "reference": entry["id"],
                    "duplicate_of": other["id"],
                    "kind": kind,
                    "similarity": score,
                }
            )
            finding(entry, code, "warning", message)
            break
    usable = sorted(embeddings)
    distinct = {}
    for index in usable:
        distinct.setdefault(entries[index]["sha256"], index)
    consistency = {"model": model, "basis": CONSISTENCY_BASIS, "available": False}
    limit = thresholds["outlier_similarity"]
    if len(distinct) >= 3:
        consistency["available"] = True
        for index in usable:
            others = [
                unit(embeddings[j])
                for j in usable
                if entries[j]["sha256"] != entries[index]["sha256"]
            ]
            score = round(similarity(embeddings[index], [sum(c) for c in zip(*others)]), 1)
            entries[index]["consistency_score"] = score
            if score < limit:
                message = (
                    f"Scores {score} against the other references' mean, below {limit}: it may "
                    "show another person or an unusual view or lighting. Review its crop; "
                    "nothing was removed"
                )
                finding(entries[index], "outlier", "warning", message)
    elif len(distinct) == 2:
        first, second = distinct.values()
        score = round(similarity(embeddings[first], embeddings[second]), 1)
        consistency["pair_similarity"] = score
        consistency["reason"] = (
            "Outlier scores need at least 3 distinct usable references; with 2, a disagreement "
            "cannot be attributed to either"
        )
        if score < limit:
            for index, other in ((first, second), (second, first)):
                message = (
                    f"Scores {score} against {entries[other]['id']}, below {limit}: one of the "
                    "two may show another person; review both"
                )
                finding(entries[index], "inconsistent_pair", "warning", message)
    else:
        consistency["reason"] = "Consistency needs at least 2 distinct usable references"
    return duplicates, consistency


def analyze_references(window, request, directory):
    """Per-reference face count, size, sharpness, duplicates and embedding consistency.

    References are decoded from the bytes fingerprinted at submission and are never altered or
    dropped: findings only report. Writes a crop per usable reference, a labelled contact sheet
    of every reference and analysis.json.
    """
    import numpy as np

    parameters = layouts()[1]
    swap_model = request.get("swap_model") or parameters["SwapModelSelection"]["default"]
    model = window.function_worker.get_arcface_model(swap_model)
    thresholds = {
        **ANALYSIS_DEFAULTS,
        "outlier_similarity": parameters["SimilarityThresholdSlider"]["default"],
        **(request.get("thresholds") or {}),
    }
    paths = request["references"]
    expected = request.get("reference_sha256") or [None] * len(paths)
    artifacts = directory / "artifacts"
    entries, embeddings, shown = [], {}, []
    for index, (path, digest) in enumerate(zip(paths, expected)):
        atomic_json(
            directory / "progress.json",
            {"stage": "analyzing", "references": index, "total_references": len(paths)},
        )
        entry = {
            "id": f"r{index}",
            "path": path,
            "sha256": None,
            "usable": False,
            "faces": None,
            "image_size": None,
            "box": None,
            "face_pixels": None,
            "sharpness": None,
            "consistency_score": None,
            "artifact": None,
            "findings": [],
        }
        entry["sha256"], image = reference_image(path, digest)
        picture = np.full((RECOGNITION_SIDE, RECOGNITION_SIDE, 3), 48, np.uint8)
        if image is None:
            message = "Not a decodable image (PNG, JPEG, WebP or BMP); submit_swap refuses it"
            finding(entry, "unreadable", "error", message)
        else:
            entry["image_size"] = [int(image.shape[1]), int(image.shape[0])]
            embedding = inspect_reference(window, entry, image, model, thresholds)
            picture = image
            if embedding is not None:
                embeddings[index] = embedding
                bounds = crop_bounds(entry["box"], PADDING[0], (0, 0, *entry["image_size"]))
                picture = region(image, bounds)
                entry["artifact"] = f"reference-{index}.png"
                save_image(artifacts / entry["artifact"], picture)
        entries.append(entry)
        shown.append(picture)
    duplicates, consistency = assess_collection(entries, embeddings, thresholds, model)
    tiles = []
    for entry, picture in zip(entries, shown):
        codes = [f["code"] for f in entry["findings"]]
        label = LABELS.get(codes[0], f"{entry['faces']} faces") if codes else "ok"
        more = "+" if len(codes) > 1 else ""
        tiles.append((f"{entry['id']} {label}{more}", picture, entry["usable"]))
    contact_sheet(artifacts / "contact.jpg", tiles)
    severities = [f["severity"] for entry in entries for f in entry["findings"]]
    result = {
        "swap_model": swap_model,
        "model": model,
        "usable": len(embeddings),
        "errors": severities.count("error"),
        "warnings": severities.count("warning"),
        "references": entries,
        "duplicates": duplicates,
        "consistency": consistency,
        "thresholds": thresholds,
        "metrics": METRICS,
        "artifacts": ["contact.jpg", "analysis.json"]
        + [entry["artifact"] for entry in entries if entry["artifact"]],
    }
    atomic_json(artifacts / "analysis.json", {**result, "revision": REVISION})
    return result


class TargetFace:
    def __init__(self, face_id, embeddings, sources):
        self.face_id = face_id
        self.embeddings = embeddings
        self.assigned_input_embedding = sources
        self.assigned_kv_map = None
        self.assigned_input_faces = {}
        self.assigned_merged_embeddings = {}

    def get_embedding(self, model):
        return self.embeddings[model]


def reference_image(path, digest=None):
    """(SHA-256, BGR image or None if undecodable) of a reference, decoded from the bytes hashed.

    With `digest`, those bytes must have it: a file changed since submission is refused. cv2's
    imdecode applies EXIF orientation as imread does.
    """
    import cv2
    import numpy as np

    data = Path(input_path(path)).read_bytes()
    found = hashlib.sha256(data).hexdigest()
    if digest is not None and found != digest:
        raise ValueError(
            f"Reference {path} changed after submission (SHA-256 {found}, expected {digest}); "
            "analyze it again and resubmit"
        )
    return found, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def merge_embeddings(vectors, weights):
    """Fusion's source merge (EmbMergeMethodSelection "Mean", then L2 normalization, as in
    calculate_assigned_input_embedding) with explicit weights; equal weights are its mean."""
    total = sum(weights)
    mean = [
        sum(w * float(v[i]) for v, w in zip(vectors, weights)) / total
        for i in range(len(vectors[0]))
    ]
    return unit(mean)


def configure_faces(window, mappings):
    """Bind each mapping's target and weighted source embeddings; returns what each mapping used.

    Sources with weight 0 are excluded and never read. Omitted weights are equal.
    """
    import numpy as np
    from app.helpers.miscellaneous import ParametersDict

    def embeddings(path, digest, models):
        sha256, image = reference_image(path, digest)
        if image is None:
            raise ValueError(f"Face reference must be an image: {path}")
        tensor, boxes, points = detect_reference(window, image)
        if len(boxes) != 1:
            raise ValueError(
                f"Reference must contain exactly one detected face: {path} ({len(boxes)})"
            )
        recognize = window.function_worker.run_recognize_direct
        return sha256, {model: recognize(tensor, points[0], "Auto", model)[0] for model in models}

    used = []
    for index, mapping in enumerate(mappings):
        parameters = dict(window.default_parameters)
        parameters.update(mapping.get("parameters", {}))
        models = {
            window.control["RecognitionModelSelection"],
            window.function_worker.get_arcface_model(parameters["SwapModelSelection"]),
        }
        target_sha256, target = embeddings(
            mapping["target_face"], mapping.get("target_sha256"), models
        )
        paths = mapping["source_faces"]
        weights = mapping.get("source_weights") or [1] * len(paths)
        expected = mapping.get("source_sha256") or [None] * len(paths)
        sources, report = [], []
        for path, digest, weight in zip(paths, expected, weights):
            sha256 = digest
            if weight > 0:
                sha256, store = embeddings(path, digest, models)
                sources.append((store, weight))
            share = round(weight / sum(weights), 6)
            report.append({"path": path, "sha256": sha256, "weight": weight, "share": share})
        merged = {
            model: np.asarray(
                merge_embeddings([store[model] for store, _ in sources], [w for _, w in sources]),
                dtype=np.float32,
            )
            for model in models
        }
        face_id = str(index)
        window.target_faces[face_id] = TargetFace(face_id, target, merged)
        window.parameters[face_id] = ParametersDict(parameters, dict(window.default_parameters))
        used.append(
            {
                "mapping": index,
                "target_face": mapping["target_face"],
                "target_sha256": target_sha256,
                "weighted": mapping.get("source_weights") is not None,
                "sources": report,
            }
        )
    return used


def make_worker(window, sequential=False):
    """Sequential workers mirror Fusion's video pipeline: one SequentialDetector feeds
    precomputed, identity-filtered detections (optional ByteTrack and landmark EMA) in order."""
    from app.processors.video_utils.sequential_detector import SequentialDetector
    from app.processors.workers.frame_worker import FrameWorker

    class CheckedWorker(FrameWorker):
        """Records what Fusion's own matching and swapping decide, without extra inference."""

        swap_count = 0

        def _find_best_target_match(self, embedding, control, *args, **kwargs):
            match = super()._find_best_target_match(embedding, control, *args, **kwargs)
            # Fusion asks once per target for each face when swapping only best matches.
            if embedding is not None and not self.telemetry.scored(embedding):
                model = control["RecognitionModelSelection"]
                scores = {
                    face_id: float(
                        self.function_worker.findCosineDistance(
                            embedding, target.get_embedding(model)
                        )
                    )
                    for face_id, target in self.main_window.target_faces.items()
                }
                self.telemetry.match(embedding, match[0] and str(match[0].face_id), scores)
            return match

        def swap_core(self, *args, **kwargs):
            # The target embedding Fusion passes is the mapping's own array.
            embedding = kwargs.get("t_e")
            face_id = next(
                (
                    face_id
                    for face_id, target in self.main_window.target_faces.items()
                    if any(value is embedding for value in target.embeddings.values())
                ),
                None,
            )
            try:
                result = super().swap_core(*args, **kwargs)
            except Exception:
                self.telemetry.swap(face_id, failed=True)
                raise
            self.swap_count += 1
            self.telemetry.swap(face_id)
            return result

    worker = CheckedWorker(window, is_single_frame=not sequential)
    worker.parameters = window.parameters
    worker.telemetry = Telemetry(
        {
            face_id: float(window.parameters[face_id]["SimilarityThresholdSlider"])
            for face_id in window.target_faces
        },
        sequential,
    )
    # Both paths detect through the window's FunctionWorker: the frame worker per frame, or
    # the SequentialDetector before identity filtering.
    function_worker, run_detect = window.function_worker, window.function_worker.run_detect

    def counted_detect(*args, **kwargs):
        result = run_detect(*args, **kwargs)
        worker.telemetry.detected(0 if result[0] is None else len(result[0]))
        return result

    function_worker.run_detect = counted_detect
    worker.detector = None
    if sequential:
        worker.detector = SequentialDetector(window)
        worker.detector.reset_state()
    return worker


def process(worker, bgr, index):
    import numpy as np

    worker.frame = np.ascontiguousarray(bgr[..., ::-1])
    worker.frame_number = index
    control = worker.main_window.control
    # Mappings a segment leaves unbound are not requested in this frame.
    worker.telemetry.begin(index, worker.main_window.target_faces)
    # Fusion catches some per-face exceptions. Turn its logged errors into job failures.
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            if worker.detector is not None:
                (
                    worker.precomputed_bboxes,
                    worker.precomputed_kpss_5,
                    worker.precomputed_kpss,
                    worker.precomputed_kpss_203,
                ) = worker.detector.run(
                    worker.frame, control, worker.parameters, frame_number=index
                )
            result = worker.process_frame(control, threading.Event())
    finally:
        output = captured.getvalue()
        print(output, end="", flush=True)
    if "[ERROR]" in output or "Traceback (most recent call last)" in output:
        failed = ", ".join(worker.telemetry.failed())
        raise RuntimeError(
            f"Fusion reported a per-face processing error at source frame {index}"
            + (f" (mapping {failed})" if failed else "")
            + "; see worker.log"
        )
    worker.telemetry.end()
    return np.ascontiguousarray(result)


# Why a mapping was or was not swapped in a frame. Faces that reach Fusion's matching are scored
# against every mapping; others (detected but too small or without valid landmarks, or filtered
# out by the SequentialDetector) are unscored, and their scores are unknown.
REASONS = (
    "swapped",  # Fusion's swap step completed for at least one face of this mapping.
    "no_faces",  # The detector found no face: absence, not a failure.
    "unscored",  # No scored face cleared the threshold, but some detected faces were unscored.
    "below_threshold",  # Every detected face was scored, and none cleared the threshold.
    "claimed_by_other",  # A face cleared this threshold but matched another mapping better.
    "not_swapped",  # A face matched this mapping best, yet Fusion swapped none.
    "error",  # Fusion's swap step raised; the job fails.
    "not_requested",  # The mapping does not apply to this frame; excluded from coverage.
)
# Suspicious interval kinds, in review-image priority after never-swapped mappings.
INTERVALS = ("expected_missing", "dropout", "claimed_by_other", "not_swapped", "ambiguous")
REPORT_VERSION = 1
MAX_INTERVALS = 100  # Per mapping; counts stay exact, further intervals are only counted.
REVIEW_IMAGES = 6


class MappingTelemetry:
    """One mapping's per-frame outcomes as counters and suspicious intervals.

    Intervals are half-open source-frame ranges [start, end) of consecutive processed frames:
    `expected_missing` (unswapped inside an expected-visible range), `dropout` (unswapped
    frames, at least one with faces, between two swapped frames), `claimed_by_other`,
    `not_swapped` and `ambiguous`. Frames must arrive consecutively.
    """

    def __init__(self, threshold):
        self.threshold = threshold
        self.frames = dict.fromkeys(REASONS, 0)
        self.swaps = self.ambiguous = self.expected = self.expected_swapped = 0
        self.scores = []  # Count, sum, minimum and maximum of swapped faces' scores.
        self.near = None  # (score, frame) of the best-scoring unswapped frame.
        self.ranges, self.intervals, self.omitted, self.open = [], [], 0, {}
        self.after_swap = False

    def add(self, frame, reason, score, ambiguous, swaps):
        self.frames[reason] += 1
        self.swaps += swaps
        requested, swapped = reason != "not_requested", reason == "swapped"
        expected = requested and any(start <= frame < end for start, end in self.ranges)
        self.expected += expected
        self.expected_swapped += expected and swapped
        ambiguous = requested and ambiguous
        self.ambiguous += ambiguous
        if score is not None and swapped:
            count, total, low, high = self.scores or (0, 0.0, score, score)
            self.scores = (count + 1, total + score, min(low, score), max(high, score))
        elif score is not None and requested and (self.near is None or score > self.near[0]):
            self.near = (score, frame)
        for kind, active in (
            ("expected_missing", expected and not swapped),
            ("claimed_by_other", reason == "claimed_by_other"),
            ("not_swapped", reason == "not_swapped"),
            ("ambiguous", ambiguous),
        ):
            if active:
                self.extend(kind, frame, reason)
            elif kind in self.open:
                self.close(kind, frame)
        if swapped:
            if any(r != "no_faces" for r in self.open.get("dropout", {}).get("reasons", {})):
                self.close("dropout", frame)
            self.open.pop("dropout", None)
            self.after_swap = True
        elif not requested:
            self.open.pop("dropout", None)
            self.after_swap = False
        elif self.after_swap:
            self.extend("dropout", frame, reason)

    def extend(self, kind, frame, reason):
        run = self.open.setdefault(kind, {"start": frame, "reasons": {}})
        run["reasons"][reason] = run["reasons"].get(reason, 0) + 1

    def close(self, kind, end):
        run = self.open.pop(kind)
        if len(self.intervals) == MAX_INTERVALS:
            self.omitted += 1
            return
        self.intervals.append(
            {"kind": kind, "start_frame": run["start"], "end_frame": end, "reasons": run["reasons"]}
        )

    def finish(self, end):
        """Close runs at `end`, one past the last frame; a dropout needs a swap after it."""
        self.open.pop("dropout", None)
        for kind in list(self.open):
            self.close(kind, end)

    def state(self):
        """Everything but the expected ranges, which come from the request (checkpoints)."""
        return {key: value for key, value in vars(self).items() if key != "ranges"}

    def restore(self, state):
        for key, value in state.items():
            setattr(self, key, value)


class Telemetry:
    """Fusion's matching and swapping observed per processed frame, attributed to mappings.

    Mapping IDs are Fusion face IDs, the request's mapping index as a string. Scores are what
    Fusion's FunctionWorker.findCosineDistance gives each face that reaches matching; the best
    match is Fusion's own choice. Nothing is inferred again.
    """

    def __init__(self, thresholds, sequential=False):
        self.thresholds, self.sequential = thresholds, sequential
        self.mappings = {face_id: MappingTelemetry(t) for face_id, t in thresholds.items()}
        self.processed = self.with_faces = self.detections = self.scored_faces = 0
        self.frame = None

    def expect(self, ranges):
        """Expected-visible [start, end) source-frame ranges by mapping ID."""
        for face_id, values in ranges.items():
            self.mappings[face_id].ranges = values

    def begin(self, frame, requested=None):
        """Start a frame; mappings outside `requested` (default all) are not_requested."""
        self.frame, self.requested = frame, set(self.mappings if requested is None else requested)
        self.detections_now, self.faces, self.swaps, self.errors = None, {}, {}, {}

    def detected(self, count):
        if self.frame is not None:
            self.detections_now = count

    def scored(self, embedding):
        return id(embedding) in self.faces

    def match(self, embedding, best, scores):
        # The embedding is kept so its id cannot be reused within the frame.
        if self.frame is not None:
            self.faces[id(embedding)] = (embedding, best, scores)

    def swap(self, face_id, failed=False):
        if self.frame is not None:
            counts = self.errors if failed else self.swaps
            counts[face_id] = counts.get(face_id, 0) + 1

    def failed(self):
        return [face_id for face_id in self.errors if face_id is not None]

    def classify(self, face_id, detected):
        """(reason, score, ambiguous) for one mapping in the current frame."""
        threshold = self.thresholds[face_id]
        faces = [(best, scores) for _, best, scores in self.faces.values()]
        assigned = [scores[face_id] for best, scores in faces if best == face_id]
        eligible = [scores for _, scores in faces if scores.get(face_id, -1) >= threshold]
        # Observed-score ambiguity: several faces matched this mapping best, or a face clearing
        # this threshold also clears another mapping's.
        ambiguous = len(assigned) > 1 or any(
            score >= self.thresholds.get(other, math.inf)
            for scores in eligible
            for other, score in scores.items()
            if other != face_id
        )
        observed = [scores[face_id] for _, scores in faces if face_id in scores]
        score = max(assigned or observed, default=None)
        if face_id not in self.requested:
            reason = "not_requested"
        elif self.errors.get(face_id):
            reason = "error"
        elif self.swaps.get(face_id):
            reason = "swapped"
        elif not detected:
            reason = "no_faces"
        elif assigned:
            reason = "not_swapped"
        elif eligible:
            reason = "claimed_by_other"
        elif detected > len(faces):
            reason = "unscored"
        else:
            reason = "below_threshold"
        return reason, score, ambiguous

    def end(self):
        detected = max(self.detections_now or 0, len(self.faces))
        self.processed += 1
        self.with_faces += detected > 0
        self.detections += detected
        self.scored_faces += len(self.faces)
        for face_id, mapping in self.mappings.items():
            reason, score, ambiguous = self.classify(face_id, detected)
            mapping.add(self.frame, reason, score, ambiguous, self.swaps.get(face_id, 0))
        self.frame = None

    def finish(self, end):
        for mapping in self.mappings.values():
            mapping.finish(end)

    COUNTERS = ("processed", "with_faces", "detections", "scored_faces")

    def state(self):
        """Counters and open runs between frames, which a render checkpoint stores with each
        chunk so a resumed attempt reports the whole range as one uninterrupted run would."""
        state = {key: getattr(self, key) for key in self.COUNTERS}
        state["mappings"] = {face_id: m.state() for face_id, m in self.mappings.items()}
        return state

    def restore(self, state):
        if set(state["mappings"]) != set(self.mappings):
            raise ValueError("The checkpointed swap telemetry has other mappings")
        for key in self.COUNTERS:
            setattr(self, key, state[key])
        for face_id, values in state["mappings"].items():
            self.mappings[face_id].restore(values)


CONVENTIONS = (
    "Frames are zero-based source frame indices. Intervals are half-open: start_frame is the "
    "first frame and end_frame the first frame after it. Times are seconds, frame / fps on the "
    "source video's clock (start_time, end_time) or (frame - render start_frame) / fps in the "
    "output (output_start_time); previews have none."
)
EVIDENCE = {
    "per_frame": (
        "Fusion detects every frame. Faces below its minimum size or without valid landmarks "
        "are not recognized, so they are unscored."
    ),
    "sequential": (
        "The SequentialDetector detects every frame and passes on only faces that clear a "
        "mapping's threshold or continue a tracked face; the others are unscored."
    ),
}
SCORES = (
    "Scores are Fusion's similarity (100 - 50 x cosine distance) of each scored face against "
    "each mapping's target; swapped means Fusion's swap step completed, not that the result "
    "looks right. Unscored faces have unknown scores."
)


def rounded(value, digits=2):
    return None if value is None else round(value, digits)


def ratio(part, whole):
    return round(part / whole, 4) if whole else None


def frame_times(frame, fps, start):
    """Source-clock and output seconds of zero-based source frame `frame`, from the exact rate."""
    if fps is None:
        return None, None
    return round(float(frame / fps), 6), round(float((frame - start) / fps), 6)


def mapping_report(index, mapping, stats, fps, start):
    frames = dict(stats.frames)
    requested = sum(frames.values()) - frames["not_requested"]
    with_faces = requested - frames["no_faces"]
    count, total, low, high = stats.scores or (0, 0.0, None, None)
    near_score, near_frame = stats.near or (None, None)
    intervals = []
    for interval in sorted(stats.intervals, key=lambda i: (i["start_frame"], i["kind"])):
        first, end = interval["start_frame"], interval["end_frame"]
        start_time, output_time = frame_times(first, fps, start)
        intervals.append(
            {
                **interval,
                "frames": end - first,
                "start_time": start_time,
                "end_time": frame_times(end, fps, start)[0],
                "output_start_time": output_time,
            }
        )
    suspicious = {kind: 0 for kind in INTERVALS}
    for interval in intervals:
        suspicious[interval["kind"]] += 1
    return {
        "mapping": index,
        "target_face": mapping["target_face"],
        "threshold": stats.threshold,
        "expected_visible": [list(r) for r in stats.ranges],
        "frames": {
            "requested": requested,
            "with_faces": with_faces,
            "expected_visible": stats.expected,
            "expected_visible_swapped": stats.expected_swapped,
            **frames,
        },
        "swaps": stats.swaps,
        "ambiguous_frames": stats.ambiguous,
        "coverage": {
            "requested": ratio(frames["swapped"], requested),
            "with_faces": ratio(frames["swapped"], with_faces),
            "expected_visible": ratio(stats.expected_swapped, stats.expected),
        },
        "scores": {
            "swapped_min": rounded(low),
            "swapped_mean": rounded(total / count) if count else None,
            "swapped_max": rounded(high),
            "best_unswapped": rounded(near_score),
            "best_unswapped_frame": near_frame,
        },
        "suspicious": suspicious,
        "intervals": intervals,
        "intervals_omitted": stats.omitted,
    }


def report_warnings(mappings):
    """Advisory per-mapping warnings; they never change the job's terminal state."""
    found = []
    for m in mappings:
        index, frames = m["mapping"], m["frames"]
        if frames["requested"] and not frames["swapped"]:
            best = m["scores"]["best_unswapped"]
            detail = (
                f"best observed score {best:g} against threshold {m['threshold']:g}"
                if best is not None
                else "no face reached matching"
            )
            message = (
                f"Mapping {index} was never swapped in {frames['requested']} frames "
                f"({frames['with_faces']} with faces); {detail}"
            )
            found.append({"mapping": index, "code": "never_swapped", "message": message})
            continue
        missing = frames["expected_visible"] - frames["expected_visible_swapped"]
        if missing:
            message = (
                f"Mapping {index} was not swapped in {missing} of its "
                f"{frames['expected_visible']} expected-visible frames"
            )
            found.append({"mapping": index, "code": "expected_missing", "message": message})
        kinds = {k: v for k, v in m["suspicious"].items() if v and k != "expected_missing"}
        if kinds or m["intervals_omitted"]:
            listed = ", ".join(f"{v} {k}" for k, v in kinds.items())
            omitted = m["intervals_omitted"]
            message = f"Mapping {index} has suspicious intervals ({listed}" + (
                f", {omitted} more" if omitted else ""
            )
            found.append(
                {"mapping": index, "code": "suspicious_intervals", "message": message + ")"}
            )
    return found


def swap_report(request, telemetry, fps, start):
    """The versioned swap report of a successful preview or render (review images added later)."""
    options = request.get("report") or {}
    mappings = [
        mapping_report(index, mapping, telemetry.mappings[str(index)], fps, start)
        for index, mapping in enumerate(request["mappings"])
    ]
    path = "sequential" if telemetry.sequential else "per_frame"
    return {
        "version": REPORT_VERSION,
        "complete": True,
        "kind": request["kind"],
        "media_path": request["media_path"],
        "fps": None if fps is None else str(fps),
        "start_frame": start,
        "end_frame": start + telemetry.processed,
        "conventions": CONVENTIONS,
        "detection_path": path,
        "evidence": {"detection": EVIDENCE[path], "scores": SCORES},
        "frames": {
            "processed": telemetry.processed,
            "with_faces": telemetry.with_faces,
            "faces_detected": telemetry.detections,
            "faces_scored": telemetry.scored_faces,
        },
        "options": {
            "review_images": options.get("review_images", REVIEW_IMAGES),
            "expected_visible": options.get("expected_visible", []),
            "min_coverage": options.get("min_coverage"),
        },
        "mappings": mappings,
        "warnings": report_warnings(mappings),
        "review_images": [],
    }


def coverage_shortfalls(report):
    """Mappings below min_coverage: of expected-visible frames when the mapping has ranges,
    else of requested frames. A mapping with no such frames has nothing to cover."""
    minimum = report["options"]["min_coverage"]
    found = []
    for m in report["mappings"] if minimum is not None else ():
        basis = "expected_visible" if m["expected_visible"] else "requested"
        value, frames = m["coverage"][basis], m["frames"]
        if value is not None and value < minimum:
            swapped = frames["swapped" if basis == "requested" else "expected_visible_swapped"]
            name = basis.replace("_", "-")
            found.append(f"mapping {m['mapping']} {swapped} of {frames[basis]} {name} frames")
    return found


def review_frames(report, limit):
    """Up to `limit` (mapping, frame, reason): each never-swapped mapping's best-scoring frame,
    then the middle frame of suspicious intervals by kind priority, longest first."""
    candidates = []
    for m in report["mappings"]:
        frame = m["scores"]["best_unswapped_frame"]
        if m["frames"]["requested"] and not m["frames"]["swapped"] and frame is not None:
            candidates.append((0, 0, m["mapping"], frame, "never_swapped"))
        for interval in m["intervals"]:
            middle = interval["start_frame"] + (interval["frames"] - 1) // 2
            priority = 1 + INTERVALS.index(interval["kind"])
            candidates.append(
                (priority, -interval["frames"], m["mapping"], middle, interval["kind"])
            )
    chosen, seen = [], set()
    for *_, mapping, frame, reason in sorted(candidates):
        if len(chosen) < limit and (mapping, frame) not in seen:
            seen.add((mapping, frame))
            chosen.append((mapping, frame, reason))
    return chosen


def review_image(path, before, after, label):
    """Write the source and output frames side by side under a label bar."""
    import cv2
    import numpy as np

    pair = np.hstack([before, after])
    scale = min(1.0, 1920 / pair.shape[1])
    if scale < 1:
        size = (round(pair.shape[1] * scale), round(pair.shape[0] * scale))
        pair = cv2.resize(pair, size, interpolation=cv2.INTER_AREA)
    bar = np.full((40, pair.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    save_image(path, np.vstack([bar, pair]))


def report_summary(report):
    """The compact part of a report that job_status returns."""
    mappings = []
    for m in report["mappings"]:
        entry = {
            "mapping": m["mapping"],
            "swapped_frames": m["frames"]["swapped"],
            "requested_frames": m["frames"]["requested"],
            "coverage": m["coverage"]["requested"],
            "ambiguous_frames": m["ambiguous_frames"],
            "suspicious_intervals": len(m["intervals"]) + m["intervals_omitted"],
        }
        if m["expected_visible"]:
            entry["expected_visible_coverage"] = m["coverage"]["expected_visible"]
        mappings.append(entry)
    return {
        "artifact": "swap-report.json",
        "version": report["version"],
        "mappings": mappings,
        "warnings": [w["message"] for w in report["warnings"]],
        "review_images": [r["artifact"] for r in report["review_images"]],
    }


def expected_ranges(options, end):
    """Expected-visible ranges by mapping ID; each must end by `end`, one past the last frame."""
    ranges = {}
    for item in options.get("expected_visible", []):
        if item["end_frame"] > end:
            raise Refusal(
                "expected_visible_beyond_range",
                f"expected_visible range {item['start_frame']}-{item['end_frame']} for mapping "
                f"{item['mapping']} ends after the rendered range, which ends at frame {end}",
            )
        ranges.setdefault(str(item["mapping"]), []).append((item["start_frame"], item["end_frame"]))
    return {face_id: sorted(values) for face_id, values in ranges.items()}


def save_image(path, image):
    import cv2

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def contact_sheet(path, tiles, size=200, columns=6):
    """Write labelled face crops as one grid image, so a single view shows every face.

    Tiles are (label, image, verified); unverified ones are framed and labelled in red."""
    import cv2
    import numpy as np

    cell, label = size + 8, 36
    rows = -(-len(tiles) // columns)
    sheet = np.full((rows * (cell + label), min(len(tiles), columns) * cell, 3), 32, np.uint8)
    for number, (text, image, verified) in enumerate(tiles):
        top, left = [v * c for v, c in zip(divmod(number, columns), (cell + label, cell))]
        scale = size / max(image.shape[:2])
        width, height = [max(1, round(v * scale)) for v in (image.shape[1], image.shape[0])]
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        y, x = top + 4 + (size - height) // 2, left + 4 + (size - width) // 2
        sheet[y : y + height, x : x + width] = cv2.resize(
            image, (width, height), None, 0, 0, interpolation
        )
        colour = (255, 255, 255) if verified else (0, 0, 255)
        if not verified:
            corner = (left + cell - 2, top + cell + label - 2)
            cv2.rectangle(sheet, (left + 1, top + 1), corner, colour, 2)
        cv2.putText(
            sheet,
            text,
            (left + 8, top + cell + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            colour,
            2,
            cv2.LINE_AA,
        )
    save_image(path, sheet)


class Timeline:
    """Presentation times (seconds, ascending) of the first video stream's frames.

    times[n] is decoded frame n on the stream's own clock, whatever the container start.
    """

    def __init__(self, times):
        if not times:
            raise Refusal("no_video_frames", "No video timestamps found")
        self.times = times
        self.start = times[0]

    def constant(self, fps, tolerance):
        """Whether frame n lies n/fps after the start for every frame read."""
        return all(abs(t - self.start - n / fps) <= tolerance for n, t in enumerate(self.times))

    @property
    def duration(self):
        """Seconds through the end of the last frame, taken to last the average interval."""
        count = len(self.times)
        return (self.times[-1] - self.start) * count / (count - 1) if count > 1 else None


def timeline(source, end=None):
    """Read the Timeline of the first `end` frames (all if None) from packets, without decoding.

    Packets arrive in decode order, so a bounded read takes enough extra packets to cover
    B-frame reordering, then sorts. Discarded (edit-list) packets are skipped. A packet without
    a timestamp leaves frame order unknown.
    """
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if end is not None:
        command += ["-read_intervals", f"%+#{end + 32}"]
    command += ["-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", *ffmpeg_input(source)]
    output = subprocess.check_output(command, text=True, timeout=120)
    times = []
    for line in output.splitlines():
        value, _, flags = line.partition(",")
        value = value.strip()
        if not value or "D" in flags:
            continue
        if value == "N/A":
            message = "Video packets lack timestamps; run normalize_media first"
            raise Refusal("missing_timestamps", message, normalize=True)
        times.append(float(value))
    return Timeline(sorted(times)[:end])


def validate_timing(source, fps, stream, end=None):
    """Require CFR timestamps from zero over the first `end` frames (all if None); metadata can
    hide VFR. Returns the number of frames checked."""
    timing = timeline(source, end)
    tick = float(Fraction(stream["time_base"]))
    tolerance = max(0.00001, tick * 1.1)
    if abs(timing.start) > tolerance:
        message = "Video timestamps must start at zero; run normalize_media first"
        raise Refusal("nonzero_start", message, normalize=True)
    if not timing.constant(float(fps), tolerance):
        message = "Variable frame timestamps detected; run normalize_media first"
        raise Refusal("variable_timestamps", message, normalize=True)
    return len(timing.times)


def audio_window(source, start, duration):
    """Input arguments reading `duration` seconds of `source` from `start` on the video clock.

    Without -seek_timestamp FFmpeg offsets -ss by the container start, which an earlier audio
    stream can move.
    """
    return ["-seek_timestamp", "1", "-ss", str(start), "-t", str(duration), *ffmpeg_input(source)]


def audio_overlaps(source, metadata, start, duration):
    """Whether the first audio stream, the one muxed, has samples in the video interval.

    It can end before the interval or begin after it. Decoding one frame through the mux's
    own input window avoids durations that metadata can omit or offset by encoder priming.
    """
    if not any(s["codec_type"] == "audio" for s in metadata["streams"]):
        return False
    command = ["ffmpeg", "-v", "error", "-nostdin", *audio_window(source, start, duration)]
    command += ["-map", "0:a:0", "-frames:a", "1", "-f", "s16le", "pipe:1"]
    return bool(subprocess.run(command, stdout=subprocess.PIPE, check=True).stdout)


def mux(video, output, audio=None, options=()):
    """Copy the video of input arguments `video` into `output` with faststart, adding `audio` =
    (source, start, duration).

    The first audio stream of that source window becomes AAC; silence pads audio that starts
    later than the video, so it stays aligned with the first frame.
    """
    inputs, outputs = list(video), ["-map", "0:v:0", "-c:v", "copy"]
    if audio:
        inputs += audio_window(*audio)
        outputs += ["-map", "1:a:0", "-af", "aresample=async=1:first_pts=0"]
        outputs += ["-c:a", "aac", "-b:a", "192k"]
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", *inputs, *outputs, *options]
    subprocess.run(command + ["-movflags", "+faststart", str(output)], check=True)


def output_audio(metadata, expected, end, missing):
    """The output's audio stream, if any. Expected audio must be present, and no audio may run
    more than one AAC frame (the encoder's padding) past the video's end at `end` seconds."""
    audio = next((s for s in metadata["streams"] if s["codec_type"] == "audio"), None)
    if audio is None:
        if expected:
            raise RuntimeError(missing)
        return None
    finish = float(audio.get("start_time") or 0) + float(audio.get("duration") or 0)
    if finish > end + 1024 / int(audio["sample_rate"]) + 1e-5:
        raise RuntimeError(f"Audio ends at {finish:.3f} s, after the video's end at {end:.3f} s")
    return audio


SEGMENT_CONVENTION = (
    "Zero-based source frame indices in decoded order, in half-open ranges: start_frame is the "
    "first frame and end_frame the first frame after it. Segments are clipped to the rendered "
    "frames; applied gives the frames each one covers."
)
GAPS = "Frames outside every segment use the base mappings, parameters and controls."
BOUNDARIES = (
    "Every run starts with fresh temporal state. With tracking or keypoint smoothing on, the "
    "SequentialDetector is reset at a run's first frame (ByteTrack tracks and history, identity "
    "memory and keypoint smoothing), and ByteTrack is re-created with the run's controls. The "
    "per-frame path keeps no state between frames. Passthrough frames are encoded as decoded, "
    "without detection."
)


class Segments:
    """A render's segment plan applied frame by frame (DESIGN.md, Segments).

    A frame's run follows from its index alone. Entering a run other than passthrough binds its
    mappings with their parameters and the segment's overrides, sets its controls and the
    thresholds telemetry classifies with, and resets the sequential detector.
    """

    def __init__(self, worker, runs):
        window = worker.main_window
        self.worker, self.runs = worker, runs
        self.starts = [run["start_frame"] for run in runs]
        self.targets = dict(window.target_faces)
        self.parameters = {face_id: copy.copy(p) for face_id, p in window.parameters.items()}
        self.control = dict(window.control)
        self.thresholds = dict(worker.telemetry.thresholds)
        self.current = None
        self.stats = [{"frames": 0, "swaps": 0, "swapped_frames": 0} for _ in runs]

    def at(self, frame):
        """The number of the run holding `frame`; entering it unless it is the current run."""
        number = bisect.bisect_right(self.starts, frame) - 1
        if number != self.current:
            self.current = number
            if not self.runs[number]["passthrough"]:
                self.enter(self.runs[number])
        return number

    def settings(self, run):
        """(controls, {mapping ID: parameters}) a run swaps with."""
        parameters = {}
        for index in run["mappings"]:
            values = copy.copy(self.parameters[str(index)])
            values.update(run["parameters"])
            parameters[str(index)] = values
        return {**self.control, **run["controls"]}, parameters

    def enter(self, run):
        window = self.worker.main_window
        controls, parameters = self.settings(run)
        # Fusion and the worker read these live, so they are refilled in place.
        window.target_faces.clear()
        window.target_faces.update((face_id, self.targets[face_id]) for face_id in parameters)
        window.parameters.clear()
        window.parameters.update(parameters)
        window.control.update(controls)
        thresholds = self.worker.telemetry.thresholds
        thresholds.update(self.thresholds)
        for face_id, values in parameters.items():
            thresholds[face_id] = float(values["SimilarityThresholdSlider"])
        if self.worker.detector is not None:
            self.worker.detector.reset_state()

    def count(self, number, swaps):
        stats = self.stats[number]
        stats["frames"] += 1
        stats["swaps"] += swaps
        stats["swapped_frames"] += swaps > 0

    def document(self, segments, start, end):
        """plan.json: each segment with the frames it covers, and every run with the effective
        allowlisted controls and per-mapping parameters it swaps with."""
        applied = {run["segment"]: run for run in self.runs if run["segment"] is not None}
        entries = []
        for segment in segments:
            run = applied[segment["id"]]
            span = [run["start_frame"], run["end_frame"]]
            clipped = span != [segment["start_frame"], segment["end_frame"]]
            entries.append({**segment, "applied": span, "clipped": clipped})
        runs = []
        for run in self.runs:
            controls, parameters = self.settings(run)
            effective = {
                "controls": {key: controls[key] for key in sorted(CONTROLS) if key in controls},
                "parameters": {
                    face_id: {key: values[key] for key in sorted(PARAMETERS) if key in values}
                    for face_id, values in parameters.items()
                },
            }
            runs.append({**run, "effective": None if run["passthrough"] else effective})
        return {
            "version": 1,
            "conventions": SEGMENT_CONVENTION,
            "gaps": GAPS,
            "boundaries": BOUNDARIES,
            "start_frame": start,
            "end_frame": end,
            "segments": entries,
            "runs": runs,
        }

    def summary(self):
        """Every run with its frame and swap counts, for the result and the swap report."""
        keys = ("segment", "start_frame", "end_frame", "passthrough", "mappings")
        return [{**{k: run[k] for k in keys}, **s} for run, s in zip(self.runs, self.stats)]

    def warnings(self):
        """Advisory: a segment that requested swaps and swapped no face."""
        found = []
        for run, stats in zip(self.runs, self.stats):
            if run["segment"] is not None and not run["passthrough"] and not stats["swaps"]:
                message = (
                    f"Segment {run['segment']} [{run['start_frame']}, {run['end_frame']}) "
                    f"swapped no face in its {stats['frames']} frames"
                )
                found.append(
                    {"segment": run["segment"], "code": "segment_never_swapped", "message": message}
                )
        return found


def clip_range(source, request):
    """(metadata, video stream, rate, start, frame count) of the CFR range a render processes:
    `start_frame` and at most `max_frames` frames (None: through the end). Renders, comparison
    clips and preflight share these checks, made before any frame is loaded: CFR metadata and
    timestamps over the requested range only, a nonempty range, expected-visible ranges within
    it, even dimensions and supported colors."""
    metadata = probe(source)
    stream = video_stream(metadata)
    fps = Fraction(stream["avg_frame_rate"])
    if fps <= 0 or fps != Fraction(stream["r_frame_rate"]):
        message = "Render requires constant frame rate video; run normalize_media first"
        raise Refusal("variable_frame_rate", message, normalize=True)
    start = request.get("start_frame", 0)
    limit = request.get("max_frames")
    total = validate_timing(source, fps, stream, start + limit if limit else None)
    expected = min(limit or total, total - start)
    if expected <= 0:
        raise Refusal("empty_range", "Requested frame range is empty")
    expected_ranges(request.get("report") or {}, start + expected)
    if int(stream["width"]) % 2 or int(stream["height"]) % 2:
        message = "H.264 output requires even frame dimensions; run normalize_media first"
        raise Refusal("odd_dimensions", message, normalize=True)
    output_color(stream)  # Rejects HDR and matrices the scaler lacks, as encoding would.
    return metadata, stream, fps, start, expected


def preview_source(source, frame):
    """The checks a preview makes of its media before decoding, which preflight also runs: an
    image has only frame 0; a video's frame must exist (timestamps up to it are read) and its
    colors must be supported. Returns the video stream, or None for an image."""
    metadata = probe(source)
    if metadata.get("format", {}).get("format_name", "").endswith(("_pipe", "image2")):
        if frame:
            raise Refusal("frame_out_of_range", "Images only have frame 0")
        return None
    stream = video_stream(metadata)
    color(stream)
    if frame:
        count = len(timeline(source, frame + 1).times)
        if frame >= count:
            message = f"Frame {frame} is out of range; the video has {count} frames"
            raise Refusal("frame_out_of_range", message)
    return stream


def stop_encoder(process):
    """Kill an encoder that is still running and close its pipe; the worker may go on."""
    if process is None:
        return
    if process.poll() is None:
        process.kill()
        process.wait()
    with contextlib.suppress(OSError):
        process.stdin.close()


class Encoder:
    """One H.264 encode of a whole clip into a video-only file (comparison clips)."""

    def __init__(self, path):
        self.path, self.process = path, None

    def begin(self, stream, fps, start, end, worker, plan):
        self.process = subprocess.Popen(
            encode_command(stream, fps, self.path), stdin=subprocess.PIPE
        )
        return start

    def write(self, index, frame):
        self.process.stdin.write(frame.tobytes())

    def finish(self):
        """Input arguments of the finished video."""
        self.process.stdin.close()
        if self.process.wait(timeout=120) != 0:
            raise RuntimeError("FFmpeg video encoding failed")
        return ffmpeg_input(self.path)

    def close(self):
        stop_encoder(self.process)

    def cleanup(self):
        self.path.unlink()


# Render checkpoints (DESIGN.md, Checkpoints).
CHUNK_FRAMES = 250  # Source frames per chunk; libx264's default keyframe interval.
CHUNKING = (
    "Chunk n holds source frames [start_frame + n * chunk_frames, start_frame + (n + 1) * "
    "chunk_frames), clipped to the rendered range, encoded alone as video-only H.264 that "
    "starts with a keyframe. Chunks are rendered in order, and the output joins them by stream "
    "copy in index order."
)
# Codec parameters, the decoder configuration (SPS and PPS) included, that must be identical
# for chunks to join without re-encoding.
CODEC_KEYS = (
    "codec_name",
    "profile",
    "level",
    "width",
    "height",
    "pix_fmt",
    "sample_aspect_ratio",
    "color_range",
    "color_space",
    "color_transfer",
    "color_primaries",
    "time_base",
    "r_frame_rate",
    "extradata_hash",
)


class Checkpoint:
    """A render's validated chunks and the manifest that commits them (DESIGN.md, Checkpoints).

    Each chunk is published after its frame count, timing and codec parameters are checked,
    with the render's accounting after its last frame. A resumed attempt starts from a copy of
    the earlier manifest (inherited.json) and hard links of its chunks, both made at
    submission, and reuses the longest prefix of chunks that verify; it renders the rest.
    """

    def __init__(self, directory, request, record=None):
        self.job_id, self.request, self.record = directory.name, request, record
        self.root = directory / "checkpoint"
        self.chunks = self.root / "chunks"
        self.chunks.mkdir(parents=True, exist_ok=True)
        self.encoder = self.inherited = None
        self.reused = []
        resume = request.get("resume")
        if resume:
            source = resume["from_job_id"]
            try:
                self.inherited = read_checkpoint(self.root / "inherited.json")
            except ValueError as error:
                raise Refusal("checkpoint_unusable", f"Cannot resume {source}: {error}") from None
            problems = resume_problems(self.inherited, request, record)
            if problems:
                raise Refusal(
                    "checkpoint_incompatible",
                    f"Cannot reuse the chunks of {source}:\n- "
                    + "\n- ".join(problems)
                    + "\nSubmit a fresh render with submit_swap instead",
                )

    def bounds(self, number):
        low = self.start + number * self.size
        return low, min(low + self.size, self.end)

    def begin(self, stream, fps, start, end, worker, plan):
        """Write the manifest with the verified inherited prefix, restore the accounting at its
        end and return the first frame left to render."""
        self.stream, self.fps, self.start, self.end = stream, fps, start, end
        self.worker, self.plan = worker, plan
        self.size = self.inherited["chunk_frames"] if self.inherited else CHUNK_FRAMES
        controls = self.record["settings"]["controls"] if self.record else {}
        self.manifest = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "job_id": self.job_id,
            "created_at": time.time(),
            "resumed_from": (self.request.get("resume") or {}).get("from_job_id"),
            "intent_sha256": intent_digest(self.request),
            "range": {
                "start_frame": start,
                "end_frame": end,
                "frames": end - start,
                "fps": str(fps),
            },
            "chunk_frames": self.size,
            "chunking": CHUNKING,
            "temporal": bool(temporal(controls or self.request.get("controls") or {})),
            "runtime": ffmpeg_version(),
            "codec": None,
            "provenance": self.record,
            "chunks": [],
        }
        state = None
        if self.inherited:
            if self.inherited["range"] != self.manifest["range"]:
                raise Refusal(
                    "checkpoint_incompatible",
                    f"The checkpointed range {self.inherited['range']} differs from this "
                    f"render's {self.manifest['range']}; submit a fresh render instead",
                )
            for number, entry in enumerate(self.inherited["chunks"]):
                found = self.trusted(number, entry)
                if found is None:
                    break  # Later chunks' accounting builds on this one, so they are redone too.
                state, self.reused = found, self.reused + [entry]
            if self.reused:
                self.manifest["codec"] = self.inherited["codec"]
        self.manifest["chunks"] = list(self.reused)
        kept = {entry[key] for entry in self.reused for key in ("file", "state")}
        for path in self.chunks.iterdir():
            if path.name not in kept:
                path.unlink()  # Partial writes and unlisted or untrusted chunks.
        atomic_json(self.root / "manifest.json", self.manifest)
        if state is None:
            return start
        worker.swap_count = state["swap_count"]
        worker.telemetry.restore(state["telemetry"])
        if plan:
            if len(state["runs"]) != len(plan.stats):
                raise ValueError("The checkpointed segment counts do not match the plan")
            plan.stats = [dict(stats) for stats in state["runs"]]
        return state["end_frame"]

    def trusted(self, number, entry):
        """The accounting a listed chunk ends with, if its files have the recorded bytes."""
        low, high = self.bounds(number)
        name = f"{number:05d}"
        expected = {
            "index": number,
            "start_frame": low,
            "end_frame": high,
            "frames": high - low,
            "file": f"{name}.mp4",
            "state": f"{name}.json",
        }
        if not isinstance(entry, dict) or low >= self.end:
            return None
        if any(entry.get(key) != value for key, value in expected.items()):
            return None
        video, record = self.chunks / expected["file"], self.chunks / expected["state"]
        try:
            if video.is_symlink() or record.is_symlink() or video.stat().st_size != entry["bytes"]:
                return None
            if file_sha256(video) != entry["sha256"]:
                return None
            if file_sha256(record) != entry["state_sha256"]:
                return None
            state = read_json(record)
        except (OSError, ValueError, KeyError):
            return None
        return state if isinstance(state, dict) and state.get("end_frame") == high else None

    def write(self, index, frame):
        number = (index - self.start) // self.size
        if self.encoder is None:
            self.partial = self.chunks / f"{number:05d}.partial.mp4"
            command = encode_command(self.stream, self.fps, self.partial)
            self.encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        self.encoder.stdin.write(frame.tobytes())
        if index + 1 == self.bounds(number)[1]:
            self.publish(number)

    def publish(self, number):
        """Validate a finished chunk, then publish it, its accounting and its manifest entry, each
        atomically and in that order. A crash between steps leaves a chunk the manifest does not
        list, which a resumed attempt deletes and renders again."""
        low, high = self.bounds(number)
        encoder, self.encoder = self.encoder, None
        encoder.stdin.close()
        if encoder.wait(timeout=120) != 0:
            raise RuntimeError(f"FFmpeg encoding of chunk {number} failed")
        codec = self.validate(number, high - low)
        known = self.manifest["codec"]
        if known is None:
            self.manifest["codec"] = codec
        elif codec != known:
            keys = ", ".join(key for key in CODEC_KEYS if codec.get(key) != known.get(key))
            raise RuntimeError(
                f"Chunk {number} differs from the earlier chunks in {keys}, so they cannot be "
                "joined without re-encoding; submit a fresh render"
            )
        with self.partial.open("rb") as stream:
            os.fsync(stream.fileno())  # The bytes are durable before the chunk is named.
        digest, size = file_sha256(self.partial), self.partial.stat().st_size
        name = f"{number:05d}"
        video, record = self.chunks / f"{name}.mp4", self.chunks / f"{name}.json"
        os.replace(self.partial, video)
        state = {
            "chunk": number,
            "end_frame": high,
            "swap_count": self.worker.swap_count,
            "telemetry": self.worker.telemetry.state(),
            "runs": self.plan.stats if self.plan else None,
        }
        atomic_json(record, state)
        self.manifest["chunks"].append(
            {
                "index": number,
                "start_frame": low,
                "end_frame": high,
                "frames": high - low,
                "file": video.name,
                "sha256": digest,
                "bytes": size,
                "state": record.name,
                "state_sha256": file_sha256(record),
                "job_id": self.job_id,
                "published_at": time.time(),
            }
        )
        atomic_json(self.root / "manifest.json", self.manifest)

    def validate(self, number, frames):
        """A chunk's codec parameters, after checking its frame count, constant-rate timestamps
        from zero, dimensions and pixel aspect."""
        command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams"]
        command += ["-show_data_hash", "SHA256", "-of", "json"]
        command += ffmpeg_input(self.partial, ffprobe=True)
        video = json.loads(subprocess.check_output(command, timeout=60))["streams"][0]
        try:
            counted = validate_timing(self.partial, self.fps, video)
        except ValueError as error:
            raise RuntimeError(f"Chunk {number} failed timestamp validation: {error}") from None
        if int(video.get("nb_frames") or 0) != frames or counted != frames:
            raise RuntimeError(f"Chunk {number} has {counted} frames, expected {frames}")
        size = (int(video["width"]), int(video["height"]))
        if size != (int(self.stream["width"]), int(self.stream["height"])) or sample_aspect(
            video
        ) != sample_aspect(self.stream):
            raise RuntimeError(f"Chunk {number} has other dimensions or pixel aspect")
        return {key: video.get(key) for key in CODEC_KEYS}

    def finish(self):
        """Input arguments joining every chunk by stream copy. Each chunk's listed duration is
        rounded so the running total stays exact to the microsecond; FFmpeg's own sum of
        rounded chunk durations drifts by a tick every few dozen chunks at NTSC rates."""
        chunks = self.manifest["chunks"]
        count = -(-(self.end - self.start) // self.size)
        if [entry["index"] for entry in chunks] != list(range(count)):
            raise RuntimeError("The checkpoint does not cover the rendered range")
        lines, edge = ["ffconcat version 1.0"], 0
        for entry in chunks:
            following = round(Fraction(entry["end_frame"] - self.start) / self.fps * 10**6)
            span, edge = following - edge, following
            lines += [
                f"file 'chunks/{entry['file']}'",
                f"duration {span // 10**6}.{span % 10**6:06d}",
            ]
        listing = self.root / "concat.txt"
        listing.write_text("\n".join(lines) + "\n")
        return concat_input(listing)

    def close(self):
        stop_encoder(self.encoder)

    def cleanup(self):
        """Chunks stay until the supervisor publishes completion, then it removes them."""

    def summary(self):
        return {
            "chunks": len(self.manifest["chunks"]),
            "chunk_frames": self.size,
            "reused_chunks": len(self.reused),
            "reused_frames": sum(entry["frames"] for entry in self.reused),
            "resumed_from": self.manifest["resumed_from"],
        }


def render_video(window, request, directory, worker, checkpoint=None):
    artifacts = directory / "artifacts"
    checkpoint = checkpoint or Checkpoint(directory, request)
    clip = encode_range(request, directory, worker, "result.mp4", checkpoint=checkpoint)
    report, fps, start = clip.report, clip.fps, clip.start
    # The first processed frame, after any leading passthrough, from the source and the
    # published output, so an interrupted and resumed render shows the same pair.
    runs = clip.plan.runs if clip.plan else []
    shown = next((run["start_frame"] for run in runs if not run["passthrough"]), start)
    save_image(artifacts / "before.jpg", read_frame(clip.source, clip.stream, shown))
    save_image(artifacts / "preview.jpg", read_frame(clip.output, clip.video, shown - start))
    # Review pairs come from the source and the published output, decoded at the same frame.
    limit = report["options"]["review_images"]
    for number, (mapping, frame, reason) in enumerate(review_frames(report, limit), 1):
        name = f"review-{number:02d}.jpg"
        seconds, output_seconds = frame_times(frame, fps, start)
        label = f"mapping {mapping} | frame {frame} | {seconds:.3f} s | {reason} | before, after"
        before = read_frame(clip.source, clip.stream, frame)
        after = read_frame(clip.output, clip.video, frame - start)
        review_image(artifacts / name, before, after, label)
        report["review_images"].append(
            {
                "artifact": name,
                "mapping": mapping,
                "frame": frame,
                "time": seconds,
                "output_time": output_seconds,
                "reason": reason,
            }
        )
    atomic_json(artifacts / "swap-report.json", report)
    names = ["plan.json"] if clip.plan else []
    names += [r["artifact"] for r in report["review_images"]]
    result = {
        "frames": clip.frames,
        "fps": str(fps),
        "duration": clip.duration,
        "swapped_faces": worker.swap_count,
        "audio": clip.audio,
        "swap_report": report_summary(report),
        "checkpoint": checkpoint.summary(),
        "artifacts": ["result.mp4", "preview.jpg", "before.jpg", "swap-report.json", *names],
    }
    if clip.plan:
        result["plan"] = report["plan"]
    return result


def encode_range(request, directory, worker, name, observe=None, progress=None, checkpoint=None):
    """Process a render request's range into the validated video artifact `name`.

    `observe(offset, source frame, swapped frame)` sees each processed frame before it is
    encoded; passthrough frames of a segmented render are encoded as decoded, unobserved.
    `progress` adds fields to each progress record (stage "rendering" by default). With a
    `checkpoint` the frames are encoded in chunks, starting after any it reuses; otherwise in
    one piece. Returns the clip's source, streams, rate, range, audio, segment plan (None for a
    uniform render) and swap report (review images not yet added).
    """
    source = input_path(request["media_path"])
    metadata, stream, fps, start, expected = clip_range(source, request)
    end = start + expected
    options = request.get("report") or {}
    telemetry = worker.telemetry
    telemetry.expect(expected_ranges(options, end))
    segments, plan = request.get("segments"), None
    artifacts = directory / "artifacts"
    if segments:
        # The whole plan is resolved, and refused if a segment misses the frames, before work.
        runs = segment_plan(segments, start, end, len(request["mappings"]))
        plan = Segments(worker, runs)
        atomic_json(artifacts / "plan.json", plan.document(segments, start, end))
    sink = checkpoint or Encoder(directory / f"{Path(name).stem}.video-only.mp4")
    width, height = int(stream["width"]), int(stream["height"])
    frames = (frame for frame in ())
    try:
        first = sink.begin(stream, fps, start, end, worker, plan)
        if first < end:
            frames = decode(source, stream, first, end - first)
        started = time.monotonic()
        for index in range(first, end):
            frame = next(frames, None)
            if frame is None:
                raise RuntimeError(f"Input decoding stopped at {index}, expected {expected} frames")
            number = plan.at(index) if plan else None
            swaps = worker.swap_count
            if plan and plan.runs[number]["passthrough"]:
                # Decoded pixels go to the encoder unchanged, and no mapping is requested.
                telemetry.begin(index, ())
                telemetry.end()
                swapped = frame
            else:
                swapped = process(worker, frame, index)
                if swapped.shape[:2] != (height, width):
                    raise RuntimeError("Fusion changed the frame dimensions")
                if observe:
                    observe(index - start, frame, swapped)
            if plan:
                plan.count(number, worker.swap_count - swaps)
            rate = (index + 1 - first) / max(time.monotonic() - started, 1e-6)
            atomic_json(
                directory / "progress.json",
                {
                    "stage": "rendering",
                    **(progress or {}),
                    "frames": index + 1 - start,
                    "total_frames": expected,
                    "swapped_faces": worker.swap_count,
                    "frames_per_second": round(rate, 2),
                    "eta_seconds": round((end - index - 1) / rate, 1),
                    **({"segment": plan.runs[number]["segment"]} if plan else {}),
                    **({"reused_frames": first - start} if checkpoint else {}),
                },
            )
            sink.write(index, swapped)
        joined = sink.finish()
    finally:
        frames.close()
        sink.close()  # A comparison's worker goes on to other variants, so nothing may leak.
    if worker.swap_count == 0:
        # Passthrough frames request no swap; the frames that do must swap a face somewhere.
        raise RuntimeError(
            "No faces matched the target references in the requested range"
            + (" outside passthrough segments" if plan else "")
        )
    telemetry.finish(end)
    report = swap_report(request, telemetry, fps, start)
    if plan:
        report["plan"] = {"artifact": "plan.json", "runs": plan.summary()}
        report["warnings"] += plan.warnings()
    shortfalls = coverage_shortfalls(report)
    if shortfalls:
        raise RuntimeError(
            f"Swap coverage is below min_coverage {report['options']['min_coverage']:g}: "
            + "; ".join(shortfalls)
        )
    output = artifacts / name
    duration = float(Fraction(expected, 1) / fps)
    offset = float(Fraction(start, 1) / fps)
    audio = audio_overlaps(source, metadata, offset, duration)
    mux(joined, output, (source, offset, duration) if audio else None)
    verified = probe(output)
    video = next(s for s in verified["streams"] if s["codec_type"] == "video")
    if int(video["nb_frames"]) != expected or abs(float(video["duration"]) - duration) > 1 / float(
        fps
    ):
        raise RuntimeError("Encoded frame count or duration validation failed")
    try:
        # Every frame at its exact time from zero: no gap or duplicate where pieces join.
        counted = validate_timing(output, fps, video)
    except ValueError as error:
        raise RuntimeError(f"Encoded timestamps failed validation: {error}") from None
    if counted != expected:
        raise RuntimeError(f"Encoded output has {counted} timestamps, expected {expected}")
    if sample_aspect(video) != sample_aspect(stream):
        raise RuntimeError("Encoded pixel aspect ratio differs from the source")
    emitted = output_audio(verified, audio, duration, "Audio missing from output")
    sink.cleanup()
    return SimpleNamespace(
        source=source,
        stream=stream,
        output=output,
        video=video,
        fps=fps,
        start=start,
        frames=expected,
        duration=duration,
        audio="AAC" if emitted else "none",
        plan=plan,
        report=report,
    )


# Variant comparisons (DESIGN.md, Variant comparisons).
COMPARE_VERSION = 1
COMPARE_TILE = 480  # Grid cells fit a display-aspect thumbnail in this square.
COMPARE_COLUMNS = 4
COMPARE_ORDER = (
    "Variants run in request order (index 1 first) in one worker under one GPU admission, each "
    "on the same decoded source frames. Frames are zero-based source indices in ascending "
    "order; a clip's stills are its first, middle and last frames. Grid tiles are display-"
    "aspect thumbnails of the original and of Fusion's output before encoding; full-resolution "
    "PNGs and clips are separate artifacts."
)
AUDIO_POLICY = (
    "Each variant clip carries the source's first audio stream over the compared interval, cut "
    "and encoded as renders do, so all clips have the same audio. No composite combines clips, "
    "so no audio is duplicated. Stills carry no audio."
)


def frame_digest(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()


def thumbnail(image, aspect):
    """`image` at its display aspect (pixel aspect ratio `aspect`), fit inside a grid cell."""
    import cv2

    height, width = image.shape[:2]
    scale = min(1.0, COMPARE_TILE / (width * aspect), COMPARE_TILE / height)
    size = (max(1, round(width * aspect * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def blank(shape):
    import numpy as np

    return np.full(shape, 48, np.uint8)


def configure_variant(window, request, specs):
    """Apply one variant's request to the shared window: Fusion reads window.control and
    window.parameters live, so they are refilled in place. Returns what each mapping used and
    the variant's effective adjustable settings."""
    window.control.clear()
    window.control.update(initial_controls(specs, request.get("controls", {})))
    window.target_faces.clear()
    window.parameters.clear()
    used = configure_faces(window, request["mappings"])
    defaults = dict(window.default_parameters)
    mappings = []
    for mapping in request["mappings"]:
        merged = {**defaults, **mapping.get("parameters", {})}
        mappings.append({key: merged[key] for key in sorted(PARAMETERS) if key in merged})
    controls = {key: window.control[key] for key in sorted(CONTROLS) if key in window.control}
    return used, {"controls": controls, "mappings": mappings}


def follow_up(original, resolved, variant):
    """submit_swap arguments that repeat one variant alone, from the request rather than labels.
    Each mapping pins the source fingerprints the comparison used."""
    single = variant_request(original, variant)
    mappings = []
    for mapping, pinned in zip(single["mappings"], resolved["mappings"]):
        mapping = {key: value for key, value in mapping.items() if key != "target_sha256"}
        mappings.append({**mapping, "source_sha256": pinned["source_sha256"]})
    keys = ("frame",) if single["kind"] == "preview" else ("start_frame", "max_frames")
    return {
        "media_path": single["media_path"],
        "mode": single["kind"],
        **{key: single[key] for key in keys},
        "mappings": mappings,
        "controls": single["controls"],
    }


def compare_variants(window, request, directory):
    """Run each variant over the same decoded source frames, in request order.

    `request` is the resolved request. A failed variant is recorded with its reason and its
    partial artifacts are removed; the comparison fails only when every variant fails.
    """
    import cv2

    original = read_json(directory / "request.json")
    source = input_path(request["media_path"])
    artifacts = directory / "artifacts"
    variants, clip = request["variants"], "max_frames" in request
    fps = start = count = None
    if clip:
        _, stream, fps, start, count = clip_range(source, request)
        frames = sorted({start, start + (count - 1) // 2, start + count - 1})
        aspect = sample_aspect(stream)
        originals = {number: read_frame(source, stream, number) for number in frames}
    else:
        frames, aspect, timing = sorted(request["frames"]), Fraction(1), None
        if cv2.imread(str(source)) is None:
            aspect = sample_aspect(video_stream(probe(source)))
            timing = timeline(source, frames[-1] + 1) if frames[-1] else None
        originals = {number: load_frame(source, number, timing) for number in frames}
    height, width = originals[frames[0]].shape[:2]
    if width * height > MAX_COMPARE_PIXELS:
        raise ValueError(
            f"Comparisons accept frames of at most {MAX_COMPARE_PIXELS} pixels (3840x2160); "
            f"this source is {width}x{height}"
        )
    digests = {number: frame_digest(frame) for number, frame in originals.items()}
    names, tiles = [], {}
    for number, frame in originals.items():
        save_image(artifacts / f"original-f{number}.png", frame)
        names.append(f"original-f{number}.png")
        tiles[number] = [(f"original f{number}", thumbnail(frame, aspect), True)]
    specs = layouts()[0]
    run_detect = window.function_worker.run_detect
    entries, reference = [], None
    for index, variant in enumerate(variants, 1):
        prefix = f"v{index}-{variant['name']}"
        single = variant_request(request, variant)
        entry = {
            "index": index,
            "name": variant["name"],
            "status": "failed",
            "error": None,
            "overrides": {key: variant.get(key, {}) for key in ("parameters", "controls")},
            "settings": None,
            "submit_swap": follow_up(original, request, variant),
            "references": None,
            "input_sha256": None,
            "artifacts": [],
        }
        progress = {"stage": "comparing", "variant": index, "variants": len(variants)}
        progress["name"] = variant["name"]
        total = count if clip else len(frames)
        atomic_json(directory / "progress.json", {**progress, "frames": 0, "total_frames": total})
        written, shown, seen = [], {}, hashlib.sha256()
        try:
            entry["references"], entry["settings"] = configure_variant(window, single, specs)
            if clip:

                def observe(offset, frame, swapped):
                    seen.update(frame.tobytes())
                    number = start + offset
                    if number in digests:
                        if frame_digest(frame) != digests[number]:
                            raise RuntimeError(
                                f"Decoded source frame {number} differs from the compared original"
                            )
                        written.append(f"{prefix}-f{number}.png")
                        save_image(artifacts / written[-1], swapped)
                        shown[number] = thumbnail(swapped, aspect)

                tracking = ("FaceTrackingEnableToggle", "KPSSmoothingEnableToggle")
                worker = make_worker(window, sequential=any(window.control[k] for k in tracking))
                written.append(f"{prefix}.mp4")
                internal = {**single, "report": {"review_images": 0}}
                result = encode_range(internal, directory, worker, written[-1], observe, progress)
                if reference is not None and seen.hexdigest() != reference:
                    raise RuntimeError("Decoded clip differs from the other variants' input")
                report = result.report
                summaries = [report_summary(report)]
                outcome = {
                    "swapped_faces": worker.swap_count,
                    "clip": {
                        "artifact": f"{prefix}.mp4",
                        "frames": result.frames,
                        "fps": str(result.fps),
                        "duration": result.duration,
                        "audio": result.audio,
                    },
                    "frames": [{"frame": n, "artifact": f"{prefix}-f{n}.png"} for n in frames],
                    "swap_report": summaries[0],
                }
            else:
                report, cells = [], []
                for done, number in enumerate(frames, 1):
                    frame = originals[number]
                    seen.update(frame.tobytes())
                    window.function_worker.run_detect = run_detect  # Unwrap the last worker.
                    worker = make_worker(window)
                    swapped = process(worker, frame, number)
                    if swapped.shape != frame.shape:
                        raise RuntimeError("Fusion changed the frame dimensions")
                    written.append(f"{prefix}-f{number}.png")
                    save_image(artifacts / written[-1], swapped)
                    shown[number] = thumbnail(swapped, aspect)
                    worker.telemetry.finish(number + 1)
                    cell = variant_request(request, variant, number)
                    report.append(swap_report(cell, worker.telemetry, None, number))
                    cells.append(
                        {
                            "frame": number,
                            "artifact": written[-1],
                            "swapped_faces": worker.swap_count,
                            "swap_report": report_summary(report[-1]),
                        }
                    )
                    atomic_json(
                        directory / "progress.json",
                        {**progress, "frames": done, "total_frames": total},
                    )
                swapped_faces = sum(cell["swapped_faces"] for cell in cells)
                if not swapped_faces:
                    raise RuntimeError("No faces matched the target references in any frame")
                summaries = [cell["swap_report"] for cell in cells]
                outcome = {"swapped_faces": swapped_faces, "frames": cells}
            # Clips: the clip's report; stills: one preview report per frame.
            written.append(f"{prefix}-swap-report.json")
            atomic_json(artifacts / written[-1], report)
            for summary in summaries:
                summary["artifact"] = written[-1]
            entry.update(status="completed", input_sha256=seen.hexdigest(), artifacts=written)
            entry.update(outcome)
            reference = reference or entry["input_sha256"]
            names += written
        except Exception as error:
            traceback.print_exc()  # The reason is in the manifest; the traceback in worker.log.
            entry["error"] = str(error) or type(error).__name__
            for name in written:
                (artifacts / name).unlink(missing_ok=True)
            (directory / f"{prefix}.video-only.mp4").unlink(missing_ok=True)
            shown = {}
        finally:
            window.function_worker.run_detect = run_detect
        label = f"{index} {variant['name']}" + ("" if shown else " FAILED")
        for number in frames:
            image = shown.get(number)
            image = blank(tiles[number][0][1].shape) if image is None else image
            tiles[number].append((label, image, bool(shown)))
        entries.append(entry)
    failed = [entry for entry in entries if entry["status"] == "failed"]
    if len(failed) == len(entries):
        reasons = "; ".join(f"{entry['name']}: {entry['error']}" for entry in failed)
        raise RuntimeError(f"Every variant failed: {reasons}")
    grids = []
    for number in frames:
        grids.append({"frame": number, "artifact": f"compare-f{number}.jpg"})
        contact_sheet(
            artifacts / grids[-1]["artifact"], tiles[number], COMPARE_TILE, COMPARE_COLUMNS
        )
    selection = {"frames": frames}
    if clip:
        selection.update(
            fps=str(fps),
            start_frame=start,
            end_frame=start + count,
            frame_count=count,
            start_time=float(start / fps),
            duration=float(count / fps),
        )
    manifest = {
        "version": COMPARE_VERSION,
        "kind": "clip" if clip else "stills",
        "complete": not failed,
        "failed_variants": [entry["name"] for entry in failed],
        "media_path": original["media_path"],
        "source": {
            "width": width,
            "height": height,
            "sample_aspect_ratio": f"{aspect.numerator}:{aspect.denominator}",
            **selection,
            "frame_sha256": {str(number): digest for number, digest in digests.items()},
        },
        "order": COMPARE_ORDER,
        "audio_policy": AUDIO_POLICY,
        "original": [{"frame": number, "artifact": f"original-f{number}.png"} for number in frames],
        "grids": grids,
        "variants": entries,
    }
    if failed:
        listed = ", ".join(f"{entry['name']} ({entry['error']})" for entry in failed)
        manifest["warning"] = (
            f"{len(failed)} of {len(entries)} variants failed: {listed}. The comparison is "
            "incomplete; failed variants appear as FAILED tiles and have no artifacts."
        )
    names = [grid["artifact"] for grid in grids] + ["compare.json"] + names
    manifest["artifacts"] = names
    atomic_json(artifacts / "compare.json", {**manifest, "revision": REVISION})
    return manifest


STANDARD_RATES = [
    Fraction(24000, 1001),
    Fraction(24),
    Fraction(25),
    Fraction(30000, 1001),
    Fraction(30),
    Fraction(48),
    Fraction(50),
    Fraction(60000, 1001),
    Fraction(60),
    Fraction(120),
]


def normalized_rate(stream, override=None):
    """Use the override, else the stream's average rate snapped to a standard rate within 2%."""
    if override is not None:
        return Fraction(str(override))
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            rate = Fraction(stream.get(key, "0/0"))
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            break
    else:
        raise ValueError("Cannot determine the frame rate; pass fps")
    nearest = min(STANDARD_RATES, key=lambda value: abs(value - rate))
    return nearest if abs(nearest - rate) <= rate / 50 else rate.limit_denominator(1001)


def normalize_command(source, output, stream, rate, extent=None):
    """Encode the video alone. FFmpeg applies display matrices (rotations and flips) by default;
    the filters then rebase time, resample to CFR, pad and convert to the output's colors.

    The fps filter rounds the video's end to the nearest frame, which leaves nothing of a clip
    under half a target frame period. Passing the last frame at the end rounds up instead, so
    such a clip keeps one frame. Both give one frame from half to one and a half periods, so the
    switch at three quarters tolerates an approximate `extent` in seconds.
    """
    eof = ":eof_action=pass" if extent and extent * rate < 0.75 else ""
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-nostats", "-progress", "pipe:1"]
    # Keep source timestamps so the filters see the start times ffprobe reported.
    command += ["-copyts", *ffmpeg_input(source), "-map", "0:v:0"]
    # Padding by at most one pixel on the right/bottom keeps every face pixel, unlike cropping.
    # An explicit conversion fixes the matrix and range, which FFmpeg versions negotiate apart.
    convert = output_filters(stream, "in", "out")
    pad = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    command += ["-vf", f"setpts=PTS-STARTPTS,fps={rate}{eof},{pad},{convert}"]
    pixels = output_pixels(stream)
    command += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", pixels]
    command += color_tags(stream)
    command += ["-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1"]
    return command + ["-movflags", "+faststart", str(output)]


def video_extent(source, stream):
    """Seconds spanned by the first video stream, or None if unknown.

    Uses the stream's own duration, else a scan of its packet timestamps; the container
    duration can come from a longer audio track.
    """
    duration = float(stream.get("duration") or 0)
    if duration > 0:
        return duration
    try:
        return timeline(source).duration
    except (ValueError, subprocess.TimeoutExpired):
        return None


def normalize_video(request, directory):
    source = input_path(request["media_path"])
    metadata = probe(source)
    stream = next((s for s in metadata["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise ValueError("Media has no video stream")
    rate = normalized_rate(stream, request.get("fps"))
    rotation, reflection = display_transform(stream)
    extent = video_extent(source, stream)
    # An estimate for progress and a truncation check; output validation establishes the count.
    expected = max(round(extent * rate), 1) if extent else None
    video_only, temporary = directory / "normalized.video.mp4", directory / "normalized.partial.mp4"
    encoder = subprocess.Popen(
        normalize_command(source, video_only, stream, rate, extent),
        stdout=subprocess.PIPE,
        text=True,
    )
    started = time.monotonic()
    try:
        for line in encoder.stdout:
            key, _, value = line.strip().partition("=")
            if key == "frame" and value.isdigit():
                frames = int(value)
                speed = frames / max(time.monotonic() - started, 1e-6)
                eta = expected and round(max(expected - frames, 0) / max(speed, 1e-6), 1)
                atomic_json(
                    directory / "progress.json",
                    {
                        "stage": "normalizing",
                        "frames": frames,
                        "estimated_total_frames": expected,
                        "frames_per_second": round(speed, 2),
                        "eta_seconds": eta,
                    },
                )
        if encoder.wait(timeout=60) != 0:
            raise RuntimeError("FFmpeg normalization failed; see worker.log")
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    encoded = next((s for s in probe(video_only)["streams"] if s["codec_type"] == "video"), None)
    if encoded is None or not int(encoded.get("nb_frames") or 0):
        raise RuntimeError("Normalization produced no video frames")
    # The first audio stream is cut to the resampled video, which can end before or after the
    # source's video and is known exactly only now.
    start, end = float(stream.get("start_time") or 0), float(int(encoded["nb_frames"]) / rate)
    audio = audio_overlaps(source, metadata, start, end)
    strip = ["-map_metadata", "-1", "-map_chapters", "-1"]
    mux(ffmpeg_input(video_only), temporary, (source, start, end) if audio else None, strip)
    video_only.unlink()
    verified = probe(temporary)
    video = video_stream(verified)
    width, height = int(video["width"]), int(video["height"])
    if width % 2 or height % 2:
        raise RuntimeError("Normalized output has odd dimensions")
    if Fraction(video["avg_frame_rate"]) != rate or Fraction(video["r_frame_rate"]) != rate:
        raise RuntimeError("Normalized output is not constant frame rate")
    frames = validate_timing(temporary, rate, video)
    output_duration = float(video["duration"])
    if int(video["nb_frames"]) != frames or abs(output_duration - frames / rate) > 1 / rate:
        raise RuntimeError("Normalized frame count or duration validation failed")
    if expected is not None and abs(frames - expected) > max(2, expected / 100):
        raise RuntimeError(f"Normalized output has {frames} frames, expected about {expected}")
    missing = "Audio missing from normalized output"
    emitted = output_audio(verified, audio, float(frames / rate), missing)
    output = directory / "artifacts/normalized.mp4"
    os.replace(temporary, output)
    return {
        "media_path": str(output),
        "frames": frames,
        "fps": str(rate),
        "duration": output_duration,
        "width": width,
        "height": height,
        "rotation_applied": rotation,
        "reflection_applied": reflection,
        "source": {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "r_frame_rate": stream.get("r_frame_rate"),
            "start_time": float(stream.get("start_time") or 0),
        },
        "audio": "AAC" if emitted else "none",
        "audio_start_time": float(emitted["start_time"]) if emitted else None,
        "audio_duration": float(emitted["duration"]) if emitted else None,
        "artifacts": ["normalized.mp4"],
    }


def loaded_models(window):
    """Names of the Fusion models loaded when the job finished, which recipes fingerprint."""
    models = getattr(getattr(window, "models_processor", None), "models", None) or {}
    return sorted(name for name, model in models.items() if model is not None)


def run(directory):
    # Linux kills the worker if its supervisor dies, including the startup race.
    parent = int(os.environ["VM_SUPERVISOR_PID"])
    if sys.platform.startswith("linux") and ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0:
        raise RuntimeError("Cannot set the worker's parent-death signal")
    if os.getppid() != parent:
        try:
            os.kill(parent, 0)
        except OSError:
            raise RuntimeError("Supervisor disappeared before worker startup") from None
        raise RuntimeError(
            "The configured runner must exec the Fusion Python interpreter so the worker is "
            "the supervisor's direct child (see README, Runner)"
        )
    if CONFIG_ERROR:
        raise RuntimeError(CONFIG_ERROR)
    if SETTINGS.sha256 != os.environ.get("VM_CONFIG_SHA256"):
        raise RuntimeError("Remote configuration changed after submission; resubmit the job")
    # Inputs resolved at submission, which retention protects; older jobs keep the raw request.
    resolved = directory / "resolved.json"
    request = read_json(resolved if resolved.exists() else directory / "request.json")
    request = validate_request(request)
    if request["kind"] == "normalize":
        # FFmpeg only: Fusion and the GPU are never loaded.
        atomic_json(directory / "result.json", normalize_video(request, directory))
        return
    expected, checkpoint, digest = request.get("media_sha256"), None, None
    if request["kind"] == "render":
        # A render records its provenance with its chunks before Fusion loads, and a resumed
        # attempt must match the earlier attempt's (DESIGN.md, Checkpoints).
        atomic_json(directory / "progress.json", {"stage": "fingerprinting"})
        record = provenance(request)
        digest = record["inputs"]["media_path"]["sha256"]
        checkpoint = Checkpoint(directory, request, record)
    elif expected is not None:
        atomic_json(directory / "progress.json", {"stage": "verifying_media"})
        digest = file_sha256(input_path(request["media_path"]))
    if expected is not None and digest != expected:
        # Recipe jobs name the media bytes they reproduce; hashed once, before Fusion loads.
        raise ValueError(
            f"Media {request['media_path']} changed after submission (expected SHA-256 "
            f"{expected}); verify the recipe again and resubmit"
        )
    atomic_json(directory / "progress.json", {"stage": "loading_models"})
    window = initialize(request.get("controls", {}))
    artifacts = directory / "artifacts"
    if request["kind"] == "detect" and (request.get("frames") or request.get("sample_frames")):
        result = detect_identities(window, request, directory)
    elif request["kind"] == "detect":
        result = detect_frame(window, request, directory)
    elif request["kind"] == "analyze":
        result = analyze_references(window, request, directory)
    elif request["kind"] == "compare":
        result = compare_variants(window, request, directory)
    else:
        used = configure_faces(window, request["mappings"])
        # Fusion's tracking is opt-in and only meaningful across consecutive render frames.
        tracking = any(
            window.control[k] for k in ("FaceTrackingEnableToggle", "KPSSmoothingEnableToggle")
        )
        worker = make_worker(window, sequential=request["kind"] == "render" and tracking)
        if request["kind"] == "preview":
            number = request.get("frame", 0)
            frame = load_frame(input_path(request["media_path"]), number)
            swapped = process(worker, frame, number)
            if worker.swap_count == 0:
                raise RuntimeError("No faces matched the target references")
            save_image(artifacts / "before.jpg", frame)
            save_image(artifacts / "preview.jpg", swapped)
            save_image(artifacts / "result.png", swapped)
            worker.telemetry.finish(number + 1)
            report = swap_report(request, worker.telemetry, None, number)
            atomic_json(artifacts / "swap-report.json", report)
            result = {
                "swapped_faces": worker.swap_count,
                "swap_report": report_summary(report),
                "artifacts": ["result.png", "before.jpg", "preview.jpg", "swap-report.json"],
            }
        else:
            result = render_video(window, request, directory, worker, checkpoint)
        result["references"] = used
        result["models"] = loaded_models(window)
        if request.get("recipe"):
            result["recipe"] = request["recipe"]
    result["revision"] = REVISION
    atomic_json(directory / "result.json", result)


def main(directory):
    # Persist the failure for the supervisor; the traceback still reaches worker.log.
    (directory / "error.json").unlink(missing_ok=True)
    try:
        run(directory)
    except Exception as error:
        atomic_json(
            directory / "error.json",
            {"type": type(error).__name__, "message": str(error) or type(error).__name__},
        )
        raise


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

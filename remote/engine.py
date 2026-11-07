"""Version-pinned adapter to Fusion's FrameWorker and FunctionWorker."""

import contextlib
import ctypes
import io
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from common import (
    APP,
    CONFIG_ERROR,
    REVISION,
    SETTINGS,
    atomic_json,
    ffmpeg_input,
    fusion_compatibility,
    input_path,
    layouts,
    probe,
    read_json,
    validate_request,
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
    control = {k: v["default"] for k, v in control_specs.items()}
    control.update(
        {
            "DetectorModelSelection": "Yolov8",
            "MaxFacesToDetectSlider": 20,
            "LandmarkDetectToggle": False,
            "ForceSwapToggle": False,
            "SwapOnlyBestMatchEnableToggle": False,
        }
    )
    control.update(controls)
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


def video_stream(metadata):
    stream = next((s for s in metadata["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise ValueError("Media has no video stream")
    if any(d.get("rotation") for d in stream.get("side_data_list", [])):
        raise ValueError("Rotated video is not supported; run normalize_media first")
    return stream


HDR = {"smpte2084": "PQ", "arib-std-b67": "HLG"}
# FFprobe colorspace names mapped to swscale matrix names.
MATRICES = {
    "bt709": "bt709",
    "smpte170m": "smpte170m",
    "bt470bg": "bt470",
    "fcc": "fcc",
    "smpte240m": "smpte240m",
    "bt2020nc": "bt2020",
}
SWS_FLAGS = "accurate_rnd+full_chroma_int+full_chroma_inp"


def color(stream):
    """Source YUV matrix and range; decode and encode share them so unswapped pixels keep color."""
    transfer = stream.get("color_transfer")
    if transfer in HDR:
        raise ValueError(
            f"HDR ({HDR[transfer]}) video is not supported because faces are processed as 8-bit "
            "SDR; tone-map the input to SDR first"
        )
    full = stream.get("color_range") == "pc" or stream.get("pix_fmt", "").startswith("yuvj")
    return MATRICES.get(stream.get("color_space")), "pc" if full else "tv"


def scale_filter(stream, side):
    matrix, value = color(stream)
    options = [f"{side}_range={value}", f"flags={SWS_FLAGS}"]
    if matrix:
        options.insert(0, f"{side}_color_matrix={matrix}")
    return "scale=" + ":".join(options)


def deep_color(stream):
    return re.search(r"p1[0-6](le|be)$", stream.get("pix_fmt", "")) is not None


def encode_command(stream, fps, output):
    """Encode bgr24 frames with the source's matrix, range, tags and (8- or 10-bit) depth."""
    width, height = int(stream["width"]), int(stream["height"])
    pixels = "yuv420p10le" if deep_color(stream) else "yuv420p"
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", "2", "-f", "rawvideo"]
    command += ["-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
    command += ["-an", "-vf", scale_filter(stream, "out") + f",format={pixels}", "-c:v", "libx264"]
    command += ["-threads", "2", "-preset", "medium", "-crf", "18", "-pix_fmt", pixels]
    return command + color_tags(stream) + [str(output)]


def color_tags(stream):
    """Output options that keep the source's range, matrix, primaries and transfer tags."""
    tags = ["-color_range", color(stream)[1]]
    for option, key in (
        ("-colorspace", "color_space"),
        ("-color_primaries", "color_primaries"),
        ("-color_trc", "color_transfer"),
    ):
        if stream.get(key, "unknown") not in ("unknown", "reserved"):
            tags += [option, stream[key]]
    return tags


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
    decoded = decode(path, stream, frame_number, 1, times)
    try:
        frame = next(decoded, None)
    finally:
        decoded.close()
    if frame is None:
        raise ValueError(f"Cannot read frame {frame_number} from {path}")
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


def crop_face(image, box):
    """Pad a detection box by 30% so the crop re-detects as exactly one face."""
    x1, y1, x2, y2 = box
    pad = int(max(x2 - x1, y2 - y1) * 0.3)
    return image[
        max(0, y1 - pad) : min(image.shape[0], y2 + pad),
        max(0, x1 - pad) : min(image.shape[1], x2 + pad),
    ]


def cluster_faces(embeddings, frames, threshold):
    """Greedily group faces into identities on Fusion's similarity scale.

    Fusion scores a match as 100 - 50 * cosine distance (50 + 50 * cosine similarity) and swaps
    at or above SimilarityThresholdSlider. Faces should arrive best first. Each joins the most
    similar identity whose normalized mean embedding scores at least `threshold` and that has
    no face from the same frame; otherwise it starts a new identity. Returns one label per face.
    """

    def unit(vector):
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

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
    model = window.control["RecognitionModelSelection"]
    faces = []
    for done, number in enumerate(frames):
        atomic_json(
            directory / "progress.json",
            {"stage": "detecting", "frames": done, "total_frames": len(frames)},
        )
        image = load_frame(path, number, timing)
        tensor, boxes, points = detect(window, image)
        for box, point in zip(boxes, points):
            embedding = window.function_worker.run_recognize_direct(tensor, point, "Auto", model)[0]
            if embedding is None:
                continue
            box = [int(v) for v in box[:4]]
            faces.append(
                {
                    "frame": number,
                    "box": box,
                    "area": (box[2] - box[0]) * (box[3] - box[1]),
                    "embedding": [float(v) for v in embedding.ravel()],
                    "crop": crop_face(image, box).copy(),
                }
            )
    faces.sort(key=lambda face: -face["area"])
    threshold = layouts()[1]["SimilarityThresholdSlider"]["default"]
    labels = cluster_faces([f["embedding"] for f in faces], [f["frame"] for f in faces], threshold)
    groups = {}
    for face, label in zip(faces, labels):
        groups.setdefault(label, []).append(face)
    artifacts = directory / "artifacts"
    identities, tiles = [], []
    for number, members in enumerate(sorted(groups.values(), key=lambda m: -len(m))):
        # Members are largest first; keep the first crop that re-detects as one face, as
        # submit_swap requires of target_face.
        best, verified = members[0], False
        for member in members[:5]:
            if len(detect(window, member["crop"], max_faces=20)[1]) == 1:
                best, verified = member, True
                break
        name = f"identity-{number}.png"
        save_image(artifacts / name, best["crop"])
        tiles.append((f"id {number}: {len(members)}x" + ("" if verified else " *"), best["crop"]))
        identities.append(
            {
                "identity": number,
                "appearances": len(members),
                "frames": sorted({m["frame"] for m in members}),
                "frame": best["frame"],
                "box": best["box"],
                "reference_path": str(artifacts / name),
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
    return {
        "frames": frames,
        "detections": len(faces),
        "similarity_threshold": threshold,
        "identities": identities,
        "artifacts": names,
    }


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


def configure_faces(window, mappings):
    import cv2
    import numpy as np
    from app.helpers.miscellaneous import ParametersDict

    def embeddings(path, models):
        image = cv2.imread(str(input_path(path)))
        if image is None:
            raise ValueError(f"Face reference must be an image: {path}")
        tensor, boxes, points = detect(window, image, max_faces=20)
        if len(boxes) != 1:
            raise ValueError(
                f"Reference must contain exactly one detected face: {path} ({len(boxes)})"
            )
        return {
            model: window.function_worker.run_recognize_direct(tensor, points[0], "Auto", model)[0]
            for model in models
        }

    for index, mapping in enumerate(mappings):
        parameters = dict(window.default_parameters)
        parameters.update(mapping.get("parameters", {}))
        models = {
            window.control["RecognitionModelSelection"],
            window.function_worker.get_arcface_model(parameters["SwapModelSelection"]),
        }
        target = embeddings(mapping["target_face"], models)
        sources = [embeddings(path, models) for path in mapping["source_faces"]]
        merged = {model: np.mean([source[model] for source in sources], axis=0) for model in models}
        face_id = str(index)
        window.target_faces[face_id] = TargetFace(face_id, target, merged)
        window.parameters[face_id] = ParametersDict(parameters, dict(window.default_parameters))


def make_worker(window, sequential=False):
    """Sequential workers mirror Fusion's video pipeline: one SequentialDetector feeds
    precomputed, identity-filtered detections (optional ByteTrack and landmark EMA) in order."""
    from app.processors.video_utils.sequential_detector import SequentialDetector
    from app.processors.workers.frame_worker import FrameWorker

    class CheckedWorker(FrameWorker):
        swap_count = 0

        def swap_core(self, *args, **kwargs):
            result = super().swap_core(*args, **kwargs)
            self.swap_count += 1
            return result

    worker = CheckedWorker(window, is_single_frame=not sequential)
    worker.parameters = window.parameters
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
        raise RuntimeError("Fusion reported a per-face processing error; see worker.log")
    return np.ascontiguousarray(result)


def save_image(path, image):
    import cv2

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def contact_sheet(path, tiles, size=200, columns=6):
    """Write labelled face crops as one grid image, so a single view shows every face."""
    import cv2
    import numpy as np

    cell, label = size + 8, 36
    rows = -(-len(tiles) // columns)
    sheet = np.full((rows * (cell + label), min(len(tiles), columns) * cell, 3), 32, np.uint8)
    for number, (text, image) in enumerate(tiles):
        top, left = [v * c for v, c in zip(divmod(number, columns), (cell + label, cell))]
        scale = size / max(image.shape[:2])
        width, height = [max(1, round(v * scale)) for v in (image.shape[1], image.shape[0])]
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        y, x = top + 4 + (size - height) // 2, left + 4 + (size - width) // 2
        sheet[y : y + height, x : x + width] = cv2.resize(
            image, (width, height), None, 0, 0, interpolation
        )
        cv2.putText(
            sheet,
            text,
            (left + 8, top + cell + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
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
            raise ValueError("No video timestamps found")
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
            raise ValueError("Video packets lack timestamps; run normalize_media first")
        times.append(float(value))
    return Timeline(sorted(times)[:end])


def validate_timing(source, fps, stream, end=None):
    """Require CFR timestamps from zero over the first `end` frames (all if None); metadata can
    hide VFR. Returns the number of frames checked."""
    timing = timeline(source, end)
    tick = float(Fraction(stream["time_base"]))
    tolerance = max(0.00001, tick * 1.1)
    if abs(timing.start) > tolerance:
        raise ValueError("Video timestamps must start at zero; run normalize_media first")
    if not timing.constant(float(fps), tolerance):
        raise ValueError("Variable frame timestamps detected; run normalize_media first")
    return len(timing.times)


def render_video(window, request, directory, worker):
    source = input_path(request["media_path"])
    metadata = probe(source)
    stream = video_stream(metadata)
    fps = Fraction(stream["avg_frame_rate"])
    if fps <= 0 or fps != Fraction(stream["r_frame_rate"]):
        raise ValueError("Render requires constant frame rate video; run normalize_media first")
    start = request.get("start_frame", 0)
    limit = request.get("max_frames")
    total = validate_timing(source, fps, stream, start + limit if limit else None)
    expected = min(limit or total, total - start)
    if expected <= 0:
        raise ValueError("Requested frame range is empty")
    video_only = directory / "video-only.mp4"
    artifacts = directory / "artifacts"
    encoder = None
    frames = decode(source, stream, start, expected)
    processed = 0
    width, height = int(stream["width"]), int(stream["height"])
    if width % 2 or height % 2:
        raise ValueError("H.264 output requires even frame dimensions; run normalize_media first")
    try:
        encoder = subprocess.Popen(
            encode_command(stream, fps, video_only),
            stdin=subprocess.PIPE,
        )
        started = time.monotonic()
        for offset in range(expected):
            frame = next(frames, None)
            if frame is None:
                raise RuntimeError(
                    f"Input decoding stopped at {start + offset}, expected {expected} frames"
                )
            swapped = process(worker, frame, start + offset)
            if swapped.shape[:2] != (height, width):
                raise RuntimeError("Fusion changed the frame dimensions")
            if offset == 0:
                save_image(artifacts / "before.jpg", frame)
                save_image(artifacts / "preview.jpg", swapped)
            encoder.stdin.write(swapped.tobytes())
            processed += 1
            rate = processed / max(time.monotonic() - started, 1e-6)
            atomic_json(
                directory / "progress.json",
                {
                    "stage": "rendering",
                    "frames": processed,
                    "total_frames": expected,
                    "swapped_faces": worker.swap_count,
                    "frames_per_second": round(rate, 2),
                    "eta_seconds": round((expected - processed) / rate, 1),
                },
            )
        encoder.stdin.close()
        if encoder.wait(timeout=120) != 0:
            raise RuntimeError("FFmpeg video encoding failed")
    finally:
        frames.close()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    if worker.swap_count == 0:
        raise RuntimeError("No faces matched the target references in the requested range")
    output = artifacts / "result.mp4"
    duration = float(Fraction(processed, 1) / fps)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            *ffmpeg_input(video_only),
            # Frame `start` on the video clock; the container start may be earlier audio.
            "-seek_timestamp",
            "1",
            "-ss",
            str(float(Fraction(start, 1) / fps)),
            *ffmpeg_input(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            str(duration),
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
        timeout=300,
    )
    verified = probe(output)
    video = next(s for s in verified["streams"] if s["codec_type"] == "video")
    if int(video["nb_frames"]) != processed or abs(float(video["duration"]) - duration) > 1 / float(
        fps
    ):
        raise RuntimeError("Encoded frame count or duration validation failed")
    has_audio = any(s["codec_type"] == "audio" for s in metadata["streams"])
    if has_audio and not any(s["codec_type"] == "audio" for s in verified["streams"]):
        raise RuntimeError("Audio missing from output")
    video_only.unlink()
    return {
        "frames": processed,
        "fps": str(fps),
        "duration": duration,
        "swapped_faces": worker.swap_count,
        "audio": "AAC" if has_audio else "none",
        "artifacts": ["result.mp4", "preview.jpg", "before.jpg"],
    }


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


def normalize_command(source, output, stream, audio, rate):
    """FFmpeg autorotates by default; the filters then rebase time, resample to CFR and pad."""
    start = float(stream.get("start_time") or 0)
    duration = float(stream.get("duration") or 0)
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-nostats", "-progress", "pipe:1"]
    # Keep source timestamps so the filters see the start times ffprobe reported.
    command += ["-copyts", *ffmpeg_input(source), "-map", "0:v:0"]
    # Padding by at most one pixel on the right/bottom keeps every face pixel, unlike cropping.
    command += ["-vf", f"setpts=PTS-STARTPTS,fps={rate},pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    pixels = "yuv420p10le" if deep_color(stream) else "yuv420p"
    command += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", pixels]
    command += color_tags(stream)
    if audio:
        # Align audio with the first video frame: trim earlier audio, pad later audio with silence.
        trim = f"atrim=start={start}" + (f":end={start + duration}" if duration > 0 else "")
        filters = f"{trim},asetpts=PTS-{start}/TB,aresample=async=1:first_pts=0"
        command += ["-map", "0:a:0", "-af", filters, "-c:a", "aac", "-b:a", "192k"]
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
    audio = any(s["codec_type"] == "audio" for s in metadata["streams"])
    rate = normalized_rate(stream, request.get("fps"))
    rotation = next(
        (d["rotation"] for d in stream.get("side_data_list", []) if d.get("rotation")), 0
    )
    extent = video_extent(source, stream)
    # An estimate for progress and a truncation check; output validation establishes the count.
    expected = max(round(extent * rate), 1) if extent else None
    temporary = directory / "normalized.partial.mp4"
    encoder = subprocess.Popen(
        normalize_command(source, temporary, stream, audio, rate),
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
    output_audio = next((s for s in verified["streams"] if s["codec_type"] == "audio"), None)
    if audio and output_audio is None:
        raise RuntimeError("Audio missing from normalized output")
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
        "source": {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "r_frame_rate": stream.get("r_frame_rate"),
            "start_time": float(stream.get("start_time") or 0),
        },
        "audio": "AAC" if output_audio else "none",
        "audio_start_time": float(output_audio["start_time"]) if output_audio else None,
        "artifacts": ["normalized.mp4"],
    }


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
    atomic_json(directory / "progress.json", {"stage": "loading_models"})
    window = initialize(request.get("controls", {}))
    artifacts = directory / "artifacts"
    if request["kind"] == "detect" and (request.get("frames") or request.get("sample_frames")):
        result = detect_identities(window, request, directory)
    elif request["kind"] == "detect":
        image = load_frame(input_path(request["media_path"]), request.get("frame", 0))
        _, boxes, _ = detect(window, image)
        faces, tiles = [], []
        for index, box in enumerate(boxes):
            box = [int(v) for v in box[:4]]
            name = f"face-{index}.png"
            crop = crop_face(image, box)
            save_image(artifacts / name, crop)
            tiles.append((f"face {index}", crop))
            faces.append(
                {
                    "index": index,
                    "box": box,
                    "reference_path": str(artifacts / name),
                    "artifact": name,
                }
            )
        save_image(artifacts / "frame.jpg", image)
        names = ["frame.jpg"] + [f["artifact"] for f in faces]
        if tiles:
            contact_sheet(artifacts / "contact.jpg", tiles)
            names.insert(1, "contact.jpg")
        result = {"faces": faces, "artifacts": names}
    else:
        configure_faces(window, request["mappings"])
        # Fusion's tracking is opt-in and only meaningful across consecutive render frames.
        tracking = any(
            window.control[k] for k in ("FaceTrackingEnableToggle", "KPSSmoothingEnableToggle")
        )
        worker = make_worker(window, sequential=request["kind"] == "render" and tracking)
        if request["kind"] == "preview":
            frame = load_frame(input_path(request["media_path"]), request.get("frame", 0))
            swapped = process(worker, frame, request.get("frame", 0))
            if worker.swap_count == 0:
                raise RuntimeError("No faces matched the target references")
            save_image(artifacts / "before.jpg", frame)
            save_image(artifacts / "preview.jpg", swapped)
            save_image(artifacts / "result.png", swapped)
            result = {
                "swapped_faces": worker.swap_count,
                "artifacts": ["result.png", "before.jpg", "preview.jpg"],
            }
        else:
            result = render_video(window, request, directory, worker)
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

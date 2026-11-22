from __future__ import annotations
from training.data.samples import (
    validate_rgb_frames,
    DecodedVideo,
    PreparedSample,
    Sample,
    VideoMapping,
)
from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    with Image.open(Path(path)) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def load_video(path: str | Path) -> DecodedVideo:
    """Decode the complete sequence; a header or decode mismatch is an error."""
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames or len(frames) != expected:
        raise ValueError(
            f"{path}: decoded {len(frames)} frames, container declares {expected}"
        )
    return DecodedVideo(np.stack(frames), fps)


def _spec_value(spec: Any, name: str) -> Any:
    return spec[name] if isinstance(spec, Mapping) else getattr(spec, name)


def _make_video_mapping(
    source_width: int,
    source_height: int,
    source_count: int,
    source_fps: float,
    spec: Any,
) -> VideoMapping:
    width, height, count = (
        _spec_value(spec, name) for name in ("width", "height", "num_frames")
    )
    fps = float(_spec_value(spec, "fps"))
    if (
        any(type(value) is not int or value < 1 for value in (width, height, count))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError(
            "model video dimensions, frame count, and fps must be positive"
        )
    if (
        any(
            type(value) is not int or value < 1
            for value in (source_width, source_height, source_count)
        )
        or not math.isfinite(source_fps)
        or source_fps <= 0
    ):
        raise ValueError(
            "source delivery dimensions, frame count, and fps must be positive"
        )
    if count < source_count:
        raise ValueError(
            f"model budget {count} would discard source process frames ({source_count}); "
            "increase num_frames to preserve the complete video"
        )
    ratio = min(width / source_width, height / source_height)
    resized_width = max(1, min(width, round(source_width * ratio)))
    resized_height = max(1, min(height, round(source_height * ratio)))
    forward = tuple(
        np.rint(np.linspace(0, source_count - 1, count)).astype(int).tolist()
    )
    reverse = tuple(
        np.rint(np.linspace(0, count - 1, source_count)).astype(int).tolist()
    )
    return VideoMapping(
        source_width,
        source_height,
        source_count,
        source_fps,
        width,
        height,
        count,
        fps,
        resized_width,
        resized_height,
        (width - resized_width) // 2,
        (height - resized_height) // 2,
        forward,
        reverse,
    )


def make_video_mapping(video: DecodedVideo, spec: Any) -> VideoMapping:
    source_count, source_height, source_width, _ = video.frames.shape
    return _make_video_mapping(
        source_width, source_height, source_count, video.fps, spec
    )


def _format_image(image: np.ndarray, mapping: VideoMapping) -> np.ndarray:
    resized = cv2.resize(
        image,
        (mapping.resized_width, mapping.resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    result = np.zeros(
        (mapping.model_height, mapping.model_width, 3), dtype=np.uint8
    )
    x, y = mapping.offset_x, mapping.offset_y
    result[y : y + mapping.resized_height, x : x + mapping.resized_width] = resized
    return result


def prepare_sample(sample: Sample, spec: Any) -> PreparedSample:
    image, video = load_image(sample.input_image), load_video(sample.reference_video)
    if image.shape[:2] != video.frames.shape[1:3]:
        raise ValueError(
            f"{sample.id}: conditioning canvas and reference video dimensions differ"
        )
    if (
        sample.canonical_sample["video_num_frames"] != len(video.frames)
        or abs(float(sample.canonical_sample["video_fps"]) - float(video.fps)) > 0.01
    ):
        raise ValueError(
            f"{sample.id}: decoded reference metadata differs from canonical_sample"
        )
    mapping = make_video_mapping(video, spec)
    source_frames = np.stack([_format_image(frame, mapping) for frame in video.frames])
    model_frames = source_frames[np.asarray(mapping.model_to_source_indices)]
    return PreparedSample(
        sample,
        _format_image(image, mapping),
        DecodedVideo(model_frames, mapping.model_fps),
        mapping,
    )


def prepare_condition(
    sample: Sample,
    spec: Any,
    *,
    source_num_frames: int | None = None,
    source_fps: float | None = None,
) -> PreparedSample:
    """Prepare inference from the prompt image and public delivery metadata.

    No reference asset, answer, or semantic target is opened or inspected.
    Explicit delivery arguments take precedence over canonical frame-count/fps
    metadata. The zero video is only a shape placeholder; inference must start
    from fresh noise and consume ``input_image`` as its first-frame condition.
    """
    image = load_image(sample.input_image)
    if source_num_frames is None:
        source_num_frames = sample.canonical_sample.get("video_num_frames")
    if source_fps is None:
        source_fps = sample.canonical_sample.get("video_fps")
    if source_num_frames is None or source_fps is None:
        raise ValueError(
            "condition preparation requires public delivery frame count and fps"
        )
    height, width = image.shape[:2]
    mapping = _make_video_mapping(
        width, height, source_num_frames, float(source_fps), spec
    )
    placeholder = np.broadcast_to(
        np.uint8(0),
        (mapping.model_num_frames, mapping.model_height, mapping.model_width, 3),
    )
    return PreparedSample(
        sample,
        _format_image(image, mapping),
        DecodedVideo(placeholder, mapping.model_fps),
        mapping,
    )


def restore_video(
    frames: np.ndarray | DecodedVideo, mapping: VideoMapping
) -> DecodedVideo:
    """Undo letterbox and retiming before the original task verifier runs."""
    if isinstance(frames, DecodedVideo):
        if abs(frames.fps - mapping.model_fps) > 0.01:
            raise ValueError("generated video fps disagrees with its model mapping")
        frames = frames.frames
    validate_rgb_frames(frames)
    if frames.shape[:3] != (
        mapping.model_num_frames,
        mapping.model_height,
        mapping.model_width,
    ):
        raise ValueError("generated video shape disagrees with its model mapping")
    x, y = mapping.offset_x, mapping.offset_y
    cropped = frames[:, y : y + mapping.resized_height, x : x + mapping.resized_width]
    restored = np.stack(
        [
            cv2.resize(
                cropped[index],
                (mapping.source_width, mapping.source_height),
                interpolation=cv2.INTER_LINEAR,
            )
            for index in mapping.source_to_model_indices
        ]
    )
    return DecodedVideo(restored, mapping.source_fps)

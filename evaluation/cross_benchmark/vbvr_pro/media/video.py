"""Complete video decoding. No prefix caps, frame dropping, or GT-dependent crops."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_FPS = 15

ENCODING = {
    "backend": "imageio-ffmpeg",
    "codec": "libx264",
    "pixel_format": "yuv420p",
    "quality": 5.0,
    "macro_block_size": 16,
    "threads": 1,
}


def encode_h264(frames, path: str | Path, fps: float = DEFAULT_FPS):
    """Encode BGR uint8 frames with the recorded codec parameters."""
    import imageio

    with imageio.get_writer(
        str(path),
        fps=fps,
        codec=ENCODING["codec"],
        pixelformat=ENCODING["pixel_format"],
        quality=ENCODING["quality"],
        macro_block_size=ENCODING["macro_block_size"],
        output_params=["-threads", str(ENCODING["threads"])],
    ) as writer:
        for frame in frames:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


class InvalidMedia(ValueError):
    """A candidate is missing, undecodable, or truncated."""


@dataclass
class Video:
    frames: list[np.ndarray]
    fps: float
    declared_frames: int


def decode(path: str | Path) -> Video:
    path = Path(path)
    if not path.is_file():
        raise InvalidMedia(f"Missing video: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise InvalidMedia(f"Cannot open video: {path}")
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise InvalidMedia("Invalid decoded frame")
            if frames and frame.shape != frames[0].shape:
                raise InvalidMedia("Frame dimensions change within video")
            frames.append(frame)
        if not frames:
            raise InvalidMedia("Video has no decodable frames")
        if declared > len(frames):
            raise InvalidMedia(f"Incomplete decode: {len(frames)}/{declared} frames")
        if not np.isfinite(fps) or fps <= 0:
            raise InvalidMedia("Invalid frame rate")
        return Video(frames, fps, declared)
    finally:
        capture.release()


def image(path: str | Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot decode reference image: {path}")
    return frame


def normalize(frames: list[np.ndarray], first: np.ndarray) -> list[np.ndarray]:
    """Resize candidate only, retaining the complete canvas and aspect ratio."""
    height, width = first.shape[:2]
    source_height, source_width = frames[0].shape[:2]
    if abs(source_width / source_height - width / height) > 0.02:
        raise InvalidMedia("Candidate aspect ratio differs from the input canvas")
    if (source_height, source_width) == (height, width):
        return frames
    return [cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in frames]


def input_preserved(candidate: np.ndarray, first: np.ndarray) -> tuple[bool, dict]:
    """Check local input features so white background cannot hide missing objects.

    This is only an I2V conditioning check; it never establishes task correctness.
    Every occupied 32x32 tile must retain its edges and colors. Codec/resize
    tolerance applies to the observations, never to the symbolic answer.
    """
    a = cv2.GaussianBlur(candidate, (3, 3), 0).astype(np.float32)
    b = cv2.GaussianBlur(first, (3, 3), 0).astype(np.float32)
    delta = np.max(np.abs(a - b), axis=2)
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 100)
    errors = []
    for y in range(0, gray.shape[0], 32):
        for x in range(0, gray.shape[1], 32):
            tile = delta[y : y + 32, x : x + 32]
            edge_tile = edges[y : y + 32, x : x + 32]
            if np.count_nonzero(edge_tile) >= 4:
                errors.append(float(np.mean(tile > 55)))
    largest = max(errors, default=0.0)
    global_error = float(np.mean(delta > 55))
    return bool(largest <= 0.38 and global_error <= 0.025), {
        "worst_input_tile_error": largest,
        "input_changed_fraction": global_error,
    }

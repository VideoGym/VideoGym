"""Candidate image/video decoding; shared by task verifiers."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List


def _read_bgr(image_path: Path) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for semantic artifact evaluation.") from exc

    image = cv2.imread(image_path.as_posix())
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return image


def _align_bgr(image: np.ndarray, reference_shape: tuple[int, int, int]) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for semantic artifact evaluation.") from exc

    if image.shape != reference_shape:
        image = cv2.resize(image, (reference_shape[1], reference_shape[0]))
    return image


def _video_frames_bgr(video_path: Path) -> List[np.ndarray]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for semantic artifact evaluation.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames processed from candidate video.")
    return frames


def _video_fps(video_path: Path) -> float:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for video frame-rate parsing") from exc
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    cap.release()
    return fps

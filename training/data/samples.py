from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any
import numpy as np
from data.common.paths import to_relative
from data.common.schemas import CanonicalSample


@dataclass(frozen=True)
class Sample:
    id: str
    task_name: str
    prompt: str
    input_image: Path
    reference_video: Path
    canonical_sample: dict[str, Any]

    def canonical(self) -> CanonicalSample:
        """Reconstruct verifier inputs without sharing mutable source state."""
        return CanonicalSample.from_dict(deepcopy(self.canonical_sample))

    def to_dict(self, *, base_dir: Path | None = None) -> dict[str, Any]:
        canonical_payload = deepcopy(self.canonical_sample)
        if base_dir is not None:
            for key in ("image", "video"):
                if canonical_payload.get(key):
                    canonical_payload[key] = to_relative(canonical_payload[key], base_dir)
            input_image = to_relative(self.input_image, base_dir)
            reference_video = to_relative(self.reference_video, base_dir)
        else:
            input_image = str(self.input_image)
            reference_video = str(self.reference_video)
        return {
            "id": self.id,
            "task_name": self.task_name,
            "prompt": self.prompt,
            "input_image": input_image,
            "reference_video": reference_video,
            "canonical_sample": canonical_payload,
        }


@dataclass(frozen=True)
class DecodedVideo:
    """All decoded frames in RGB uint8 T,H,W,C order."""

    frames: np.ndarray
    fps: float

    def __post_init__(self) -> None:
        validate_rgb_frames(self.frames)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("video fps must be finite and positive")


@dataclass(frozen=True)
class VideoMapping:
    """Pixel-canvas and frame-index mappings used to undo model formatting.

    Letterboxing preserves the entire source canvas. Temporal expansion uses
    nearest-neighbor holds, preserving every source frame and both endpoints.
    """

    source_width: int
    source_height: int
    source_num_frames: int
    source_fps: float
    model_width: int
    model_height: int
    model_num_frames: int
    model_fps: float
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int
    model_to_source_indices: tuple[int, ...]
    source_to_model_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedSample:
    sample: Sample
    input_image: np.ndarray
    reference_video: DecodedVideo
    mapping: VideoMapping


def validate_rgb_frames(frames: np.ndarray) -> None:
    if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8:
        raise ValueError("video frames must be a numpy RGB uint8 array")
    if (
        frames.ndim != 4
        or frames.shape[-1] != 3
        or any(size < 1 for size in frames.shape)
    ):
        raise ValueError("video frames must have nonempty shape T,H,W,3")

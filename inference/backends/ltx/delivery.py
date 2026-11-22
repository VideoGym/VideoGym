from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Delivery:
    # Shared 24 FPS / 768x512 native LTX canvas and 16 FPS / 512x512 delivery
    # transform across Fixed Length and Reference Length; only target frames differ.
    native_frames: int = 129
    native_fps: float = 24.0
    native_width: int = 768
    native_height: int = 512
    frames: int = 81
    fps: float = 16.0
    width: int = 512
    height: int = 512

    @classmethod
    def from_reference_timing(cls, frames, fps=16):
        """Map target 16 FPS frame count to the smallest covering 8k+1 bucket at 24 FPS."""
        if type(frames) is not int or frames < 1:
            raise ValueError("delivery timing requires positive integer frames")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps != 16:
            raise ValueError("delivery timing requires the canonical 16 FPS")
        display = math.ceil(frames * 24 / fps)
        native = max(25, 1 + 8 * math.ceil((display - 1) / 8))
        return cls(native_frames=native, native_fps=24.0, frames=frames, fps=float(fps))


def plan_deliveries(timings):
    return {
        identity: Delivery.from_reference_timing(row["frames"], row["fps"])
        for identity, row in timings.items()
    }


def delivery_indices(delivery=Delivery()):
    """Nearest-neighbor frame indices mapping native 24 FPS output to 16 FPS."""
    return np.minimum(
        delivery.native_frames - 1,
        np.floor(
            np.arange(delivery.frames) * delivery.native_fps / delivery.fps + 0.5
        ).astype(int),
    )


def pad_input(image, delivery=Delivery()):
    if image.shape != (delivery.height, delivery.width, 3):
        raise ValueError(
            f"expected public input {delivery.width}x{delivery.height} RGB, got {image.shape}"
        )
    left = (delivery.native_width - delivery.width) // 2
    canvas = np.zeros(
        (delivery.native_height, delivery.native_width, 3), dtype=np.uint8
    )
    canvas[:, left : left + delivery.width] = image
    return canvas


def deliver_video(decoded, delivery=Delivery()):
    """Center-crop and resample decoded native LTX frames to 512x512 at 16 FPS."""
    expected = (
        delivery.native_frames,
        delivery.native_height,
        delivery.native_width,
        3,
    )
    if decoded.shape != expected:
        raise ValueError(
            f"native decoder shape {decoded.shape} differs from {expected}"
        )
    left = (delivery.native_width - delivery.width) // 2
    return decoded[delivery_indices(delivery), :, left : left + delivery.width].copy()

"""Family-level visual transition primitives for ICL reasoning states."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def changed_region_opacity_frames(
    previous: np.ndarray,
    stable: np.ndarray,
    *,
    fractions: Iterable[float] = (1.0 / 3.0, 2.0 / 3.0),
    delta_threshold: int = 2,
) -> list[np.ndarray]:
    """Fade only newly changed pixels while preserving the rest of the frame bit-for-bit.

    This is used for atomic semantic commits such as adding a complete evidence
    layer or filling one predicted panel.  It is deliberately not a full-frame
    crossfade and has no scan direction: unchanged objects, labels, panels,
    background, and camera remain exactly stable.
    """

    before = np.asarray(previous, dtype=np.uint8)
    after = np.asarray(stable, dtype=np.uint8)
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3:
        raise ValueError("changed-region opacity transition requires equal RGB frame shapes")
    changed = np.max(
        np.abs(before.astype(np.int16) - after.astype(np.int16)),
        axis=2,
    ) > int(delta_threshold)
    frames: list[np.ndarray] = []
    for raw_fraction in fractions:
        fraction = float(raw_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("opacity transition fractions must lie strictly between 0 and 1")
        frame = before.copy()
        blended = np.rint(
            before.astype(np.float32) * (1.0 - fraction)
            + after.astype(np.float32) * fraction
        ).astype(np.uint8)
        frame[changed] = blended[changed]
        frames.append(frame)
    return frames


__all__ = ["changed_region_opacity_frames",]

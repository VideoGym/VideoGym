import cv2
import numpy as np


def _result(checks, evidence=None):
    for name, value in checks.items():
        if type(value) not in (bool, np.bool_):
            raise TypeError(
                f"Predicate {name!r} must be a boolean, received {type(value).__name__}"
            )
    return {
        "passed": bool(checks and all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "evidence": evidence or {},
    }


def _mask(frame, rgb, tolerance=35):
    return np.max(np.abs(frame.astype(np.int16) - np.array(rgb[::-1], np.int16)), axis=2) <= tolerance


def _block_mask(frame, rgb):
    """Join a solid block's detached, narrow codec boundary fringe."""
    raw = _mask(frame, rgb).astype(np.uint8)
    return cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)


def _blobs(mask, minimum=20):
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return [
        (center, int(stat[cv2.CC_STAT_AREA]), stat[:4])
        for stat, center in zip(stats[1:count], centers[1:count])
        if stat[cv2.CC_STAT_AREA] >= minimum
    ]


def _static(frames, first, allowed):
    """No changed connected feature outside the allowed task regions."""
    reference = cv2.GaussianBlur(first, (3, 3), 0).astype(np.int16)
    worst = 0
    for frame in frames:
        difference = np.max(np.abs(cv2.GaussianBlur(frame, (3, 3), 0).astype(np.int16) - reference), axis=2)
        unexpected = (difference > 65) & ~allowed
        areas = [blob[1] for blob in _blobs(unexpected, minimum=1)]
        worst = max(worst, max(areas, default=0))
    return worst <= 30, worst


def _layers(metadata, phase="first_frame"):
    return metadata.get("generic_declarative_render", {}).get(phase, {}).get("layers", [])

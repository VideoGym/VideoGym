"""Read a visible numeric glyph against a source-rendered vocabulary."""
from __future__ import annotations

from typing import Mapping

import cv2
import numpy as np


def numeric_glyph_descriptor(patch: np.ndarray, *, light: bool = False) -> np.ndarray | None:
    """Remove background, normalize position/size, and retain the ink shape.

    The supplied crop owns the complete glyph and excludes surrounding labels.
    Components touching its boundary belong to the surrounding cell/slot border.
    """
    rgb = np.asarray(patch, dtype=np.int16)
    if not rgb.size:
        return None
    if light:
        ink = (rgb.min(axis=2) > 150) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 120)
    else:
        ink = (rgb.min(axis=2) < 115) & (rgb.max(axis=2) < 200)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    kept = np.zeros(ink.shape, dtype=np.uint8)
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area >= 5 and x > 0 and y > 0 and x + width < ink.shape[1] and y + height < ink.shape[0]:
            kept[labels == index] = 1
    ys, xs = np.where(kept)
    if not len(xs):
        return None
    tight = kept[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = tight.shape
    normalized_width = max(1, round(width * 28 / height))
    if normalized_width > 80:
        return None
    scaled = cv2.resize(tight.astype(np.float32), (normalized_width, 28), interpolation=cv2.INTER_AREA)
    descriptor = np.zeros((36, 80), dtype=np.float32)
    left = (80 - normalized_width) // 2
    descriptor[4:32, left:left + normalized_width] = scaled
    return cv2.GaussianBlur(descriptor, (3, 3), 0.7)


def _closed_counter_centres(descriptor: np.ndarray) -> list[float]:
    """Read substantial enclosed background regions inside the ink shape."""
    background = (descriptor <= .5).astype(np.uint8)
    count, _labels, stats, centres = cv2.connectedComponentsWithStats(background, 8)
    height, width = background.shape
    return sorted(float(centres[index][1]) for index in range(1, count)
                  if stats[index, cv2.CC_STAT_AREA] >= 8
                  and stats[index, cv2.CC_STAT_LEFT] > 0 and stats[index, cv2.CC_STAT_TOP] > 0
                  and stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH] < width
                  and stats[index, cv2.CC_STAT_TOP] + stats[index, cv2.CC_STAT_HEIGHT] < height)


def _closed_counter_compatible(observed: list[float], template: np.ndarray) -> bool:
    expected = _closed_counter_centres(template)
    # 0 has one centred counter, 6/9 one low/high counter, and 8 two. The
    # allowance is the same two-pixel alignment plus one contour pixel used
    # by the shape matcher; it does not require identical counter areas.
    return len(observed) == len(expected) and all(abs(left - right) <= 3.0
                                                for left, right in zip(observed, expected))


def _contour_similarities(observed: np.ndarray, templates: list[np.ndarray]) -> np.ndarray:
    """Match complete ink contours with bounded raster correspondence.

    Tight glyph crops can move the stem inside their normalized box when a
    serif grows by one native pixel. Small translations and one normalized
    pixel of edge tolerance retain the stroke shape without requiring the
    font's exact thickness. Both observed and expected ink must be supported.
    """
    ink = observed > .35
    kernel = np.ones((3, 3), dtype=np.uint8)
    nearby = cv2.dilate(ink.astype(np.uint8), kernel).astype(bool)
    padded = np.pad(ink, ((2, 2), (3, 3)))
    padded_nearby = np.pad(nearby, ((2, 2), (3, 3)))
    height, width = ink.shape
    shifts, near_shifts = [], []
    for dy in range(-2, 3):
        for dx in range(-3, 4):
            shifts.append(padded[2 + dy:2 + dy + height, 3 + dx:3 + dx + width].ravel())
            near_shifts.append(padded_nearby[2 + dy:2 + dy + height, 3 + dx:3 + dx + width].ravel())
    source = np.stack([template > .35 for template in templates])
    source_nearby = np.stack([cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) for mask in source])
    # Compute all small translations and vocabulary matches together, rather
    # than warping and dilating an image separately for every candidate digit.
    observed_ink = np.asarray(shifts, dtype=np.float32)
    observed_nearby = np.asarray(near_shifts, dtype=np.float32)
    source_ink = source.reshape(len(templates), -1).astype(np.float32)
    source_near = source_nearby.reshape(len(templates), -1).astype(np.float32)
    precision = observed_ink @ source_near.T / np.maximum(1.0, observed_ink.sum(axis=1))[:, None]
    recall = observed_nearby @ source_ink.T / np.maximum(1.0, source_ink.sum(axis=1))[None, :]
    return np.minimum(precision, recall).max(axis=0)


def read_numeric_glyph(
    patch: np.ndarray,
    vocabulary: Mapping[int, np.ndarray],
    *,
    light: bool = False,
) -> tuple[int | None, float, float]:
    """Decode independently of the expected answer, allowing raster variation."""
    observed = numeric_glyph_descriptor(patch, light=light)
    if observed is None:
        return None, 0.0, 0.0
    if not vocabulary:
        return None, 0.0, 0.0
    entries = list(vocabulary.items())
    pixel_scores = [
        (float(np.sum(observed * template) / max(1e-6, np.linalg.norm(observed) * np.linalg.norm(template))), value)
        for value, template in entries
    ]
    ranked = sorted(pixel_scores)
    similarity, value = ranked[-1]
    margin = similarity - (ranked[-2][0] if len(ranked) > 1 else 0.0)
    # Confident pixel-shape readings already distinguish thin internal strokes
    # such as 0/8. Keep those identities; use bounded contour correspondence
    # when font/serif variation makes the literal ink overlap inconclusive.
    if (similarity >= .78 and margin >= .025) or (value == -1 and similarity >= .65 and margin >= .10):
        return value, similarity, margin
    contour_scores = _contour_similarities(observed, [template for _value, template in entries])
    # Counter evidence resolves contour matches that would otherwise fill the
    # waist of an 8 with the sides of a narrow 0. Use it when the observation
    # has substantial enclosed regions; a raster gap in an otherwise clearly
    # matched glyph must not invalidate a confident pixel-shape reading.
    observed_counters = _closed_counter_centres(observed) if any(value in {0, 6, 8, 9} for value in vocabulary) else []
    if observed_counters:
        compatible = [value not in {0, 6, 8, 9} or _closed_counter_compatible(observed_counters, template)
                      for value, template in entries]
        contour_scores = np.where(compatible, contour_scores, 0.0)
    ranked = sorted(
        (max(similarity, float(contour_score)), value)
        for (similarity, value), contour_score in zip(pixel_scores, contour_scores)
    )
    similarity, value = ranked[-1]
    margin = similarity - (ranked[-2][0] if len(ranked) > 1 else 0.0)
    # The separated minus and narrow 1 change their relative spacing under
    # small raster shifts; require a wider runner-up margin for that glyph.
    recognized = (similarity >= .78 and margin >= .025) or (value == -1 and similarity >= .65 and margin >= .10)
    return (value if recognized else None), similarity, margin

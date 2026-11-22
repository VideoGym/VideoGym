"""Shared pixel and state-sequence measurements, independent of task goals."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image
from data.common.verifiers.keyframes import visual_delta
from typing import Any, List, Mapping, Optional, Sequence


CANDIDATE_STATE_MAX_UNRECOGNIZED_FRACTION = 0.25


def _pixels_near_rgb(frame_bgr: np.ndarray, rgb: Sequence[int], *, tolerance: float) -> np.ndarray:
    target = np.asarray([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
    delta = frame_bgr.astype(np.float32) - target
    return np.sqrt(np.sum(delta * delta, axis=2)) <= float(tolerance)


def _analyze_fixed_visual_state_sequence(
    *,
    frame_count: int,
    recognized: Sequence[tuple[int, Any]],
    expected_states: Sequence[Any],
) -> dict[str, Any]:
    compact_groups: List[dict[str, Any]] = []
    for frame_index, state in recognized:
        if compact_groups and compact_groups[-1]["state"] == state:
            compact_groups[-1]["frame_indices"].append(frame_index)
        else:
            compact_groups.append({"state": state, "frame_indices": [frame_index]})
    observed_states = [group["state"] for group in compact_groups]
    recognized_count = len(recognized)
    unrecognized_fraction = 1.0 - (recognized_count / frame_count)
    starts_at_input = bool(observed_states and observed_states[0] == expected_states[0])
    ends_at_final = bool(observed_states and observed_states[-1] == expected_states[-1])
    has_initial_hold = bool(
        len(recognized) >= 2
        and recognized[0] == (0, expected_states[0])
        and recognized[1] == (1, expected_states[0])
    )
    has_final_hold = bool(
        len(recognized) >= 2
        and recognized[-2] == (frame_count - 2, expected_states[-1])
        and recognized[-1] == (frame_count - 1, expected_states[-1])
    )
    expected_indices: List[int] = []
    for state in observed_states:
        try:
            expected_indices.append(expected_states.index(state))
        except ValueError:
            expected_indices.append(-1)
    skipped_layers = any(
        previous >= 0 and current >= 0 and current > previous + 1
        for previous, current in zip(expected_indices, expected_indices[1:])
    )
    return {
        "compact_groups": compact_groups,
        "observed_states": observed_states,
        "recognized_count": recognized_count,
        "unrecognized_fraction": unrecognized_fraction,
        "starts_at_input": starts_at_input,
        "ends_at_final": ends_at_final,
        "has_initial_hold": has_initial_hold,
        "has_final_hold": has_final_hold,
        "expected_indices": expected_indices,
        "skipped_layers": skipped_layers,
        "exact_sequence": observed_states == list(expected_states),
    }


def _printed_number_signature(
    frame_bgr: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    token_width: float,
    token_height: float,
) -> Optional[np.ndarray]:
    """Read the dark printed glyph in one fixed number token.

    The signature comes from the visible source image, not from metadata or a
    renderer-side digit palette.  This makes terminal verification sensitive
    to the actual printed number while allowing ordinary antialiasing and
    video compression noise.
    """

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3 or token_width <= 0 or token_height <= 0:
        return None
    height, width = frame_bgr.shape[:2]
    scale_x = width / 512.0
    scale_y = height / 512.0
    half_width = token_width * scale_x / 2.0
    half_height = token_height * scale_y / 2.0
    left = max(0, int(round(center_x * scale_x - half_width)))
    right = min(width, int(round(center_x * scale_x + half_width + 1)))
    top = max(0, int(round(center_y * scale_y - half_height)))
    bottom = min(height, int(round(center_y * scale_y + half_height + 1)))
    patch = frame_bgr[top:bottom, left:right, :3]
    if patch.size == 0:
        return None
    rgb = patch[:, :, ::-1].astype(np.float32)
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    # Keep the teal visited-cell background out of the signature; only the
    # dark printed glyph should survive this threshold.
    ink = luminance <= 110.0
    target_width = max(8, int(round(token_width)) + 1)
    target_height = max(8, int(round(token_height)) + 1)
    if ink.shape != (target_height, target_width):
        resized = Image.fromarray((ink.astype(np.uint8) * 255), mode="L").resize(
            (target_width, target_height),
            resample=Image.Resampling.NEAREST,
        )
        ink = np.asarray(resized, dtype=np.uint8) >= 128
    if int(ink.sum()) < 8:
        return None
    return ink


def _printed_number_templates(
    frame_bgr: np.ndarray,
    *,
    matrix: Sequence[Sequence[int]],
    grid_bbox: Sequence[Any],
    cell_size: float,
    token_width: float,
    token_height: float,
) -> Optional[dict[int, np.ndarray]]:
    if len(grid_bbox) != 4 or cell_size <= 0:
        return None
    try:
        left = float(grid_bbox[0])
        top = float(grid_bbox[1])
    except Exception:
        return None
    templates: dict[int, np.ndarray] = {}
    for row, values in enumerate(matrix):
        for col, raw_value in enumerate(values):
            value = int(raw_value)
            signature = _printed_number_signature(
                frame_bgr,
                center_x=left + (col + 0.5) * cell_size,
                center_y=top + (row + 0.5) * cell_size,
                token_width=token_width,
                token_height=token_height,
            )
            if signature is None or value in templates:
                return None
            templates[value] = signature
    return templates


def _classify_printed_number(
    frame_bgr: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    token_width: float,
    token_height: float,
    templates: Mapping[int, np.ndarray],
) -> Optional[int]:
    observed = _printed_number_signature(
        frame_bgr,
        center_x=center_x,
        center_y=center_y,
        token_width=token_width,
        token_height=token_height,
    )
    if observed is None or not templates:
        return None
    scores: List[tuple[float, int]] = []
    for value, template in templates.items():
        if template.shape != observed.shape:
            continue
        union = int(np.logical_or(observed, template).sum())
        if union <= 0:
            continue
        distance = float(np.logical_xor(observed, template).sum()) / float(union)
        scores.append((distance, int(value)))
    if not scores:
        return None
    scores.sort()
    best_distance, best_value = scores[0]
    second_distance = scores[1][0] if len(scores) > 1 else 1.0
    if best_distance > 0.34 or second_distance - best_distance < 0.035:
        return None
    return best_value


def static_mutation_fraction(
    reference_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    *,
    allowed_change_bboxes: Sequence[Sequence[float]],
    codec_tolerant: bool = False,
) -> float:
    if reference_bgr.shape != frame_bgr.shape:
        return 1.0
    height, width = frame_bgr.shape[:2]
    scale_x = width / 512.0
    scale_y = height / 512.0
    allowed = np.zeros((height, width), dtype=bool)
    padding = max(4, int(round(7 * min(scale_x, scale_y))))
    for bbox in allowed_change_bboxes:
        if len(bbox) != 4:
            return 1.0
        x0 = max(0, int(math.floor(float(bbox[0]) * scale_x)) - padding)
        y0 = max(0, int(math.floor(float(bbox[1]) * scale_y)) - padding)
        x1 = min(width, int(math.ceil(float(bbox[2]) * scale_x)) + padding)
        y1 = min(height, int(math.ceil(float(bbox[3]) * scale_y)) + padding)
        allowed[y0:y1, x0:x1] = True
    static = ~allowed
    if not static.any():
        return 1.0
    if codec_tolerant:
        delta = visual_delta(frame_bgr, reference_bgr).mean(axis=2)
        # Keep a local obligation for fixed content; a painted patch cannot be
        # averaged away by the rest of a nearly unchanged white canvas.
        masked = np.where(static, delta, 0.0)
        if any(float(masked[y:y + 32, x:x + 32].mean()) > 12
               for y in range(0, height, 32) for x in range(0, width, 32)):
            return 1.0
    else:
        delta = np.mean(
            np.abs(frame_bgr.astype(np.int16) - reference_bgr.astype(np.int16)),
            axis=2,
        )
    return float((delta[static] >= 18.0).mean())

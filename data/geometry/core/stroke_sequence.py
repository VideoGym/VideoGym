"""Source-derived checks for serial endpoint-to-endpoint Geometry strokes.

The helper is deliberately unaware of task names.  A task verifier supplies
the ordered centerline of every answer-bearing construction stroke, while this
module checks the rendered video against its own source and terminal frames.
Static input lines are therefore ignored, and no saved reference video is
used as an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class EndpointStroke:
    """One stroke whose point order declares its start and moving endpoint."""

    label: str
    points: tuple[tuple[float, float], ...]
    width: float


def _resample_polyline(
    points: Sequence[Sequence[float]],
    *,
    sample_count: int = 65,
) -> np.ndarray:
    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    segment_lengths = np.linalg.norm(raw[1:] - raw[:-1], axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.empty((0, 2), dtype=np.float64)
    distances = np.linspace(0.0, total, max(9, int(sample_count)))
    result = np.empty((len(distances), 2), dtype=np.float64)
    segment_index = 0
    for index, distance in enumerate(distances):
        while (
            segment_index + 1 < len(segment_lengths)
            and distance > cumulative[segment_index + 1]
        ):
            segment_index += 1
        length = float(segment_lengths[segment_index])
        ratio = 0.0 if length <= 1e-6 else (distance - cumulative[segment_index]) / length
        result[index] = raw[segment_index] + (raw[segment_index + 1] - raw[segment_index]) * ratio
    return result


def _stroke_bin_masks(
    shape: tuple[int, int],
    stroke: EndpointStroke,
    expected_dynamic: np.ndarray,
    *,
    bin_count: int = 8,
) -> list[np.ndarray]:
    points = _resample_polyline(stroke.points, sample_count=bin_count * 12 + 1)
    if len(points) < bin_count + 1:
        return []
    masks: list[np.ndarray] = []
    edges = np.linspace(0, len(points) - 1, bin_count + 1).round().astype(int)
    thickness = max(3, int(round(float(stroke.width) + 6.0)))
    for index in range(bin_count):
        start = max(0, int(edges[index]))
        end = min(len(points) - 1, int(edges[index + 1]))
        path = np.rint(points[start : end + 1]).astype(np.int32)
        mask = np.zeros(shape, dtype=np.uint8)
        if len(path) >= 2:
            cv2.polylines(mask, [path], False, 1, thickness=thickness, lineType=cv2.LINE_8)
        # Construction strokes frequently meet at shared vertices.  Endpoint
        # markers or a previously completed incident stroke must not make a
        # later stroke look active before its own pen tip arrives.
        endpoint_mask = np.zeros(shape, dtype=np.uint8)
        endpoint_radius = max(3, int(round(float(stroke.width) + 3.0)))
        for endpoint in (points[0], points[-1]):
            cv2.circle(
                endpoint_mask,
                tuple(np.rint(endpoint).astype(int)),
                endpoint_radius,
                1,
                thickness=-1,
            )
        masks.append(mask.astype(bool) & ~endpoint_mask.astype(bool) & expected_dynamic)
    return masks


def _pixel_terminal_projection(
    frame: np.ndarray,
    source: np.ndarray,
    terminal: np.ndarray,
    *, pixel_reach: int = 0,
) -> np.ndarray:
    if pixel_reach:
        # Match nearby observed pixels before projecting: a projection mask
        # restricted to the exact centerline cannot recover displaced ink.
        return np.maximum.reduce([_pixel_terminal_projection(
            np.roll(frame,(dy,dx),axis=(0,1)),source,terminal)
            for dy in range(-pixel_reach,pixel_reach+1)
            for dx in range(-pixel_reach,pixel_reach+1)])
    delta = terminal.astype(np.float32) - source.astype(np.float32)
    current = frame.astype(np.float32) - source.astype(np.float32)
    energy = np.sum(delta * delta, axis=2)
    projection = np.zeros(energy.shape, dtype=np.float32)
    valid = energy >= 36.0
    projection[valid] = np.sum(current[valid] * delta[valid], axis=1) / energy[valid]
    return np.clip(projection, 0.0, 1.0)


def analyze_serial_endpoint_strokes(
    frames: Sequence[np.ndarray],
    source: np.ndarray,
    terminal: np.ndarray,
    strokes: Sequence[EndpointStroke],
    *, pixel_reach: int = 0,
) -> tuple[dict[str, Any], bool]:
    """Verify ordered completion of source-declared construction strokes.

    Progress is measured only where the supplied terminal differs from the
    supplied source.  This makes the check invariant to unchanged input lines
    and independent of generator metadata such as a saved correct option.
    """

    arrays = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    shape_valid = bool(
        arrays
        and source.shape == terminal.shape == arrays[0].shape
        and source.ndim == 3
        and source.shape[2] == 3
        and all(frame.shape == source.shape for frame in arrays)
        and bool(strokes)
    )
    if not shape_valid:
        metrics = {
            "serial_stroke_check_pass": False,
            "serial_stroke_reason": "frames, endpoints, or stroke specifications are missing or malformed",
        }
        return metrics, False

    expected_dynamic = np.max(
        np.abs(terminal.astype(np.int16) - source.astype(np.int16)), axis=2
    ) >= 12
    masks_by_stroke = [
        _stroke_bin_masks(source.shape[:2], stroke, expected_dynamic)
        for stroke in strokes
    ]
    mask_valid = bool(
        all(
            len(masks) == 8 and all(int(mask.sum()) >= 4 for mask in masks)
            for masks in masks_by_stroke
        )
    )
    if not mask_valid:
        metrics = {
            "serial_stroke_check_pass": False,
            "serial_stroke_reason": "a declared stroke is not visible in the source-derived terminal",
            "serial_stroke_labels": [stroke.label for stroke in strokes],
            "serial_stroke_bin_pixel_counts": [
                [int(mask.sum()) for mask in masks] for masks in masks_by_stroke
            ],
        }
        return metrics, False

    # Read completion of each source-declared stroke. Its precise drawing
    # speed, opacity curve and within-stroke tip raster are not hard gates.
    progress = []
    def project(frame):
        return _pixel_terminal_projection(frame, source, terminal, pixel_reach=pixel_reach)
    terminal_projection = project(arrays[-1])
    terminal_strength = [float(np.mean([terminal_projection[mask].mean() for mask in masks]))
                         for masks in masks_by_stroke]
    # Retain broad color/coverage evidence at the endpoint.  The per-stroke
    # observed strength then sets the scale of that same stroke through time,
    # so a thinner or paler retained line does not become an absent state.
    terminal_coverage = [float(np.mean([np.quantile(terminal_projection[mask], 0.65)
                                       for mask in masks])) for masks in masks_by_stroke]
    for frame in arrays:
        projection = project(frame)
        progress.append([float(np.mean([projection[mask].mean() for mask in masks])) / max(strength, 0.1)
                         for masks, strength in zip(masks_by_stroke, terminal_strength)])
    complete_frames = [next((i for i, row in enumerate(progress) if row[j] >= .78), None)
                       for j in range(len(strokes))]
    start_clear = max(progress[0]) <= .15
    final_complete = min(terminal_coverage) >= .78
    ordered = all(a is not None and b is not None and a < b
                  for a, b in zip(complete_frames, complete_frames[1:]))
    # A completed semantic element must not be persistently erased. Ignore
    # a single ambiguous codec frame; two consecutive absent observations fail.
    regressions = 0
    for j, start in enumerate(complete_frames):
        if start is not None:
            regressions += sum(progress[i][j] < .45 and progress[i + 1][j] < .45
                               for i in range(start, len(progress) - 1))
    order_violations = sum(row[j] > .30 and min(row[:j]) < .60
                           for row in progress for j in range(1, len(strokes)))
    passed = bool(start_clear and final_complete and ordered and not order_violations and not regressions)
    reason = ("construction is already present in the initial state" if not start_clear else
              "required construction strokes are missing from the final state" if not final_complete else
              "required stroke completions are skipped or out of order" if not ordered or order_violations else
              "a completed construction stroke is erased" if regressions else "ok")
    return {
        "serial_stroke_check_pass": passed,
        "serial_stroke_reason": reason,
        "serial_stroke_labels": [stroke.label for stroke in strokes],
        "serial_stroke_progress_by_frame": progress,
        "serial_stroke_completion_frames": complete_frames,
        "serial_stroke_order_violation_count": order_violations,
        "serial_stroke_regressions": regressions,
        "serial_stroke_input_clear": start_clear,
        "serial_stroke_terminal_complete": final_complete,
        "serial_stroke_terminal_strength": terminal_strength,
        "serial_stroke_terminal_coverage": terminal_coverage,
    }, passed


__all__ = [
    "EndpointStroke",
    "analyze_serial_endpoint_strokes",
]

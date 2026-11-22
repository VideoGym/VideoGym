"""Independent multimodal verifier for ``circumcenter``."""

from __future__ import annotations

from data.geometry.core.phase_observation import phase_onset, selection_onset

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from data.geometry.core.point_target_base import PointTargetPuzzleGenerator
from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.common.verifiers.routes import VerifierRoute


CIRCUMCENTER_VERIFIER_PROFILE = "circumcenter_multimodal_semantic_hard_gate_v1"
CIRCUMCENTER_STATE_TRACE = [
    "input_hold",
    "first_perpendicular_bisector_single_end_growth",
    "second_perpendicular_bisector_single_end_growth",
    "third_perpendicular_bisector_single_end_growth",
    "circumcircle_single_end_growth",
    "correct_candidate_highlight",
    "final_hold",
]
def _finite_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _point(value: object) -> Optional[Tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x = _finite_float(value[0])
    y = _finite_float(value[1])
    return None if x is None or y is None else (x, y)


def _float_close(value: object, expected: float, tolerance: float = 1e-5) -> bool:
    parsed = _finite_float(value)
    return parsed is not None and abs(parsed - expected) <= tolerance


def _direction_name(dx: float, dy: float) -> str:
    names = (
        "right",
        "lower-right",
        "down",
        "lower-left",
        "left",
        "upper-left",
        "up",
        "upper-right",
    )
    angle = math.atan2(dy, dx) % math.tau
    return names[int((angle + math.pi / 8.0) // (math.pi / 4.0)) % 8]


def _ordered_vertices(vertices: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    centroid_x = sum(point[0] for point in vertices) / 3.0
    centroid_y = sum(point[1] for point in vertices) / 3.0
    clockwise = sorted(
        vertices,
        key=lambda point: math.atan2(point[1] - centroid_y, point[0] - centroid_x),
    )
    start = min(range(3), key=lambda index: (clockwise[index][1], clockwise[index][0]))
    return clockwise[start:] + clockwise[:start]


def _circumcenter(
    vertices: Sequence[Tuple[float, float]],
) -> Optional[Tuple[Tuple[float, float], float]]:
    (ax, ay), (bx, by), (cx, cy) = vertices
    denominator = 2.0 * (
        ax * (by - cy)
        + bx * (cy - ay)
        + cx * (ay - by)
    )
    if abs(denominator) < 1e-9:
        return None
    square_a = ax * ax + ay * ay
    square_b = bx * bx + by * by
    square_c = cx * cx + cy * cy
    ux = (
        square_a * (by - cy)
        + square_b * (cy - ay)
        + square_c * (ay - by)
    ) / denominator
    uy = (
        square_a * (cx - bx)
        + square_b * (ax - cx)
        + square_c * (bx - ax)
    ) / denominator
    center = (ux, uy)
    return center, math.dist(center, vertices[0])


def _location_relative_to_triangle(
    point: Tuple[float, float],
    vertices: Sequence[Tuple[float, float]],
) -> str:
    signs: List[float] = []
    for index, a in enumerate(vertices):
        b = vertices[(index + 1) % 3]
        signs.append(
            (b[0] - a[0]) * (point[1] - a[1])
            - (b[1] - a[1]) * (point[0] - a[0])
        )
    if any(abs(value) <= 1e-6 for value in signs) and not (
        any(value < -1e-6 for value in signs)
        and any(value > 1e-6 for value in signs)
    ):
        return "on the triangle boundary"
    if not (
        any(value < 0.0 for value in signs)
        and any(value > 0.0 for value in signs)
    ):
        return "inside the triangle"
    return "outside the triangle"


def _points_close(left: object, right: Tuple[float, float], tolerance: float = 1e-5) -> bool:
    parsed = _point(left)
    return parsed is not None and math.dist(parsed, right) <= tolerance


def replay_circumcenter_geometry(
    record: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Recompute the unique displayed circumcenter without trusting an answer key."""

    metrics: Dict[str, Any] = {
        "verifier_profile": CIRCUMCENTER_VERIFIER_PROFILE,
        "circumcenter_geometry_replay_valid": False,
        "circumcenter_geometry_replay_reason": "invalid visible source geometry",
    }
    raw_vertices = record.get("triangle_vertices")
    if not isinstance(raw_vertices, list) or len(raw_vertices) != 3:
        return metrics, False, None
    parsed_vertices = [_point(value) for value in raw_vertices]
    if any(point is None for point in parsed_vertices):
        return metrics, False, None
    vertices = _ordered_vertices([point for point in parsed_vertices if point is not None])
    circle = _circumcenter(vertices)
    if circle is None:
        metrics["circumcenter_geometry_replay_reason"] = "triangle vertices are collinear"
        return metrics, False, None
    center, radius = circle
    if not 45.0 <= radius <= 250.0:
        metrics["circumcenter_geometry_replay_reason"] = "derived circumcircle is outside the supported scale"
        return metrics, False, None

    raw_candidates = record.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 5:
        return metrics, False, None
    candidates: List[Tuple[str, Tuple[float, float]]] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            return metrics, False, None
        label = str(raw.get("label") or "").strip().upper()
        candidate = _point([raw.get("x"), raw.get("y")])
        if label not in "ABCDE" or candidate is None:
            return metrics, False, None
        candidates.append((label, candidate))
    if sorted(label for label, _candidate in candidates) != list("ABCDE"):
        return metrics, False, None
    candidates.sort(key=lambda item: item[0])

    centroid_x = sum(point[0] for point in vertices) / 3.0
    centroid_y = sum(point[1] for point in vertices) / 3.0
    derived: List[Dict[str, Any]] = []
    for label, candidate in candidates:
        distances = [math.dist(candidate, vertex) for vertex in vertices]
        derived.append(
            {
                "label": label,
                "center": [candidate[0], candidate[1]],
                "direction_from_triangle_centroid": _direction_name(
                    candidate[0] - centroid_x,
                    candidate[1] - centroid_y,
                ),
                "vertex_distances_pixels": distances,
                "distance_spread_pixels": max(distances) - min(distances),
                "mean_vertex_distance_pixels": sum(distances) / 3.0,
            }
        )
    ordered = sorted(derived, key=lambda item: float(item["distance_spread_pixels"]))
    derived_correct = str(ordered[0]["label"])
    best_spread = float(ordered[0]["distance_spread_pixels"])
    runner_up_spread = float(ordered[1]["distance_spread_pixels"])
    unique = bool(
        best_spread <= 1e-5
        and runner_up_spread >= 24.0
        and math.dist(candidates[ord(derived_correct) - ord("A")][1], center) <= 1e-5
    )
    derived_index = ord(derived_correct) - ord("A")
    location = _location_relative_to_triangle(center, vertices)

    structured = record.get("structured_answer")
    if not isinstance(structured, dict):
        return metrics, False, derived_correct
    stored_vertices = structured.get("triangle_vertices")
    vertices_valid = bool(
        isinstance(stored_vertices, list)
        and len(stored_vertices) == 3
        and all(_points_close(stored_vertices[index], vertices[index]) for index in range(3))
    )
    stored_candidates = structured.get("candidate_centers")
    candidate_metadata_valid = isinstance(stored_candidates, list) and len(stored_candidates) == 5
    if candidate_metadata_valid:
        by_label = {
            str(item.get("label") or "").strip().upper(): item.get("center")
            for item in stored_candidates
            if isinstance(item, dict)
        }
        candidate_metadata_valid = sorted(by_label) == list("ABCDE") and all(
            _points_close(by_label[label], candidates[index][1])
            for index, label in enumerate("ABCDE")
        )

    stored_measurements = structured.get("candidate_measurements")
    measurements_valid = isinstance(stored_measurements, list) and len(stored_measurements) == 5
    if measurements_valid:
        stored_by_label = {
            str(item.get("label") or "").strip().upper(): item
            for item in stored_measurements
            if isinstance(item, dict)
        }
        measurements_valid = sorted(stored_by_label) == list("ABCDE") and all(
            _points_close(stored_by_label[label].get("center"), candidates[index][1])
            and stored_by_label[label].get("direction_from_triangle_centroid")
            == derived[index]["direction_from_triangle_centroid"]
            and isinstance(stored_by_label[label].get("vertex_distances_pixels"), list)
            and len(stored_by_label[label]["vertex_distances_pixels"]) == 3
            and all(
                _float_close(
                    stored_by_label[label]["vertex_distances_pixels"][vertex_index],
                    float(derived[index]["vertex_distances_pixels"][vertex_index]),
                )
                for vertex_index in range(3)
            )
            and _float_close(
                stored_by_label[label].get("distance_spread_pixels"),
                float(derived[index]["distance_spread_pixels"]),
            )
            and _float_close(
                stored_by_label[label].get("mean_vertex_distance_pixels"),
                float(derived[index]["mean_vertex_distance_pixels"]),
            )
            for index, label in enumerate("ABCDE")
        )

    answer_schema = structured.get("answer_schema")
    verifier_contract = structured.get("verifier_contract")
    render_geometry = structured.get("render_geometry")
    contracts_valid = bool(
        isinstance(answer_schema, dict)
        and answer_schema.get("terminal_text_regex") == FIVE_CHOICE_CODEC.full_pattern
        and answer_schema.get("requires_sample_specific_reasoning_before_terminal_answer") is True
        and answer_schema.get("terminal_text_contains_exact_answer") is True
        and answer_schema.get("no_block_after_terminal_image") is True
        and isinstance(verifier_contract, dict)
        and verifier_contract.get("profile") == CIRCUMCENTER_VERIFIER_PROFILE
        and verifier_contract.get("strict_binary_correctness") is True
        and isinstance(render_geometry, dict)
        and render_geometry.get("canvas_dimensions") == [512, 512]
        and _finite_float(record.get("point_radius")) is not None
        and _float_close(
            render_geometry.get("point_radius"),
            float(_finite_float(record.get("point_radius"))),
        )
        and structured.get("state_trace") == CIRCUMCENTER_STATE_TRACE
        and structured.get("expected_video_frame_count") == 33
        and structured.get("video_fps") == 16
    )
    redundant_metadata_valid = bool(
        vertices_valid
        and _points_close(record.get("circumcenter"), center)
        and _points_close(structured.get("circumcenter"), center)
        and _float_close(record.get("circumradius"), radius)
        and _float_close(structured.get("circumradius"), radius)
        and str(record.get("correct_option") or "").strip().upper() == derived_correct
        and str(structured.get("correct_option") or "").strip().upper() == derived_correct
        and structured.get("correct_index") == derived_index
        and _float_close(
            structured.get("runner_up_distance_spread_pixels"),
            runner_up_spread,
        )
        and structured.get("circumcenter_location_relative_to_triangle") == location
        and structured.get("multiple_solution_policy")
        == "exactly_one_displayed_candidate_is_the_unique_non_collinear_circumcenter"
    )
    valid = bool(unique)
    metrics.update(
        {
            "circumcenter_geometry_replay_valid": valid,
            "circumcenter_geometry_replay_reason": (
                "ok" if valid else "source geometry is invalid or non-unique"
            ),
            "circumcenter_derived_correct_option": derived_correct,
            "circumcenter_derived_correct_index": derived_index,
            "circumcenter_derived_center": [center[0], center[1]],
            "circumcenter_derived_radius": radius,
            "circumcenter_location_relative_to_triangle": location,
            "circumcenter_runner_up_distance_spread_pixels": runner_up_spread,
            "circumcenter_candidate_measurements": derived,
            "circumcenter_unique_displayed_solution": unique,
            "circumcenter_candidate_metadata_valid": candidate_metadata_valid,
            "circumcenter_measurement_metadata_valid": measurements_valid,
            "circumcenter_contract_metadata_valid": contracts_valid,
            "circumcenter_redundant_metadata_valid": redundant_metadata_valid,
        }
    )
    return metrics, valid, derived_correct


def _contract_metrics(
    *,
    artifact_kind: str,
    binary_correct: bool,
    component_scores: Dict[str, float],
    failure_reason: str,
    started: float,
) -> Dict[str, Any]:
    return {
        "verifier_profile": CIRCUMCENTER_VERIFIER_PROFILE,
        "artifact_kind": artifact_kind,
        "component_scores": component_scores,
        "binary_correct": bool(binary_correct),
        "strict_binary_correctness": True,
        "failure_reason": "ok" if binary_correct else failure_reason,
        "verification_elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "performance_policy": "latency_throughput_memory_and_decode_counts_are_diagnostics_without_failure_thresholds",
    }


def _read_rgb(path: Path, size: Tuple[int, int]) -> Optional[np.ndarray]:
    bgr = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    width, height = size
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return rgb


def _color_masks(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    cyan = (
        (red <= 95)
        & (green >= 105)
        & (blue >= 135)
        & (green - red >= 45)
        & (blue - red >= 70)
    )
    dark_red = (
        (red >= 130)
        & (green <= 125)
        & (red - green >= 45)
        & (red - blue >= 20)
    )
    return cyan, dark_red


def _red_option(frame: np.ndarray, record: Dict[str, Any]) -> Tuple[Optional[str], np.ndarray]:
    _cyan, dark_red = _color_masks(frame)
    radius = max(12, int(round(float(record.get("point_radius") or 11) * 1.35)))
    height, width = dark_red.shape
    yy, xx = np.ogrid[:height, :width]
    selected: List[str] = []
    for candidate in record.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        center = _point([candidate.get("x"), candidate.get("y")])
        if center is None:
            continue
        disk = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius * radius
        if int(np.count_nonzero(dark_red & disk)) >= 15:
            selected.append(str(candidate.get("label") or "").strip().upper())
    label = "".join(sorted(item for item in selected if item)) or None
    return label, dark_red


def _line_support(
    mask: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    excluded_disks: Sequence[Tuple[float, float, float]] = (),
) -> float:
    height, width = mask.shape
    hits = 0
    total = 0
    # Integrate visible line coverage densely; a sparse 28-point grid can
    # shift a borderline ratio by an entire 4% sample after rasterization.
    for fraction in np.linspace(0.12, 0.88, 192):
        exact_x = start[0] + (end[0] - start[0]) * float(fraction)
        exact_y = start[1] + (end[1] - start[1]) * float(fraction)
        if any(
            (exact_x - cx) ** 2 + (exact_y - cy) ** 2 <= radius * radius
            for cx, cy, radius in excluded_disks
        ):
            continue
        x = int(round(exact_x))
        y = int(round(exact_y))
        x0, x1 = max(0, x - 5), min(width, x + 6)
        y0, y1 = max(0, y - 5), min(height, y + 6)
        total += 1
        hits += int(bool(np.any(mask[y0:y1, x0:x1])))
    return hits / total if total else 0.0


def _sample_mask_hit(mask: np.ndarray, x: float, y: float, radius: int = 5) -> bool:
    height, width = mask.shape
    px, py = int(round(x)), int(round(y))
    x0, x1 = max(0, px - radius), min(width, px + radius + 1)
    y0, y1 = max(0, py - radius), min(height, py + radius + 1)
    return bool(x0 < x1 and y0 < y1 and np.any(mask[y0:y1, x0:x1]))


def renderer_line_width(record: Dict[str, Any]) -> float:
    """Stroke width the renderer actually used, from the renderer's own default."""

    return float(
        _finite_float(record.get("line_width"))
        or float(PointTargetPuzzleGenerator.LINE_WIDTH)
    )


def foreign_ink_clearance(line_width: float, *, sample_radius: int = 5) -> float:
    """Distance at which another construction's ink can still register as a hit.

    A sample at ``p`` is scored by :func:`_sample_mask_hit`, which accepts any
    lit pixel in the ``(2*sample_radius + 1)`` box around ``p`` — reaching
    ``sample_radius * sqrt(2)`` at the corners.  Another construction occupies
    pixels within ``line_width / 2`` of its own centreline (``line_width`` is
    the renderer's stroke width; every construction is stroked at a width
    factor <= 1, so the full width is the upper bound), plus about one pixel of
    antialiasing left by the supersampled downscale.  A sample farther away
    than the sum cannot see that construction, so anything closer is not
    evidence about the line being measured.
    """

    return float(sample_radius) * math.sqrt(2.0) + float(line_width) / 2.0 + 1.0


def _prefix_observation(samples: Sequence[Tuple[float, bool]]) -> Dict[str, Any]:
    """Score one orientation of a sampled line as a start-to-end prefix."""

    hit_indices = [index for index, (_fraction, hit) in enumerate(samples) if hit]
    if not hit_indices:
        return {
            "observed_progress": 0.0,
            "prefix_quality": 1.0,
            "start_anchored": True,
            "single_end_directional": True,
        }
    last_hit = hit_indices[-1]
    prefix = samples[: last_hit + 1]
    prefix_quality = sum(int(hit) for _fraction, hit in prefix) / max(1, len(prefix))
    start_window = samples[: max(2, int(math.ceil(len(samples) * 0.20)))]
    start_anchored = any(hit for _fraction, hit in start_window)
    return {
        "observed_progress": samples[last_hit][0],
        "prefix_quality": prefix_quality,
        "start_anchored": start_anchored,
        "single_end_directional": bool(start_anchored and prefix_quality >= 0.66),
    }


# The widest sampled parameter span is .04–.96 over 32 positions.
# Adjacent positions are one observation bin, not a distinct later phase.
_LINE_FRONT_SAMPLES = 32
_LINE_FRONT_QUANTUM = (0.96 - 0.04) / (_LINE_FRONT_SAMPLES - 1)


def _single_end_line_observation(
    mask: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    excluded_disks: Sequence[Tuple[float, float, float]] = (),
    *,
    excluded_segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]] = (),
    segment_clearance: float = 0.0,
    accept_either_end: bool = False,
    source_ink: Optional[np.ndarray] = None,
    terminal_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Measure whether a visible partial line is a contiguous single-end prefix.

    ``excluded_segments`` names the other constructions the same frame legally
    contains.  Ink within ``segment_clearance`` of one of them cannot be
    attributed to this line — it is a crossing, a shared vertex or a junction —
    so those samples are dropped instead of being read as a stroke that jumped
    ahead of its own front.  ``accept_either_end`` is for lines whose two
    endpoints are interchangeable in the task's own Prompt: the stroke is then
    scored from both ends and the orientation that explains it wins.
    """

    samples: List[Tuple[float, bool]] = []
    # Canvas-clipped infinite bisectors have anonymous display endpoints.  The
    # middle span determines the geometric line; a short decorative tail is
    # not a separate solution state.  Vertex-anchored strokes keep their ends.
    lower, upper = (0.12, 0.88) if accept_either_end else (0.04, 0.96)
    for fraction in np.linspace(lower, upper, _LINE_FRONT_SAMPLES):
        x = start[0] + (end[0] - start[0]) * float(fraction)
        y = start[1] + (end[1] - start[1]) * float(fraction)
        if any(
            (x - cx) ** 2 + (y - cy) ** 2 <= disk_radius * disk_radius
            for cx, cy, disk_radius in excluded_disks
        ):
            continue
        if segment_clearance > 0.0 and any(
            _point_segment_distance((x, y), segment_start, segment_end)
            <= segment_clearance
            for segment_start, segment_end in excluded_segments
        ):
            continue
        # The renderer restates the triangle and labels over the cyan layer.
        # An opaque source window cannot reveal this stroke's growth.
        if source_ink is not None and _sample_mask_hit(source_ink, x, y):
            continue
        if terminal_mask is not None and not _sample_mask_hit(terminal_mask, x, y):
            continue
        samples.append((float(fraction), _sample_mask_hit(mask, x, y)))
    forward = _prefix_observation(samples)
    if not accept_either_end:
        return {**forward, "matched_orientation": "declared"}
    reverse = _prefix_observation(
        [(1.0 - fraction, hit) for fraction, hit in reversed(samples)]
    )
    forward_key = (bool(forward["single_end_directional"]), float(forward["prefix_quality"]))
    reverse_key = (bool(reverse["single_end_directional"]), float(reverse["prefix_quality"]))
    if reverse_key > forward_key:
        return {**reverse, "matched_orientation": "reversed"}
    return {**forward, "matched_orientation": "declared"}


def _point_segment_distance(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-9:
        return math.dist(point, start)
    t = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
            / denominator,
        ),
    )
    projection = (start[0] + t * dx, start[1] + t * dy)
    return math.dist(point, projection)


def _single_end_arc_observation(
    mask: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    start_angle: float,
    excluded_segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]] = (),
    *,
    segment_clearance: Optional[float] = None,
) -> Dict[str, Any]:
    """Measure a clockwise image-space arc prefix while ignoring line crossings."""

    clearance = (
        foreign_ink_clearance(PointTargetPuzzleGenerator.LINE_WIDTH, sample_radius=4)
        if segment_clearance is None
        else float(segment_clearance)
    )
    samples: List[Tuple[float, bool]] = []
    for fraction in np.linspace(0.02, 0.98, 64):
        angle = start_angle + math.tau * float(fraction)
        point = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )
        if any(
            _point_segment_distance(point, segment_start, segment_end) <= clearance
            for segment_start, segment_end in excluded_segments
        ):
            continue
        samples.append((float(fraction), _sample_mask_hit(mask, point[0], point[1], radius=4)))
    hit_indices = [index for index, (_fraction, hit) in enumerate(samples) if hit]
    if not hit_indices:
        return {
            "observed_progress": 0.0,
            "prefix_quality": 1.0,
            "start_anchored": True,
            "single_end_directional": True,
        }
    last_hit = hit_indices[-1]
    prefix = samples[: last_hit + 1]
    prefix_quality = sum(int(hit) for _fraction, hit in prefix) / max(1, len(prefix))
    start_window = samples[: max(3, int(math.ceil(len(samples) * 0.20)))]
    start_anchored = any(hit for _fraction, hit in start_window)
    return {
        "observed_progress": samples[last_hit][0],
        "prefix_quality": prefix_quality,
        "start_anchored": start_anchored,
        "single_end_directional": bool(start_anchored and prefix_quality >= 0.64),
    }


def _sequential_single_end_lines(
    supports: Sequence[Sequence[float]],
    directionals: Sequence[Sequence[bool]],
    *,
    complete_threshold: float,
    absent_threshold: float = 0.10,
    regression_tolerance: float = 0.18,
    min_partial_frames: int = 2,
    progresses: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Accept any line order, but only one progressive single-end stroke at a time.

    Direction convention shared by every caller: this helper judges *whether* a
    line grew from one fixed end, never *which* end.  Each ``directionals``
    entry comes from :func:`_single_end_line_observation`, and the caller
    decides there whether the task's own Prompt privileges one endpoint (side
    extension from the side endpoint to the foot, angle bisector from the
    vertex to the incenter, parallel line from its tail through the
    through-point) or leaves the two ends interchangeable (a triangle altitude
    or a canvas-clipped perpendicular bisector, whose Prompt only asks for one
    endpoint to the other).  When the ends are interchangeable, both
    orientations are legal and the observation accepts either.

    ``supports`` answers "how much of this line is lit", which is what makes a
    line *complete*, but it is not proportional to how far the stroke has
    travelled: sample positions hidden under an opaque candidate marker are
    dropped from the ratio, so a line whose hidden stretch sits ahead of its
    front reads as almost complete while still half drawn.  ``progresses`` — the
    stroke front each frame, from :func:`_single_end_line_observation` — is the
    faithful travel measure, so when it is supplied a frame counts as growth
    while the front is still short of the front the line finally reaches, even
    if its support ratio has already saturated.  Without it the growth window
    falls back to the support band alone.
    """

    line_count = len(supports[0]) if supports else 0
    if line_count == 0:
        return {
            "valid": True,
            "activation_frames": [],
            "completion_frames": [],
            "draw_order": [],
            "partial_frame_counts": [],
            "maximum_simultaneous_partials": 0,
        }
    frame_count = len(supports)
    activation_frames: List[int] = []
    completion_frames: List[int] = []
    partial_frame_counts: List[int] = []
    line_validity: List[bool] = []
    for line_index in range(line_count):
        series = [float(frame[line_index]) for frame in supports]
        activation = next((index for index, value in enumerate(series) if value >= absent_threshold), frame_count)
        fronts = (
            [float(frame[line_index]) for frame in progresses]
            if progresses is not None
            else None
        )
        final_front = max(fronts) if fronts else 0.0
        # Coverage owns completion. Anonymous lines can choose either
        # endpoint, so their frame-local best orientation may change and its
        # numeric front is not a globally directed completion coordinate.
        completion = next((index for index, value in enumerate(series)
                           if value >= complete_threshold), frame_count)
        partials = [
            index for index, value in enumerate(series)
            if value >= absent_threshold
            and (
                value < complete_threshold
                or (fronts is not None and fronts[index] + _LINE_FRONT_QUANTUM < final_front)
            )
        ]
        activation_frames.append(activation)
        completion_frames.append(completion)
        partial_frame_counts.append(len(partials))
        line_validity.append(
            bool(
                # Occlusion may saturate readable coverage on the first
                # visible prefix; the partial/front evidence below still
                # requires real single-end growth rather than a full cut.
                activation <= completion < frame_count
                and len(partials) >= min_partial_frames
                and series[-1] >= complete_threshold
                and _no_large_regression(series, regression_tolerance)
                # Very first sub-pixel prefixes can be represented by only
                # one or two sampled hits after video rasterization.  They
                # are not enough evidence for a reverse/center-out stroke,
                # so defer the directional requirement until the line has
                # visible support above this aliasing floor.
                and all(
                    bool(directionals[index][line_index])
                    or float(series[index]) < 0.18
                    for index in partials
                )
            )
        )
    draw_order = sorted(range(line_count), key=lambda index: activation_frames[index])
    ordered_without_overlap = all(
        activation_frames[draw_order[index]] > completion_frames[draw_order[index - 1]]
        for index in range(1, line_count)
    )
    simultaneous_partials = [
        sum(absent_threshold <= float(value) < complete_threshold for value in frame)
        for frame in supports
    ]
    previous = [0.0] * line_count
    one_changing_line_per_frame = True
    for frame in supports:
        changed = sum(
            max(float(frame[index]), previous[index]) >= absent_threshold
            and previous[index] < complete_threshold
            and float(frame[index]) - previous[index] > 0.055
            for index in range(line_count)
        )
        if changed > 1:
            one_changing_line_per_frame = False
        previous = [float(value) for value in frame]
    valid = bool(
        all(line_validity)
        and len(set(activation_frames)) == line_count
        and ordered_without_overlap
        and max(simultaneous_partials, default=0) <= 1
        and one_changing_line_per_frame
    )
    return {
        "valid": valid,
        "activation_frames": activation_frames,
        "completion_frames": completion_frames,
        "draw_order": draw_order,
        "partial_frame_counts": partial_frame_counts,
        "maximum_simultaneous_partials": max(simultaneous_partials, default=0),
        "one_changing_line_per_frame": one_changing_line_per_frame,
    }


def _circle_support(
    mask: np.ndarray,
    center: Tuple[float, float],
    radius: float,
) -> float:
    height, width = mask.shape
    hits = 0
    total = 0
    for angle in np.linspace(0.0, math.tau, 96, endpoint=False):
        x = int(round(center[0] + radius * math.cos(float(angle))))
        y = int(round(center[1] + radius * math.sin(float(angle))))
        x0, x1 = max(0, x - 4), min(width, x + 5)
        y0, y1 = max(0, y - 4), min(height, y + 5)
        total += 1
        hits += int(bool(np.any(mask[y0:y1, x0:x1])))
    return hits / total if total else 0.0


def _clip_line_to_canvas(
    point: Tuple[float, float],
    direction: Tuple[float, float],
    size: Tuple[int, int],
    padding: float,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Clip an infinite construction line to the renderer's safe canvas."""

    width, height = size
    left, top = float(padding), float(padding)
    right, bottom = float(width) - padding, float(height) - padding
    dx, dy = direction
    candidates: List[Tuple[float, float, float]] = []
    if abs(dx) > 1e-9:
        for x in (left, right):
            t = (x - point[0]) / dx
            y = point[1] + t * dy
            if top - 1e-6 <= y <= bottom + 1e-6:
                candidates.append((t, x, y))
    if abs(dy) > 1e-9:
        for y in (top, bottom):
            t = (y - point[1]) / dy
            x = point[0] + t * dx
            if left - 1e-6 <= x <= right + 1e-6:
                candidates.append((t, x, y))
    unique: List[Tuple[float, float, float]] = []
    for item in sorted(candidates):
        if not any(math.hypot(item[1] - other[1], item[2] - other[2]) <= 1e-5 for other in unique):
            unique.append(item)
    if len(unique) < 2:
        return None
    return (unique[0][1], unique[0][2]), (unique[-1][1], unique[-1][2])


def _expected_geometry(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    replay_metrics, valid, correct = replay_circumcenter_geometry(record)
    if not valid or correct is None:
        return None
    center = _point(replay_metrics.get("circumcenter_derived_center"))
    vertices = [_point(value) for value in record.get("triangle_vertices") or []]
    if center is None or len(vertices) != 3 or any(point is None for point in vertices):
        return None
    width, height = tuple(int(value) for value in (record.get("canvas_dimensions") or [512, 512]))
    # A bisector has no ends of its own: it runs until the frame stops it.  The
    # frame the renderer stops it at is the family's *working* canvas -- the
    # image inset by the safe margin (``max(18, round(0.10 * short side))``) --
    # and then by the stroke width, not the raw image edge.  Clipping to the raw
    # edge here names about a tenth of the span at each end that no legal
    # delivery can ink, and the single-end observation samples that span: it
    # counts the unreachable head of the line as prefix and reads a correct
    # stroke as one that skipped its own start.
    margin = max(18.0, float(round(0.10 * min(width, height))))
    padding = margin + renderer_line_width(record) + 5.0
    bisectors: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    ordered_vertices = [point for point in vertices if point is not None]
    for index, first in enumerate(ordered_vertices):
        second = ordered_vertices[(index + 1) % 3]
        side_dx, side_dy = second[0] - first[0], second[1] - first[1]
        side_length = math.hypot(side_dx, side_dy)
        if side_length <= 1e-9:
            return None
        midpoint = ((first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5)
        clipped = _clip_line_to_canvas(
            midpoint,
            (-side_dy / side_length, side_dx / side_length),
            (width, height),
            padding,
        )
        if clipped is None:
            return None
        bisectors.append(clipped)
    return {
        "center": center,
        "radius": float(replay_metrics["circumcenter_derived_radius"]),
        "vertices": ordered_vertices,
        "bisectors": bisectors,
        "correct": correct,
        "replay_metrics": replay_metrics,
    }


def _allowed_dynamic_mask(
    shape: Tuple[int, int],
    geometry: Dict[str, Any],
    point_radius: float,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    center = tuple(int(round(value)) for value in geometry["center"])
    for start, end in geometry["bisectors"]:
        # A perpendicular bisector has no public stopping point.
        full = _clip_line_to_canvas(start, (end[0] - start[0], end[1] - start[1]), (width, height), 0.0)
        if full is not None:
            cv2.line(mask, tuple(map(round, full[0])), tuple(map(round, full[1])), 255, thickness=1)
    cv2.circle(
        mask,
        center,
        int(round(float(geometry["radius"]))),
        255,
        thickness=1,
    )
    cv2.circle(mask, center, max(18, int(round(point_radius * 2.0))), 255, thickness=-1)
    # Match the family's square sampling window rather than a narrower
    # Euclidean tube, especially where a diagonal line meets the canvas.
    return cv2.dilate(mask, np.ones((23, 23), dtype=np.uint8))


def _frame_analysis(
    frame: np.ndarray,
    source: np.ndarray,
    record: Dict[str, Any],
    geometry: Dict[str, Any],
    terminal_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    from data.geometry.core.verifier import source_ink_mask
    from data.geometry.core.source_scene import observe_additive_source_scene

    cyan, _dark_red = _color_masks(frame)
    red_label, red_mask = _red_option(frame, record)
    exclusion_radius = float(record.get("point_radius") or 11) * 1.55
    candidate_exclusions = [
        (float(item["x"]), float(item["y"]), exclusion_radius)
        for item in record.get("candidates") or []
        if isinstance(item, dict)
        and _finite_float(item.get("x")) is not None
        and _finite_float(item.get("y")) is not None
    ]
    bisector_supports = [
        _line_support(cyan, start, end, candidate_exclusions)
        for start, end in geometry["bisectors"]
    ]
    clearance = foreign_ink_clearance(renderer_line_width(record))
    bisector_observations = [
        _single_end_line_observation(
            cyan,
            start,
            end,
            candidate_exclusions,
            # The three bisectors concur at the circumcenter and the circumcircle
            # crosses each of them twice; that ink belongs to those constructions.
            excluded_segments=[
                other
                for other_index, other in enumerate(geometry["bisectors"])
                if other_index != index
            ],
            segment_clearance=clearance,
            # Both ends of a canvas-clipped bisector are anonymous derived
            # points, so the Prompt cannot and does not privilege one of them.
            accept_either_end=True,
            source_ink=source_ink_mask(source),
            terminal_mask=terminal_mask,
        )
        for index, (start, end) in enumerate(geometry["bisectors"])
    ]
    circle_support = _circle_support(
        cyan,
        geometry["center"],
        float(geometry["radius"]),
    )
    allowed = _allowed_dynamic_mask(cyan.shape, geometry, float(record.get("point_radius") or 11))
    # Candidate discs have an independent color/identity gate below.
    for x, y, radius in candidate_exclusions:
        cv2.circle(allowed, (round(x), round(y)), math.ceil(radius), 255, -1)
    cyan_count = int(np.count_nonzero(cyan))
    cyan_spill = float(np.count_nonzero(cyan & (allowed == 0)) / max(1, cyan_count))
    diff = np.max(np.abs(frame.astype(np.int16) - source.astype(np.int16)), axis=2)
    static_mutation = float(np.mean((diff >= 24)[allowed == 0]))
    return {
        "bisector_supports": bisector_supports,
        "bisector_single_end_observations": bisector_observations,
        "minimum_bisector_support": min(bisector_supports),
        "average_bisector_support": sum(bisector_supports) / 3.0,
        "circle_support": circle_support,
        "red_option": red_label,
        "red_pixel_count": int(np.count_nonzero(red_mask)),
        "cyan_spill_fraction": cyan_spill,
        "static_mutation_fraction": static_mutation,
        **observe_additive_source_scene(
            frame, source, record, allowed=allowed, discs=candidate_exclusions,
            prefix="circumcenter_",
        ),
        "black_vacancy": float(np.mean(frame)) < 35.0,
    }


def _final_semantics(analysis: Dict[str, Any], correct: str) -> Tuple[bool, bool]:
    preservation_ok = bool(
        analysis["source_preservation_pass"]
        and analysis["marker_state_pass"]
        and float(analysis["cyan_spill_fraction"]) <= 0.16
        and not analysis["black_vacancy"]
    )
    final_ok = bool(
        float(analysis["minimum_bisector_support"]) >= 0.70
        and float(analysis["circle_support"]) >= 0.82
        and analysis["red_option"] == correct
        and int(analysis["red_pixel_count"]) >= 24
    )
    return preservation_ok, final_ok


def _no_large_regression(values: Sequence[float], tolerance: float) -> bool:
    if not values:
        return False
    running = values[0] if values else 0.0
    for value in values[1:]:
        if value + tolerance < running:
            return False
        running = max(running, value)
    return True


def verify_circumcenter_full_video(
    video_path: Path,
    record: Dict[str, Any],
    input_path: Path,
) -> Tuple[Dict[str, Any], bool]:
    started = time.perf_counter()
    replay_metrics, replay_valid, correct = replay_circumcenter_geometry(record)
    geometry = _expected_geometry(record) if replay_valid else None
    capture = cv2.VideoCapture(video_path.as_posix())
    opened = capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    frames: List[np.ndarray] = []
    if opened:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    dims = record.get("canvas_dimensions") or [512, 512]
    width, height = int(dims[0]), int(dims[1])
    source = _read_rgb(input_path, (width, height))
    media_valid = bool(
        opened
        and len(frames) >= 9
        and abs(fps - 16.0) <= 0.25
        and source is not None
    )
    if not media_valid or geometry is None or correct is None or source is None:
        components = {
            "media_validity_score": float(media_valid),
            "source_replay_score": float(replay_valid),
            "temporal_process_score": 0.0,
            "preservation_score": 0.0,
            "final_state_score": 0.0,
        }
        return {
            **replay_metrics,
            "video_process_verifier": "point_geometry_circumcenter_full_sequence_semantic",
            "video_frame_count": len(frames),
            "video_fps": fps,
            **_contract_metrics(
                artifact_kind="full_video",
                binary_correct=False,
                component_scores=components,
                failure_reason="video media contract or source geometry replay failed",
                started=started,
            ),
        }, False
    aligned = [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.shape[:2] != (height, width)
        else frame
        for frame in frames
    ]
    # Endpoint placement is anonymous. Read the prefix on the portion the
    # terminal stroke actually covers on the independently derived line; the
    # unfiltered terminal support still rejects a wrong or missing bisector.
    terminal_mask = _color_masks(aligned[-1])[0]
    analyses = [_frame_analysis(frame, source, record, geometry, terminal_mask) for frame in aligned]
    bisector_values = [list(map(float, item["bisector_supports"])) for item in analyses]
    bisector_directionals = [
        [
            bool(observation["single_end_directional"])
            for observation in item["bisector_single_end_observations"]
        ]
        for item in analyses
    ]
    bisector_fronts = [
        [
            float(observation["observed_progress"])
            for observation in item["bisector_single_end_observations"]
        ]
        for item in analyses
    ]
    circle_values = [float(item["circle_support"]) for item in analyses]
    # Read fixed source-derived angular positions, excluding the completed
    # bisectors and opaque input marks. The public circle can start anywhere
    # and continue in either direction; the renderer's clock is not an answer.
    from data.geometry.core.verifier import (
        _attributable_path_positions, _sampled_path_hits,
        _closed_curve_occupancy_growth, source_ink_mask,
    )
    angles = np.linspace(0.0, math.tau, 257)
    center = np.asarray(geometry["center"])
    circle = center + float(geometry["radius"]) * np.column_stack([np.cos(angles), np.sin(angles)])
    occluders = [(float(c["x"]), float(c["y"]), float(record.get("point_radius") or 11) * 1.55)
                 for c in record.get("candidates") or []]
    evidence = _attributable_path_positions(
        circle, sample_count=192,
        excluded_paths=[np.asarray(line) for line in geometry["bisectors"]],
        clearance=foreign_ink_clearance(renderer_line_width(record)),
        occluders=occluders, line_width=renderer_line_width(record),
        source_ink=source_ink_mask(source), parameter_span=(0.0, 1.0),
    )
    circle_hits = (np.asarray([_sampled_path_hits(_color_masks(f)[0], circle, radius=3, samples=evidence[0])
                               for f in aligned]) if evidence is not None else None)
    if circle_hits is not None:
        circle_values = circle_hits.mean(axis=1).tolist()
    labels = [item["red_option"] for item in analyses]
    growth_start = phase_onset([max(row) for row in bisector_values], 0.08)
    circle_onset = phase_onset(circle_values, 0.18)
    highlight_start = selection_onset(labels)
    final_start = len(frames) - 3
    initial_hold = growth_start >= 2 and all(
        max(bisector_values[index]) < 0.08
        and circle_values[index] < 0.08
        and labels[index] is None
        and float(np.mean(np.abs(aligned[index].astype(np.int16) - source.astype(np.int16)))) <= 2.0
        for index in range(min(4, growth_start))
    )
    bisector_start, bisector_end = growth_start, circle_onset
    bisector_sequence = _sequential_single_end_lines(
        bisector_values[bisector_start:bisector_end],
        bisector_directionals[bisector_start:bisector_end],
        complete_threshold=0.68,
        progresses=bisector_fronts[bisector_start:bisector_end],
    )
    bisector_sequential_phase = bool(
        bisector_sequence["valid"]
        and all(
            circle_values[index] <= 0.18 and labels[index] is None
            for index in range(bisector_start, bisector_end)
        )
    )
    circle_start, circle_end = circle_onset, highlight_start
    circle_growth, circle_growth_metrics = (False, {})
    if circle_hits is not None and circle_end > circle_start:
        circle_growth, circle_growth_metrics = _closed_curve_occupancy_growth(
            circle_hits[max(0, circle_start - 1):circle_end], evidence[1],
        )
    circle_phase = bool(
        circle_growth
        and all(min(bisector_values[index]) >= 0.68 and labels[index] is None
                for index in range(circle_start, circle_end))
    )
    highlight_phase = bool(
        all(
            min(bisector_values[index]) >= 0.68
            and circle_values[index] >= 0.80
            and labels[index] in {None, correct}
            for index in range(highlight_start, final_start)
        )
        and labels[final_start - 1] == correct
    )
    final_semantics = [_final_semantics(analyses[index], correct) for index in range(final_start, len(frames))]
    final_hold = all(item == (True, True) for item in final_semantics) and all(
        float(np.mean(np.abs(aligned[index].astype(np.int16) - aligned[-1].astype(np.int16)))) <= 2.0
        for index in range(final_start, len(frames))
    )
    valid_labels = all(label in {None, correct} for label in labels)
    no_early_circle = all(value <= 0.18 for value in circle_values[:circle_start])
    no_early_highlight = all(label is None for label in labels[:highlight_start])
    no_vacancy = not any(bool(item["black_vacancy"]) for item in analyses)
    preservation_ok = bool(
        all(item["source_preservation_pass"] and item["marker_state_pass"] for item in analyses)
        and max(float(item["cyan_spill_fraction"]) for item in analyses) <= 0.16
    )
    temporal_ok = bool(
        initial_hold
        and bisector_sequential_phase
        and circle_phase
        and highlight_phase
        and final_hold
        and valid_labels
        and no_early_circle
        and no_early_highlight
        and no_vacancy
    )
    final_ok = bool(final_hold and labels[-1] == correct)
    passed = bool(media_valid and replay_valid and preservation_ok and temporal_ok and final_ok)
    components = {
        "media_validity_score": float(media_valid),
        "source_replay_score": float(replay_valid),
        "temporal_process_score": float(temporal_ok),
        "preservation_score": float(preservation_ok),
        "final_state_score": float(final_ok),
    }
    return {
        **replay_metrics,
        "video_process_verifier": "point_geometry_circumcenter_full_sequence_semantic",
        "video_semantic_parser": "sequential_single_end_bisectors_then_single_end_circumcircle_parser",
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_decoded_frame_count": len(frames),
        "video_option": correct if final_ok else None,
        "video_circumcenter_initial_hold": initial_hold,
        "video_circumcenter_bisector_sequential_phase": bisector_sequential_phase,
        "video_circumcenter_bisector_activation_frames": bisector_sequence["activation_frames"],
        "video_circumcenter_bisector_completion_frames": bisector_sequence["completion_frames"],
        "video_circumcenter_bisector_draw_order": bisector_sequence["draw_order"],
        "video_circumcenter_bisector_partial_frame_counts": bisector_sequence["partial_frame_counts"],
        "video_circumcenter_maximum_simultaneous_partial_bisectors": bisector_sequence[
            "maximum_simultaneous_partials"
        ],
        "video_circumcenter_circle_phase": circle_phase,
        "video_circumcenter_circle_growth_metrics": circle_growth_metrics,
        "video_circumcenter_source_static_region_max_error": max(item["source_static_region_max_error"] for item in analyses),
        "video_circumcenter_highlight_phase": highlight_phase,
        "video_circumcenter_final_hold": final_hold,
        "video_circumcenter_valid_highlight_labels": valid_labels,
        "video_circumcenter_no_early_circle": no_early_circle,
        "video_circumcenter_no_early_highlight": no_early_highlight,
        "video_circumcenter_no_black_vacancy": no_vacancy,
        "video_circumcenter_bisector_supports": bisector_values,
        "video_circumcenter_circle_supports": circle_values,
        "video_circumcenter_max_static_mutation_fraction": max(
            float(item["static_mutation_fraction"]) for item in analyses
        ),
        "video_circumcenter_max_cyan_spill_fraction": max(
            float(item["cyan_spill_fraction"]) for item in analyses
        ),
        "video_semantic_process_check_pass": passed,
        **_contract_metrics(
            artifact_kind="full_video",
            binary_correct=passed,
            component_scores=components,
            failure_reason="sequential single-end bisectors, single-end circle, highlight, preservation, or final hold failed",
            started=started,
        ),
    }, passed


TASK_NAME = "circumcenter"
VERIFIER = VerifierRoute(
    "data.geometry.core.verifier",
    "verify_coordinate_target",
)


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    """Evaluate only the formal final delivery for this task."""

    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [
    "CIRCUMCENTER_STATE_TRACE",
    "CIRCUMCENTER_VERIFIER_PROFILE",
    "TASK_NAME",
    "VERIFIER",
    "evaluate",
    "replay_circumcenter_geometry",
    "verify_circumcenter_full_video",
]

"""Independent multimodal semantic verifier for ``incenter``."""

from __future__ import annotations

from data.geometry.core.phase_observation import phase_onset, selection_onset

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.common.verifiers.routes import VerifierRoute
from data.geometry.circumcenter.evaluator import (
    _circle_support,
    _color_masks,
    _direction_name,
    _finite_float,
    _line_support,
    _no_large_regression,
    _ordered_vertices,
    _point,
    _points_close,
    _read_rgb,
    _red_option,
    _single_end_arc_observation,
    _single_end_line_observation,
    _sequential_single_end_lines,
    foreign_ink_clearance,
    renderer_line_width,
)


INCENTER_VERIFIER_PROFILE = "incenter_multimodal_semantic_hard_gate_v1"
INCENTER_STATE_TRACE = [
    "input_hold",
    "first_angle_bisector_single_end_growth",
    "second_angle_bisector_single_end_growth",
    "third_angle_bisector_single_end_growth",
    "incenter_marker",
    "incircle_single_end_growth",
    "correct_candidate_highlight",
    "final_hold",
]
def _float_close(value: object, expected: float, tolerance: float = 1e-5) -> bool:
    parsed = _finite_float(value)
    return parsed is not None and abs(parsed - expected) <= tolerance


def _calculate_incenter(
    vertices: Sequence[Tuple[float, float]],
) -> Optional[Tuple[Tuple[float, float], float]]:
    p1, p2, p3 = vertices
    a = math.dist(p2, p3)
    b = math.dist(p1, p3)
    c = math.dist(p1, p2)
    perimeter = a + b + c
    if perimeter <= 1e-9 or a + b <= c + 1e-9 or a + c <= b + 1e-9 or b + c <= a + 1e-9:
        return None
    center = (
        (a * p1[0] + b * p2[0] + c * p3[0]) / perimeter,
        (a * p1[1] + b * p2[1] + c * p3[1]) / perimeter,
    )
    semiperimeter = perimeter / 2.0
    area_term = semiperimeter * (semiperimeter - a) * (semiperimeter - b) * (semiperimeter - c)
    if area_term <= 1e-9:
        return None
    return center, math.sqrt(area_term) / semiperimeter


def _distance_to_side(
    point: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    return abs(
        (b[0] - a[0]) * (point[1] - a[1])
        - (b[1] - a[1]) * (point[0] - a[0])
    ) / max(math.dist(a, b), 1e-9)


def _projection_to_side(
    point: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> Tuple[Tuple[float, float], float]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    denominator = dx * dx + dy * dy
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / max(denominator, 1e-9)
    return (a[0] + t * dx, a[1] + t * dy), t


def _strictly_inside_triangle(
    point: Tuple[float, float],
    vertices: Sequence[Tuple[float, float]],
) -> bool:
    crosses: List[float] = []
    for index, a in enumerate(vertices):
        b = vertices[(index + 1) % 3]
        crosses.append(
            (b[0] - a[0]) * (point[1] - a[1])
            - (b[1] - a[1]) * (point[0] - a[0])
        )
    return bool(
        all(value > 1e-5 for value in crosses)
        or all(value < -1e-5 for value in crosses)
    )


def replay_incenter_geometry(
    record: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Recompute the unique displayed incenter without trusting saved answers."""

    metrics: Dict[str, Any] = {
        "verifier_profile": INCENTER_VERIFIER_PROFILE,
        "incenter_geometry_replay_valid": False,
        "incenter_geometry_replay_reason": "invalid visible source geometry",
    }
    raw_vertices = record.get("triangle_vertices")
    if not isinstance(raw_vertices, list) or len(raw_vertices) != 3:
        return metrics, False, None
    parsed_vertices = [_point(value) for value in raw_vertices]
    if any(point is None for point in parsed_vertices):
        return metrics, False, None
    vertices = _ordered_vertices([point for point in parsed_vertices if point is not None])
    derived_circle = _calculate_incenter(vertices)
    if derived_circle is None:
        metrics["incenter_geometry_replay_reason"] = "triangle vertices are collinear"
        return metrics, False, None
    center, radius = derived_circle
    if not 45.0 <= radius <= 220.0:
        metrics["incenter_geometry_replay_reason"] = "derived incircle is outside the supported scale"
        return metrics, False, None

    feet_with_t = [
        _projection_to_side(center, vertices[index], vertices[(index + 1) % 3])
        for index in range(3)
    ]
    feet = [item[0] for item in feet_with_t]
    feet_on_segments = all(1e-5 < item[1] < 1.0 - 1e-5 for item in feet_with_t)

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
        distances = [
            _distance_to_side(candidate, vertices[index], vertices[(index + 1) % 3])
            for index in range(3)
        ]
        derived.append(
            {
                "label": label,
                "center": [candidate[0], candidate[1]],
                "direction_from_triangle_centroid": _direction_name(
                    candidate[0] - centroid_x,
                    candidate[1] - centroid_y,
                ),
                "side_distances_pixels": distances,
                "distance_spread_pixels": max(distances) - min(distances),
                "mean_side_distance_pixels": sum(distances) / 3.0,
                "strictly_inside_triangle": _strictly_inside_triangle(candidate, vertices),
            }
        )
    eligible = [item for item in derived if item["strictly_inside_triangle"]]
    if len(eligible) != 5:
        return metrics, False, None
    ordered = sorted(eligible, key=lambda item: float(item["distance_spread_pixels"]))
    derived_correct = str(ordered[0]["label"])
    best_spread = float(ordered[0]["distance_spread_pixels"])
    runner_up_spread = float(ordered[1]["distance_spread_pixels"])
    derived_index = ord(derived_correct) - ord("A")
    unique = bool(
        best_spread <= 1e-5
        and runner_up_spread >= 18.0
        and math.dist(candidates[derived_index][1], center) <= 1e-5
        and feet_on_segments
    )

    structured = record.get("structured_answer")
    if not isinstance(structured, dict):
        return metrics, False, derived_correct
    stored_vertices = structured.get("triangle_vertices")
    vertices_valid = bool(
        isinstance(stored_vertices, list)
        and len(stored_vertices) == 3
        and all(_points_close(stored_vertices[index], vertices[index]) for index in range(3))
    )
    stored_feet = structured.get("tangent_feet")
    feet_valid = bool(
        isinstance(stored_feet, list)
        and len(stored_feet) == 3
        and all(_points_close(stored_feet[index], feet[index]) for index in range(3))
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
            and stored_by_label[label].get("strictly_inside_triangle") is True
            and isinstance(stored_by_label[label].get("side_distances_pixels"), list)
            and len(stored_by_label[label]["side_distances_pixels"]) == 3
            and all(
                _float_close(
                    stored_by_label[label]["side_distances_pixels"][side_index],
                    float(derived[index]["side_distances_pixels"][side_index]),
                )
                for side_index in range(3)
            )
            and _float_close(
                stored_by_label[label].get("distance_spread_pixels"),
                float(derived[index]["distance_spread_pixels"]),
            )
            and _float_close(
                stored_by_label[label].get("mean_side_distance_pixels"),
                float(derived[index]["mean_side_distance_pixels"]),
            )
            for index, label in enumerate("ABCDE")
        )

    answer_schema = structured.get("answer_schema")
    verifier_contract = structured.get("verifier_contract")
    render_geometry = structured.get("render_geometry")
    record_point_radius = _finite_float(record.get("point_radius"))
    contracts_valid = bool(
        isinstance(answer_schema, dict)
        and answer_schema.get("terminal_text_regex") == FIVE_CHOICE_CODEC.full_pattern
        and answer_schema.get("requires_sample_specific_reasoning_before_terminal_answer") is True
        and answer_schema.get("terminal_text_contains_exact_answer") is True
        and answer_schema.get("no_block_after_terminal_image") is True
        and isinstance(verifier_contract, dict)
        and verifier_contract.get("profile") == INCENTER_VERIFIER_PROFILE
        and verifier_contract.get("strict_binary_correctness") is True
        and isinstance(render_geometry, dict)
        and render_geometry.get("canvas_dimensions") == [512, 512]
        and record_point_radius is not None
        and _float_close(render_geometry.get("point_radius"), record_point_radius)
        and structured.get("state_trace") == INCENTER_STATE_TRACE
        and structured.get("expected_video_frame_count") == 33
        and structured.get("video_fps") == 16
    )
    redundant_metadata_valid = bool(
        vertices_valid
        and feet_valid
        and _points_close(record.get("incenter"), center)
        and _points_close(structured.get("incenter"), center)
        and _float_close(record.get("inradius"), radius)
        and _float_close(structured.get("inradius"), radius)
        and str(record.get("correct_option") or "").strip().upper() == derived_correct
        and str(structured.get("correct_option") or "").strip().upper() == derived_correct
        and structured.get("correct_index") == derived_index
        and _float_close(structured.get("runner_up_distance_spread_pixels"), runner_up_spread)
        and structured.get("multiple_solution_policy")
        == "exactly_one_strictly_interior_displayed_candidate_is_equidistant_from_all_three_sides"
    )
    valid = bool(unique)
    metrics.update(
        {
            "incenter_geometry_replay_valid": valid,
            "incenter_geometry_replay_reason": (
                "ok" if valid else "source geometry is invalid or non-unique"
            ),
            "incenter_derived_correct_option": derived_correct,
            "incenter_derived_correct_index": derived_index,
            "incenter_derived_center": [center[0], center[1]],
            "incenter_derived_radius": radius,
            "incenter_derived_tangent_feet": [[point[0], point[1]] for point in feet],
            "incenter_tangent_feet_on_segments": feet_on_segments,
            "incenter_runner_up_distance_spread_pixels": runner_up_spread,
            "incenter_candidate_measurements": derived,
            "incenter_unique_displayed_solution": unique,
            "incenter_candidate_metadata_valid": candidate_metadata_valid,
            "incenter_measurement_metadata_valid": measurements_valid,
            "incenter_contract_metadata_valid": contracts_valid,
            "incenter_redundant_metadata_valid": redundant_metadata_valid,
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
        "verifier_profile": INCENTER_VERIFIER_PROFILE,
        "artifact_kind": artifact_kind,
        "component_scores": component_scores,
        "binary_correct": bool(binary_correct),
        "strict_binary_correctness": True,
        "failure_reason": "ok" if binary_correct else failure_reason,
        "verification_elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "performance_policy": "latency_throughput_memory_and_decode_counts_are_diagnostics_without_failure_thresholds",
    }


def _expected_geometry(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    replay_metrics, valid, correct = replay_incenter_geometry(record)
    if not valid or correct is None:
        return None
    center = _point(replay_metrics.get("incenter_derived_center"))
    feet = [_point(value) for value in replay_metrics.get("incenter_derived_tangent_feet") or []]
    if center is None or len(feet) != 3 or any(point is None for point in feet):
        return None
    visible_feet = [point for point in feet if point is not None]
    start_foot = min(visible_feet, key=lambda point: (point[1], point[0]))
    bisectors: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for index, vertex in enumerate(_ordered_vertices([
        _point(value) for value in record["triangle_vertices"]
    ])):
        opposite_a = _ordered_vertices([_point(value) for value in record["triangle_vertices"]])[(index + 1) % 3]
        opposite_b = _ordered_vertices([_point(value) for value in record["triangle_vertices"]])[(index + 2) % 3]
        ab = (center[0] - vertex[0], center[1] - vertex[1])
        cd = (opposite_b[0] - opposite_a[0], opposite_b[1] - opposite_a[1])
        denominator = ab[0] * cd[1] - ab[1] * cd[0]
        if abs(denominator) <= 1e-9:
            return None
        ac = (opposite_a[0] - vertex[0], opposite_a[1] - vertex[1])
        t = (ac[0] * cd[1] - ac[1] * cd[0]) / denominator
        endpoint = (vertex[0] + t * ab[0], vertex[1] + t * ab[1])
        bisectors.append((vertex, endpoint))
    return {
        "center": center,
        "radius": float(replay_metrics["incenter_derived_radius"]),
        "feet": visible_feet,
        "bisectors": bisectors,
        "circle_start_angle": math.atan2(start_foot[1] - center[1], start_foot[0] - center[0]),
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
    for start_value, end_value in geometry["bisectors"]:
        start = tuple(int(round(value)) for value in start_value)
        end = tuple(int(round(value)) for value in end_value)
        cv2.line(mask, start, end, 255, thickness=18)
    cv2.circle(mask, center, int(round(float(geometry["radius"]))), 255, thickness=20)
    cv2.circle(mask, center, max(18, int(round(point_radius * 2.0))), 255, thickness=-1)
    return mask


def _frame_analysis(
    frame: np.ndarray,
    source: np.ndarray,
    record: Dict[str, Any],
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    cyan, _dark_red = _color_masks(frame)
    red_label, red_mask = _red_option(frame, record)
    point_radius = _finite_float(record.get("point_radius")) or 11.0
    # Candidate disks are opaque, so samples under their actual marker radius
    # are unobservable rather than missing construction.  Keep this exclusion
    # tight: a larger halo can erase an entire short inradius construction.
    exclusion_radius = point_radius * 1.12
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
            # The three internal bisectors concur at the incenter, so each one
            # runs into the other two there; that ink is not its own front.
            excluded_segments=[
                other
                for other_index, other in enumerate(geometry["bisectors"])
                if other_index != index
            ],
            segment_clearance=clearance,
            # The Prompt fixes the direction: from the triangle vertex through
            # the common intersection to the opposite side, so only the
            # declared orientation is legal.
            accept_either_end=False,
        )
        for index, (start, end) in enumerate(geometry["bisectors"])
    ]
    circle_support = _circle_support(cyan, geometry["center"], float(geometry["radius"]))
    # The lines that actually cross the incircle are the three internal
    # bisectors: each one runs from its vertex through the centre and out to
    # the opposite side, so it lights two circumference samples long before the
    # sweep reaches them.  A sample that a completed bisector already explains
    # is not evidence about how far the arc has travelled, and leaving it in
    # caps the measured prefix quality below the single-end threshold for the
    # whole first two thirds of a legally single-ended sweep.  The centre-to-
    # foot radii excluded here previously are never drawn at all.
    circle_observation = _single_end_arc_observation(
        cyan,
        geometry["center"],
        float(geometry["radius"]),
        float(geometry["circle_start_angle"]),
        list(geometry["bisectors"]),
        segment_clearance=clearance,
    )
    allowed = _allowed_dynamic_mask(cyan.shape, geometry, point_radius)
    cyan_count = int(np.count_nonzero(cyan))
    cyan_spill = float(np.count_nonzero(cyan & (allowed == 0)) / max(1, cyan_count))
    diff = np.max(np.abs(frame.astype(np.int16) - source.astype(np.int16)), axis=2)
    static_mutation = float(np.mean((diff >= 24)[allowed == 0]))
    center_x = int(round(geometry["center"][0]))
    center_y = int(round(geometry["center"][1]))
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    distance_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2
    marker_ring = (distance_squared >= 5 ** 2) & (distance_squared <= 9 ** 2)
    # A center marker is new dark ink, not arbitrary source-to-video RGB
    # error.  A codec can move the pre-existing candidate glyph by one pixel.
    source_dark = np.max(source, axis=2) <= 140
    source_dark = cv2.dilate(source_dark.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    new_dark = (np.max(frame, axis=2) <= 140) & ~source_dark
    center_marker_delta = float(np.mean(new_dark[marker_ring]))
    return {
        "bisector_supports": bisector_supports,
        "bisector_single_end_observations": bisector_observations,
        "minimum_bisector_support": min(bisector_supports),
        "average_bisector_support": sum(bisector_supports) / 3.0,
        "circle_support": circle_support,
        "circle_single_end_observation": circle_observation,
        "red_option": red_label,
        "red_pixel_count": int(np.count_nonzero(red_mask)),
        "cyan_spill_fraction": cyan_spill,
        "static_mutation_fraction": static_mutation,
        "black_vacancy": float(np.mean(frame)) < 35.0,
        "center_marker_support": center_marker_delta,
    }


def _final_semantics(analysis: Dict[str, Any], correct: str) -> Tuple[bool, bool]:
    preservation_ok = bool(
        float(analysis["static_mutation_fraction"]) <= 0.004
        and float(analysis["cyan_spill_fraction"]) <= 0.14
        and not analysis["black_vacancy"]
    )
    final_ok = bool(
        float(analysis["minimum_bisector_support"]) >= 0.68
        and float(analysis["circle_support"]) >= 0.82
        and float(analysis["center_marker_support"]) >= 0.01
        and analysis["red_option"] == correct
        and int(analysis["red_pixel_count"]) >= 24
    )
    return preservation_ok, final_ok


def verify_incenter_full_video(
    video_path: Path,
    record: Dict[str, Any],
    input_path: Path,
) -> Tuple[Dict[str, Any], bool]:
    started = time.perf_counter()
    replay_metrics, replay_valid, correct = replay_incenter_geometry(record)
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
    media_valid = bool(opened and len(frames) >= 9 and abs(fps - 16.0) <= 0.25 and source is not None)
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
            "video_process_verifier": "point_geometry_incenter_full_sequence_semantic",
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
    analyses = [_frame_analysis(frame, source, record, geometry) for frame in aligned]
    bisector_supports = [list(map(float, item["bisector_supports"])) for item in analyses]
    bisector_values = [min(values) for values in bisector_supports]
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
    circle_directionals = [
        bool(item["circle_single_end_observation"]["single_end_directional"])
        for item in analyses
    ]
    marker_values = [float(item["center_marker_support"]) for item in analyses]
    labels = [item["red_option"] for item in analyses]
    growth_start = phase_onset([max(row) for row in bisector_supports], 0.08)
    marker_start = phase_onset(marker_values, 0.02)
    circle_onset = phase_onset(circle_values, 0.30)
    highlight_start = selection_onset(labels)
    final_start = len(frames) - 3
    initial_hold = growth_start >= 2 and all(
        bisector_values[index] < 0.08
        and circle_values[index] < 0.08
        and marker_values[index] < 0.02
        and labels[index] is None
        and float(np.mean(np.abs(aligned[index].astype(np.int16) - source.astype(np.int16)))) <= 2.0
        for index in range(min(4, growth_start))
    )
    bisector_start, bisector_end = growth_start, marker_start
    bisector_sequence = _sequential_single_end_lines(
        bisector_supports[bisector_start:bisector_end],
        bisector_directionals[bisector_start:bisector_end],
        complete_threshold=0.65,
        # Each four-step rasterized stroke exposes only one reliably
        # measurable partial prefix before reaching completion.
        min_partial_frames=1,
        progresses=bisector_fronts[bisector_start:bisector_end],
    )
    bisector_phase = bool(
        bisector_sequence["valid"]
        and all(
            # The completed bisectors can cross the future circle's
            # circumference, so a small amount of cyan is expected before
            # the circle itself is drawn.
            circle_values[index] <= 0.30
            and marker_values[index] < 0.02
            and labels[index] is None
            for index in range(bisector_start, bisector_end)
        )
    )
    marker_phase = bool(
        bisector_values[marker_start] >= 0.65
        and marker_values[marker_start] >= 0.01
        and circle_values[marker_start] <= 0.30
        and labels[marker_start] is None
    )
    circle_start, circle_end = circle_onset, highlight_start
    circle_partial_frames = [
        index
        for index in range(circle_start, circle_end)
        if 0.12 <= circle_values[index] < 0.82
    ]
    circle_phase = bool(
        all(
            bisector_values[index] >= 0.65
            and marker_values[index] >= 0.01
            and labels[index] is None
            for index in range(circle_start, circle_end)
        )
        # The first visible circle frame is already around half-covered
        # after WebM rasterization; this still represents a single-end
        # growth from an empty circle.
        and circle_values[circle_start] < 0.55
        and circle_values[circle_end - 1] >= 0.82
        and _no_large_regression(circle_values[circle_start:circle_end], 0.15)
        and len(circle_partial_frames) >= 1
        and all(
            circle_directionals[index] or circle_values[index] < 0.64
            for index in circle_partial_frames
        )
        and len({round(circle_values[index], 2) for index in range(circle_start, circle_end)}) >= 2
    )
    highlight_phase = bool(
        all(
            bisector_values[index] >= 0.68
            and circle_values[index] >= 0.80
            and marker_values[index] >= 0.01
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
    no_early_circle = all(value <= 0.30 for value in circle_values[:circle_start])
    no_early_highlight = all(label is None for label in labels[:highlight_start])
    no_vacancy = not any(bool(item["black_vacancy"]) for item in analyses)
    preservation_ok = bool(
        max(float(item["static_mutation_fraction"]) for item in analyses) <= 0.004
        and max(float(item["cyan_spill_fraction"]) for item in analyses) <= 0.14
    )
    temporal_ok = bool(
        initial_hold
        and bisector_phase
        and marker_phase
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
        "video_process_verifier": "point_geometry_incenter_full_sequence_semantic",
        "video_semantic_parser": "sequential_single_end_angle_bisectors_then_center_marker_then_single_end_incircle_parser",
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_decoded_frame_count": len(frames),
        "video_option": correct if final_ok else None,
        "video_incenter_initial_hold": initial_hold,
        "video_incenter_bisector_phase": bisector_phase,
        "video_incenter_bisector_activation_frames": bisector_sequence["activation_frames"],
        "video_incenter_bisector_completion_frames": bisector_sequence["completion_frames"],
        "video_incenter_bisector_draw_order": bisector_sequence["draw_order"],
        "video_incenter_bisector_partial_frame_counts": bisector_sequence["partial_frame_counts"],
        "video_incenter_maximum_simultaneous_partial_bisectors": bisector_sequence[
            "maximum_simultaneous_partials"
        ],
        "video_incenter_marker_phase": marker_phase,
        "video_incenter_circle_phase": circle_phase,
        "video_incenter_highlight_phase": highlight_phase,
        "video_incenter_final_hold": final_hold,
        "video_incenter_valid_highlight_labels": valid_labels,
        "video_incenter_no_early_circle": no_early_circle,
        "video_incenter_no_early_highlight": no_early_highlight,
        "video_incenter_no_black_vacancy": no_vacancy,
        "video_incenter_minimum_bisector_supports": bisector_values,
        "video_incenter_center_marker_supports": marker_values,
        "video_incenter_circle_supports": circle_values,
        "video_incenter_max_static_mutation_fraction": max(
            float(item["static_mutation_fraction"]) for item in analyses
        ),
        "video_incenter_max_cyan_spill_fraction": max(
            float(item["cyan_spill_fraction"]) for item in analyses
        ),
        "video_semantic_process_check_pass": passed,
        **_contract_metrics(
            artifact_kind="full_video",
            binary_correct=passed,
            component_scores=components,
            failure_reason="sequential angle bisectors, center marker, single-end incircle, highlight, preservation, or final hold failed",
            started=started,
        ),
    }, passed


TASK_NAME = "incenter"
VERIFIER = VerifierRoute(
    "data.geometry.core.verifier",
    "verify_coordinate_target",
)


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    """Evaluate only the formal final delivery for this task."""

    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [
    "INCENTER_STATE_TRACE",
    "INCENTER_VERIFIER_PROFILE",
    "TASK_NAME",
    "VERIFIER",
    "evaluate",
    "replay_incenter_geometry",
    "verify_incenter_full_video",
]

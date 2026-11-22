"""Independent multimodal verifier for ``circle_tangent_point``."""

from __future__ import annotations

from data.geometry.core.phase_observation import phase_onset, selection_onset, serial_construction_observation

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.common.verifiers.routes import VerifierRoute
from data.geometry.core.source_scene import observe_additive_source_scene
from data.geometry.circumcenter.evaluator import (
    foreign_ink_clearance,
    renderer_line_width,
)


CIRCLE_TANGENT_POINT_VERIFIER_PROFILE = "circle_tangent_point_multimodal_semantic_hard_gate_v1"
CIRCLE_TANGENT_POINT_STATE_TRACE = [
    "input_hold",
    "external_tangent_growth",
    "center_and_radius_growth",
    "right_angle_marker_growth",
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


def _angle_close(value: object, expected: float, tolerance: float = 1e-5) -> bool:
    parsed = _finite_float(value)
    if parsed is None:
        return False
    return abs((parsed - expected + 180.0) % 360.0 - 180.0) <= tolerance


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


def replay_circle_tangent_point_geometry(
    record: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Derive the displayed tangent winner without trusting a saved answer."""

    metrics: Dict[str, Any] = {
        "verifier_profile": CIRCLE_TANGENT_POINT_VERIFIER_PROFILE,
        "circle_tangent_point_geometry_replay_valid": False,
        "circle_tangent_point_geometry_replay_reason": "invalid structured source geometry",
    }
    structured = record.get("structured_answer")
    if not isinstance(structured, dict):
        return metrics, False, None
    center = _point(record.get("circle_center"))
    external = _point(record.get("external_point"))
    radius = _finite_float(record.get("circle_radius"))
    if center is None or external is None or radius is None or not 90.0 <= radius <= 190.0:
        return metrics, False, None
    center_external_distance = math.dist(center, external)
    if center_external_distance <= radius * 1.6:
        metrics["circle_tangent_point_geometry_replay_reason"] = "external point is not safely outside the circle"
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
        if abs(math.dist(center, candidate) - radius) > 1e-5:
            metrics["circle_tangent_point_geometry_replay_reason"] = "candidate center is not on the circle"
            return metrics, False, None
        candidates.append((label, candidate))
    if sorted(label for label, _candidate in candidates) != list("ABCDE"):
        return metrics, False, None
    candidates.sort(key=lambda item: item[0])

    derived: List[Dict[str, Any]] = []
    for label, candidate in candidates:
        inward_x = center[0] - candidate[0]
        inward_y = center[1] - candidate[1]
        external_x = external[0] - candidate[0]
        external_y = external[1] - candidate[1]
        inward_length = math.hypot(inward_x, inward_y)
        external_length = math.hypot(external_x, external_y)
        cosine = (inward_x * external_x + inward_y * external_y) / max(
            1e-9,
            inward_length * external_length,
        )
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        residual = abs(inward_x * external_x + inward_y * external_y) / max(1e-9, inward_length)
        derived.append(
            {
                "label": label,
                "center": [candidate[0], candidate[1]],
                "relative_direction_from_circle_center": _direction_name(
                    candidate[0] - center[0],
                    candidate[1] - center[1],
                ),
                "direction_from_candidate_to_external": _direction_name(external_x, external_y),
                "candidate_radius_bearing_degrees": math.degrees(
                    math.atan2(candidate[1] - center[1], candidate[0] - center[0]) % math.tau
                ),
                "candidate_to_external_bearing_degrees": math.degrees(
                    math.atan2(external_y, external_x) % math.tau
                ),
                "candidate_to_external_distance_pixels": external_length,
                "radius_external_angle_degrees": angle,
                "right_angle_error_degrees": abs(angle - 90.0),
                "perpendicularity_residual_pixels": residual,
            }
        )
    ordered = sorted(derived, key=lambda item: float(item["right_angle_error_degrees"]))
    derived_correct = str(ordered[0]["label"])
    runner_up_error = float(ordered[1]["right_angle_error_degrees"])
    unique = bool(
        float(ordered[0]["right_angle_error_degrees"]) <= 1e-5
        and runner_up_error >= 15.0
    )

    structured_candidates = structured.get("candidate_centers")
    candidate_metadata_valid = isinstance(structured_candidates, list) and len(structured_candidates) == 5
    if candidate_metadata_valid:
        by_label = {
            str(item.get("label") or "").strip().upper(): _point(item.get("center"))
            for item in structured_candidates
            if isinstance(item, dict)
        }
        candidate_metadata_valid = sorted(by_label) == list("ABCDE") and all(
            by_label[label] is not None
            and math.dist(by_label[label], candidates[index][1]) <= 1e-5  # type: ignore[arg-type]
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
            _point(stored_by_label[label].get("center")) is not None
            and math.dist(
                _point(stored_by_label[label]["center"]),  # type: ignore[arg-type]
                tuple(derived[index]["center"]),
            ) <= 1e-5
            and stored_by_label[label].get("relative_direction_from_circle_center")
            == derived[index]["relative_direction_from_circle_center"]
            and stored_by_label[label].get("direction_from_candidate_to_external")
            == derived[index]["direction_from_candidate_to_external"]
            and _angle_close(
                stored_by_label[label].get("candidate_radius_bearing_degrees"),
                float(derived[index]["candidate_radius_bearing_degrees"]),
            )
            and _angle_close(
                stored_by_label[label].get("candidate_to_external_bearing_degrees"),
                float(derived[index]["candidate_to_external_bearing_degrees"]),
            )
            and abs(
                float(stored_by_label[label].get("candidate_to_external_distance_pixels", 1e9))
                - float(derived[index]["candidate_to_external_distance_pixels"])
            ) <= 1e-5
            and abs(
                float(stored_by_label[label].get("radius_external_angle_degrees", 1e9))
                - float(derived[index]["radius_external_angle_degrees"])
            ) <= 1e-5
            and abs(
                float(stored_by_label[label].get("right_angle_error_degrees", 1e9))
                - float(derived[index]["right_angle_error_degrees"])
            ) <= 1e-5
            and abs(
                float(stored_by_label[label].get("perpendicularity_residual_pixels", 1e9))
                - float(derived[index]["perpendicularity_residual_pixels"])
            ) <= 1e-5
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
        and verifier_contract.get("profile") == CIRCLE_TANGENT_POINT_VERIFIER_PROFILE
        and verifier_contract.get("strict_binary_correctness") is True
        and isinstance(render_geometry, dict)
        and render_geometry.get("canvas_dimensions") == [512, 512]
        and abs(float(render_geometry.get("right_angle_marker_size", -1.0)) - 24.0) <= 1e-5
        and structured.get("state_trace") == CIRCLE_TANGENT_POINT_STATE_TRACE
        and structured.get("expected_video_frame_count") == 29
        and structured.get("video_fps") == 16
    )
    derived_index = next(index for index, item in enumerate(derived) if item["label"] == derived_correct)
    redundant_metadata_valid = bool(
        _point(record.get("circle_center")) == center
        and _point(record.get("external_point")) == external
        and abs(float(record.get("circle_radius", -1.0)) - radius) <= 1e-5
        and str(record.get("correct_option") or "").strip().upper() == derived_correct
        and str(structured.get("correct_option") or "").strip().upper() == derived_correct
        and structured.get("correct_index") == derived_index
        and abs(
            float(structured.get("runner_up_right_angle_error_degrees", -1.0))
            - runner_up_error
        ) <= 1e-5
        and structured.get("multiple_solution_policy")
        == "the_other_mathematical_tangent_point_is_not_present_among_A_to_E"
    )
    valid = bool(unique)
    metrics.update(
        {
            "circle_tangent_point_geometry_replay_valid": valid,
            "circle_tangent_point_geometry_replay_reason": (
                "ok" if valid else "source geometry is invalid or non-unique"
            ),
            "circle_tangent_point_derived_correct_option": derived_correct,
            "circle_tangent_point_derived_correct_index": derived_index,
            "circle_tangent_point_center_external_distance_pixels": center_external_distance,
            "circle_tangent_point_runner_up_right_angle_error_degrees": runner_up_error,
            "circle_tangent_point_candidate_measurements": derived,
            "circle_tangent_point_unique_displayed_solution": unique,
            "circle_tangent_point_candidate_metadata_valid": candidate_metadata_valid,
            "circle_tangent_point_measurement_metadata_valid": measurements_valid,
            "circle_tangent_point_contract_metadata_valid": contracts_valid,
            "circle_tangent_point_redundant_metadata_valid": redundant_metadata_valid,
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
        "verifier_profile": CIRCLE_TANGENT_POINT_VERIFIER_PROFILE,
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


def _point_segment_distance(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-9:
        return math.dist(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator))
    return math.dist(point, (start[0] + t * dx, start[1] + t * dy))


def _line_support(
    mask: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    excluded_segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]] = (),
    segment_clearance: float = 0.0,
) -> float:
    """Fraction of this segment that is lit, ignoring ink another stroke owns.

    ``excluded_segments`` names the other constructions the same frame legally
    contains.  A sample within ``segment_clearance`` of one of them can see that
    stroke's ink and so is no evidence about this segment; those samples are
    dropped rather than counted.  The right-angle mark needs this: its corner
    sits one marker width from the tangent point, so the leg leaving the radius
    starts inside one sample window of the radius and the leg reaching the
    tangent ends inside one window of the tangent.  Counted, those samples make
    a mark that has not been drawn yet read as partly present, by an amount that
    depends only on how the leg's direction falls on the pixel grid.
    """

    height, width = mask.shape
    hits = 0
    total = 0
    for fraction in np.linspace(0.12, 0.88, 28):
        exact_x = start[0] + (end[0] - start[0]) * float(fraction)
        exact_y = start[1] + (end[1] - start[1]) * float(fraction)
        if segment_clearance > 0.0 and any(
            _point_segment_distance((exact_x, exact_y), segment_start, segment_end)
            <= segment_clearance
            for segment_start, segment_end in excluded_segments
        ):
            continue
        x = int(round(exact_x))
        y = int(round(exact_y))
        x0, x1 = max(0, x - 3), min(width, x + 4)
        y0, y1 = max(0, y - 3), min(height, y + 4)
        total += 1
        hits += int(bool(np.any(mask[y0:y1, x0:x1])))
    return hits / total if total else 0.0


def _expected_geometry(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    replay_metrics, valid, correct = replay_circle_tangent_point_geometry(record)
    if not valid or correct is None:
        return None
    structured = record["structured_answer"]
    center = _point(structured["circle_center"])
    external = _point(structured["external_point"])
    centers = {
        str(item.get("label") or ""): _point(item.get("center"))
        for item in structured.get("candidate_centers") or []
        if isinstance(item, dict)
    }
    target = centers.get(correct)
    if center is None or external is None or target is None:
        return None
    radius = float(structured["circle_radius"])
    inward = ((center[0] - target[0]) / radius, (center[1] - target[1]) / radius)
    external_distance = math.dist(target, external)
    toward_external = (
        (external[0] - target[0]) / external_distance,
        (external[1] - target[1]) / external_distance,
    )
    # Fixed task visual grammar.  Canonical semantic input contains only the
    # visible source geometry, never answer-side render metadata.
    marker_size = 24.0
    q1 = (target[0] + inward[0] * marker_size, target[1] + inward[1] * marker_size)
    q2 = (
        q1[0] + toward_external[0] * marker_size,
        q1[1] + toward_external[1] * marker_size,
    )
    q3 = (
        target[0] + toward_external[0] * marker_size,
        target[1] + toward_external[1] * marker_size,
    )
    return {
        "center": center,
        "external": external,
        "target": target,
        "correct": correct,
        "right_angle_points": [q1, q2, q3],
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
    external = tuple(int(round(value)) for value in geometry["external"])
    target = tuple(int(round(value)) for value in geometry["target"])
    cv2.line(mask, center, target, 255, thickness=18)
    cv2.line(mask, external, target, 255, thickness=18)
    q1, q2, q3 = [
        tuple(int(round(value)) for value in point)
        for point in geometry["right_angle_points"]
    ]
    cv2.line(mask, q1, q2, 255, thickness=16)
    cv2.line(mask, q2, q3, 255, thickness=16)
    cv2.circle(mask, center, 12, 255, thickness=-1)
    cv2.circle(mask, target, max(18, int(round(point_radius * 2.0))), 255, thickness=-1)
    return mask


def _frame_analysis(
    frame: np.ndarray,
    source: np.ndarray,
    record: Dict[str, Any],
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    cyan, _dark_red = _color_masks(frame)
    red_label, red_mask = _red_option(frame, record)
    q1, q2, q3 = geometry["right_angle_points"]
    radius_support = _line_support(cyan, geometry["center"], geometry["target"])
    tangent_support = _line_support(cyan, geometry["external"], geometry["target"])
    # The radius and the tangent are the two strokes the right-angle mark is
    # built on, and they are already drawn when it appears, so ink within one
    # sample window of either belongs to them.
    clearance = foreign_ink_clearance(renderer_line_width(record))
    corner_neighbours = [
        (geometry["center"], geometry["target"]),
        (geometry["external"], geometry["target"]),
    ]
    corner_first = _line_support(
        cyan, q1, q2, excluded_segments=corner_neighbours, segment_clearance=clearance
    )
    corner_second = _line_support(
        cyan, q2, q3, excluded_segments=corner_neighbours, segment_clearance=clearance
    )
    cx, cy = (int(round(value)) for value in geometry["center"])
    y0, y1 = max(0, cy - 8), min(cyan.shape[0], cy + 9)
    x0, x1 = max(0, cx - 8), min(cyan.shape[1], cx + 9)
    center_dot_pixels = int(np.count_nonzero(cyan[y0:y1, x0:x1]))
    allowed = _allowed_dynamic_mask(cyan.shape, geometry, float(record.get("point_radius") or 11))
    cyan_count = int(np.count_nonzero(cyan))
    cyan_spill = float(np.count_nonzero(cyan & (allowed == 0)) / max(1, cyan_count))
    diff = np.max(np.abs(frame.astype(np.int16) - source.astype(np.int16)), axis=2)
    static_mutation = float(np.mean((diff >= 24)[allowed == 0]))
    point_radius = float(record.get("point_radius") or 11)
    preservation = observe_additive_source_scene(
        frame, source, record, allowed=allowed,
        discs=[(*geometry["center"], 12.0), (*geometry["target"], max(18.0, point_radius*2.0))],
        prefix="circle_tangent_point",
    )
    return {
        **preservation,
        "radius_support": radius_support,
        "tangent_support": tangent_support,
        "corner_first_support": corner_first,
        "corner_second_support": corner_second,
        "center_dot_pixels": center_dot_pixels,
        "red_option": red_label,
        "red_pixel_count": int(np.count_nonzero(red_mask)),
        "cyan_spill_fraction": cyan_spill,
        "static_mutation_fraction": static_mutation,
        "black_vacancy": float(np.mean(frame)) < 35.0,
    }


def _final_semantics(analysis: Dict[str, Any], correct: str) -> Tuple[bool, bool]:
    preservation_ok = bool(
        bool(analysis["source_preservation_pass"])
        and bool(analysis["marker_state_pass"])
        and float(analysis["cyan_spill_fraction"]) <= 0.16
        and not analysis["black_vacancy"]
    )
    final_ok = bool(
        float(analysis["radius_support"]) >= 0.80
        and float(analysis["tangent_support"]) >= 0.80
        and float(analysis["corner_first_support"]) >= 0.68
        and float(analysis["corner_second_support"]) >= 0.68
        and int(analysis["center_dot_pixels"]) >= 18
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


def verify_circle_tangent_point_full_video(
    video_path: Path,
    record: Dict[str, Any],
    input_path: Path,
) -> Tuple[Dict[str, Any], bool]:
    started = time.perf_counter()
    replay_metrics, replay_valid, correct = replay_circle_tangent_point_geometry(record)
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
            "video_process_verifier": "point_geometry_circle_tangent_point_full_sequence_semantic",
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
    radius_values = [float(item["radius_support"]) for item in analyses]
    tangent_values = [float(item["tangent_support"]) for item in analyses]
    corner_values = [
        min(float(item["corner_first_support"]), float(item["corner_second_support"]))
        for item in analyses
    ]
    labels = [item["red_option"] for item in analyses]
    first_start = phase_onset(tangent_values, 0.08)
    second_start = phase_onset(radius_values, 0.08)
    corner_start = phase_onset([
        max(float(item["corner_first_support"]), float(item["corner_second_support"]))
        for item in analyses
    ], 0.25)
    highlight_start = selection_onset(labels)
    final_start = len(frames) - 3
    growth_metrics, growth_pass = serial_construction_observation(
        aligned, source, record, task_name="circle_tangent_point",
        expected=str(correct), end=highlight_start,
    )
    initial_hold = first_start >= 2 and all(
        radius_values[index] < 0.08
        and tangent_values[index] < 0.08
        and corner_values[index] < 0.08
        and labels[index] is None
        and float(np.mean(np.abs(aligned[index].astype(np.int16) - source.astype(np.int16)))) <= 2.0
        for index in range(min(4, first_start))
    )
    tangent_phase = bool(
        all(
            radius_values[index] < 0.08
            and float(analyses[index]["corner_first_support"]) <= 0.25
            and float(analyses[index]["corner_second_support"]) <= 0.25
            and labels[index] is None
            for index in range(first_start, second_start)
        )
        and tangent_values[second_start - 1] >= 0.80
        and _no_large_regression(tangent_values[first_start:second_start], 0.18)
    )
    radius_phase = bool(
        all(
            tangent_values[index] >= 0.78
            and float(analyses[index]["corner_first_support"]) <= 0.25
            and float(analyses[index]["corner_second_support"]) <= 0.25
            and labels[index] is None
            for index in range(second_start, corner_start)
        )
        and radius_values[corner_start - 1] >= 0.80
        and _no_large_regression(radius_values[second_start:corner_start], 0.18)
    )
    corner_phase = bool(
        all(
            radius_values[index] >= 0.78
            and tangent_values[index] >= 0.78
            and labels[index] is None
            for index in range(corner_start, highlight_start)
        )
        and corner_values[highlight_start - 1] >= 0.68
        and _no_large_regression(corner_values[corner_start:highlight_start], 0.22)
    )
    highlight_phase = bool(
        all(
            radius_values[index] >= 0.78
            and tangent_values[index] >= 0.78
            and corner_values[index] >= 0.66
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
    no_vacancy = not any(bool(item["black_vacancy"]) for item in analyses)
    preservation_ok = bool(
        all(bool(item["source_preservation_pass"]) for item in analyses)
        and max(float(item["cyan_spill_fraction"]) for item in analyses) <= 0.16
    )
    temporal_ok = bool(
        initial_hold
        and growth_pass is not False
        and tangent_phase
        and radius_phase
        and corner_phase
        and highlight_phase
        and final_hold
        and valid_labels
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
        "video_process_verifier": "point_geometry_circle_tangent_point_full_sequence_semantic",
        "video_semantic_parser": "radius_external_tangent_right_angle_unique_highlight_phase_parser",
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_decoded_frame_count": len(frames),
        "video_option": correct if final_ok else None,
        **growth_metrics,
        "video_circle_tangent_point_single_endpoint_growth": growth_pass is not False,
        "video_circle_tangent_point_initial_hold": initial_hold,
        "video_circle_tangent_point_radius_phase": radius_phase,
        "video_circle_tangent_point_tangent_phase": tangent_phase,
        "video_circle_tangent_point_right_angle_phase": corner_phase,
        "video_circle_tangent_point_highlight_phase": highlight_phase,
        "video_circle_tangent_point_final_hold": final_hold,
        "video_circle_tangent_point_valid_highlight_labels": valid_labels,
        "video_circle_tangent_point_no_black_vacancy": no_vacancy,
        "video_circle_tangent_point_source_preservation_pass": all(
            bool(item["source_preservation_pass"]) for item in analyses
        ),
        "video_circle_tangent_point_max_static_region_error": max(
            float(item["source_static_region_max_error"]) for item in analyses
        ),
        "video_circle_tangent_point_scene_failure_frames": [
            index for index, item in enumerate(analyses) if not item["source_preservation_pass"]
        ],
        "video_circle_tangent_point_max_static_mutation_fraction": max(
            float(item["static_mutation_fraction"]) for item in analyses
        ),
        "video_circle_tangent_point_max_cyan_spill_fraction": max(
            float(item["cyan_spill_fraction"]) for item in analyses
        ),
        "video_semantic_process_check_pass": passed,
        **_contract_metrics(
            artifact_kind="full_video",
            binary_correct=passed,
            component_scores=components,
            failure_reason="ordered radius, tangent, right-angle, highlight, preservation, or final hold failed",
            started=started,
        ),
    }, passed


TASK_NAME = "circle_tangent_point"
VERIFIER = VerifierRoute(
    "data.geometry.core.verifier",
    "verify_coordinate_target",
)


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    """Evaluate only the formal final delivery for this task."""

    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [
    "CIRCLE_TANGENT_POINT_STATE_TRACE",
    "CIRCLE_TANGENT_POINT_VERIFIER_PROFILE",
    "TASK_NAME",
    "VERIFIER",
    "evaluate",
    "replay_circle_tangent_point_geometry",
    "verify_circle_tangent_point_full_video",
]

"""Independent multimodal verifier for ``isosceles_trapezoid``."""

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


ISOSCELES_TRAPEZOID_VERIFIER_PROFILE = (
    "isosceles_trapezoid_multimodal_semantic_hard_gate_v1"
)
ISOSCELES_TRAPEZOID_STATE_TRACE = [
    "input_hold",
    "shorter_base_growth",
    "equal_leg_growth",
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


def _points_close(value: object, expected: Tuple[float, float], tolerance: float = 1e-5) -> bool:
    parsed = _point(value)
    return parsed is not None and math.dist(parsed, expected) <= tolerance


def _cross(
    first: Tuple[float, float],
    second: Tuple[float, float],
    third: Tuple[float, float],
) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _strictly_convex(points: Sequence[Tuple[float, float]]) -> bool:
    if len(points) != 4:
        return False
    values = [
        _cross(points[index], points[(index + 1) % 4], points[(index + 2) % 4])
        for index in range(4)
    ]
    return all(value > 1e-5 for value in values) or all(value < -1e-5 for value in values)


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


def _candidate_measurements(
    junction: Tuple[float, float],
    base_endpoint: Tuple[float, float],
    leg_endpoint: Tuple[float, float],
    candidates: Sequence[Tuple[str, Tuple[float, float]]],
) -> Tuple[List[Dict[str, Any]], float, float]:
    base_dx = base_endpoint[0] - junction[0]
    base_dy = base_endpoint[1] - junction[1]
    base_length = math.hypot(base_dx, base_dy)
    if base_length <= 1e-9:
        return [], base_length, 0.0
    unit_x, unit_y = base_dx / base_length, base_dy / base_length
    known_leg = math.dist(junction, leg_endpoint)
    measurements: List[Dict[str, Any]] = []
    for label, candidate in candidates:
        top_dx = candidate[0] - leg_endpoint[0]
        top_dy = candidate[1] - leg_endpoint[1]
        forward = top_dx * unit_x + top_dy * unit_y
        signed_offset = top_dx * unit_y - top_dy * unit_x
        right_leg = math.dist(candidate, base_endpoint)
        mismatch = abs(right_leg - known_leg)
        convex = _strictly_convex((junction, base_endpoint, candidate, leg_endpoint))
        shorter_positive = 1e-5 < forward < base_length - 1e-5
        eligible = bool(
            abs(signed_offset) <= 1e-5
            and mismatch <= 1e-5
            and convex
            and shorter_positive
        )
        residual = math.hypot(abs(signed_offset), mismatch)
        if not convex or not shorter_positive:
            residual += 40.0
        measurements.append(
            {
                "label": label,
                "center": [candidate[0], candidate[1]],
                "direction_from_V": _direction_name(top_dx, top_dy),
                "parallel_signed_offset_pixels": signed_offset,
                "parallel_offset_pixels": abs(signed_offset),
                "top_base_forward_length_pixels": forward,
                "closing_leg_length_pixels": right_leg,
                "known_visible_leg_length_pixels": known_leg,
                "leg_length_mismatch_pixels": mismatch,
                "strict_convex": convex,
                "shorter_positive_top_base": shorter_positive,
                "eligible": eligible,
                "decision_residual_pixels": residual,
            }
        )
    return measurements, base_length, known_leg


def _derive_missing_vertex(
    junction: Tuple[float, float],
    base_endpoint: Tuple[float, float],
    leg_endpoint: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Solve the parallel-line/equal-leg constraints and exclude the parallelogram root."""

    base_dx = base_endpoint[0] - junction[0]
    base_dy = base_endpoint[1] - junction[1]
    base_length = math.hypot(base_dx, base_dy)
    known_leg = math.dist(junction, leg_endpoint)
    if base_length <= 1e-9 or known_leg <= 1e-9:
        return None
    unit_x, unit_y = base_dx / base_length, base_dy / base_length
    wx = leg_endpoint[0] - base_endpoint[0]
    wy = leg_endpoint[1] - base_endpoint[1]
    projection = wx * unit_x + wy * unit_y
    constant = wx * wx + wy * wy - known_leg * known_leg
    discriminant = projection * projection - constant
    if discriminant < -1e-6:
        return None
    root = math.sqrt(max(0.0, discriminant))
    candidates: List[Tuple[float, Tuple[float, float]]] = []
    for forward in (-projection - root, -projection + root):
        point = (
            leg_endpoint[0] + forward * unit_x,
            leg_endpoint[1] + forward * unit_y,
        )
        if (
            1e-5 < forward < base_length - 1e-5
            and _strictly_convex((junction, base_endpoint, point, leg_endpoint))
        ):
            candidates.append((forward, point))
    if len(candidates) != 1:
        return None
    return candidates[0][1]


def replay_isosceles_trapezoid_geometry(
    record: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Recompute the winner from visible vertices and candidate centers."""

    metrics: Dict[str, Any] = {
        "verifier_profile": ISOSCELES_TRAPEZOID_VERIFIER_PROFILE,
        "isosceles_trapezoid_geometry_replay_valid": False,
        "isosceles_trapezoid_geometry_replay_reason": "invalid visible source geometry",
    }
    visible = record.get("visible_vertices")
    if not isinstance(visible, dict):
        return metrics, False, None
    junction = _point(visible.get("J"))
    base_endpoint = _point(visible.get("U"))
    leg_endpoint = _point(visible.get("V"))
    if junction is None or base_endpoint is None or leg_endpoint is None:
        return metrics, False, None
    long_base = math.dist(junction, base_endpoint)
    visible_leg = math.dist(junction, leg_endpoint)
    if (
        long_base < 180.0
        or visible_leg < 90.0
        or long_base - visible_leg < 35.0
        or abs(_cross(junction, base_endpoint, leg_endpoint)) < 5000.0
    ):
        metrics["isosceles_trapezoid_geometry_replay_reason"] = "visible polyline is degenerate or endpoint naming is ambiguous"
        return metrics, False, None

    derived_vertex = _derive_missing_vertex(junction, base_endpoint, leg_endpoint)
    if derived_vertex is None:
        metrics["isosceles_trapezoid_geometry_replay_reason"] = "strict shorter-base solution is not unique"
        return metrics, False, None

    raw_candidates = record.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 5:
        return metrics, False, None
    candidates: List[Tuple[str, Tuple[float, float]]] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            return metrics, False, None
        label = str(raw.get("label") or "").strip().upper()
        center = _point([raw.get("x"), raw.get("y")])
        if label not in "ABCDE" or center is None:
            return metrics, False, None
        candidates.append((label, center))
    if sorted(label for label, _center in candidates) != list("ABCDE"):
        return metrics, False, None
    candidates.sort(key=lambda item: item[0])

    derived, base_length, known_leg = _candidate_measurements(
        junction,
        base_endpoint,
        leg_endpoint,
        candidates,
    )
    if len(derived) != 5:
        return metrics, False, None
    eligible = [item for item in derived if bool(item["eligible"])]
    ordered = sorted(derived, key=lambda item: float(item["decision_residual_pixels"]))
    derived_correct = str(ordered[0]["label"])
    runner_up = float(ordered[1]["decision_residual_pixels"])
    winner_center = candidates[ord(derived_correct) - ord("A")][1]
    unique = bool(
        len(eligible) == 1
        and str(eligible[0]["label"]) == derived_correct
        and float(ordered[0]["decision_residual_pixels"]) <= 1e-5
        and runner_up >= 18.0
        and math.dist(winner_center, derived_vertex) <= 1e-5
    )

    structured = record.get("structured_answer")
    if not isinstance(structured, dict):
        return metrics, False, derived_correct
    stored_visible = structured.get("visible_vertices")
    visible_metadata_valid = bool(
        isinstance(stored_visible, dict)
        and _points_close(stored_visible.get("J"), junction)
        and _points_close(stored_visible.get("U"), base_endpoint)
        and _points_close(stored_visible.get("V"), leg_endpoint)
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
            and stored_by_label[label].get("direction_from_V") == derived[index]["direction_from_V"]
            and all(
                _float_close(stored_by_label[label].get(field), float(derived[index][field]))
                for field in (
                    "parallel_signed_offset_pixels",
                    "parallel_offset_pixels",
                    "top_base_forward_length_pixels",
                    "closing_leg_length_pixels",
                    "known_visible_leg_length_pixels",
                    "leg_length_mismatch_pixels",
                    "decision_residual_pixels",
                )
            )
            and stored_by_label[label].get("strict_convex") is derived[index]["strict_convex"]
            and stored_by_label[label].get("shorter_positive_top_base") is derived[index]["shorter_positive_top_base"]
            and stored_by_label[label].get("eligible") is derived[index]["eligible"]
            for index, label in enumerate("ABCDE")
        )

    answer_schema = structured.get("answer_schema")
    verifier_contract = structured.get("verifier_contract")
    render_geometry = structured.get("render_geometry")
    record_point_radius = _finite_float(record.get("point_radius"))
    record_line_width = _finite_float(record.get("line_width"))
    contracts_valid = bool(
        isinstance(answer_schema, dict)
        and answer_schema.get("terminal_text_regex") == FIVE_CHOICE_CODEC.full_pattern
        and answer_schema.get("requires_sample_specific_reasoning_before_terminal_answer") is True
        and answer_schema.get("terminal_text_contains_exact_answer") is True
        and answer_schema.get("no_block_after_terminal_image") is True
        and isinstance(verifier_contract, dict)
        and verifier_contract.get("profile") == ISOSCELES_TRAPEZOID_VERIFIER_PROFILE
        and verifier_contract.get("strict_binary_correctness") is True
        and isinstance(render_geometry, dict)
        and render_geometry.get("canvas_dimensions") == [512, 512]
        and record_point_radius is not None
        and record_line_width is not None
        and _float_close(render_geometry.get("point_radius"), record_point_radius)
        and _float_close(render_geometry.get("line_width"), record_line_width)
        and structured.get("state_trace") == ISOSCELES_TRAPEZOID_STATE_TRACE
        and structured.get("expected_video_frame_count") == 25
        and structured.get("video_fps") == 16
    )
    derived_index = ord(derived_correct) - ord("A")
    top_length = math.dist(leg_endpoint, derived_vertex)
    redundant_metadata_valid = bool(
        visible_metadata_valid
        and _points_close(structured.get("derived_missing_vertex"), derived_vertex)
        and _float_close(structured.get("long_base_length_pixels"), base_length)
        and _float_close(structured.get("short_base_length_pixels"), top_length)
        and _float_close(structured.get("equal_leg_length_pixels"), known_leg)
        and str(record.get("correct_option") or "").strip().upper() == derived_correct
        and str(structured.get("correct_option") or "").strip().upper() == derived_correct
        and structured.get("correct_index") == derived_index
        and _float_close(structured.get("runner_up_decision_residual_pixels"), runner_up)
        and structured.get("multiple_solution_policy")
        == "exactly_one_displayed_candidate_forms_a_strict_convex_shorter_top_base_isosceles_trapezoid"
    )
    valid = bool(unique)
    metrics.update(
        {
            "isosceles_trapezoid_geometry_replay_valid": valid,
            "isosceles_trapezoid_geometry_replay_reason": (
                "ok" if valid else "source geometry is invalid or non-unique"
            ),
            "isosceles_trapezoid_derived_correct_option": derived_correct,
            "isosceles_trapezoid_derived_correct_index": derived_index,
            "isosceles_trapezoid_derived_missing_vertex": [derived_vertex[0], derived_vertex[1]],
            "isosceles_trapezoid_long_base_length_pixels": base_length,
            "isosceles_trapezoid_short_base_length_pixels": top_length,
            "isosceles_trapezoid_equal_leg_length_pixels": known_leg,
            "isosceles_trapezoid_runner_up_decision_residual_pixels": runner_up,
            "isosceles_trapezoid_candidate_measurements": derived,
            "isosceles_trapezoid_unique_displayed_solution": unique,
            "isosceles_trapezoid_visible_metadata_valid": visible_metadata_valid,
            "isosceles_trapezoid_candidate_metadata_valid": candidate_metadata_valid,
            "isosceles_trapezoid_measurement_metadata_valid": measurements_valid,
            "isosceles_trapezoid_contract_metadata_valid": contracts_valid,
            "isosceles_trapezoid_redundant_metadata_valid": redundant_metadata_valid,
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
        "verifier_profile": ISOSCELES_TRAPEZOID_VERIFIER_PROFILE,
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


def _construction_and_red_masks(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    cyan = (
        (red <= 90)
        & (green >= 95)
        & (green <= 215)
        & (blue >= 120)
        & (blue <= 235)
        & (green - red >= 35)
        & (blue - red >= 55)
    )
    dark_red = (
        (red >= 130)
        & (green <= 125)
        & (red - green >= 45)
        & (red - blue >= 20)
    )
    return cyan, dark_red


def _red_option(frame: np.ndarray, record: Dict[str, Any]) -> Tuple[Optional[str], np.ndarray]:
    _construction, dark_red = _construction_and_red_masks(frame)
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
    excluded_disks: Sequence[Tuple[float, float, float]],
) -> float:
    height, width = mask.shape
    hits = 0
    total = 0
    for fraction in np.linspace(0.08, 0.92, 32):
        x_value = start[0] + (end[0] - start[0]) * float(fraction)
        y_value = start[1] + (end[1] - start[1]) * float(fraction)
        if any(
            (x_value - cx) ** 2 + (y_value - cy) ** 2 <= radius * radius
            for cx, cy, radius in excluded_disks
        ):
            continue
        x = int(round(x_value))
        y = int(round(y_value))
        x0, x1 = max(0, x - 3), min(width, x + 4)
        y0, y1 = max(0, y - 3), min(height, y + 4)
        total += 1
        hits += int(bool(np.any(mask[y0:y1, x0:x1])))
    return hits / total if total else 0.0


def _expected_geometry(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    replay_metrics, valid, correct = replay_isosceles_trapezoid_geometry(record)
    if not valid or correct is None:
        return None
    visible = record.get("visible_vertices")
    if not isinstance(visible, dict):
        return None
    junction = _point(visible.get("J"))
    base_endpoint = _point(visible.get("U"))
    leg_endpoint = _point(visible.get("V"))
    target = _point(replay_metrics.get("isosceles_trapezoid_derived_missing_vertex"))
    if None in (junction, base_endpoint, leg_endpoint, target):
        return None
    return {
        "junction": junction,
        "base_endpoint": base_endpoint,
        "leg_endpoint": leg_endpoint,
        "target": target,
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
    leg_endpoint = tuple(int(round(value)) for value in geometry["leg_endpoint"])
    target = tuple(int(round(value)) for value in geometry["target"])
    base_endpoint = tuple(int(round(value)) for value in geometry["base_endpoint"])
    cv2.line(mask, leg_endpoint, target, 255, thickness=18)
    cv2.line(mask, target, base_endpoint, 255, thickness=18)
    cv2.circle(mask, target, max(18, int(round(point_radius * 2.0))), 255, thickness=-1)
    return mask


def _frame_analysis(
    frame: np.ndarray,
    source: np.ndarray,
    record: Dict[str, Any],
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    construction, _dark_red = _construction_and_red_masks(frame)
    source_construction, _source_red = _construction_and_red_masks(source)
    new_construction = construction & ~source_construction
    red_label, red_mask = _red_option(frame, record)
    exclusion_radius = float(record.get("point_radius") or 11) * 1.55
    exclusions = [
        (float(item["x"]), float(item["y"]), exclusion_radius)
        for item in record.get("candidates") or []
        if isinstance(item, dict)
        and _finite_float(item.get("x")) is not None
        and _finite_float(item.get("y")) is not None
    ]
    top_support = _line_support(
        new_construction,
        geometry["leg_endpoint"],
        geometry["target"],
        exclusions,
    )
    right_support = _line_support(
        new_construction,
        geometry["target"],
        geometry["base_endpoint"],
        exclusions,
    )
    allowed = _allowed_dynamic_mask(
        construction.shape,
        geometry,
        float(record.get("point_radius") or 11),
    )
    construction_count = int(np.count_nonzero(new_construction))
    construction_spill = float(
        np.count_nonzero(new_construction & (allowed == 0)) / max(1, construction_count)
    )
    diff = np.max(np.abs(frame.astype(np.int16) - source.astype(np.int16)), axis=2)
    static_mutation = float(np.mean((diff >= 24)[allowed == 0]))
    point_radius = float(record.get("point_radius") or 11)
    preservation = observe_additive_source_scene(
        frame, source, record, allowed=allowed,
        discs=[(*geometry["target"], max(18.0, point_radius*2.0))],
        prefix="isosceles_trapezoid",
    )
    return {
        **preservation,
        "top_base_support": top_support,
        "right_leg_support": right_support,
        "red_option": red_label,
        "red_pixel_count": int(np.count_nonzero(red_mask)),
        "new_construction_spill_fraction": construction_spill,
        "static_mutation_fraction": static_mutation,
        "black_vacancy": float(np.mean(frame)) < 35.0,
    }


def _final_semantics(analysis: Dict[str, Any], correct: str) -> Tuple[bool, bool]:
    preservation_ok = bool(
        bool(analysis["source_preservation_pass"])
        and bool(analysis["marker_state_pass"])
        and float(analysis["new_construction_spill_fraction"]) <= 0.18
        and not analysis["black_vacancy"]
    )
    final_ok = bool(
        float(analysis["top_base_support"]) >= 0.72
        and float(analysis["right_leg_support"]) >= 0.72
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


def verify_isosceles_trapezoid_full_video(
    video_path: Path,
    record: Dict[str, Any],
    input_path: Path,
) -> Tuple[Dict[str, Any], bool]:
    started = time.perf_counter()
    replay_metrics, replay_valid, correct = replay_isosceles_trapezoid_geometry(record)
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
            "video_process_verifier": "point_geometry_isosceles_trapezoid_full_sequence_semantic",
            "video_frame_count": len(frames),
            "video_fps": fps,
            **_contract_metrics(
                artifact_kind="full_video",
                binary_correct=False,
                component_scores=components,
                failure_reason="video media contract or source replay failed",
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
    top_values = [float(item["top_base_support"]) for item in analyses]
    right_values = [float(item["right_leg_support"]) for item in analyses]
    labels = [item["red_option"] for item in analyses]
    top_start = phase_onset(top_values, 0.08)
    right_start = phase_onset(right_values, 0.16)
    highlight_start = selection_onset(labels)
    final_start = len(frames) - 3
    growth_metrics, growth_pass = serial_construction_observation(
        aligned, source, record, task_name="isosceles_trapezoid",
        expected=str(correct), end=highlight_start,
    )
    initial_hold = top_start >= 2 and all(
        top_values[index] < 0.08
        and right_values[index] < 0.08
        and labels[index] is None
        and float(np.mean(np.abs(aligned[index].astype(np.int16) - source.astype(np.int16)))) <= 2.0
        for index in range(min(4, top_start))
    )
    top_phase = bool(
        all(right_values[index] <= 0.16 and labels[index] is None for index in range(top_start, right_start))
        and top_values[right_start - 1] >= 0.72
        and _no_large_regression(top_values[top_start:right_start], 0.18)
    )
    right_phase = bool(
        all(top_values[index] >= 0.70 and labels[index] is None for index in range(right_start, highlight_start))
        and right_values[highlight_start - 1] >= 0.72
        and _no_large_regression(right_values[right_start:highlight_start], 0.18)
    )
    highlight_phase = bool(
        all(
            top_values[index] >= 0.70
            and right_values[index] >= 0.70
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
    no_early_highlight = all(label is None for label in labels[:highlight_start])
    no_vacancy = not any(bool(item["black_vacancy"]) for item in analyses)
    preservation_ok = bool(
        all(bool(item["source_preservation_pass"]) for item in analyses)
        and max(float(item["new_construction_spill_fraction"]) for item in analyses) <= 0.18
    )
    temporal_ok = bool(
        initial_hold
        and growth_pass is not False
        and top_phase
        and right_phase
        and highlight_phase
        and final_hold
        and valid_labels
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
        "video_process_verifier": "point_geometry_isosceles_trapezoid_full_sequence_semantic",
        "video_semantic_parser": "shorter_base_then_equal_leg_then_unique_highlight_phase_parser",
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_decoded_frame_count": len(frames),
        "video_option": correct if final_ok else None,
        **growth_metrics,
        "video_isosceles_trapezoid_single_endpoint_growth": growth_pass is not False,
        "video_isosceles_trapezoid_initial_hold": initial_hold,
        "video_isosceles_trapezoid_top_base_phase": top_phase,
        "video_isosceles_trapezoid_right_leg_phase": right_phase,
        "video_isosceles_trapezoid_highlight_phase": highlight_phase,
        "video_isosceles_trapezoid_final_hold": final_hold,
        "video_isosceles_trapezoid_valid_highlight_labels": valid_labels,
        "video_isosceles_trapezoid_no_early_highlight": no_early_highlight,
        "video_isosceles_trapezoid_no_black_vacancy": no_vacancy,
        "video_isosceles_trapezoid_top_base_supports": top_values,
        "video_isosceles_trapezoid_right_leg_supports": right_values,
        "video_isosceles_trapezoid_source_preservation_pass": all(
            bool(item["source_preservation_pass"]) for item in analyses
        ),
        "video_isosceles_trapezoid_max_static_region_error": max(
            float(item["source_static_region_max_error"]) for item in analyses
        ),
        "video_isosceles_trapezoid_scene_failure_frames": [
            index for index, item in enumerate(analyses) if not item["source_preservation_pass"]
        ],
        "video_isosceles_trapezoid_max_static_mutation_fraction": max(
            float(item["static_mutation_fraction"]) for item in analyses
        ),
        "video_isosceles_trapezoid_max_new_construction_spill_fraction": max(
            float(item["new_construction_spill_fraction"]) for item in analyses
        ),
        "video_semantic_process_check_pass": passed,
        **_contract_metrics(
            artifact_kind="full_video",
            binary_correct=passed,
            component_scores=components,
            failure_reason="ordered closing edges, highlight, preservation, or final hold failed",
            started=started,
        ),
    }, passed


TASK_NAME = "isosceles_trapezoid"
VERIFIER = VerifierRoute(
    "data.geometry.core.verifier",
    "verify_coordinate_target",
)


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    """Evaluate only the formal final delivery for this task."""

    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [
    "ISOSCELES_TRAPEZOID_STATE_TRACE",
    "ISOSCELES_TRAPEZOID_VERIFIER_PROFILE",
    "TASK_NAME",
    "VERIFIER",
    "evaluate",
    "replay_isosceles_trapezoid_geometry",
    "verify_isosceles_trapezoid_full_video",
]

"""Independent multimodal verifier for ``midpoint``."""

from __future__ import annotations

from data.geometry.core.phase_observation import selection_onset

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from data.common.verifiers.keyframes import visual_delta

from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.common.verifiers.routes import VerifierRoute


MIDPOINT_VERIFIER_PROFILE = "midpoint_multimodal_semantic_hard_gate_v1"
MIDPOINT_STATE_TRACE = [
    "input_hold",
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


def _direction_name(dx: float, dy: float) -> str:
    if math.hypot(dx, dy) <= 1e-5:
        return "center"
    names = (
        "right", "lower-right", "down", "lower-left",
        "left", "upper-left", "up", "upper-right",
    )
    angle = math.atan2(dy, dx) % math.tau
    return names[int((angle + math.pi / 8.0) // (math.pi / 4.0)) % 8]


def _candidate_measurements(
    p: Tuple[float, float],
    q: Tuple[float, float],
    candidates: Sequence[Tuple[str, Tuple[float, float]]],
) -> Tuple[List[Dict[str, Any]], Tuple[float, float], float]:
    midpoint = ((p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5)
    vx, vy = q[0] - p[0], q[1] - p[1]
    length_sq = vx * vx + vy * vy
    segment_length = math.sqrt(length_sq)
    half = segment_length * 0.5
    measurements: List[Dict[str, Any]] = []
    for label, center in candidates:
        dx, dy = center[0] - midpoint[0], center[1] - midpoint[1]
        distance_p = math.dist(center, p)
        distance_q = math.dist(center, q)
        along = (
            ((center[0] - p[0]) * vx + (center[1] - p[1]) * vy) / length_sq
            if length_sq > 1e-9 else float("nan")
        )
        signed_perpendicular = (
            ((center[0] - p[0]) * vy - (center[1] - p[1]) * vx) / segment_length
            if segment_length > 1e-9 else float("nan")
        )
        midpoint_error = math.hypot(dx, dy)
        on_segment = abs(signed_perpendicular) <= 1e-5 and -1e-5 <= along <= 1.0 + 1e-5
        strictly_between = on_segment and 1e-5 < along < 1.0 - 1e-5
        measurements.append(
            {
                "label": label,
                "center": [center[0], center[1]],
                "direction_from_M": _direction_name(dx, dy),
                "offset_from_M": [dx, dy],
                "midpoint_error_pixels": midpoint_error,
                "distance_to_P_pixels": distance_p,
                "distance_to_Q_pixels": distance_q,
                "endpoint_distance_difference_pixels": abs(distance_p - distance_q),
                "half_length_pixels": half,
                "along_segment_fraction": along,
                "perpendicular_offset_pixels": abs(signed_perpendicular),
                "on_segment": on_segment,
                "strictly_between_endpoints": strictly_between,
                "eligible": bool(midpoint_error <= 1e-5 and strictly_between),
            }
        )
    return measurements, midpoint, segment_length


def replay_midpoint_geometry(
    record: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Derive the unique winner from visible endpoints and candidate centers."""

    metrics: Dict[str, Any] = {
        "verifier_profile": MIDPOINT_VERIFIER_PROFILE,
        "midpoint_geometry_replay_valid": False,
        "midpoint_geometry_replay_reason": "invalid visible endpoint geometry",
    }
    segment = record.get("segment")
    if not isinstance(segment, dict):
        return metrics, False, None
    p = _point(segment.get("start"))
    q = _point(segment.get("end"))
    if p is None or q is None:
        return metrics, False, None
    segment_length = math.dist(p, q)
    if segment_length < 100.0 or (p[0], p[1]) > (q[0], q[1]):
        metrics["midpoint_geometry_replay_reason"] = "degenerate segment or noncanonical visible endpoint naming"
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
    candidates.sort(key=lambda item: item[0])
    if [label for label, _center in candidates] != list("ABCDE"):
        return metrics, False, None

    derived, midpoint, derived_length = _candidate_measurements(p, q, candidates)
    ordered = sorted(derived, key=lambda item: float(item["midpoint_error_pixels"]))
    correct = str(ordered[0]["label"])
    runner_up = float(ordered[1]["midpoint_error_pixels"])
    eligible = [item for item in derived if bool(item["eligible"])]
    unique = bool(
        len(eligible) == 1
        and str(eligible[0]["label"]) == correct
        and float(ordered[0]["midpoint_error_pixels"]) <= 1e-5
        and runner_up >= 32.0
    )

    structured = record.get("structured_answer")
    if not isinstance(structured, dict):
        return metrics, False, correct
    visible = structured.get("visible_endpoints")
    visible_valid = bool(
        isinstance(visible, dict)
        and _points_close(visible.get("P"), p)
        and _points_close(visible.get("Q"), q)
    )
    stored_centers = structured.get("candidate_centers")
    centers_valid = isinstance(stored_centers, list) and len(stored_centers) == 5
    if centers_valid:
        by_label = {
            str(item.get("label") or "").strip().upper(): item.get("center")
            for item in stored_centers
            if isinstance(item, dict)
        }
        centers_valid = sorted(by_label) == list("ABCDE") and all(
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
        numeric_fields = (
            "midpoint_error_pixels",
            "distance_to_P_pixels",
            "distance_to_Q_pixels",
            "endpoint_distance_difference_pixels",
            "half_length_pixels",
            "along_segment_fraction",
            "perpendicular_offset_pixels",
        )
        measurements_valid = sorted(stored_by_label) == list("ABCDE") and all(
            _points_close(stored_by_label[label].get("center"), candidates[index][1])
            and _points_close(stored_by_label[label].get("offset_from_M"), tuple(derived[index]["offset_from_M"]))
            and stored_by_label[label].get("direction_from_M") == derived[index]["direction_from_M"]
            and all(
                _float_close(stored_by_label[label].get(field), float(derived[index][field]))
                for field in numeric_fields
            )
            and stored_by_label[label].get("on_segment") is derived[index]["on_segment"]
            and stored_by_label[label].get("strictly_between_endpoints") is derived[index]["strictly_between_endpoints"]
            and stored_by_label[label].get("eligible") is derived[index]["eligible"]
            for index, label in enumerate("ABCDE")
        )

    schema = structured.get("answer_schema")
    verifier = structured.get("verifier_contract")
    render = structured.get("render_geometry")
    contracts_valid = bool(
        isinstance(schema, dict)
        and schema.get("terminal_text_regex") == FIVE_CHOICE_CODEC.full_pattern
        and schema.get("requires_sample_specific_reasoning_before_terminal_answer") is True
        and schema.get("terminal_text_contains_exact_answer") is True
        and schema.get("no_block_after_terminal_image") is True
        and isinstance(verifier, dict)
        and verifier.get("profile") == MIDPOINT_VERIFIER_PROFILE
        and verifier.get("strict_binary_correctness") is True
        and isinstance(render, dict)
        and render.get("canvas_dimensions") == [512, 512]
        and _float_close(render.get("point_radius"), float(record.get("point_radius")))
        and _float_close(render.get("line_width"), float(record.get("line_width")))
        and structured.get("state_trace") == MIDPOINT_STATE_TRACE
        and structured.get("expected_video_frame_count") == 17
        and structured.get("video_fps") == 16
    )
    correct_index = ord(correct) - ord("A")
    redundant_valid = bool(
        _points_close(record.get("midpoint"), midpoint)
        and _points_close(structured.get("derived_midpoint"), midpoint)
        and _float_close(structured.get("segment_length_pixels"), derived_length)
        and _float_close(structured.get("half_length_pixels"), derived_length * 0.5)
        and str(record.get("correct_option") or "").strip().upper() == correct
        and str(structured.get("correct_option") or "").strip().upper() == correct
        and structured.get("correct_index") == correct_index
        and _float_close(structured.get("runner_up_midpoint_error_pixels"), runner_up)
        and structured.get("multiple_solution_policy")
        == "exactly_one_displayed_candidate_center_equals_the_coordinate_midpoint"
        and record.get("construction_trace") == MIDPOINT_STATE_TRACE
    )
    valid = bool(unique and visible_valid)
    metrics.update(
        {
            "midpoint_geometry_replay_valid": valid,
            "midpoint_geometry_replay_reason": (
                "ok" if valid else "source geometry is invalid or non-unique"
            ),
            "midpoint_derived_correct_option": correct,
            "midpoint_derived_correct_index": correct_index,
            "midpoint_derived": [midpoint[0], midpoint[1]],
            "midpoint_segment_length_pixels": derived_length,
            "midpoint_half_length_pixels": derived_length * 0.5,
            "midpoint_runner_up_error_pixels": runner_up,
            "midpoint_candidate_measurements": derived,
            "midpoint_unique_displayed_solution": unique,
            "midpoint_visible_metadata_valid": visible_valid,
            "midpoint_candidate_metadata_valid": centers_valid,
            "midpoint_measurement_metadata_valid": measurements_valid,
            "midpoint_contract_metadata_valid": contracts_valid,
            "midpoint_redundant_metadata_valid": redundant_valid,
        }
    )
    return metrics, valid, correct


def _contract_metrics(
    *,
    artifact_kind: str,
    binary_correct: bool,
    component_scores: Dict[str, float],
    failure_reason: str,
    started: float,
) -> Dict[str, Any]:
    return {
        "verifier_profile": MIDPOINT_VERIFIER_PROFILE,
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
    return rgb if rgb.shape[:2] == (height, width) else None


def _color_masks(frame: np.ndarray) -> Dict[str, np.ndarray]:
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    return {
        "dark": (red <= 115) & (green <= 115) & (blue <= 115),
        "cyan": (blue >= 135) & (green >= 105) & (blue - red >= 55) & (green - red >= 45),
        "orange": (red >= 175) & (green >= 75) & (green <= 190) & (blue <= 110) & (red - blue >= 90),
        "red": (red >= 130) & (green <= 105) & (red - green >= 65) & (red - blue >= 25),
    }


def _red_option(frame: np.ndarray, record: Dict[str, Any]) -> Tuple[Optional[str], np.ndarray]:
    red_mask = _color_masks(frame)["red"]
    radius = max(12, int(round(float(record.get("point_radius") or 11) * 1.35)))
    height, width = red_mask.shape
    yy, xx = np.ogrid[:height, :width]
    selected: List[str] = []
    for candidate in record.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        center = _point([candidate.get("x"), candidate.get("y")])
        if center is None:
            continue
        disk = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius * radius
        if int(np.count_nonzero(red_mask & disk)) >= 15:
            selected.append(str(candidate.get("label") or "").strip().upper())
    return ("".join(sorted(label for label in selected if label)) or None), red_mask


def _line_support(
    mask: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    exclusions: Sequence[Tuple[float, float, float]],
) -> float:
    height, width = mask.shape
    hits = total = 0
    for fraction in np.linspace(0.06, 0.94, 36):
        x_value = start[0] + (end[0] - start[0]) * float(fraction)
        y_value = start[1] + (end[1] - start[1]) * float(fraction)
        if any((x_value - cx) ** 2 + (y_value - cy) ** 2 <= radius * radius for cx, cy, radius in exclusions):
            continue
        x, y = int(round(x_value)), int(round(y_value))
        x0, x1 = max(0, x - 3), min(width, x + 4)
        y0, y1 = max(0, y - 3), min(height, y + 4)
        total += 1
        hits += int(bool(np.any(mask[y0:y1, x0:x1])))
    return hits / total if total else 0.0


def _expected_geometry(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    replay_metrics, valid, correct = replay_midpoint_geometry(record)
    segment = record.get("segment")
    p = _point(segment.get("start")) if isinstance(segment, dict) else None
    q = _point(segment.get("end")) if isinstance(segment, dict) else None
    midpoint = _point(replay_metrics.get("midpoint_derived"))
    if not valid or correct is None or p is None or q is None or midpoint is None:
        return None
    return {"p": p, "q": q, "midpoint": midpoint, "correct": correct, "replay_metrics": replay_metrics}


def _tube(shape: Tuple[int, int], start: Tuple[float, float], end: Tuple[float, float], thickness: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(mask, tuple(map(round, start)), tuple(map(round, end)), 255, thickness=thickness)
    return mask


def _frame_analysis(
    frame: np.ndarray,
    source: np.ndarray,
    record: Dict[str, Any],
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    masks = _color_masks(frame)
    source_masks = _color_masks(source)
    # The complete source segment is not a new construction when codec
    # rasterization moves its edge into the adjacent pixel.
    retained_dark = cv2.dilate(source_masks["dark"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    new_dark = masks["dark"] & ~retained_dark
    red_label, red_mask = _red_option(frame, record)
    point_radius = float(record.get("point_radius") or 11)
    anchor_radius = max(point_radius + 6.0, min(frame.shape[:2]) * 0.028)
    exclusions = [
        (float(item["x"]), float(item["y"]), point_radius * 1.55)
        for item in record.get("candidates") or []
        if isinstance(item, dict) and _finite_float(item.get("x")) is not None and _finite_float(item.get("y")) is not None
    ] + [
        (geometry["p"][0], geometry["p"][1], anchor_radius * 1.25),
        (geometry["q"][0], geometry["q"][1], anchor_radius * 1.25),
    ]
    segment_support = _line_support(new_dark, geometry["p"], geometry["q"], exclusions)
    cyan_support = _line_support(masks["cyan"], geometry["p"], geometry["midpoint"], exclusions)
    orange_support = _line_support(masks["orange"], geometry["q"], geometry["midpoint"], exclusions)
    ink = new_dark | masks["cyan"] | masks["orange"]
    total_support = _line_support(ink, geometry["p"], geometry["q"], exclusions)
    full_tube = _tube(masks["dark"].shape, geometry["p"], geometry["q"], 18)
    first_tube = _tube(masks["dark"].shape, geometry["p"], geometry["midpoint"], 14)
    second_tube = _tube(masks["dark"].shape, geometry["q"], geometry["midpoint"], 14)
    # The source already contains the complete segment.  The only legal
    # delivery mutation is the selected midpoint marker itself.
    allowed = np.zeros_like(full_tube)
    cv2.circle(
        allowed,
        tuple(map(round, geometry["midpoint"])),
        max(18, int(round(point_radius * 2.0))),
        255,
        thickness=-1,
    )
    diff = np.max(visual_delta(frame, source), axis=2)
    static_mutation = float(np.mean((diff >= 24)[allowed == 0]))
    fixed_disk = np.zeros_like(allowed)
    cv2.circle(fixed_disk, tuple(map(round, geometry["p"])), int(round(anchor_radius * 1.2)), 255, thickness=-1)
    cv2.circle(fixed_disk, tuple(map(round, geometry["q"])), int(round(anchor_radius * 1.2)), 255, thickness=-1)
    for item in record.get("candidates") or []:
        if not isinstance(item, dict) or str(item.get("label") or "").strip().upper() == geometry["correct"]:
            continue
        center = _point([item.get("x"), item.get("y")])
        if center is not None:
            cv2.circle(fixed_disk, tuple(map(round, center)), int(round(point_radius * 1.3)), 255, thickness=-1)
    fixed_object_mutation = float(np.mean(diff[fixed_disk > 0]) / 255.0) if np.any(fixed_disk) else 1.0
    new_dark_count = int(np.count_nonzero(new_dark))
    cyan_count = int(np.count_nonzero(masks["cyan"]))
    orange_count = int(np.count_nonzero(masks["orange"]))
    return {
        "segment_dark_support": segment_support,
        "cyan_support": cyan_support,
        "orange_support": orange_support,
        "total_segment_support": total_support,
        "red_option": red_label,
        "red_pixel_count": int(np.count_nonzero(red_mask)),
        "new_dark_spill_fraction": float(np.count_nonzero(new_dark & (full_tube == 0)) / max(1, new_dark_count)),
        "cyan_spill_fraction": float(np.count_nonzero(masks["cyan"] & (first_tube == 0)) / max(1, cyan_count)),
        "orange_spill_fraction": float(np.count_nonzero(masks["orange"] & (second_tube == 0)) / max(1, orange_count)),
        "static_mutation_fraction": static_mutation,
        "fixed_object_mutation_fraction": fixed_object_mutation,
        "black_vacancy": float(np.mean(frame)) < 35.0,
    }


def _final_semantics(analysis: Dict[str, Any], correct: str) -> Tuple[bool, bool]:
    preservation_ok = bool(
        float(analysis["static_mutation_fraction"]) <= 0.004
        and float(analysis["fixed_object_mutation_fraction"]) <= 0.030
        and not analysis["black_vacancy"]
    )
    final_ok = bool(
        float(analysis["cyan_support"]) < 0.08
        and float(analysis["orange_support"]) < 0.08
        and float(analysis["segment_dark_support"]) < 0.08
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


def verify_midpoint_full_video(
    video_path: Path,
    record: Dict[str, Any],
    input_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], bool]:
    started = time.perf_counter()
    replay_metrics, replay_valid, correct = replay_midpoint_geometry(record)
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
    exact_frame_sizes = bool(frames) and all(frame.shape[:2] == (height, width) for frame in frames)
    source = _read_rgb(input_path, (width, height)) if input_path is not None else (frames[0] if frames else None)
    media_valid = bool(
        opened
        and len(frames) >= 9
        and abs(fps - 16.0) <= 0.25
        and exact_frame_sizes
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
            "video_process_verifier": "point_geometry_midpoint_full_sequence_semantic",
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

    analyses = [_frame_analysis(frame, source, record, geometry) for frame in frames]
    labels = [item["red_option"] for item in analyses]
    red_counts = [int(item["red_pixel_count"]) for item in analyses]
    highlight_start = selection_onset(labels)
    final_start = len(frames) - 3
    initial_hold = highlight_start >= 2 and all(
        float(analyses[index]["segment_dark_support"]) < 0.08
        and float(analyses[index]["cyan_support"]) < 0.08
        and float(analyses[index]["orange_support"]) < 0.08
        and labels[index] is None
        and float(np.mean(np.abs(frames[index].astype(np.int16) - source.astype(np.int16)))) <= 2.0
        for index in range(min(4, highlight_start))
    )
    highlight_phase = bool(
        all(
            float(analyses[index]["segment_dark_support"]) < 0.08
            and float(analyses[index]["cyan_support"]) < 0.08
            and float(analyses[index]["orange_support"]) < 0.08
            and labels[index] in {None, correct}
            for index in range(min(4, highlight_start), final_start)
        )
        and labels[final_start - 1] == correct
        # Thresholded red area fluctuates along a reconstructed marker edge.
        # Require the visible commit to remain, using the same relative
        # withdrawal budget as the common point-target selection observer.
        and _no_large_regression(
            [float(value) for value in red_counts[min(4, highlight_start):final_start]],
            max(8.0, red_counts[-1] * 0.25),
        )
    )
    final_semantics = [_final_semantics(analyses[index], correct) for index in range(final_start, len(frames))]
    final_hold = all(item == (True, True) for item in final_semantics) and all(
        float(np.mean(np.abs(frames[index].astype(np.int16) - frames[-1].astype(np.int16)))) <= 2.0
        for index in range(final_start, len(frames))
    )
    valid_labels = all(label in {None, correct} for label in labels)
    no_construction_ink = all(
        float(item["segment_dark_support"]) < 0.08
        and float(item["cyan_support"]) < 0.08
        and float(item["orange_support"]) < 0.08
        for item in analyses
    )
    no_early_highlight = all(label is None for label in labels[:min(4, highlight_start)])
    no_vacancy = not any(bool(item["black_vacancy"]) for item in analyses)
    preservation_ok = bool(
        max(float(item["static_mutation_fraction"]) for item in analyses) <= 0.004
        and max(float(item["fixed_object_mutation_fraction"]) for item in analyses) <= 0.030
    )
    temporal_ok = bool(
        initial_hold
        and highlight_phase
        and final_hold
        and valid_labels
        and no_construction_ink
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
        "video_process_verifier": "point_geometry_midpoint_full_sequence_semantic",
        "video_semantic_parser": "direct_unique_midpoint_highlight_phase_parser",
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_decoded_frame_count": len(frames),
        "video_option": correct if final_ok else None,
        "video_midpoint_initial_hold": initial_hold,
        "video_midpoint_highlight_phase": highlight_phase,
        "video_midpoint_final_hold": final_hold,
        "video_midpoint_valid_highlight_labels": valid_labels,
        "video_midpoint_no_construction_ink": no_construction_ink,
        "video_midpoint_no_early_highlight": no_early_highlight,
        "video_midpoint_no_black_vacancy": no_vacancy,
        "video_midpoint_red_pixel_counts": red_counts,
        "video_midpoint_max_static_mutation_fraction": max(float(item["static_mutation_fraction"]) for item in analyses),
        "video_midpoint_max_fixed_object_mutation_fraction": max(float(item["fixed_object_mutation_fraction"]) for item in analyses),
        "video_semantic_process_check_pass": passed,
        **_contract_metrics(
            artifact_kind="full_video",
            binary_correct=passed,
            component_scores=components,
            failure_reason="direct highlight, source preservation, or final hold failed",
            started=started,
        ),
    }, passed


TASK_NAME = "midpoint"
VERIFIER = VerifierRoute(
    "data.geometry.core.verifier",
    "verify_coordinate_target",
)


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    """Evaluate only the formal final delivery for this task."""

    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [
    "MIDPOINT_STATE_TRACE",
    "MIDPOINT_VERIFIER_PROFILE",
    "TASK_NAME",
    "VERIFIER",
    "evaluate",
    "replay_midpoint_geometry",
    "verify_midpoint_full_video",
]

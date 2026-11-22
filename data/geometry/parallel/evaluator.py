"""Independent multimodal semantic verifier for ``parallel``."""

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
    _sequential_single_end_lines,
    _single_end_line_observation,
)


PARALLEL_VERIFIER_PROFILE = "parallel_multimodal_semantic_hard_gate_v1"
PARALLEL_STATE_TRACE = [
    "input_hold",
    "parallel_line_single_end_growth",
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


def _points_close(
    value: object,
    expected: Tuple[float, float],
    tolerance: float = 1e-5,
) -> bool:
    parsed = _point(value)
    return parsed is not None and math.dist(parsed, expected) <= tolerance


def _direction_name(dx: float, dy: float) -> str:
    if math.hypot(dx, dy) <= 1e-5:
        return "center"
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
    through_point: Tuple[float, float],
    first: Tuple[float, float],
    second: Tuple[float, float],
    candidates: Sequence[Tuple[str, Tuple[float, float]]],
) -> Tuple[List[Dict[str, Any]], Tuple[float, float], float]:
    vx, vy = second[0] - first[0], second[1] - first[1]
    reference_length = math.hypot(vx, vy)
    ux, uy = vx / reference_length, vy / reference_length
    measurements: List[Dict[str, Any]] = []
    for label, center in candidates:
        dx, dy = center[0] - through_point[0], center[1] - through_point[1]
        along = dx * ux + dy * uy
        signed_offset = ux * dy - uy * dx
        measurements.append(
            {
                "label": label,
                "center": [center[0], center[1]],
                "direction_from_T": _direction_name(dx, dy),
                "along_line_projection_pixels": along,
                "signed_perpendicular_offset_pixels": signed_offset,
                "perpendicular_offset_pixels": abs(signed_offset),
                "eligible": abs(signed_offset) <= 1e-5,
            }
        )
    return measurements, (ux, uy), reference_length


def _measurement_table_valid(
    stored: object,
    derived: Sequence[Dict[str, Any]],
) -> bool:
    if not isinstance(stored, list) or len(stored) != 5:
        return False
    by_label = {
        str(item.get("label") or "").strip().upper(): item
        for item in stored
        if isinstance(item, dict)
    }
    if sorted(by_label) != list("ABCDE"):
        return False
    numeric_fields = (
        "along_line_projection_pixels",
        "signed_perpendicular_offset_pixels",
        "perpendicular_offset_pixels",
    )
    return all(
        _points_close(by_label[label].get("center"), tuple(item["center"]))
        and by_label[label].get("direction_from_T") == item["direction_from_T"]
        and all(
            _float_close(by_label[label].get(field), float(item[field]))
            for field in numeric_fields
        )
        and by_label[label].get("eligible") is item["eligible"]
        for label, item in zip("ABCDE", derived)
    )


def replay_parallel_geometry(
    record: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Recover the unique winner from visible source geometry, not the saved oracle."""

    metrics: Dict[str, Any] = {
        "verifier_profile": PARALLEL_VERIFIER_PROFILE,
        "parallel_geometry_replay_valid": False,
        "parallel_geometry_replay_reason": "invalid visible reference, T, or candidate geometry",
    }
    raw_reference = record.get("source_line_endpoints") or record.get("reference_line_endpoints")
    if not isinstance(raw_reference, list) or len(raw_reference) != 2:
        return metrics, False, None
    first = _point(raw_reference[0])
    second = _point(raw_reference[1])
    through_point = _point(record.get("through_point"))
    if first is None or second is None or through_point is None:
        return metrics, False, None
    if (first[0], first[1]) > (second[0], second[1]):
        metrics["parallel_geometry_replay_reason"] = "reference endpoints are not in canonical reporting order"
        return metrics, False, None
    reference_length = math.dist(first, second)
    if reference_length < 100.0:
        metrics["parallel_geometry_replay_reason"] = "reference segment is degenerate or too short"
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

    derived, unit, derived_length = _candidate_measurements(
        through_point,
        first,
        second,
        candidates,
    )
    ordered = sorted(derived, key=lambda item: float(item["perpendicular_offset_pixels"]))
    correct = str(ordered[0]["label"])
    correct_index = ord(correct) - ord("A")
    runner_up = float(ordered[1]["perpendicular_offset_pixels"])
    eligible = [item for item in derived if item["eligible"]]
    unique = bool(
        len(eligible) == 1
        and eligible[0]["label"] == correct
        and float(ordered[0]["perpendicular_offset_pixels"]) <= 1e-5
        and runner_up >= 24.0
    )

    structured = record.get("structured_answer")
    if not isinstance(structured, dict):
        return metrics, False, correct
    stored_reference = structured.get("reference_endpoints")
    stored_centers = structured.get("candidate_centers")
    reference_valid = bool(
        isinstance(stored_reference, dict)
        and _points_close(stored_reference.get("R1"), first)
        and _points_close(stored_reference.get("R2"), second)
        and _points_close(structured.get("reference_vector"), (second[0] - first[0], second[1] - first[1]))
        and _points_close(structured.get("reference_unit_vector"), unit)
        and _points_close(structured.get("through_point"), through_point)
    )
    centers_valid = isinstance(stored_centers, list) and len(stored_centers) == 5
    if centers_valid:
        center_by_label = {
            str(item.get("label") or "").strip().upper(): item.get("center")
            for item in stored_centers
            if isinstance(item, dict)
        }
        centers_valid = sorted(center_by_label) == list("ABCDE") and all(
            _points_close(center_by_label[label], center)
            for label, center in candidates
        )
    measurements_valid = bool(
        _measurement_table_valid(record.get("candidate_measurements"), derived)
        and _measurement_table_valid(structured.get("candidate_measurements"), derived)
    )

    schema = structured.get("answer_schema")
    verifier = structured.get("verifier_contract")
    render = structured.get("render_geometry")
    record_point_radius = _finite_float(record.get("point_radius"))
    record_line_width = _finite_float(record.get("line_width"))
    contracts_valid = bool(
        isinstance(schema, dict)
        and schema.get("terminal_text_regex") == FIVE_CHOICE_CODEC.full_pattern
        and schema.get("requires_sample_specific_reasoning_before_terminal_answer") is True
        and schema.get("terminal_text_contains_exact_answer") is True
        and schema.get("no_block_after_terminal_image") is True
        and isinstance(verifier, dict)
        and verifier.get("profile") == PARALLEL_VERIFIER_PROFILE
        and verifier.get("judged_artifacts") == ["text", "final_image", "full_video"]
        and verifier.get("strict_binary_correctness") is True
        and isinstance(render, dict)
        and render.get("canvas_dimensions") == [512, 512]
        and record_point_radius is not None
        and record_line_width is not None
        and _float_close(render.get("point_radius"), record_point_radius)
        and _float_close(render.get("through_point_radius"), 7.0)
        and _float_close(render.get("line_width"), record_line_width)
        and structured.get("state_trace") == PARALLEL_STATE_TRACE
        and structured.get("expected_video_frame_count") == 21
        and structured.get("video_fps") == 16
    )
    redundant_valid = bool(
        str(record.get("correct_option") or "").strip().upper() == correct
        and str(structured.get("correct_option") or "").strip().upper() == correct
        and structured.get("correct_index") == correct_index
        and _float_close(structured.get("runner_up_perpendicular_offset_pixels"), runner_up)
        and structured.get("multiple_solution_policy")
        == "exactly_one_displayed_candidate_center_is_on_the_parallel_through_T"
        and record.get("construction_trace") == PARALLEL_STATE_TRACE
    )
    valid = bool(unique)
    metrics.update(
        {
            "parallel_geometry_replay_valid": valid,
            "parallel_geometry_replay_reason": (
                "ok" if valid else "source geometry is invalid or non-unique"
            ),
            "parallel_derived_correct_option": correct,
            "parallel_derived_correct_index": correct_index,
            "parallel_reference_length_pixels": derived_length,
            "parallel_reference_unit_vector": [unit[0], unit[1]],
            "parallel_runner_up_perpendicular_offset_pixels": runner_up,
            "parallel_candidate_measurements": derived,
            "parallel_unique_displayed_solution": unique,
            "parallel_reference_metadata_valid": reference_valid,
            "parallel_candidate_metadata_valid": centers_valid,
            "parallel_measurement_metadata_valid": measurements_valid,
            "parallel_contract_metadata_valid": contracts_valid,
            "parallel_redundant_metadata_valid": redundant_valid,
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
        "verifier_profile": PARALLEL_VERIFIER_PROFILE,
        "artifact_kind": artifact_kind,
        "component_scores": component_scores,
        "binary_correct": bool(binary_correct),
        "strict_binary_correctness": True,
        "failure_reason": "ok" if binary_correct else failure_reason,
        "verification_elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "performance_policy": "latency_throughput_memory_and_decode_counts_are_diagnostics_without_failure_thresholds",
    }


def _read_rgb(path: Optional[Path], size: Tuple[int, int]) -> Optional[np.ndarray]:
    if path is None:
        return None
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
        "cyan": (red <= 100) & (green >= 105) & (blue >= 130) & (green - red >= 35) & (blue - red >= 60),
        "red": (red >= 130) & (green <= 125) & (red - green >= 45) & (red - blue >= 20),
    }


def _red_option(frame: np.ndarray, record: Dict[str, Any]) -> Tuple[Optional[str], np.ndarray]:
    red_mask = _color_masks(frame)["red"]
    radius = max(12, int(round(float(record.get("point_radius") or 11) * 1.40)))
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
) -> float:
    height, width = mask.shape
    hits = total = 0
    for fraction in np.linspace(0.03, 0.97, 40):
        x = int(round(start[0] + (end[0] - start[0]) * float(fraction)))
        y = int(round(start[1] + (end[1] - start[1]) * float(fraction)))
        x0, x1 = max(0, x - 3), min(width, x + 4)
        y0, y1 = max(0, y - 3), min(height, y + 4)
        total += 1
        hits += int(bool(np.any(mask[y0:y1, x0:x1])))
    return hits / total if total else 0.0




def _expected_geometry(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    replay_metrics, valid, correct = replay_parallel_geometry(record)
    through_point = _point(record.get("through_point"))
    candidate = next(
        (
            _point([item.get("x"), item.get("y")])
            for item in record.get("candidates") or []
            if isinstance(item, dict) and str(item.get("label") or "").strip().upper() == correct
        ),
        None,
    )
    # Candidate disks use the fixed Geometry point-target grammar.  The
    # canonical source omits renderer configuration by design.
    point_radius = _finite_float(record.get("point_radius")) or 11.0
    if not valid or correct is None or through_point is None or candidate is None:
        return None
    dx, dy = candidate[0] - through_point[0], candidate[1] - through_point[1]
    length = math.hypot(dx, dy)
    if length <= 72.0 + point_radius + 2.0:
        return None
    ux, uy = dx / length, dy / length
    tail = (through_point[0] - ux * 36.0, through_point[1] - uy * 36.0)
    guide_front = (through_point[0] + ux * 36.0, through_point[1] + uy * 36.0)
    final_end = (
        candidate[0] - ux * (point_radius + 2.0),
        candidate[1] - uy * (point_radius + 2.0),
    )
    return {
        "through_point": through_point,
        "candidate": candidate,
        "tail": tail,
        "guide_front": guide_front,
        "final_end": final_end,
        "correct": correct,
        "replay_metrics": replay_metrics,
    }


def _tube(
    shape: Tuple[int, int],
    start: Tuple[float, float],
    end: Tuple[float, float],
    thickness: int,
) -> np.ndarray:
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
    red_label, red_mask = _red_option(frame, record)
    allowed_line = _tube(masks["cyan"].shape, geometry["tail"], geometry["final_end"], 18)
    allowed_dynamic = allowed_line.copy()
    point_radius = float(record.get("point_radius") or 11)
    cv2.circle(
        allowed_dynamic,
        tuple(map(round, geometry["candidate"])),
        max(18, int(round(point_radius * 1.8))),
        255,
        thickness=-1,
    )
    cyan_count = int(np.count_nonzero(masks["cyan"]))
    # The opaque through-point marker is repainted over the construction, so
    # samples it fully covers are unobservable rather than undrawn.  A sample
    # window of Chebyshev radius 3 still sees the line just outside the marker,
    # so only samples closer than (marker radius - window radius) are blind.
    through_marker_radius = float(record.get("through_point_radius") or 7.0)
    line_observation = _single_end_line_observation(
        masks["cyan"],
        geometry["tail"],
        geometry["final_end"],
        [
            (
                float(geometry["through_point"][0]),
                float(geometry["through_point"][1]),
                max(0.0, through_marker_radius - 3.0),
            )
        ],
        # The Prompt fixes the direction: from the short tail on one side of T,
        # through T, to the winning candidate centre on the other side.
        accept_either_end=False,
    )
    diff = np.max(np.abs(frame.astype(np.int16) - source.astype(np.int16)), axis=2)
    static_region = allowed_dynamic == 0
    static_mutation = float(np.mean((diff >= 24)[static_region])) if np.any(static_region) else 1.0
    return {
        "guide_support": _line_support(masks["cyan"], geometry["tail"], geometry["guide_front"]),
        "extension_support": _line_support(
            masks["cyan"], geometry["guide_front"], geometry["final_end"]
        ),
        "line_support": _line_support(masks["cyan"], geometry["tail"], geometry["final_end"]),
        "line_single_end_observation": line_observation,
        "cyan_pixel_count": cyan_count,
        "cyan_spill_fraction": float(
            np.count_nonzero(masks["cyan"] & (allowed_line == 0)) / max(1, cyan_count)
        ),
        "red_option": red_label,
        "red_pixel_count": int(np.count_nonzero(red_mask)),
        "static_mutation_fraction": static_mutation,
        "black_vacancy": float(np.mean(frame)) < 35.0,
    }


def _final_semantics(analysis: Dict[str, Any], correct: str) -> Tuple[bool, bool]:
    preservation_ok = bool(
        float(analysis["static_mutation_fraction"]) <= 0.004
        and float(analysis["cyan_spill_fraction"]) <= 0.16
        and not analysis["black_vacancy"]
    )
    final_ok = bool(
        float(analysis["guide_support"]) >= 0.72
        and float(analysis["extension_support"]) >= 0.72
        and float(analysis["line_support"]) >= 0.72
        and int(analysis["cyan_pixel_count"]) >= 100
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


def verify_parallel_full_video(
    video_path: Path,
    record: Dict[str, Any],
    input_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], bool]:
    started = time.perf_counter()
    replay_metrics, replay_valid, correct = replay_parallel_geometry(record)
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
    exact_sizes = bool(frames) and all(frame.shape[:2] == (height, width) for frame in frames)
    source = _read_rgb(input_path, (width, height)) if input_path is not None else (frames[0] if frames else None)
    media_valid = bool(
        opened
        and len(frames) >= 9
        and abs(fps - 16.0) <= 0.25
        and exact_sizes
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
            "video_process_verifier": "point_geometry_parallel_full_sequence_semantic",
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
    supports = [float(item["line_support"]) for item in analyses]
    line_directionals = [
        bool(item["line_single_end_observation"]["single_end_directional"])
        for item in analyses
    ]
    line_fronts = [
        float(item["line_single_end_observation"]["observed_progress"])
        for item in analyses
    ]
    guide_supports = [float(item["guide_support"]) for item in analyses]
    extension_supports = [float(item["extension_support"]) for item in analyses]
    cyan_counts = [int(item["cyan_pixel_count"]) for item in analyses]
    labels = [item["red_option"] for item in analyses]
    red_counts = [int(item["red_pixel_count"]) for item in analyses]
    # A short visible prefix may not yet reach the sampled coverage floor.
    # Its ink still starts the operation under the unchanged-input ink gate.
    first_visible_ink = next((index for index, count in enumerate(cyan_counts) if count >= 20), len(frames) - 3)
    growth_start = min(phase_onset(supports, 0.08), first_visible_ink)
    highlight_start = selection_onset(labels)
    final_start = len(frames) - 3
    initial_hold = growth_start >= 2 and all(
        supports[index] < 0.08
        and cyan_counts[index] < 20
        and labels[index] is None
        and float(np.mean(np.abs(frames[index].astype(np.int16) - source.astype(np.int16)))) <= 2.0
        for index in range(min(4, growth_start))
    )
    line_start, line_end = growth_start, highlight_start
    line_sequence = _sequential_single_end_lines(
        [[supports[index]] for index in range(line_start, line_end)],
        [[line_directionals[index]] for index in range(line_start, line_end)],
        complete_threshold=0.70,
        regression_tolerance=0.14,
        progresses=[[line_fronts[index]] for index in range(line_start, line_end)],
    )
    line_phase_counts = cyan_counts[line_start:line_end]
    line_phase = bool(
        line_sequence["valid"]
        and all(labels[index] is None for index in range(line_start, line_end))
        and supports[line_start] < 0.35
        and _no_large_regression([float(value) for value in line_phase_counts], 20.0)
        and len(set(line_phase_counts)) >= 2
    )
    highlight_phase = bool(
        all(
            guide_supports[index] >= 0.70
            and extension_supports[index] >= 0.70
            and supports[index] >= 0.70
            for index in range(highlight_start, final_start)
        )
        and all(labels[index] in {None, correct} for index in range(highlight_start, final_start))
        and labels[final_start - 1] == correct
        and _no_large_regression([float(value) for value in red_counts[highlight_start:final_start]], 12.0)
        and len(set(red_counts[highlight_start:final_start])) >= 1
    )
    final_semantics = [_final_semantics(analyses[index], correct) for index in range(final_start, len(frames))]
    final_hold = all(item == (True, True) for item in final_semantics) and all(
        float(np.mean(np.abs(frames[index].astype(np.int16) - frames[-1].astype(np.int16)))) <= 2.0
        for index in range(final_start, len(frames))
    )
    valid_labels = all(label in {None, correct} for label in labels)
    no_early_cyan = all(value < 20 for value in cyan_counts[:growth_start])
    no_early_highlight = all(label is None for label in labels[:highlight_start])
    no_vacancy = not any(bool(item["black_vacancy"]) for item in analyses)
    preservation_ok = bool(
        all(float(item["static_mutation_fraction"]) <= 0.004 for item in analyses)
        and all(float(item["cyan_spill_fraction"]) <= 0.16 for item in analyses[growth_start:])
        and valid_labels
        and no_vacancy
    )
    temporal_ok = bool(
        initial_hold
        and line_phase
        and highlight_phase
        and no_early_cyan
        and no_early_highlight
    )
    final_ok = bool(final_hold)
    passed = bool(media_valid and replay_valid and temporal_ok and preservation_ok and final_ok)
    components = {
        "media_validity_score": 1.0,
        "source_replay_score": float(replay_valid),
        "temporal_process_score": float(temporal_ok),
        "preservation_score": float(preservation_ok),
        "final_state_score": float(final_ok),
    }
    return {
        **replay_metrics,
        "video_process_verifier": "point_geometry_parallel_full_sequence_semantic",
        "video_semantic_parser": "single_end_tail_through_T_to_winner_line_parser",
        "video_frame_count": len(frames),
        "video_decoded_frame_count": len(frames),
        "video_fps": fps,
        "video_parallel_initial_hold": initial_hold,
        "video_parallel_single_end_line_phase": line_phase,
        "video_parallel_line_activation_frames": line_sequence["activation_frames"],
        "video_parallel_line_completion_frames": line_sequence["completion_frames"],
        "video_parallel_line_partial_frame_counts": line_sequence["partial_frame_counts"],
        "video_parallel_highlight_phase": highlight_phase,
        "video_parallel_final_hold": final_hold,
        "video_parallel_valid_highlight_labels": valid_labels,
        "video_parallel_no_early_cyan": no_early_cyan,
        "video_parallel_no_early_highlight": no_early_highlight,
        "video_parallel_no_black_vacancy": no_vacancy,
        "video_parallel_line_supports": supports,
        "video_parallel_guide_supports": guide_supports,
        "video_parallel_extension_supports": extension_supports,
        "video_parallel_cyan_pixel_counts": cyan_counts,
        "video_parallel_red_pixel_counts": red_counts,
        "video_parallel_max_static_mutation_fraction": max(
            float(item["static_mutation_fraction"]) for item in analyses
        ),
        "video_semantic_process_check_pass": passed,
        **_contract_metrics(
            artifact_kind="full_video",
            binary_correct=passed,
            component_scores=components,
            failure_reason="single-end full-line growth, highlight, preservation, or final hold failed",
            started=started,
        ),
    }, passed


TASK_NAME = "parallel"
VERIFIER = VerifierRoute(
    "data.geometry.core.verifier",
    "verify_coordinate_target",
)


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    """Evaluate only the formal final delivery for this task."""

    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [
    "PARALLEL_STATE_TRACE",
    "PARALLEL_VERIFIER_PROFILE",
    "TASK_NAME",
    "VERIFIER",
    "evaluate",
    "replay_parallel_geometry",
    "verify_parallel_full_video",
]

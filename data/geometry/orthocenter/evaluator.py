"""Independent multimodal semantic verifier for ``orthocenter``."""

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
    _color_masks,
    _direction_name,
    _finite_float,
    _line_support,
    _location_relative_to_triangle,
    _ordered_vertices,
    _point,
    _points_close,
    _read_rgb,
    _red_option,
    _sequential_single_end_lines,
    _single_end_line_observation,
    foreign_ink_clearance,
    renderer_line_width,
)


ORTHOCENTER_VERIFIER_PROFILE = "orthocenter_multimodal_semantic_hard_gate_v1"
ORTHOCENTER_STATE_TRACE = [
    "input_hold",
    "required_opposite_side_extensions_sequential_single_end_growth",
    "first_altitude_single_end_growth",
    "second_altitude_single_end_growth",
    "third_altitude_single_end_growth",
    "correct_candidate_highlight",
    "final_hold",
]
def _float_close(value: object, expected: float, tolerance: float = 1e-5) -> bool:
    parsed = _finite_float(value)
    return parsed is not None and abs(parsed - expected) <= tolerance


def _calculate_orthocenter(
    vertices: Sequence[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    (x1, y1), (x2, y2), (x3, y3) = vertices
    a1, b1 = x2 - x3, y2 - y3
    a2, b2 = x1 - x3, y1 - y3
    d1 = a1 * x1 + b1 * y1
    d2 = a2 * x2 + b2 * y2
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-9:
        return None
    return (
        (d1 * b2 - d2 * b1) / determinant,
        (a1 * d2 - a2 * d1) / determinant,
    )


def _project_to_line(
    point: Tuple[float, float],
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> Tuple[Tuple[float, float], float]:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-9:
        return first, 0.0
    t = ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / denominator
    return (first[0] + t * dx, first[1] + t * dy), t


def _triangle_type(vertices: Sequence[Tuple[float, float]]) -> str:
    dots: List[float] = []
    for index, vertex in enumerate(vertices):
        first = vertices[(index + 1) % 3]
        second = vertices[(index + 2) % 3]
        dots.append(
            (first[0] - vertex[0]) * (second[0] - vertex[0])
            + (first[1] - vertex[1]) * (second[1] - vertex[1])
        )
    scale = max(math.dist(vertices[0], vertices[1]), math.dist(vertices[1], vertices[2]), math.dist(vertices[2], vertices[0]))
    tolerance = max(1e-6, scale * scale * 1e-8)
    if any(value < -tolerance for value in dots):
        return "obtuse"
    if any(abs(value) <= tolerance for value in dots):
        return "right"
    return "acute"


def _altitude_offset(
    point: Tuple[float, float],
    vertex: Tuple[float, float],
    side_first: Tuple[float, float],
    side_second: Tuple[float, float],
) -> float:
    side_dx = side_second[0] - side_first[0]
    side_dy = side_second[1] - side_first[1]
    return abs(
        (point[0] - vertex[0]) * side_dx
        + (point[1] - vertex[1]) * side_dy
    ) / max(math.hypot(side_dx, side_dy), 1e-9)


def _altitude_data(
    vertices: Sequence[Tuple[float, float]],
    center: Tuple[float, float],
) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    for index, vertex in enumerate(vertices):
        side_first = vertices[(index + 1) % 3]
        side_second = vertices[(index + 2) % 3]
        foot, side_t = _project_to_line(vertex, side_first, side_second)
        points = (vertex, center, foot)
        segment_start, segment_end = max(
            ((first, second) for first in points for second in points if first is not second),
            key=lambda pair: math.dist(pair[0], pair[1]),
        )
        extension: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        if side_t < 0.0:
            extension = (side_first, foot)
            status = f"beyond V{(index + 1) % 3 + 1}"
        elif side_t > 1.0:
            extension = (side_second, foot)
            status = f"beyond V{(index + 2) % 3 + 1}"
        else:
            status = "on the finite opposite side"
        data.append(
            {
                "vertex": vertex,
                "side_first": side_first,
                "side_second": side_second,
                "foot": foot,
                "side_parameter": side_t,
                "foot_status": status,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "extension": extension,
            }
        )
    return data


def _segment_matches(
    value: object,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    return bool(
        (_points_close(value[0], start) and _points_close(value[1], end))
        or (_points_close(value[0], end) and _points_close(value[1], start))
    )


def replay_orthocenter_geometry(
    record: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Recompute the displayed winner from vertices and candidates only."""

    metrics: Dict[str, Any] = {
        "verifier_profile": ORTHOCENTER_VERIFIER_PROFILE,
        "orthocenter_geometry_replay_valid": False,
        "orthocenter_geometry_replay_reason": "invalid visible source geometry",
    }
    raw_vertices = record.get("triangle_vertices")
    if not isinstance(raw_vertices, list) or len(raw_vertices) != 3:
        return metrics, False, None
    parsed_vertices = [_point(value) for value in raw_vertices]
    if any(point is None for point in parsed_vertices):
        return metrics, False, None
    vertices = _ordered_vertices([point for point in parsed_vertices if point is not None])
    center = _calculate_orthocenter(vertices)
    if center is None:
        metrics["orthocenter_geometry_replay_reason"] = "triangle vertices are collinear"
        return metrics, False, None
    altitude_data = _altitude_data(vertices, center)
    if any(not (35.0 <= value <= 477.0) for item in altitude_data for point in (item["foot"], item["segment_start"], item["segment_end"]) for value in point):
        metrics["orthocenter_geometry_replay_reason"] = "derived construction leaves the supported canvas"
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
    if sorted(label for label, _point_value in candidates) != list("ABCDE"):
        return metrics, False, None
    candidates.sort(key=lambda item: item[0])

    centroid_x = sum(point[0] for point in vertices) / 3.0
    centroid_y = sum(point[1] for point in vertices) / 3.0
    derived: List[Dict[str, Any]] = []
    for label, candidate in candidates:
        offsets = [
            _altitude_offset(candidate, item["vertex"], item["side_first"], item["side_second"])
            for item in altitude_data
        ]
        derived.append(
            {
                "label": label,
                "center": [candidate[0], candidate[1]],
                "direction_from_triangle_centroid": _direction_name(
                    candidate[0] - centroid_x,
                    candidate[1] - centroid_y,
                ),
                "altitude_line_offsets_pixels": offsets,
                "maximum_altitude_offset_pixels": max(offsets),
                "mean_altitude_offset_pixels": sum(offsets) / 3.0,
            }
        )
    ordered = sorted(derived, key=lambda item: float(item["maximum_altitude_offset_pixels"]))
    derived_correct = str(ordered[0]["label"])
    best_offset = float(ordered[0]["maximum_altitude_offset_pixels"])
    runner_up = float(ordered[1]["maximum_altitude_offset_pixels"])
    derived_index = ord(derived_correct) - ord("A")
    unique = bool(
        best_offset <= 1e-5
        and runner_up >= 20.0
        and math.dist(candidates[derived_index][1], center) <= 1e-5
    )
    triangle_type = _triangle_type(vertices)
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
            and isinstance(stored_by_label[label].get("altitude_line_offsets_pixels"), list)
            and len(stored_by_label[label]["altitude_line_offsets_pixels"]) == 3
            and all(
                _float_close(
                    stored_by_label[label]["altitude_line_offsets_pixels"][offset_index],
                    float(derived[index]["altitude_line_offsets_pixels"][offset_index]),
                )
                for offset_index in range(3)
            )
            and _float_close(
                stored_by_label[label].get("maximum_altitude_offset_pixels"),
                float(derived[index]["maximum_altitude_offset_pixels"]),
            )
            and _float_close(
                stored_by_label[label].get("mean_altitude_offset_pixels"),
                float(derived[index]["mean_altitude_offset_pixels"]),
            )
            for index, label in enumerate("ABCDE")
        )

    feet_valid = bool(
        isinstance(structured.get("altitude_feet"), list)
        and len(structured["altitude_feet"]) == 3
        and all(_points_close(structured["altitude_feet"][index], altitude_data[index]["foot"]) for index in range(3))
        and isinstance(record.get("altitude_feet"), list)
        and len(record["altitude_feet"]) == 3
        and all(_points_close(record["altitude_feet"][index], altitude_data[index]["foot"]) for index in range(3))
    )
    foot_contract_valid = bool(
        isinstance(structured.get("altitude_foot_side_parameters"), list)
        and len(structured["altitude_foot_side_parameters"]) == 3
        and all(_float_close(structured["altitude_foot_side_parameters"][index], float(altitude_data[index]["side_parameter"])) for index in range(3))
        and structured.get("altitude_foot_statuses") == [item["foot_status"] for item in altitude_data]
        and isinstance(structured.get("altitude_segments"), list)
        and len(structured["altitude_segments"]) == 3
        and all(_segment_matches(structured["altitude_segments"][index], altitude_data[index]["segment_start"], altitude_data[index]["segment_end"]) for index in range(3))
        and isinstance(structured.get("required_side_extensions"), list)
        and len(structured["required_side_extensions"]) == 3
    )
    if foot_contract_valid:
        for index, item in enumerate(altitude_data):
            stored_extension = structured["required_side_extensions"][index]
            if item["extension"] is None:
                foot_contract_valid = foot_contract_valid and stored_extension is None
            else:
                foot_contract_valid = foot_contract_valid and _segment_matches(
                    stored_extension,
                    item["extension"][0],
                    item["extension"][1],
                )

    answer_schema = structured.get("answer_schema")
    verifier_contract = structured.get("verifier_contract")
    render_geometry = structured.get("render_geometry")
    extension_count = sum(item["extension"] is not None for item in altitude_data)
    expected_video_frames = 25 + 4 * extension_count
    contracts_valid = bool(
        isinstance(answer_schema, dict)
        and answer_schema.get("terminal_text_regex") == FIVE_CHOICE_CODEC.full_pattern
        and answer_schema.get("requires_sample_specific_reasoning_before_terminal_answer") is True
        and answer_schema.get("terminal_text_contains_exact_answer") is True
        and answer_schema.get("no_block_after_terminal_image") is True
        and isinstance(verifier_contract, dict)
        and verifier_contract.get("profile") == ORTHOCENTER_VERIFIER_PROFILE
        and verifier_contract.get("strict_binary_correctness") is True
        and isinstance(render_geometry, dict)
        and render_geometry.get("canvas_dimensions") == [512, 512]
        and render_geometry.get("colors", {}).get("altitude_construction") == [20, 151, 188]
        and render_geometry.get("colors", {}).get("side_extension") == [20, 151, 188]
        and _float_close(render_geometry.get("point_radius"), float(_finite_float(record.get("point_radius")) or -1.0))
        and structured.get("state_trace") == ORTHOCENTER_STATE_TRACE
        and structured.get("expected_video_frame_count") == expected_video_frames
        and structured.get("video_fps") == 16
    )
    redundant_metadata_valid = bool(
        vertices_valid
        and feet_valid
        and foot_contract_valid
        and _points_close(record.get("orthocenter"), center)
        and _points_close(structured.get("orthocenter"), center)
        and structured.get("triangle_type") == triangle_type
        and structured.get("orthocenter_location_relative_to_triangle") == location
        and str(record.get("correct_option") or "").strip().upper() == derived_correct
        and str(structured.get("correct_option") or "").strip().upper() == derived_correct
        and structured.get("correct_index") == derived_index
        and _float_close(structured.get("runner_up_maximum_altitude_offset_pixels"), runner_up)
        and structured.get("multiple_solution_policy")
        == "exactly_one_displayed_candidate_is_the_common_intersection_of_three_altitude_lines"
    )
    valid = bool(unique)
    metrics.update(
        {
            "orthocenter_geometry_replay_valid": valid,
            "orthocenter_geometry_replay_reason": (
                "ok" if valid else "source geometry is invalid or non-unique"
            ),
            "orthocenter_derived_correct_option": derived_correct,
            "orthocenter_derived_correct_index": derived_index,
            "orthocenter_derived_center": [center[0], center[1]],
            "orthocenter_triangle_type": triangle_type,
            "orthocenter_location_relative_to_triangle": location,
            "orthocenter_derived_altitude_feet": [list(item["foot"]) for item in altitude_data],
            "orthocenter_derived_altitude_segments": [
                [list(item["segment_start"]), list(item["segment_end"])]
                for item in altitude_data
            ],
            "orthocenter_derived_side_extensions": [
                None if item["extension"] is None else [list(item["extension"][0]), list(item["extension"][1])]
                for item in altitude_data
            ],
            "orthocenter_altitude_foot_statuses": [item["foot_status"] for item in altitude_data],
            "orthocenter_runner_up_maximum_offset_pixels": runner_up,
            "orthocenter_candidate_measurements": derived,
            "orthocenter_unique_displayed_solution": unique,
            "orthocenter_candidate_metadata_valid": candidate_metadata_valid,
            "orthocenter_measurement_metadata_valid": measurements_valid,
            "orthocenter_contract_metadata_valid": contracts_valid,
            "orthocenter_redundant_metadata_valid": redundant_metadata_valid,
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
        "verifier_profile": ORTHOCENTER_VERIFIER_PROFILE,
        "artifact_kind": artifact_kind,
        "component_scores": component_scores,
        "binary_correct": bool(binary_correct),
        "strict_binary_correctness": True,
        "failure_reason": "ok" if binary_correct else failure_reason,
        "verification_elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "performance_policy": "latency_throughput_memory_and_decode_counts_are_diagnostics_without_failure_thresholds",
    }


def _tube(
    shape: Tuple[int, int],
    start: Tuple[float, float],
    end: Tuple[float, float],
    thickness: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(
        mask,
        tuple(int(round(value)) for value in start),
        tuple(int(round(value)) for value in end),
        255,
        thickness=thickness,
    )
    return mask


def _expected_geometry(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    replay_metrics, valid, correct = replay_orthocenter_geometry(record)
    if not valid or correct is None:
        return None
    center = _point(replay_metrics.get("orthocenter_derived_center"))
    raw_segments = replay_metrics.get("orthocenter_derived_altitude_segments") or []
    raw_extensions = replay_metrics.get("orthocenter_derived_side_extensions") or []
    if center is None or len(raw_segments) != 3 or len(raw_extensions) != 3:
        return None
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for raw in raw_segments:
        if not isinstance(raw, list) or len(raw) != 2:
            return None
        start, end = _point(raw[0]), _point(raw[1])
        if start is None or end is None:
            return None
        segments.append((start, end))
    extensions: List[Optional[Tuple[Tuple[float, float], Tuple[float, float]]]] = []
    for raw in raw_extensions:
        if raw is None:
            extensions.append(None)
            continue
        if not isinstance(raw, list) or len(raw) != 2:
            return None
        start, end = _point(raw[0]), _point(raw[1])
        if start is None or end is None:
            return None
        extensions.append((start, end))
    return {
        "center": center,
        "segments": segments,
        "extensions": extensions,
        "correct": correct,
        "replay_metrics": replay_metrics,
    }


def _allowed_dynamic_mask(
    shape: Tuple[int, int],
    geometry: Dict[str, Any],
    point_radius: float,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for start, end in geometry["segments"]:
        mask |= _tube(shape, start, end, 20)
    for extension in geometry["extensions"]:
        if extension is not None:
            mask |= _tube(shape, extension[0], extension[1], 18)
    cv2.circle(
        mask,
        tuple(int(round(value)) for value in geometry["center"]),
        max(20, int(round(point_radius * 2.0))),
        255,
        thickness=-1,
    )
    return mask


def _frame_analysis(
    frame: np.ndarray,
    source: np.ndarray,
    record: Dict[str, Any],
    geometry: Dict[str, Any],
    *,
    extension_phase: bool = False,
) -> Dict[str, Any]:
    cyan, _dark_red = _color_masks(frame)
    red_label, red_mask = _red_option(frame, record)
    point_radius = _finite_float(record.get("point_radius")) or 11.0
    candidate_exclusions = [
        (float(item["x"]), float(item["y"]), point_radius * 1.12)
        for item in record.get("candidates") or []
        if isinstance(item, dict)
        and _finite_float(item.get("x")) is not None
        and _finite_float(item.get("y")) is not None
    ]
    altitude_supports = [
        _line_support(cyan, segment[0], segment[1], candidate_exclusions)
        for segment in geometry["segments"]
    ]
    drawn_extensions = [
        extension for extension in geometry["extensions"] if extension is not None
    ]
    clearance = foreign_ink_clearance(renderer_line_width(record))
    # An altitude runs from its own vertex through the orthocenter to its foot,
    # so it necessarily touches the other two altitudes (at the shared centre)
    # and, when the orthocenter falls outside the triangle, passes straight
    # through the vertex the side extensions radiate from.  That foreign ink is
    # legal static geometry, not this altitude's stroke front, so it must not be
    # read as progress.
    altitude_observations = [
        _single_end_line_observation(
            cyan,
            segment[0],
            segment[1],
            candidate_exclusions,
            excluded_segments=[
                other
                for other_index, other in enumerate(geometry["segments"])
                if other_index != index
            ]
            + drawn_extensions,
            segment_clearance=clearance,
            # The Prompt asks for each altitude "continuously from one segment
            # endpoint to the other", so neither end is privileged.
            accept_either_end=True,
        )
        for index, segment in enumerate(geometry["segments"])
    ]
    # The broad support tube may intersect an already drawn side extension.
    # The directional reader excludes those source-declared foreign strokes;
    # without any attributable hit they cannot activate this altitude.
    altitude_supports = [
        support if float(observation["observed_progress"]) > 0.0 else 0.0
        for support, observation in zip(altitude_supports, altitude_observations)
    ]
    extension_supports = [
        1.0 if extension is None else _line_support(cyan, extension[0], extension[1])
        for extension in geometry["extensions"]
    ]
    extension_observations = [
        None
        if extension is None
        else _single_end_line_observation(
            cyan,
            extension[0],
            extension[1],
            candidate_exclusions,
            excluded_segments=[
                other for other in drawn_extensions if other is not extension
            ]
            + ([] if extension_phase else list(geometry["segments"])),
            segment_clearance=clearance,
            # The Prompt fixes the extension's direction: from the nearer
            # original side endpoint out to the perpendicular foot.
            accept_either_end=False,
        )
        for extension in geometry["extensions"]
    ]
    # As with altitudes, a neighboring stroke in the broad support tube
    # cannot activate an extension without attributable directional ink.
    extension_supports = [
        support if observation is None or float(observation["observed_progress"]) > 0.0 else 0.0
        for support, observation in zip(extension_supports, extension_observations)
    ]
    allowed = _allowed_dynamic_mask(cyan.shape, geometry, point_radius)
    cyan_count = int(np.count_nonzero(cyan))
    cyan_spill = float(np.count_nonzero(cyan & (allowed == 0)) / max(1, cyan_count))
    diff = np.max(np.abs(frame.astype(np.int16) - source.astype(np.int16)), axis=2)
    static_mutation = float(np.mean((diff >= 24)[allowed == 0]))
    return {
        "altitude_supports": altitude_supports,
        "altitude_single_end_observations": altitude_observations,
        "minimum_altitude_support": min(altitude_supports),
        "extension_supports": extension_supports,
        "extension_single_end_observations": extension_observations,
        "minimum_required_extension_support": min(extension_supports),
        "expected_extension_count": sum(item is not None for item in geometry["extensions"]),
        "red_option": red_label,
        "red_pixel_count": int(np.count_nonzero(red_mask)),
        "cyan_pixel_count": cyan_count,
        "cyan_spill_fraction": cyan_spill,
        "static_mutation_fraction": static_mutation,
        "black_vacancy": float(np.mean(frame)) < 35.0,
    }


def _final_semantics(analysis: Dict[str, Any], correct: str) -> Tuple[bool, bool]:
    preservation_ok = bool(
        float(analysis["static_mutation_fraction"]) <= 0.005
        and float(analysis["cyan_spill_fraction"]) <= 0.16
        and not analysis["black_vacancy"]
    )
    extension_ok = bool(
        float(analysis["minimum_required_extension_support"]) >= 0.50
    )
    final_ok = bool(
        float(analysis["minimum_altitude_support"]) >= 0.68
        and extension_ok
        and analysis["red_option"] == correct
        and int(analysis["red_pixel_count"]) >= 24
    )
    return preservation_ok, final_ok


def verify_orthocenter_full_video(
    video_path: Path,
    record: Dict[str, Any],
    input_path: Path,
) -> Tuple[Dict[str, Any], bool]:
    started = time.perf_counter()
    replay_metrics, replay_valid, correct = replay_orthocenter_geometry(record)
    geometry = _expected_geometry(record) if replay_valid else None
    capture = cv2.VideoCapture(video_path.as_posix())
    opened = capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    # Derive the process length from visible source geometry.  Canonical
    # manifests deliberately scrub the generator's saved frame-count field,
    # and the verifier must not require that redundant metadata to replay the
    # three altitude constructions.
    extension_count = (
        sum(extension is not None for extension in geometry.get("extensions", []))
        if geometry is not None
        else 0
    )
    expected_frames = (25 + 4 * extension_count) if geometry is not None else 0
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
        and expected_frames in {25, 29, 33, 37}
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
            "video_process_verifier": "point_geometry_orthocenter_full_sequence_semantic",
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
    # Altitudes have not been drawn during the extension phase. Their future
    # corridors must not mask the short extensions' observable growth.
    preliminary = [_frame_analysis(frame, source, record, geometry, extension_phase=True)
                   for frame in aligned]
    altitude_onset = phase_onset([max(item["altitude_supports"]) for item in preliminary], 0.18)
    analyses = [_frame_analysis(frame, source, record, geometry,
                                extension_phase=index < altitude_onset)
                for index, frame in enumerate(aligned)]
    altitude_values = [list(map(float, item["altitude_supports"])) for item in analyses]
    altitude_directionals = [
        [
            bool(observation["single_end_directional"])
            for observation in item["altitude_single_end_observations"]
        ]
        for item in analyses
    ]
    altitude_fronts = [
        [
            float(observation["observed_progress"])
            for observation in item["altitude_single_end_observations"]
        ]
        for item in analyses
    ]
    extension_values = [list(map(float, item["extension_supports"])) for item in analyses]
    extension_directionals = [
        [
            True
            if observation is None
            else bool(observation["single_end_directional"])
            for observation in item["extension_single_end_observations"]
        ]
        for item in analyses
    ]
    extension_fronts = [
        [
            0.0
            if observation is None
            else float(observation["observed_progress"])
            for observation in item["extension_single_end_observations"]
        ]
        for item in analyses
    ]
    labels = [item["red_option"] for item in analyses]
    growth_start = phase_onset([
        max(max(altitude_values[index]), max((extension_values[index][extension_index]
            for extension_index in range(3) if geometry["extensions"][extension_index] is not None), default=0.0))
        for index in range(len(frames))
    ], 0.10)
    highlight_onset = selection_onset(labels)
    final_onset = len(frames) - 3
    initial_hold = growth_start >= 2 and all(
        max(altitude_values[index]) < 0.10
        and all(
            geometry["extensions"][extension_index] is None
            or extension_values[index][extension_index] < 0.10
            for extension_index in range(3)
        )
        and labels[index] is None
        and float(np.mean(np.abs(aligned[index].astype(np.int16) - source.astype(np.int16)))) <= 2.0
        for index in range(min(4, growth_start))
    )
    cursor = growth_start
    required_extension_indices = [
        index for index, extension in enumerate(geometry["extensions"])
        if extension is not None
    ]
    extension_sequence: Dict[str, Any] = {
        "valid": True,
        "activation_frames": [],
        "completion_frames": [],
        "draw_order": [],
        "partial_frame_counts": [],
        "maximum_simultaneous_partials": 0,
    }
    if required_extension_indices:
        extension_start, extension_end = cursor, altitude_onset
        required_extension_values = [
            [extension_values[index][extension_index] for extension_index in required_extension_indices]
            for index in range(extension_start, extension_end)
        ]
        required_extension_directionals = [
            [extension_directionals[index][extension_index] for extension_index in required_extension_indices]
            for index in range(extension_start, extension_end)
        ]
        required_extension_fronts = [
            [extension_fronts[index][extension_index] for extension_index in required_extension_indices]
            for index in range(extension_start, extension_end)
        ]
        extension_sequence = _sequential_single_end_lines(
            required_extension_values,
            required_extension_directionals,
            complete_threshold=0.50,
            regression_tolerance=0.22,
            min_partial_frames=1,
            progresses=required_extension_fronts,
        )
        extension_sequential_phase = bool(
            extension_start < extension_end
            and extension_sequence["valid"]
            and all(
                max(altitude_values[index]) <= 0.18 and labels[index] is None
                for index in range(extension_start, extension_end)
            )
        )
        cursor = extension_end
    else:
        extension_sequential_phase = True

    altitude_start, altitude_end = cursor, highlight_onset
    altitude_sequence = _sequential_single_end_lines(
        # The first observed highlight frame also carries the preceding
        # stroke's endpoint. Include that shared boundary observation while
        # the label-order check below still forbids earlier selection.
        altitude_values[altitude_start:altitude_end + 1],
        altitude_directionals[altitude_start:altitude_end + 1],
        complete_threshold=0.72,
        regression_tolerance=0.20,
        progresses=altitude_fronts[altitude_start:altitude_end + 1],
    )
    altitude_sequential_phase = bool(
        altitude_sequence["valid"]
        and all(
            all(
                geometry["extensions"][extension_index] is None
                or extension_values[index][extension_index] >= 0.48
                for extension_index in range(3)
            )
            and labels[index] is None
            for index in range(altitude_start, altitude_end)
        )
    )
    cursor = altitude_end
    highlight_start, highlight_end = cursor, final_onset
    highlight_phase = bool(
        all(
            min(altitude_values[index]) >= 0.65
            and all(
                geometry["extensions"][extension_index] is None
                or extension_values[index][extension_index] >= 0.48
                for extension_index in range(3)
            )
            and labels[index] in {None, correct}
            for index in range(highlight_start, highlight_end)
        )
        and labels[highlight_end - 1] == correct
    )
    final_start = highlight_end
    final_semantics = [_final_semantics(analyses[index], correct) for index in range(final_start, len(frames))]
    final_hold = all(item == (True, True) for item in final_semantics) and all(
        float(np.mean(np.abs(aligned[index].astype(np.int16) - aligned[-1].astype(np.int16)))) <= 2.0
        for index in range(final_start, len(frames))
    )
    valid_labels = all(label in {None, correct} for label in labels)
    no_early_highlight = all(label is None for label in labels[:highlight_start])
    no_vacancy = not any(bool(item["black_vacancy"]) for item in analyses)
    preservation_ok = bool(
        max(float(item["static_mutation_fraction"]) for item in analyses) <= 0.005
        and max(float(item["cyan_spill_fraction"]) for item in analyses) <= 0.16
    )
    temporal_ok = bool(
        initial_hold
        and extension_sequential_phase
        and altitude_sequential_phase
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
        "video_process_verifier": "point_geometry_orthocenter_full_sequence_semantic",
        "video_semantic_parser": "sequential_single_end_extensions_then_sequential_single_end_altitudes_parser",
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_decoded_frame_count": len(frames),
        "video_option": correct if final_ok else None,
        "video_orthocenter_initial_hold": initial_hold,
        "video_orthocenter_extension_sequential_phase": extension_sequential_phase,
        "video_orthocenter_extension_activation_frames": extension_sequence["activation_frames"],
        "video_orthocenter_extension_completion_frames": extension_sequence["completion_frames"],
        "video_orthocenter_extension_draw_order": extension_sequence["draw_order"],
        "video_orthocenter_extension_partial_frame_counts": extension_sequence["partial_frame_counts"],
        "video_orthocenter_altitude_sequential_phase": altitude_sequential_phase,
        "video_orthocenter_altitude_activation_frames": altitude_sequence["activation_frames"],
        "video_orthocenter_altitude_completion_frames": altitude_sequence["completion_frames"],
        "video_orthocenter_altitude_draw_order": altitude_sequence["draw_order"],
        "video_orthocenter_altitude_partial_frame_counts": altitude_sequence["partial_frame_counts"],
        "video_orthocenter_maximum_simultaneous_partial_extensions": extension_sequence[
            "maximum_simultaneous_partials"
        ],
        "video_orthocenter_maximum_simultaneous_partial_altitudes": altitude_sequence[
            "maximum_simultaneous_partials"
        ],
        "video_orthocenter_highlight_phase": highlight_phase,
        "video_orthocenter_final_hold": final_hold,
        "video_orthocenter_valid_highlight_labels": valid_labels,
        "video_orthocenter_no_early_highlight": no_early_highlight,
        "video_orthocenter_no_black_vacancy": no_vacancy,
        "video_orthocenter_altitude_supports": altitude_values,
        "video_orthocenter_extension_supports": extension_values,
        "video_orthocenter_max_static_mutation_fraction": max(
            float(item["static_mutation_fraction"]) for item in analyses
        ),
        "video_orthocenter_max_cyan_spill_fraction": max(
            float(item["cyan_spill_fraction"]) for item in analyses
        ),
        "video_semantic_process_check_pass": passed,
        **_contract_metrics(
            artifact_kind="full_video",
            binary_correct=passed,
            component_scores=components,
            failure_reason="sequential single-end extensions/altitudes, highlight, preservation, or final hold failed",
            started=started,
        ),
    }, passed


TASK_NAME = "orthocenter"
VERIFIER = VerifierRoute(
    "data.geometry.core.verifier",
    "verify_coordinate_target",
)


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    """Evaluate only the formal final delivery for this task."""

    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [
    "ORTHOCENTER_STATE_TRACE",
    "ORTHOCENTER_VERIFIER_PROFILE",
    "TASK_NAME",
    "VERIFIER",
    "evaluate",
    "replay_orthocenter_geometry",
    "verify_orthocenter_full_video",
]

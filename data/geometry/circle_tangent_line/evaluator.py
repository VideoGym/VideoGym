"""Independent source-geometry evaluator for ``circle_tangent_line``."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.common.verifiers.routes import VerifierRoute


CIRCLE_TANGENT_LINE_VERIFIER_PROFILE = "circle_tangent_line_multimodal_semantic_hard_gate_v1"
CIRCLE_TANGENT_LINE_STATE_TRACE = [
    "input_hold",
    "radius_growth",
    "tangent_growth",
    "right_angle_marker_growth",
    "correct_candidate_highlight",
    "final_hold",
]


def _finite_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _point(value: object) -> Optional[Tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x = _finite_float(value[0])
    y = _finite_float(value[1])
    if x is None or y is None:
        return None
    return x, y


def _angle_close(value: object, expected: float, tolerance: float = 1e-5) -> bool:
    parsed = _finite_float(value)
    if parsed is None:
        return False
    return abs((parsed - expected + 180.0) % 360.0 - 180.0) <= tolerance


def replay_circle_tangent_line_geometry(
    record: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Derive the tangent winner without trusting any saved answer field."""

    metrics: Dict[str, Any] = {
        "verifier_profile": CIRCLE_TANGENT_LINE_VERIFIER_PROFILE,
        "circle_tangent_line_geometry_replay_valid": False,
        "circle_tangent_line_geometry_replay_reason": "invalid structured source geometry",
    }
    structured = record.get("structured_answer")
    if not isinstance(structured, dict):
        return metrics, False, None
    center = _point(record.get("circle_center"))
    contact = _point(record.get("contact_point"))
    radius = _finite_float(record.get("circle_radius"))
    if center is None or contact is None or radius is None or not 90.0 <= radius <= 190.0:
        return metrics, False, None
    vx = contact[0] - center[0]
    vy = contact[1] - center[1]
    measured_radius = math.hypot(vx, vy)
    if measured_radius <= 1e-9 or abs(measured_radius - radius) > 1e-5:
        metrics["circle_tangent_line_geometry_replay_reason"] = "contact is not on the declared circle"
        return metrics, False, None
    ux, uy = vx / measured_radius, vy / measured_radius

    top_candidates = record.get("candidates")
    if not isinstance(top_candidates, list) or len(top_candidates) != 5:
        return metrics, False, None
    candidates: List[Tuple[str, Tuple[float, float]]] = []
    for raw in top_candidates:
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

    # The tangent has two orientations. Select the one pointing toward the
    # candidate cluster from visible source coordinates, not from an oracle.
    tx, ty = -uy, ux
    projection_sum = sum(
        (candidate[0] - contact[0]) * tx + (candidate[1] - contact[1]) * ty
        for _label, candidate in candidates
    )
    if projection_sum < 0.0:
        tx, ty = -tx, -ty
    radius_bearing = math.degrees(math.atan2(uy, ux) % math.tau)
    tangent_bearing = math.degrees(math.atan2(ty, tx) % math.tau)

    derived: List[Dict[str, Any]] = []
    for label, candidate in candidates:
        dx = candidate[0] - contact[0]
        dy = candidate[1] - contact[1]
        distance = math.hypot(dx, dy)
        forward = dx * tx + dy * ty
        signed_offset = dx * ux + dy * uy
        radius_candidate_angle = math.degrees(
            math.acos(max(-1.0, min(1.0, signed_offset / max(distance, 1e-9))))
        )
        derived.append(
            {
                "label": label,
                "center": [candidate[0], candidate[1]],
                "forward_distance_pixels": forward,
                "signed_perpendicular_offset_pixels": signed_offset,
                "perpendicular_offset_pixels": abs(signed_offset),
                "radius_candidate_angle_degrees": radius_candidate_angle,
                "right_angle_error_degrees": abs(radius_candidate_angle - 90.0),
                "tangent_ray_residual_pixels": math.hypot(abs(signed_offset), max(0.0, -forward)),
            }
        )
    ordered = sorted(derived, key=lambda item: float(item["tangent_ray_residual_pixels"]))
    derived_correct = str(ordered[0]["label"])
    runner_up = float(ordered[1]["tangent_ray_residual_pixels"])
    unique = bool(
        float(ordered[0]["tangent_ray_residual_pixels"]) <= 1e-5
        and runner_up >= 18.0
        and min(float(item["forward_distance_pixels"]) for item in derived) > 12.0
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
            and abs(
                float(stored_by_label[label].get("forward_distance_pixels", 1e9))
                - float(derived[index]["forward_distance_pixels"])
            ) <= 1e-5
            and abs(
                float(stored_by_label[label].get("signed_perpendicular_offset_pixels", 1e9))
                - float(derived[index]["signed_perpendicular_offset_pixels"])
            ) <= 1e-5
            and abs(
                float(stored_by_label[label].get("perpendicular_offset_pixels", 1e9))
                - float(derived[index]["perpendicular_offset_pixels"])
            ) <= 1e-5
            and abs(
                float(stored_by_label[label].get("right_angle_error_degrees", 1e9))
                - float(derived[index]["right_angle_error_degrees"])
            ) <= 1e-5
            and abs(
                float(stored_by_label[label].get("tangent_ray_residual_pixels", 1e9))
                - float(derived[index]["tangent_ray_residual_pixels"])
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
        and verifier_contract.get("profile") == CIRCLE_TANGENT_LINE_VERIFIER_PROFILE
        and verifier_contract.get("strict_binary_correctness") is True
        and isinstance(render_geometry, dict)
        and render_geometry.get("canvas_dimensions") == [512, 512]
        and abs(float(render_geometry.get("right_angle_marker_size", -1.0)) - 24.0) <= 1e-5
        and structured.get("state_trace") == CIRCLE_TANGENT_LINE_STATE_TRACE
        and structured.get("expected_video_frame_count") == 29
        and structured.get("video_fps") == 16
    )
    derived_index = next(index for index, item in enumerate(derived) if item["label"] == derived_correct)
    redundant_metadata_valid = bool(
        _point(record.get("circle_center")) == center
        and _point(record.get("contact_point")) == contact
        and abs(float(record.get("circle_radius", -1.0)) - radius) <= 1e-5
        and str(record.get("correct_option") or "").strip().upper() == derived_correct
        and str(structured.get("correct_option") or "").strip().upper() == derived_correct
        and structured.get("correct_index") == derived_index
        and _angle_close(structured.get("radius_bearing_degrees"), radius_bearing)
        and _angle_close(structured.get("tangent_ray_bearing_degrees"), tangent_bearing)
        and abs(float(structured.get("runner_up_offset_pixels", -1.0)) - runner_up) <= 1e-5
        and structured.get("multiple_solution_policy")
        == "generator_guarantees_one_forward_candidate_center_on_the_tangent_ray"
    )
    valid = bool(unique)
    metrics.update(
        {
            "circle_tangent_line_geometry_replay_valid": valid,
            "circle_tangent_line_geometry_replay_reason": (
                "ok" if valid else "source geometry is invalid or non-unique"
            ),
            "circle_tangent_line_derived_correct_option": derived_correct,
            "circle_tangent_line_derived_correct_index": derived_index,
            "circle_tangent_line_derived_radius_bearing_degrees": radius_bearing,
            "circle_tangent_line_derived_tangent_bearing_degrees": tangent_bearing,
            "circle_tangent_line_runner_up_offset_pixels": runner_up,
            "circle_tangent_line_candidate_measurements": derived,
            "circle_tangent_line_unique_solution": unique,
            "circle_tangent_line_candidate_metadata_valid": candidate_metadata_valid,
            "circle_tangent_line_measurement_metadata_valid": measurements_valid,
            "circle_tangent_line_contract_metadata_valid": contracts_valid,
            "circle_tangent_line_redundant_metadata_valid": redundant_metadata_valid,
        }
    )
    return metrics, valid, derived_correct


TASK_NAME = "circle_tangent_line"
VERIFIER = VerifierRoute(
    "data.geometry.core.verifier",
    "verify_coordinate_target",
)


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    """Evaluate only the formal final delivery for this task."""

    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [
    "CIRCLE_TANGENT_LINE_STATE_TRACE",
    "CIRCLE_TANGENT_LINE_VERIFIER_PROFILE",
    "TASK_NAME",
    "VERIFIER",
    "evaluate",
    "replay_circle_tangent_line_geometry",
]

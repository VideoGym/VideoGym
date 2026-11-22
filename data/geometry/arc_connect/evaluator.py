"""Independent source-geometry evaluator for ``arc_connect``."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.common.verifiers.routes import VerifierRoute


ARC_CONNECT_VERIFIER_PROFILE = "arc_connect_multimodal_semantic_hard_gate_v1"
ARC_CONNECT_STATE_TRACE = [
    "input_hold",
    "mask_narrowing_and_arc_reveal",
    "revealed_geometry_hold",
    "correct_candidate_highlight",
    "final_hold",
]


def _finite_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _circle(value: object) -> Optional[Tuple[float, float, float]]:
    if not isinstance(value, dict):
        return None
    cx = _finite_float(value.get("cx"))
    cy = _finite_float(value.get("cy"))
    radius = _finite_float(value.get("radius"))
    if cx is None or cy is None or radius is None or radius <= 1.0:
        return None
    return cx, cy, radius


def replay_arc_connect_geometry(
    record: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Derive the unique answer from the left and candidate circle sources.

    Saved ``correct_option`` and measurement/oracle fields are audited only
    after the answer has been recomputed from the circle specifications.
    """

    metrics: Dict[str, Any] = {
        "verifier_profile": ARC_CONNECT_VERIFIER_PROFILE,
        "arc_connect_geometry_replay_valid": False,
        "arc_connect_geometry_replay_reason": "invalid structured source geometry",
    }
    structured = record.get("structured_answer")
    if not isinstance(structured, dict):
        return metrics, False, None
    left = _circle(structured.get("left_circle"))
    raw_candidates = structured.get("candidate_circles")
    if left is None or not isinstance(raw_candidates, list) or len(raw_candidates) != 5:
        return metrics, False, None

    candidates: List[Tuple[str, Tuple[float, float, float]]] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            return metrics, False, None
        label = str(item.get("label") or "").strip().upper()
        circle = _circle(item)
        if label not in "ABCDE" or circle is None:
            return metrics, False, None
        candidates.append((label, circle))
    if sorted(label for label, _circle_value in candidates) != list("ABCDE"):
        return metrics, False, None
    candidates.sort(key=lambda item: item[0])

    derived_measurements: List[Dict[str, Any]] = []
    for label, circle in candidates:
        center_offset = math.hypot(circle[0] - left[0], circle[1] - left[1])
        radius_offset = abs(circle[2] - left[2])
        residual = math.hypot(center_offset, radius_offset)
        derived_measurements.append(
            {
                "label": label,
                "circle_center": [circle[0], circle[1]],
                "radius": circle[2],
                "center_offset_pixels": center_offset,
                "radius_offset_pixels": radius_offset,
                "same_circle_residual_pixels": residual,
            }
        )
    ordered = sorted(derived_measurements, key=lambda item: float(item["same_circle_residual_pixels"]))
    derived_correct = str(ordered[0]["label"])
    unique = (
        float(ordered[0]["same_circle_residual_pixels"]) <= 1e-6
        and float(ordered[1]["same_circle_residual_pixels"]) >= 20.0
    )

    top_candidates = record.get("candidates")
    top_valid = isinstance(top_candidates, list) and len(top_candidates) == 5
    top_by_label = {
        str(item.get("label") or "").strip().upper(): item
        for item in top_candidates or []
        if isinstance(item, dict)
    }
    top_valid = bool(top_valid and sorted(top_by_label) == list("ABCDE"))

    stored_measurements = structured.get("candidate_measurements")
    stored_valid = isinstance(stored_measurements, list) and len(stored_measurements) == 5
    stored_by_label = {
        str(item.get("label") or "").strip().upper(): item
        for item in stored_measurements or []
        if isinstance(item, dict)
    }
    if stored_valid:
        stored_valid = sorted(stored_by_label) == list("ABCDE")
    if stored_valid:
        for index, label in enumerate("ABCDE"):
            item = stored_by_label[label]
            expected = derived_measurements[index]
            marker = item.get("marker_center")
            top = top_by_label.get(label, {})
            marker_valid = (
                isinstance(marker, list)
                and len(marker) == 2
                and _finite_float(marker[0]) is not None
                and _finite_float(marker[1]) is not None
                and abs(float(marker[0]) - float(top.get("x", 1e9))) <= 1e-5
                and abs(float(marker[1]) - float(top.get("y", 1e9))) <= 1e-5
            )
            numeric_valid = all(
                abs(float(item.get(key, 1e9)) - float(expected[key])) <= 1e-5
                for key in (
                    "radius",
                    "center_offset_pixels",
                    "radius_offset_pixels",
                    "same_circle_residual_pixels",
                )
            )
            center = item.get("circle_center")
            center_valid = (
                isinstance(center, list)
                and len(center) == 2
                and abs(float(center[0]) - float(expected["circle_center"][0])) <= 1e-5
                and abs(float(center[1]) - float(expected["circle_center"][1])) <= 1e-5
            )
            if not (marker_valid and numeric_valid and center_valid):
                stored_valid = False
                break

    mask = structured.get("mask")
    mask_valid = isinstance(mask, dict) and all(_finite_float(mask.get(key)) is not None for key in (
        "center_x", "width", "left_x", "right_x", "top_y", "bottom_y"
    ))
    if mask_valid:
        center_x = float(mask["center_x"])
        width = float(mask["width"])
        mask_valid = bool(
            width > 80.0
            and abs(float(mask["left_x"]) - (center_x - width / 2.0)) <= 1e-5
            and abs(float(mask["right_x"]) - (center_x + width / 2.0)) <= 1e-5
            and abs(float(mask["top_y"])) <= 1e-5
            and abs(float(mask["bottom_y"]) - 512.0) <= 1e-5
        )

    answer_schema = structured.get("answer_schema")
    verifier_contract = structured.get("verifier_contract")
    render_geometry = structured.get("render_geometry")
    contracts_valid = (
        isinstance(answer_schema, dict)
        and answer_schema.get("terminal_text_regex") == FIVE_CHOICE_CODEC.full_pattern
        and answer_schema.get("requires_sample_specific_reasoning_before_terminal_answer") is True
        and answer_schema.get("terminal_text_contains_exact_answer") is True
        and answer_schema.get("no_block_after_terminal_image") is True
        and isinstance(verifier_contract, dict)
        and verifier_contract.get("profile") == ARC_CONNECT_VERIFIER_PROFILE
        and verifier_contract.get("strict_binary_correctness") is True
        and isinstance(render_geometry, dict)
        and render_geometry.get("canvas_dimensions") == [512, 512]
        and structured.get("state_trace") == ARC_CONNECT_STATE_TRACE
        and structured.get("expected_video_frame_count") == 29
        and structured.get("video_fps") == 16
    )
    redundant_metadata_valid = (
        str(record.get("correct_option") or "").strip().upper() == derived_correct
        and str(structured.get("correct_option") or "").strip().upper() == derived_correct
        and str(structured.get("derived_correct_option") or "").strip().upper() == derived_correct
    )
    valid = bool(unique and top_valid and mask_valid)
    metrics.update(
        {
            "arc_connect_geometry_replay_valid": valid,
            "arc_connect_geometry_replay_reason": "ok" if valid else "source geometry is invalid or non-unique",
            "arc_connect_derived_correct_option": derived_correct,
            "arc_connect_unique_same_circle_match": unique,
            "arc_connect_runner_up_residual_pixels": float(ordered[1]["same_circle_residual_pixels"]),
            "arc_connect_candidate_measurements": derived_measurements,
            "arc_connect_top_candidate_metadata_valid": top_valid,
            "arc_connect_measurement_metadata_valid": stored_valid,
            "arc_connect_mask_metadata_valid": mask_valid,
            "arc_connect_contract_metadata_valid": contracts_valid,
            "arc_connect_redundant_answer_metadata_valid": redundant_metadata_valid,
        }
    )
    return metrics, valid, derived_correct


TASK_NAME = "arc_connect"
VERIFIER = VerifierRoute(
    "data.geometry.core.verifier",
    "verify_coordinate_target",
)


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    """Evaluate only the formal final delivery for this task."""

    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [
    "ARC_CONNECT_STATE_TRACE",
    "ARC_CONNECT_VERIFIER_PROFILE",
    "TASK_NAME",
    "VERIFIER",
    "evaluate",
    "replay_arc_connect_geometry",
]

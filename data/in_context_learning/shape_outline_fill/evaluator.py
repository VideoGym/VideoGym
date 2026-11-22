"""Task-owned source replay and visual observations."""

from __future__ import annotations

import math
import numpy as np
from data.common.verifiers.keyframes import visual_delta, visual_features
from data.common.verifiers.media import _align_bgr, _read_bgr, _video_fps, _video_frames_bgr
from data.common.verifiers.metrics import _verifier_contract_metrics
from data.common.verifiers.observations import (
    _analyze_fixed_visual_state_sequence,
    _pixels_near_rgb,
    static_mutation_fraction as measure_static_mutation,
)
from pathlib import Path
from time import perf_counter
from typing import Any, List, Optional, Sequence


_SHAPE_OUTLINE_MAX_TRANSITION_RESIDUAL = 0.0394


_SHAPE_OUTLINE_VERIFIER_PROFILE = "shape_identity_and_fill_style_visual_replay_v3"


_SHAPE_OUTLINE_STYLES = ("filled", "outline")


_SHAPE_OUTLINE_STAGES = ("input", "shape_preview", "final")


def _shape_outline_render_language() -> dict[str, Any]:
    """Import the renderer's own vocabulary and layout, never a duplicate.

    Slot centres and the shape radius are derived from the generator's layout
    fractions on its fixed 512-square canvas, so a renderer layout change can
    never silently diverge from the verifier's expected geometry.
    """

    from data.in_context_learning.shape_outline_fill.generator import (
        _COLORS,
        _PREVIEW_COLOR,
        _RADIUS_FRACTION,
        _SHAPES,
        _SLOT_FRACTIONS,
    )

    return {
        "shapes": tuple(_SHAPES),
        "colors": tuple(tuple(color) for color in _COLORS),
        "preview_rgb": tuple(_PREVIEW_COLOR),
        "slot_centers": {
            label: (512 * fx, 512 * fy) for label, (fx, fy) in _SLOT_FRACTIONS.items()
        },
        "shape_radius": 512 * _RADIUS_FRACTION,
    }


def _shape_outline_fill_replay(structured: dict[str, Any]) -> Optional[dict[str, Any]]:
    language = _shape_outline_render_language()
    shape_a = structured.get("shape_a")
    shape_c = structured.get("shape_c")
    style_from = structured.get("style_from")
    style_to = structured.get("style_to")
    source_color = structured.get("source_color")
    geometry = structured.get("render_geometry")
    if (
        shape_a not in language["shapes"]
        or shape_c not in language["shapes"]
        or shape_a == shape_c
        or style_from not in _SHAPE_OUTLINE_STYLES
        or style_to not in _SHAPE_OUTLINE_STYLES
        or style_from == style_to
        or {style_from, style_to} != set(_SHAPE_OUTLINE_STYLES)
        or not isinstance(source_color, list)
        or tuple(source_color) not in language["colors"]
        or not isinstance(geometry, dict)
    ):
        return None
    raw_centers = geometry.get("slot_centers")
    if not isinstance(raw_centers, dict) or set(raw_centers) != set(language["slot_centers"]):
        return None
    for label, expected in language["slot_centers"].items():
        raw = raw_centers.get(label)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return None
        try:
            observed = (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError):
            return None
        if any(abs(left - right) > 0.25 for left, right in zip(observed, expected)):
            return None
    try:
        radius = float(geometry.get("shape_radius"))
    except (TypeError, ValueError):
        return None
    if (
        abs(radius - language["shape_radius"]) > 0.25
        or tuple(geometry.get("preview_rgb") or ()) != language["preview_rgb"]
    ):
        return None
    answer = f"{shape_c}:{style_to}"
    return {
        "shape_a": str(shape_a),
        "shape_c": str(shape_c),
        "style_from": str(style_from),
        "style_to": str(style_to),
        "source_color": tuple(int(value) for value in source_color),
        "answer": answer,
        "slot_centers": dict(language["slot_centers"]),
        "shape_radius": language["shape_radius"],
        "preview_rgb": language["preview_rgb"],
        "allowed_change_bboxes": [(315.0, 315.0, 423.0, 423.0)],
    }


def _validate_shape_outline_fill(structured: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    replay = _shape_outline_fill_replay(structured)
    passed = replay is not None
    return {
        "video_semantic_process_check_pass": passed,
        "video_semantic_process_reason": "ok" if passed else "shape-outline visible analogy or duplicate contract metadata mismatch",
        "video_shape_outline_expected_answer": replay["answer"] if replay else None,
    }, passed


def _shape_outline_ideal_mask(
    frame_shape: Sequence[int],
    *,
    center: Sequence[float],
    radius: float,
    shape: str,
    style: str,
) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for shape-outline verification") from exc
    height, width = int(frame_shape[0]), int(frame_shape[1])
    scale_x = width / 512.0
    scale_y = height / 512.0
    scale = min(scale_x, scale_y)
    cx = float(center[0]) * scale_x
    cy = float(center[1]) * scale_y
    rr = float(radius) * scale
    thickness = max(3, int(round(4.0 * scale)))
    mask = np.zeros((height, width), dtype=np.uint8)
    fill_thickness = -1 if style == "filled" else thickness
    if shape == "circle":
        cv2.circle(
            mask,
            (int(round(cx)), int(round(cy))),
            int(round(rr)),
            255,
            thickness=fill_thickness,
            lineType=cv2.LINE_AA,
        )
    elif shape == "square":
        cv2.rectangle(
            mask,
            (int(round(cx - rr)), int(round(cy - rr))),
            (int(round(cx + rr)), int(round(cy + rr))),
            255,
            thickness=fill_thickness,
            lineType=cv2.LINE_AA,
        )
    else:
        if shape == "triangle":
            raw_points = ((cx, cy - rr), (cx - rr, cy + rr), (cx + rr, cy + rr))
        elif shape == "diamond":
            raw_points = ((cx, cy - rr), (cx - rr, cy), (cx, cy + rr), (cx + rr, cy))
        elif shape == "hexagon":
            raw_points = tuple(
                (
                    cx + rr * math.cos(math.pi / 3.0 * index - math.pi / 6.0),
                    cy + rr * math.sin(math.pi / 3.0 * index - math.pi / 6.0),
                )
                for index in range(6)
            )
        else:
            return mask.astype(bool)
        points = np.asarray(
            [[int(round(x)), int(round(y))] for x, y in raw_points],
            dtype=np.int32,
        )
        if style == "filled":
            cv2.fillPoly(mask, [points], 255, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(mask, [points], True, 255, thickness=thickness, lineType=cv2.LINE_AA)
    return mask >= 96


def _shape_outline_classify_colored_shape(
    frame_bgr: np.ndarray,
    *,
    center: Sequence[float],
    radius: float,
    rgb: Sequence[int],
) -> Optional[dict[str, Any]]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for shape-outline verification") from exc
    height, width = frame_bgr.shape[:2]
    scale_x = width / 512.0
    scale_y = height / 512.0
    cx = float(center[0]) * scale_x
    cy = float(center[1]) * scale_y
    rr = float(radius) * min(scale_x, scale_y)
    yy, xx = np.ogrid[:height, :width]
    slot_region = (np.abs(xx - cx) <= rr * 1.35) & (np.abs(yy - cy) <= rr * 1.35)
    core_observed = _pixels_near_rgb(frame_bgr, rgb, tolerance=24.0) & slot_region
    if int(core_observed.sum()) < max(16, int(round(24 * scale_x * scale_y))):
        return None
    observed = _pixels_near_rgb(frame_bgr, rgb, tolerance=48.0) & slot_region
    observed_count = int(observed.sum())
    if observed_count < max(20, int(round(40 * scale_x * scale_y))):
        return None
    kernel_size = max(3, int(round(5.0 * min(scale_x, scale_y))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    observed_dilated = cv2.dilate(observed.astype(np.uint8), kernel, iterations=1).astype(bool)
    candidates: List[tuple[float, str, str, float, float]] = []
    for shape in _shape_outline_render_language()["shapes"]:
        for style in _SHAPE_OUTLINE_STYLES:
            ideal = _shape_outline_ideal_mask(
                frame_bgr.shape,
                center=center,
                radius=radius,
                shape=shape,
                style=style,
            )
            ideal_dilated = cv2.dilate(ideal.astype(np.uint8), kernel, iterations=1).astype(bool)
            precision = float((observed & ideal_dilated).sum()) / float(max(1, observed_count))
            coverage = float((ideal & observed_dilated).sum()) / float(max(1, int(ideal.sum())))
            score = min(precision, coverage)
            candidates.append((score, shape, style, precision, coverage))
    candidates.sort(reverse=True)
    best = candidates[0]
    margin = best[0] - candidates[1][0]
    if best[0] < 0.62 or margin < 0.035:
        return None
    return {
        "shape": best[1],
        "style": best[2],
        "score": float(best[0]),
        "margin": float(margin),
        "precision": float(best[3]),
        "coverage": float(best[4]),
    }


def _shape_outline_source_frame_valid(
    frame_bgr: np.ndarray,
    replay: dict[str, Any],
) -> bool:
    expected_source = {
        "A": (replay["shape_a"], replay["style_from"]),
        "B": (replay["shape_a"], replay["style_to"]),
        "C": (replay["shape_c"], replay["style_from"]),
    }
    for label, expected in expected_source.items():
        observed = _shape_outline_classify_colored_shape(
            frame_bgr,
            center=replay["slot_centers"][label],
            radius=replay["shape_radius"],
            rgb=replay["source_color"],
        )
        if observed is None or (observed["shape"], observed["style"]) != expected:
            return False
    return True


def _parse_shape_outline_frame(
    frame_bgr: np.ndarray,
    replay: dict[str, Any],
    reference_bgr: np.ndarray,
) -> Optional[str]:
    preview = _shape_outline_classify_colored_shape(
        frame_bgr,
        center=replay["slot_centers"]["D"],
        radius=replay["shape_radius"],
        rgb=replay["preview_rgb"],
    )
    final = _shape_outline_classify_colored_shape(
        frame_bgr,
        center=replay["slot_centers"]["D"],
        radius=replay["shape_radius"],
        rgb=replay["source_color"],
    )
    height, width = frame_bgr.shape[:2]
    left, top, right, bottom = replay["allowed_change_bboxes"][0]
    x0 = max(0, int(math.floor(left * width / 512.0)))
    x1 = min(width, int(math.ceil(right * width / 512.0)))
    y0 = max(0, int(math.floor(top * height / 512.0)))
    y1 = min(height, int(math.ceil(bottom * height / 512.0)))
    current_patch = frame_bgr[y0:y1, x0:x1]
    reference_patch = reference_bgr[y0:y1, x0:x1]
    if current_patch.shape != reference_patch.shape or current_patch.size == 0:
        return None
    delta = visual_delta(current_patch, reference_patch).mean(axis=2)
    input_match = float((delta >= 18).mean()) <= 0.003
    if input_match and preview is None and final is None:
        return "input"
    if (
        preview is not None
        and (preview["shape"], preview["style"]) == (replay["shape_c"], "outline")
        and final is None
    ):
        return "shape_preview"
    if (
        final is not None
        and (final["shape"], final["style"]) == (replay["shape_c"], replay["style_to"])
        and preview is None
    ):
        return "final"
    return None


def _shape_outline_raster_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Symmetric local ink distance for already-filtered answer-slot pixels.

    One pixel of contour raster variation does not change a shape. Checking
    both directions still charges for missing strokes or added filled areas,
    and larger geometric changes remain visible in the residual.
    """
    def directed(observed, expected):
        height, width = observed.shape[:2]
        padded = np.pad(expected, ((1, 1), (1, 1), (0, 0)), mode="edge")
        errors = []
        for dy in range(3):
            for dx in range(3):
                shifted = padded[dy:dy + height, dx:dx + width]
                delta = np.maximum(np.abs(observed - shifted) - 4.0, 0.0)
                errors.append(np.sum(delta ** 2, axis=2))
        return math.sqrt(float(np.min(errors, axis=0).sum()))

    return max(directed(first, second), directed(second, first))


def _verify_shape_outline_candidate_state_video(
    video_path: Path,
    structured: dict[str, Any],
    puzzle_image: Path,
) -> tuple[dict[str, Any], bool]:
    started_at = perf_counter()
    semantic_metrics, semantic_pass = _validate_shape_outline_fill(structured)
    replay = _shape_outline_fill_replay(structured)
    if not semantic_pass or replay is None:
        scores = {
            "source_replay_score": 0.0,
            "shape_identity_score": 0.0,
            "style_direction_score": 0.0,
            "visual_parseability_score": 0.0,
            "state_order_score": 0.0,
            "transition_visibility_score": 0.0,
            "preservation_score": 0.0,
            "final_state_score": 0.0,
            "media_contract_score": 0.0,
        }
        metrics = _verifier_contract_metrics(
            verifier_profile=_SHAPE_OUTLINE_VERIFIER_PROFILE,
            artifact_kind="full_video",
            component_scores=scores,
            binary_correct=False,
            failure_reason="visible analogy source or duplicate contract metadata cannot be independently replayed",
            started_at=started_at,
            hard_gate_failures=["source_replay"],
        )
        return {
            **semantic_metrics,
            **metrics,
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": metrics["failure_reason"],
        }, False
    try:
        frames = _video_frames_bgr(video_path)
    except (OSError, RuntimeError):
        frames = []
    fps = _video_fps(video_path) if frames else 0.0
    frame_count = len(frames)
    media_contract_ok = bool(
        frames
        and abs(fps - 16.0) <= 0.05
        and frame_count % 4 == 1
        and frame_count / max(fps, 1e-9) <= 10.0
    )
    if not frames:
        scores = {
            "source_replay_score": 1.0,
            "shape_identity_score": 1.0,
            "style_direction_score": 1.0,
            "visual_parseability_score": 0.0,
            "state_order_score": 0.0,
            "transition_visibility_score": 0.0,
            "preservation_score": 0.0,
            "final_state_score": 0.0,
            "media_contract_score": 0.0,
        }
        metrics = _verifier_contract_metrics(
            verifier_profile=_SHAPE_OUTLINE_VERIFIER_PROFILE,
            artifact_kind="full_video",
            component_scores=scores,
            binary_correct=False,
            failure_reason="shape-outline candidate video could not be decoded",
            started_at=started_at,
            hard_gate_failures=["video_decode"],
        )
        return {
            **semantic_metrics,
            **metrics,
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": metrics["failure_reason"],
            "video_candidate_frame_count": 0,
        }, False
    reference_bgr = _align_bgr(_read_bgr(puzzle_image), frames[0].shape)
    source_visual_ok = _shape_outline_source_frame_valid(reference_bgr, replay)
    recognized: List[tuple[int, str]] = []
    static_mutation_fractions: List[float] = []
    for frame_index, frame in enumerate(frames):
        stage = _parse_shape_outline_frame(frame, replay, reference_bgr) if source_visual_ok else None
        if stage is not None:
            recognized.append((frame_index, stage))
        static_mutation_fractions.append(
            measure_static_mutation(
                reference_bgr,
                frame,
                codec_tolerant=True,
                allowed_change_bboxes=replay["allowed_change_bboxes"],
            )
        )
    sequence = _analyze_fixed_visual_state_sequence(
        frame_count=frame_count,
        recognized=recognized,
        expected_states=_SHAPE_OUTLINE_STAGES,
    )
    motion_gap_counts = [
        current["frame_indices"][0] - previous["frame_indices"][-1] - 1
        for previous, current in zip(sequence["compact_groups"], sequence["compact_groups"][1:])
    ]
    # A semantic gap alone is not evidence of a transition: a hard cut can
    # manufacture the same sequence with one changed frame.  Require two
    # distinct multi-frame runs of local motion in D's slot instead.
    left, top, right, bottom = replay["allowed_change_bboxes"][0]
    height, width = frames[0].shape[:2]
    x0 = max(0, int(math.floor(left * width / 512.0)))
    x1 = min(width, int(math.ceil(right * width / 512.0)))
    y0 = max(0, int(math.floor(top * height / 512.0)))
    y1 = min(height, int(math.ceil(bottom * height / 512.0)))
    dynamic_steps: List[int] = []
    for frame_index, (previous, current) in enumerate(zip(frames, frames[1:])):
        previous_patch = previous[y0:y1, x0:x1]
        current_patch = current[y0:y1, x0:x1]
        if previous_patch.size == 0 or previous_patch.shape != current_patch.shape:
            continue
        delta = np.max(
            np.abs(current_patch.astype(np.int16) - previous_patch.astype(np.int16)),
            axis=2,
        )
        if float((delta > 8).mean()) >= 0.02:
            dynamic_steps.append(frame_index)
    transition_runs: List[List[int]] = []
    for frame_index in dynamic_steps:
        if not transition_runs or frame_index != transition_runs[-1][-1] + 1:
            transition_runs.append([frame_index])
        else:
            transition_runs[-1].append(frame_index)
    has_visible_transitions = bool(
        len(transition_runs) == len(_SHAPE_OUTLINE_STAGES) - 1
        and all(len(run) >= 2 for run in transition_runs)
    )
    # Motion between two readable states is not the same as a transition *into*
    # the next state.  Each unreadable stretch must be a monotone element-local
    # approach from the state that precedes it to the state that follows, so
    # project the answer slot onto the line joining those two states and require
    # the intermediate frames to lie on it.  Frames borrowed from the other
    # transition, or replayed out of order, leave that line even when they keep
    # the readable stages, their count and their motion runs unchanged.
    transition_projections: List[dict[str, Any]] = []
    for previous_group, current_group in zip(sequence["compact_groups"], sequence["compact_groups"][1:]):
        opening = int(previous_group["frame_indices"][-1])
        closing = int(current_group["frame_indices"][0])
        if closing <= opening + 1:
            continue
        before = visual_features(frames[opening][y0:y1, x0:x1])
        after = visual_features(frames[closing][y0:y1, x0:x1])
        direction = (after - before).reshape(-1)
        energy = float(np.dot(direction, direction))
        if energy <= 1e-6:
            transition_projections.append(
                {"gap": [opening, closing], "max_off_line_residual": 1.0, "projections": []}
            )
            continue
        scale = math.sqrt(energy)
        projections: List[float] = []
        residuals: List[float] = []
        for frame_index in range(opening, closing + 1):
            observed = visual_features(frames[frame_index][y0:y1, x0:x1])
            offset = (observed - before).reshape(-1)
            value = float(np.dot(offset, direction) / energy)
            projections.append(value)
            expected = before + value * (after - before)
            residuals.append(_shape_outline_raster_distance(observed, expected) / scale)
        transition_projections.append(
            {
                "gap": [opening, closing],
                "projections": [round(value, 4) for value in projections],
                "max_off_line_residual": round(max(residuals), 4),
                "projection_low": round(min(projections), 4),
                "projection_high": round(max(projections), 4),
                "monotonic": all(right + .06 >= left for left, right in zip(projections, projections[1:])),
            }
        )
    max_off_line_residual = max(
        (float(item["max_off_line_residual"]) for item in transition_projections), default=0.0
    )
    transitions_on_line = bool(
        max_off_line_residual <= _SHAPE_OUTLINE_MAX_TRANSITION_RESIDUAL
        and all(
            -0.08 <= float(item.get("projection_low", 0.0))
            and float(item.get("projection_high", 1.0)) <= 1.08
            and bool(item.get("monotonic", False))
            for item in transition_projections
        )
    )
    max_static_mutation_fraction = max(static_mutation_fractions, default=1.0)
    preservation_ok = max_static_mutation_fraction <= 0.003
    # A frame caught mid-fade is neither the preview nor the final style, so
    # every legal element-local transition contributes unreadable frames.  Their
    # number is set by the two transitions this task declares, not by how long
    # the video is, and a frame-global fraction therefore rejects a legally
    # gradual reference purely because the holds around it are short.  The
    # readable requirement is confinement: an unreadable frame is legal only
    # while D's slot is actually moving, and the three declared states must
    # still occupy more of the video than the transitions between them.
    transition_frames = {index for step in dynamic_steps for index in (step, step + 1)}
    recognized_frames = {frame_index for frame_index, _state in recognized}
    unrecognized_outside_transitions = sorted(
        frame_index
        for frame_index in range(frame_count)
        if frame_index not in recognized_frames and frame_index not in transition_frames
    )
    readable_states_dominate = bool(sequence["unrecognized_fraction"] < 0.5)
    parseability_ok = bool(not unrecognized_outside_transitions and readable_states_dominate)
    passed = bool(
        semantic_pass
        and source_visual_ok
        and parseability_ok
        and preservation_ok
        and sequence["starts_at_input"]
        and sequence["ends_at_final"]
        and not sequence["skipped_layers"]
        and sequence["exact_sequence"]
        and sequence["has_initial_hold"]
        and sequence["has_final_hold"]
        and has_visible_transitions
        and transitions_on_line
        and media_contract_ok
    )
    if not source_visual_ok:
        reason, failures = "puzzle image does not visibly encode the declared A:B fill-style change and C shape", ["visible_source_replay"]
    elif not parseability_ok:
        reason, failures = "shape-outline video contains too many unreadable or semantically illegal frames", ["whole_video_semantic_parse"]
    elif not preservation_ok:
        reason, failures = "shape-outline video mutates A, B, C, labels, background, framing, or fixed geometry", ["fixed_content_preservation"]
    elif not sequence["starts_at_input"] or not sequence["ends_at_final"]:
        reason, failures = "shape-outline video initial or final state is missing or incorrect", ["input_or_final_state"]
    elif sequence["skipped_layers"] or not sequence["exact_sequence"]:
        reason, failures = "shape-outline video must show input, neutral C-shape preview, then the correct final style without skips or regression", ["state_order"]
    elif not sequence["has_initial_hold"] or not sequence["has_final_hold"]:
        reason, failures = "shape-outline video initial or final readable hold is missing", ["initial_or_final_hold"]
    elif not has_visible_transitions:
        reason, failures = "shape-outline semantic states use a hard cut instead of visible local transitions", ["visible_local_transitions"]
    elif not transitions_on_line:
        reason, failures = "a shape-outline transition does not move monotonically from the state before it to the state after it", ["local_transition_path"]
    elif not media_contract_ok:
        reason, failures = "shape-outline video must be 16fps, at most 10 seconds, and use a 4N+1 frame count", ["media_contract"]
    else:
        reason, failures = "ok", []
    scores = {
        "source_replay_score": 1.0 if source_visual_ok else 0.0,
        "shape_identity_score": 1.0 if sequence["ends_at_final"] else 0.0,
        "style_direction_score": 1.0 if sequence["ends_at_final"] else 0.0,
        "visual_parseability_score": 1.0 - min(1.0, sequence["unrecognized_fraction"]),
        "state_order_score": 1.0 if sequence["exact_sequence"] else 0.0,
        "transition_visibility_score": 1.0 if has_visible_transitions else 0.0,
        "preservation_score": 1.0 if preservation_ok else 0.0,
        "final_state_score": 1.0 if sequence["ends_at_final"] and sequence["has_final_hold"] else 0.0,
        "media_contract_score": 1.0 if media_contract_ok else 0.0,
    }
    return {
        **semantic_metrics,
        "video_candidate_state_check_pass": passed,
        "video_candidate_state_reason": reason,
        "video_candidate_state_verifier": "shape_identity_preview_and_fill_style_stage_parser",
        "video_candidate_frame_count": frame_count,
        "video_candidate_recognized_frame_count": sequence["recognized_count"],
        "video_candidate_unrecognized_fraction": sequence["unrecognized_fraction"],
        "video_candidate_unrecognized_outside_transitions": unrecognized_outside_transitions,
        "video_candidate_transition_frame_count": len(transition_frames),
        "video_shape_outline_transition_projections": transition_projections,
        "video_shape_outline_max_transition_residual": max_off_line_residual,
        "video_shape_outline_max_transition_residual_limit": _SHAPE_OUTLINE_MAX_TRANSITION_RESIDUAL,
        "video_shape_outline_transitions_on_line": transitions_on_line,
        "video_shape_outline_expected_stages": list(_SHAPE_OUTLINE_STAGES),
        "video_shape_outline_observed_stages": sequence["observed_states"],
        "video_shape_outline_exact_sequence": sequence["exact_sequence"],
        "video_shape_outline_skipped_stages": sequence["skipped_layers"],
        "video_shape_outline_has_initial_hold": sequence["has_initial_hold"],
        "video_shape_outline_has_final_hold": sequence["has_final_hold"],
        "video_shape_outline_motion_gap_counts": motion_gap_counts,
        "video_shape_outline_transition_runs": transition_runs,
        "video_shape_outline_has_visible_transitions": has_visible_transitions,
        "video_shape_outline_max_static_mutation_fraction": max_static_mutation_fraction,
        "video_fps": fps,
        "video_media_contract_valid": media_contract_ok,
        **_verifier_contract_metrics(
            verifier_profile=_SHAPE_OUTLINE_VERIFIER_PROFILE,
            artifact_kind="full_video",
            component_scores=scores,
            binary_correct=passed,
            failure_reason=reason,
            started_at=started_at,
            hard_gate_failures=failures,
        ),
    }, passed

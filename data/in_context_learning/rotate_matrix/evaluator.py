"""Task-owned source replay and visual observations."""

from __future__ import annotations

import math
import numpy as np
from data.common.verifiers.keyframes import visual_delta
from data.common.verifiers.media import _align_bgr, _read_bgr, _video_fps, _video_frames_bgr
from data.common.verifiers.metrics import _verifier_contract_metrics
from data.common.verifiers.numeric_glyphs import numeric_glyph_descriptor, read_numeric_glyph
from data.common.verifiers.observations import (
    CANDIDATE_STATE_MAX_UNRECOGNIZED_FRACTION,
    _analyze_fixed_visual_state_sequence,
    _classify_printed_number,
    _printed_number_templates,
    static_mutation_fraction as measure_static_mutation,
)
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, List, Mapping, Optional, Sequence


_ROTATE_MATRIX_VERIFIER_PROFILE = "rotate_matrix_printed_number_source_replay_v4"


def _rotate_matrix_replay(structured: dict[str, Any]) -> Optional[dict[str, Any]]:
    raw_input = structured.get("input_matrix")
    if not isinstance(raw_input, list) or len(raw_input) not in {3, 4, 5}:
        return None
    size = len(raw_input)
    input_matrix: List[List[int]] = []
    for row in raw_input:
        if not isinstance(row, list) or len(row) != size:
            return None
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 10
            or value > 99
            for value in row
        ):
            return None
        input_matrix.append(list(row))
    values = [value for row in input_matrix for value in row]
    if len(set(values)) != len(values):
        return None
    rotations_raw = structured.get("rotations")
    if (
        not isinstance(rotations_raw, int)
        or isinstance(rotations_raw, bool)
        or rotations_raw not in {1, 2, 3}
    ):
        return None
    expected_states = _rotate_matrix_expected_states(input_matrix, rotations_raw)
    redundant_output = structured.get("output_matrix")
    redundant_states = structured.get("rotation_states")
    metadata_consistent = bool(
        redundant_output == expected_states[-1]
        and redundant_states == expected_states
    )
    return {
        "input_matrix": input_matrix,
        "rotations": rotations_raw,
        "expected_states": expected_states,
        "output_matrix": expected_states[-1],
        # Whether the authored record happens to agree with this replay is a
        # source diagnostic only.  The replay is completely determined by the
        # visible matrix and the requested turn count, so it never becomes a
        # candidate hard gate or a reward component.
        "duplicate_metadata_consistent": metadata_consistent,
        "value_multiset": sorted(values),
    }


def _validate_rotate_matrix(structured: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    replay = _rotate_matrix_replay(structured)
    if replay is None:
        return {
            "video_semantic_process_check_pass": False,
            "video_semantic_process_reason": "rotate-matrix input matrix or rotation count is invalid",
            "rotate_matrix_source_replay_valid": False,
            "rotate_matrix_duplicate_metadata_consistent": False,
        }, False
    return {
        "video_semantic_process_check_pass": True,
        "video_semantic_process_reason": "ok",
        "rotate_matrix_source_replay_valid": True,
        "rotate_matrix_duplicate_metadata_consistent": replay["duplicate_metadata_consistent"],
        "video_rotate_matrix_expected_state_count": len(replay["expected_states"]),
    }, True


def _rotate_matrix_turn_label_ink(patch_rgb: np.ndarray) -> np.ndarray:
    from data.common.pipeline.structured_grid import GRID_ACCENT

    amber = np.max(np.abs(patch_rgb.astype(np.float32) - np.asarray(GRID_ACCENT)), axis=2) < 75
    return np.repeat(np.where(amber, 0, 255)[..., None], 3, axis=2).astype(np.uint8)


@lru_cache(maxsize=1)
def _rotate_matrix_turn_label_templates() -> dict[int, np.ndarray]:
    from data.in_context_learning.rotate_matrix.generator import RotateMatrixPattern, _TURN_LABEL_BOX

    renderer = RotateMatrixPattern(video=False)
    x0, y0, x1, y1 = _TURN_LABEL_BOX
    templates = {}
    for count in (1, 2, 3):
        # Only the immutable instruction label is consumed from these public
        # renders; no answer matrix or reference trajectory is consulted.
        image = np.asarray(renderer._render_state([[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                                                 target_turns=count))
        descriptor = numeric_glyph_descriptor(_rotate_matrix_turn_label_ink(image[y0:y1, x0:x1]))
        if descriptor is None:
            raise ValueError("source turn-count label is unreadable")
        templates[count] = descriptor
    return templates


def _rotate_matrix_turn_label_mutation(
    reference_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    *,
    turn_label_bbox: Sequence[float],
) -> float:
    """Fraction of the amber turn-count region that differs from the source.

    The requested turn count decides the whole answer, yet it occupies well
    under a percent of the canvas.  Folding it into the whole-frame static
    mutation fraction lets a rewritten count be averaged away by the unchanged
    background, so it gets its own obligation and its own gate.
    """

    if reference_bgr.shape != frame_bgr.shape or len(turn_label_bbox) != 4:
        return 1.0
    height, width = frame_bgr.shape[:2]
    scale_x = width / 512.0
    scale_y = height / 512.0
    x0 = max(0, int(math.floor(float(turn_label_bbox[0]) * scale_x)))
    y0 = max(0, int(math.floor(float(turn_label_bbox[1]) * scale_y)))
    x1 = min(width, int(math.ceil(float(turn_label_bbox[2]) * scale_x)))
    y1 = min(height, int(math.ceil(float(turn_label_bbox[3]) * scale_y)))
    if x1 <= x0 or y1 <= y0:
        return 1.0
    region_reference = reference_bgr[y0:y1, x0:x1]
    region_frame = frame_bgr[y0:y1, x0:x1]
    vocabulary = _rotate_matrix_turn_label_templates()
    expected, _, _ = read_numeric_glyph(_rotate_matrix_turn_label_ink(region_reference[..., ::-1]), vocabulary)
    observed, _, _ = read_numeric_glyph(_rotate_matrix_turn_label_ink(region_frame[..., ::-1]), vocabulary)
    if expected is None or observed != expected:
        return 1.0
    delta = visual_delta(region_frame, region_reference).mean(axis=2)
    return float((delta >= 18.0).mean())


_ROTATE_MATRIX_MAX_TURN_LABEL_MUTATION = 0.02


def _rotate_matrix_expected_states(
    input_matrix: Sequence[Sequence[int]],
    rotations: int,
) -> List[List[List[int]]]:
    states = [[list(map(int, row)) for row in input_matrix]]
    for _turn in range(rotations):
        previous = states[-1]
        states.append([list(row) for row in zip(*previous[::-1])])
    return states


def _classify_rotate_matrix_frame(
    frame_bgr: np.ndarray,
    *,
    grid_bbox: Sequence[Any],
    cell_size: float,
    number_token_geometry: dict[str, Any],
    templates: Mapping[int, np.ndarray],
    size: int,
) -> Optional[List[List[int]]]:
    if len(grid_bbox) != 4 or cell_size <= 0 or size <= 0:
        return None
    try:
        canonical_left = float(grid_bbox[0])
        canonical_top = float(grid_bbox[1])
        token_width = float(number_token_geometry["width"])
        token_height = float(number_token_geometry["height"])
    except Exception:
        return None
    # The digit sampling window scales with the cell, so a 3x3 board's window is
    # far larger than a 5x5 board's.  It must still stay inside the cell walls,
    # which the caller's geometry guarantees.
    if not 20.0 <= token_width <= 88.0 or not 14.0 <= token_height <= 60.0:
        return None
    if token_width >= cell_size or token_height >= cell_size:
        return None

    parsed: List[List[int]] = []
    for row in range(size):
        parsed_row: List[int] = []
        for col in range(size):
            cell_left = canonical_left + col * cell_size
            cell_top = canonical_top + row * cell_size
            value = _classify_printed_number(
                frame_bgr,
                center_x=cell_left + cell_size * 0.5,
                center_y=cell_top + cell_size * 0.5,
                token_width=token_width,
                token_height=token_height,
                templates=templates,
            )
            if value is None:
                return None
            parsed_row.append(value)
        parsed.append(parsed_row)
    return parsed


def _rotate_matrix_geometry(
    structured: dict[str, Any],
    *,
    size: int,
) -> Optional[dict[str, Any]]:
    grid_bbox = structured.get("grid_bbox")
    number_token_geometry = structured.get("number_token_geometry")
    allowed_change_bboxes = structured.get("allowed_change_bboxes")
    turn_label_bbox = structured.get("turn_label_bbox")
    try:
        cell_size = float(structured.get("cell_size"))
        bbox_values = [float(value) for value in grid_bbox]
        label_values = [float(value) for value in turn_label_bbox]
    except Exception:
        return None
    if (
        not isinstance(grid_bbox, (list, tuple))
        or len(bbox_values) != 4
        or cell_size <= 0
        or abs((bbox_values[2] - bbox_values[0]) - size * cell_size) > 2.0
        or abs((bbox_values[3] - bbox_values[1]) - size * cell_size) > 2.0
        or not 0.0 <= bbox_values[0] < bbox_values[2] <= 512.0
        or not 0.0 <= bbox_values[1] < bbox_values[3] <= 512.0
        or not isinstance(number_token_geometry, dict)
        or not isinstance(allowed_change_bboxes, list)
        or len(allowed_change_bboxes) != 1
        or any(
            not isinstance(bbox, (list, tuple)) or len(bbox) != 4
            for bbox in allowed_change_bboxes
        )
        or len(label_values) != 4
        or not 0.0 <= label_values[0] < label_values[2] <= 512.0
        or not 0.0 <= label_values[1] < label_values[3] <= 512.0
        # The turn label states a fact about the task, not about the board, so
        # it must sit outside the region the answer is allowed to change.
        or not (label_values[3] <= bbox_values[1] or label_values[1] >= bbox_values[3])
    ):
        return None
    return {
        "grid_bbox": grid_bbox,
        "cell_size": cell_size,
        "number_token_geometry": number_token_geometry,
        "allowed_change_bboxes": allowed_change_bboxes,
        "turn_label_bbox": turn_label_bbox,
    }


def _verify_rotate_matrix_candidate_state_video(
    video_path: Path,
    structured: dict[str, Any],
    puzzle_image: Optional[Path] = None,
) -> tuple[dict[str, Any], bool]:
    started_at = perf_counter()
    replay = _rotate_matrix_replay(structured)
    geometry = (
        _rotate_matrix_geometry(structured, size=len(replay["input_matrix"]))
        if replay is not None
        else None
    )
    if replay is None or geometry is None:
        component_scores = {
            "source_replay_score": 1.0 if replay is not None else 0.0,
            "visual_parseability_score": 0.0,
            "state_order_score": 0.0,
            "discrete_transition_score": 0.0,
            "value_preservation_score": 0.0,
            "preservation_score": 0.0,
            "final_state_score": 0.0,
            "media_contract_score": 0.0,
        }
        metrics = {
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": "missing rotate-matrix source or render geometry",
            **_verifier_contract_metrics(
                verifier_profile=_ROTATE_MATRIX_VERIFIER_PROFILE,
                artifact_kind="full_video",
                component_scores=component_scores,
                binary_correct=False,
                failure_reason="missing rotate-matrix source or render geometry",
                started_at=started_at,
                hard_gate_failures=["source_replay_or_geometry"],
            ),
        }
        return metrics, False

    expected_states = replay["expected_states"]
    rows = len(replay["input_matrix"])
    frames = _video_frames_bgr(video_path)
    frame_count = len(frames)
    fps = _video_fps(video_path)
    media_contract_ok = bool(
        frame_count > 0
        and abs(fps - 16.0) <= 0.05
        and (frame_count - 1) % 4 == 0
        and (frame_count - 1) / max(fps, 1.0) <= 10.0
    )
    if not frames:
        component_scores = {
            "source_replay_score": 1.0,
            "visual_parseability_score": 0.0,
            "state_order_score": 0.0,
            "discrete_transition_score": 0.0,
            "value_preservation_score": 0.0,
            "preservation_score": 0.0,
            "final_state_score": 0.0,
            "media_contract_score": 0.0,
        }
        metrics = {
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": "rotate-matrix video cannot be decoded",
            "video_candidate_frame_count": 0,
            "video_fps": fps,
            **_verifier_contract_metrics(
                verifier_profile=_ROTATE_MATRIX_VERIFIER_PROFILE,
                artifact_kind="full_video",
                component_scores=component_scores,
                binary_correct=False,
                failure_reason="rotate-matrix video cannot be decoded",
                started_at=started_at,
                hard_gate_failures=["video_decode"],
            ),
        }
        return metrics, False

    recognized: List[tuple[int, List[List[int]]]] = []
    static_mutation_fractions: List[float] = []
    turn_label_mutations: List[float] = []
    reference = (
        _align_bgr(_read_bgr(puzzle_image), frames[0].shape)
        if puzzle_image is not None and puzzle_image.exists()
        else frames[0]
    )
    token_width = float(geometry["number_token_geometry"].get("width", 0.0))
    token_height = float(geometry["number_token_geometry"].get("height", 0.0))
    templates = _printed_number_templates(
        reference,
        matrix=replay["input_matrix"],
        grid_bbox=geometry["grid_bbox"],
        cell_size=geometry["cell_size"],
        token_width=token_width,
        token_height=token_height,
    )
    for frame_index, frame in enumerate(frames):
        state = _classify_rotate_matrix_frame(
            frame,
            grid_bbox=geometry["grid_bbox"],
            cell_size=geometry["cell_size"],
            number_token_geometry=geometry["number_token_geometry"],
            templates=templates or {},
            size=rows,
        )
        if state is not None:
            recognized.append((frame_index, state))
        static_mutation_fractions.append(
            measure_static_mutation(
                reference,
                frame,
                codec_tolerant=True,
                allowed_change_bboxes=geometry["allowed_change_bboxes"],
            )
        )
        turn_label_mutations.append(
            _rotate_matrix_turn_label_mutation(
                reference,
                frame,
                turn_label_bbox=geometry["turn_label_bbox"],
            )
        )

    sequence = _analyze_fixed_visual_state_sequence(
        frame_count=frame_count,
        recognized=recognized,
        expected_states=expected_states,
    )
    observed_states = sequence["observed_states"]
    recognized_count = sequence["recognized_count"]
    unrecognized_fraction = sequence["unrecognized_fraction"]
    starts_at_input = sequence["starts_at_input"]
    ends_at_final = sequence["ends_at_final"]
    has_initial_hold = sequence["has_initial_hold"]
    has_final_hold = sequence["has_final_hold"]
    expected_indices = sequence["expected_indices"]
    skipped_turns = sequence["skipped_layers"]
    exact_sequence = sequence["exact_sequence"]
    # The matrix has a discrete state space, so its transition model is discrete
    # too: every frame is an exact state and each switch is direct.  A gap of one
    # or more frames between two recognized states means something was rendered
    # in between -- a part-turned, resized, faded or blended frame -- and that is
    # exactly what this gate rejects.
    state_gap_counts = [
        current["frame_indices"][0] - previous["frame_indices"][-1] - 1
        for previous, current in zip(sequence["compact_groups"], sequence["compact_groups"][1:])
    ]
    state_hold_counts = [len(group["frame_indices"]) for group in sequence["compact_groups"]]
    has_direct_switches = bool(
        len(state_gap_counts) == len(expected_states) - 1
        and all(gap == 0 for gap in state_gap_counts)
        and recognized_count == frame_count
    )
    # Only the states between the opening and closing holds must share a
    # duration; a longer first and last hold keeps the input and the answer
    # readable without privileging any intermediate state.
    intermediate_holds = state_hold_counts[1:-1]
    has_equal_state_holds = bool(
        len(set(intermediate_holds)) <= 1
        and (
            not intermediate_holds
            or (
                state_hold_counts[0] >= intermediate_holds[0]
                and state_hold_counts[-1] >= intermediate_holds[0]
            )
        )
    )
    max_static_mutation_fraction = max(static_mutation_fractions, default=1.0)
    recognized_values = [
        sorted(value for row in state for value in row)
        for _frame_index, state in recognized
    ]
    value_preservation_ok = bool(
        recognized_values
        and all(values == replay["value_multiset"] for values in recognized_values)
    )
    max_turn_label_mutation = max(turn_label_mutations, default=1.0)
    turn_label_ok = max_turn_label_mutation <= _ROTATE_MATRIX_MAX_TURN_LABEL_MUTATION
    preservation_ok = max_static_mutation_fraction <= 0.003 and turn_label_ok

    passed = (
        unrecognized_fraction <= CANDIDATE_STATE_MAX_UNRECOGNIZED_FRACTION
        and starts_at_input
        and ends_at_final
        and has_initial_hold
        and has_final_hold
        and exact_sequence
        and has_direct_switches
        and has_equal_state_holds
        and value_preservation_ok
        and preservation_ok
        and media_contract_ok
    )
    if unrecognized_fraction > CANDIDATE_STATE_MAX_UNRECOGNIZED_FRACTION:
        reason = "rotate-matrix printed numbers are not source-matchable in enough frames"
        failures = ["visual_parseability"]
    elif skipped_turns:
        reason = "rotate-matrix rollout skips a required clockwise quarter-turn"
        failures = ["ordered_clockwise_states"]
    elif not exact_sequence:
        reason = "rotate-matrix completed states are incomplete, out of order, counterclockwise, or mutated"
        failures = ["ordered_clockwise_states"]
    elif not starts_at_input or not ends_at_final:
        reason = "rotate-matrix input or final state mismatch"
        failures = ["input_or_final_state"]
    elif not has_initial_hold or not has_final_hold:
        reason = "rotate-matrix initial or final hold is missing"
        failures = ["initial_or_final_hold"]
    elif not has_direct_switches:
        reason = "rotate-matrix inserts a part-turned, resized, faded, or otherwise unrecognized frame between states"
        failures = ["direct_equal_duration_state_switches"]
    elif not has_equal_state_holds:
        reason = "rotate-matrix quarter-turn states are not held for equal durations"
        failures = ["direct_equal_duration_state_switches"]
    elif not value_preservation_ok:
        reason = "rotate-matrix video changes, duplicates, or removes a cell value"
        failures = ["cell_value_identity_preservation"]
    elif not turn_label_ok:
        reason = "rotate-matrix video rewrites the requested clockwise quarter-turn count"
        failures = ["fixed_content_preservation"]
    elif not preservation_ok:
        reason = "rotate-matrix background or fixed layout mutates"
        failures = ["fixed_content_preservation"]
    elif not media_contract_ok:
        reason = "rotate-matrix video must be 16fps, at most 10 seconds, and use 4N+1 frames"
        failures = ["media_contract"]
    else:
        reason = "ok"
        failures = []
    component_scores = {
        "source_replay_score": 1.0,
        "visual_parseability_score": 1.0 - min(1.0, unrecognized_fraction),
        "state_order_score": 1.0 if exact_sequence else 0.0,
        "discrete_transition_score": 1.0 if has_direct_switches and has_equal_state_holds else 0.0,
        "value_preservation_score": 1.0 if value_preservation_ok else 0.0,
        "preservation_score": 1.0 if preservation_ok else 0.0,
        "final_state_score": 1.0 if starts_at_input and ends_at_final else 0.0,
        "media_contract_score": 1.0 if media_contract_ok else 0.0,
    }
    metrics = {
        "video_candidate_state_check_pass": passed,
        "video_candidate_state_reason": reason,
        "video_candidate_state_verifier": "rotate_matrix_source_printed_number_state_parser",
        "video_candidate_frame_count": frame_count,
        "video_candidate_recognized_frame_count": recognized_count,
        "video_candidate_unrecognized_fraction": unrecognized_fraction,
        "video_candidate_max_unrecognized_fraction": CANDIDATE_STATE_MAX_UNRECOGNIZED_FRACTION,
        "video_rotate_matrix_expected_state_count": len(expected_states),
        "video_rotate_matrix_observed_state_count": len(observed_states),
        "video_rotate_matrix_expected_state_indices": expected_indices,
        "video_rotate_matrix_starts_at_input": starts_at_input,
        "video_rotate_matrix_ends_at_final": ends_at_final,
        "video_rotate_matrix_has_initial_hold": has_initial_hold,
        "video_rotate_matrix_has_final_hold": has_final_hold,
        "video_rotate_matrix_skipped_turns": skipped_turns,
        "video_rotate_matrix_exact_sequence": exact_sequence,
        "video_rotate_matrix_state_gap_counts": state_gap_counts,
        "video_rotate_matrix_state_hold_counts": state_hold_counts,
        "video_rotate_matrix_has_direct_switches": has_direct_switches,
        "video_rotate_matrix_has_equal_state_holds": has_equal_state_holds,
        "video_rotate_matrix_max_static_mutation_fraction": max_static_mutation_fraction,
        "video_rotate_matrix_max_turn_label_mutation": max_turn_label_mutation,
        "video_rotate_matrix_value_preservation_valid": value_preservation_ok,
        "video_rotate_matrix_duplicate_metadata_consistent": replay["duplicate_metadata_consistent"],
        "video_fps": fps,
        "video_media_contract_valid": media_contract_ok,
        **_verifier_contract_metrics(
            verifier_profile=_ROTATE_MATRIX_VERIFIER_PROFILE,
            artifact_kind="full_video",
            component_scores=component_scores,
            binary_correct=passed,
            failure_reason=reason,
            started_at=started_at,
            hard_gate_failures=failures,
        ),
    }
    return metrics, passed

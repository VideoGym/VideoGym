"""Task-owned source replay and visual observations."""

from __future__ import annotations

import math
import numpy as np
from data.common.verifiers.media import _align_bgr, _read_bgr, _video_fps, _video_frames_bgr
from data.common.verifiers.metrics import _verifier_contract_metrics
from data.common.verifiers.observations import _pixels_near_rgb, static_mutation_fraction
from data.common.verifiers.source import _to_int_grid
from pathlib import Path
from time import perf_counter
from typing import Any, List, Optional, Sequence


def _legend_rotate_clockwise(matrix: List[List[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*matrix[::-1])]


def _legend_flip_horizontal(matrix: List[List[int]]) -> List[List[int]]:
    return [list(reversed(row)) for row in matrix]


def _legend_transpose(matrix: List[List[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*matrix)]


def _apply_legend_operator(operator_name: str, matrix: List[List[int]]) -> Optional[List[List[int]]]:
    if operator_name == "rotate_cw":
        return _legend_rotate_clockwise(matrix)
    if operator_name == "flip_horizontal":
        return _legend_flip_horizontal(matrix)
    if operator_name == "transpose":
        return _legend_transpose(matrix)
    return None


_LEGEND_OPERATOR_VERIFIER_PROFILE = "legend_operator_independent_example_and_program_replay_v1"


def _legend_operator_render_language() -> dict[str, Any]:
    """Import the legend renderer's layout and colours, never a duplicate."""

    from data.in_context_learning.legend_operator_program.generator import (
        _ACTIVE_TOKEN_OUTLINE,
        program_token_bboxes,
        _RESULT_GRID_LEFT,
        _RESULT_GRID_TOP,
        _STEP_LABEL_BBOX,
        _TRANSITION_FILL,
        work_cell_px,
    )

    return {
        "token_bboxes": program_token_bboxes,
        "step_label_bbox": tuple(float(v) for v in _STEP_LABEL_BBOX),
        "result_left": float(_RESULT_GRID_LEFT),
        "result_top": float(_RESULT_GRID_TOP),
        "amber_rgb": tuple(_TRANSITION_FILL),
        "active_outline_rgb": tuple(_ACTIVE_TOKEN_OUTLINE),
        "work_cell_px": work_cell_px,
    }


def _legend_operator_program_replay(
    structured: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Replay only from the visible semantic source, ignoring saved answers and traces."""
    raw_target = structured.get("target_matrix")
    raw_tokens = structured.get("program_tokens")
    raw_examples = structured.get("legend_examples")
    raw_legend_outputs = structured.get("legend_outputs")
    operator_names = ("rotate_cw", "flip_horizontal", "transpose")
    size = len(raw_target) if isinstance(raw_target, list) else 0

    def parse_binary_square(payload: Any) -> Optional[List[List[int]]]:
        if (
            not isinstance(payload, list)
            or len(payload) != size
            or any(not isinstance(row, list) or len(row) != size for row in payload)
        ):
            return None
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value not in {0, 1}
            for row in payload
            for value in row
        ):
            return None
        return [[int(value) for value in row] for row in payload]

    target = parse_binary_square(raw_target)

    tokens = list(raw_tokens) if isinstance(raw_tokens, list) else []
    if (
        size not in {3, 4, 5}
        or target is None
        or len(tokens) not in (2, 3, 4, 5)
        or any(
            not isinstance(token, str) or token not in {"A", "B", "C"}
            for token in tokens
        )
        or not isinstance(raw_examples, dict)
        or set(raw_examples) != {"A", "B", "C"}
        or not isinstance(raw_legend_outputs, dict)
        or set(raw_legend_outputs) != {"A", "B", "C"}
    ):
        return None

    inferred_mapping: dict[str, str] = {}
    examples: dict[str, List[List[int]]] = {}
    legend_outputs: dict[str, List[List[int]]] = {}
    for token in ("A", "B", "C"):
        before = parse_binary_square(raw_examples.get(token))
        after = parse_binary_square(raw_legend_outputs.get(token))
        if before is None or after is None or before == after:
            return None
        matches = [
            operator_name
            for operator_name in operator_names
            if _apply_legend_operator(operator_name, before or []) == after
        ]
        if len(matches) != 1:
            return None
        inferred_mapping[token] = matches[0]
        examples[token] = before or []
        legend_outputs[token] = after or []
    if set(inferred_mapping.values()) != set(operator_names):
        return None

    states = [[list(row) for row in target]]
    current = [list(row) for row in target]
    for token in tokens:
        next_state = _apply_legend_operator(inferred_mapping[token], current)
        if next_state is None or next_state == current:
            return None
        states.append(next_state)
        current = next_state
    if len({tuple(tuple(row) for row in state) for state in states}) != len(states):
        return None

    reversed_current = [list(row) for row in target]
    for token in reversed(tokens):
        reversed_state = _apply_legend_operator(inferred_mapping[token], reversed_current)
        if reversed_state is None:
            return None
        reversed_current = reversed_state
    density = sum(sum(row) for row in target)
    if reversed_current == current or not 3 <= density <= size * size - 3:
        return None

    language = _legend_operator_render_language()
    token_boxes = language["token_bboxes"](len(tokens))
    cell_size = float(language["work_cell_px"](size))
    result_bbox = [
        language["result_left"],
        language["result_top"],
        language["result_left"] + size * cell_size,
        language["result_top"] + size * cell_size,
    ]
    return {
        "size": size,
        "tokens": tokens,
        "legend_examples": examples,
        "legend_outputs": legend_outputs,
        "inferred_mapping": inferred_mapping,
        "target_matrix": states[0],
        "state_trace": states,
        "output_matrix": states[-1],
        "reversed_program_output": reversed_current,
        "result_grid_bbox": result_bbox,
        "result_cell_size": cell_size,
        "program_token_bboxes": [list(box) for box in token_boxes],
        "allowed_change_bboxes": [
            result_bbox,
            *[list(box) for box in token_boxes],
            list(language["step_label_bbox"]),
        ],
    }


def _validate_legend_operator_program(structured: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    replay = _legend_operator_program_replay(structured)
    if replay is None:
        return {
            "video_semantic_process_check_pass": False,
            "video_semantic_process_reason": "legend examples or ordered program cannot be independently replayed",
        }, False
    raw_mapping = structured.get("operator_by_token")
    declared_mapping = (
        {str(token): str(name) for token, name in raw_mapping.items()}
        if isinstance(raw_mapping, dict)
        else {}
    )
    saved_trace = [_to_int_grid(state) for state in structured.get("state_trace") or []]
    try:
        saved_cell_size = float(structured.get("result_cell_size"))
    except (TypeError, ValueError):
        saved_cell_size = float("nan")
    metadata_integrity = bool(
        declared_mapping == replay["inferred_mapping"]
        and _to_int_grid(structured.get("output_matrix")) == replay["output_matrix"]
        and _to_int_grid(structured.get("reversed_program_output")) == replay["reversed_program_output"]
        and saved_trace == replay["state_trace"]
        and structured.get("grid_size") == replay["size"]
        and structured.get("result_grid_bbox") == replay["result_grid_bbox"]
        and saved_cell_size == replay["result_cell_size"]
        and structured.get("program_token_bboxes") == replay["program_token_bboxes"]
        and structured.get("allowed_change_bboxes") == replay["allowed_change_bboxes"]
    )
    # Redundant saved answers, traces, and render boxes are diagnostics only. Correctness
    # comes from the visible legend examples, target matrix, and ordered program tokens.
    passed = True
    return {
        "video_semantic_process_check_pass": passed,
        "video_semantic_process_reason": "ok",
        "video_legend_program_length": len(replay["tokens"]),
        "video_legend_operator_inferred_mapping": replay["inferred_mapping"],
        "video_legend_operator_order_load_bearing": replay["reversed_program_output"] != replay["output_matrix"],
        "video_legend_operator_metadata_integrity": metadata_integrity,
    }, passed


def _classify_legend_operator_cells(
    frame_bgr: np.ndarray,
    *,
    grid_bbox: Sequence[float],
    cell_size: float,
    size: int,
) -> Optional[List[List[int | str]]]:
    if len(grid_bbox) != 4 or cell_size <= 0.0 or size <= 0:
        return None
    height, width = frame_bgr.shape[:2]
    scale_x = width / 512.0
    scale_y = height / 512.0
    left = float(grid_bbox[0]) * scale_x
    top = float(grid_bbox[1]) * scale_y
    parsed: List[List[int | str]] = []
    for row in range(size):
        parsed_row: List[int | str] = []
        for col in range(size):
            x0 = int(round(left + (col + 0.24) * cell_size * scale_x))
            x1 = int(round(left + (col + 0.76) * cell_size * scale_x))
            y0 = int(round(top + (row + 0.24) * cell_size * scale_y))
            y1 = int(round(top + (row + 0.76) * cell_size * scale_y))
            patch = frame_bgr[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
            if patch.size == 0:
                return None
            intensity = np.mean(patch.astype(np.float32), axis=2)
            dark_fraction = float((intensity <= 105.0).mean())
            light_fraction = float((intensity >= 220.0).mean())
            amber_fraction = float(
                _pixels_near_rgb(
                    patch,
                    _legend_operator_render_language()["amber_rgb"],
                    tolerance=72.0,
                ).mean()
            )
            if amber_fraction >= 0.58:
                parsed_row.append("amber")
            elif dark_fraction >= 0.65:
                parsed_row.append(1)
            elif light_fraction >= 0.65:
                parsed_row.append(0)
            else:
                return None
        parsed.append(parsed_row)
    return parsed


def _classify_legend_operator_matrix(
    frame_bgr: np.ndarray,
    *,
    grid_bbox: Sequence[float],
    cell_size: float,
    size: int,
) -> Optional[List[List[int]]]:
    cells = _classify_legend_operator_cells(
        frame_bgr,
        grid_bbox=grid_bbox,
        cell_size=cell_size,
        size=size,
    )
    if cells is None or any(value == "amber" for row in cells for value in row):
        return None
    return [[int(value) for value in row] for row in cells]


def _classify_legend_operator_transition_cells(
    frame_bgr: np.ndarray,
    *,
    grid_bbox: Sequence[float],
    cell_size: float,
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> Optional[set[tuple[int, int]]]:
    size = len(before)
    if (
        size <= 0
        or len(after) != size
        or any(len(row) != size for row in before)
        or any(len(row) != size for row in after)
    ):
        return None
    cells = _classify_legend_operator_cells(
        frame_bgr,
        grid_bbox=grid_bbox,
        cell_size=cell_size,
        size=size,
    )
    if cells is None:
        return None
    expected_changed = {
        (row, col)
        for row in range(size)
        for col in range(size)
        if int(before[row][col]) != int(after[row][col])
    }
    observed_amber = {
        (row, col)
        for row in range(size)
        for col in range(size)
        if cells[row][col] == "amber"
    }
    if observed_amber != expected_changed:
        return None
    for row in range(size):
        for col in range(size):
            if (row, col) in expected_changed:
                continue
            if cells[row][col] != int(after[row][col]):
                return None
    return observed_amber


def _classify_legend_operator_active_token(
    frame_bgr: np.ndarray,
    token_bboxes: Sequence[Sequence[float]],
) -> Optional[int]:
    height, width = frame_bgr.shape[:2]
    scale_x = width / 512.0
    scale_y = height / 512.0
    green_mask = _pixels_near_rgb(
        frame_bgr,
        _legend_operator_render_language()["active_outline_rgb"],
        tolerance=62.0,
    )
    active: List[int] = []
    for index, bbox in enumerate(token_bboxes):
        if len(bbox) != 4:
            return None
        x0 = max(0, int(math.floor(float(bbox[0]) * scale_x)))
        y0 = max(0, int(math.floor(float(bbox[1]) * scale_y)))
        x1 = min(width, int(math.ceil(float(bbox[2]) * scale_x)))
        y1 = min(height, int(math.ceil(float(bbox[3]) * scale_y)))
        patch = green_mask[y0:y1, x0:x1]
        minimum = max(8, int(round(18 * scale_x * scale_y)))
        if patch.size and int(patch.sum()) >= minimum:
            active.append(index)
    if len(active) > 1:
        return -2
    return active[0] if active else None


def _verify_legend_operator_candidate_state_video(
    video_path: Path,
    structured: dict[str, Any],
    puzzle_path: Optional[Path] = None,
) -> tuple[dict[str, Any], bool]:
    started_at = perf_counter()
    semantic_metrics, semantic_pass = _validate_legend_operator_program(structured)
    replay = _legend_operator_program_replay(structured)
    if replay is None:
        component_scores = {
            "legend_inference_score": 0.0,
            "program_replay_score": 0.0,
            "stable_state_score": 0.0,
            "transition_cell_score": 0.0,
            "state_order_score": 0.0,
            "preservation_score": 0.0,
            "media_contract_score": 0.0,
            "artifact_readability_score": 0.0,
            "final_state_score": 0.0,
        }
        result = _verifier_contract_metrics(
            verifier_profile=_LEGEND_OPERATOR_VERIFIER_PROFILE,
            artifact_kind="video",
            component_scores=component_scores,
            binary_correct=False,
            failure_reason="legend examples or ordered program cannot be independently replayed",
            started_at=started_at,
            hard_gate_failures=["independent_replay"],
        )
        return {
            **semantic_metrics,
            **component_scores,
            **result,
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": result["failure_reason"],
        }, False

    state_trace = replay["state_trace"]
    length = len(replay["tokens"])
    expected_states = [{"matrix": matrix, "active_token": index - 1 if index else None}
                       for index, matrix in enumerate(state_trace)]
    stable_stage_names = ["stable_input"] + [f"stable_after_token_{i}" for i in range(1, length)] + ["stable_final"]
    expected_stage_sequence = ["stable_input"]
    for i in range(1, length + 1):
        expected_stage_sequence.extend([f"amber_transition_{i}", stable_stage_names[i]])
    try:
        frames = _video_frames_bgr(video_path)
    except (OSError, RuntimeError):
        frames = []
    if not frames:
        component_scores = {
            "legend_inference_score": 1.0,
            "program_replay_score": 1.0,
            "stable_state_score": 0.0,
            "transition_cell_score": 0.0,
            "state_order_score": 0.0,
            "preservation_score": 0.0,
            "media_contract_score": 0.0,
            "artifact_readability_score": 0.0,
            "final_state_score": 0.0,
        }
        reason = "legend-operator video could not be decoded"
        result = _verifier_contract_metrics(
            verifier_profile=_LEGEND_OPERATOR_VERIFIER_PROFILE,
            artifact_kind="video",
            component_scores=component_scores,
            binary_correct=False,
            failure_reason=reason,
            started_at=started_at,
            hard_gate_failures=["video_decode"],
        )
        return {
            **semantic_metrics,
            **component_scores,
            **result,
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": reason,
            "video_candidate_state_verifier": "legend_operator_exact_amber_program_parser",
            "video_candidate_frame_count": 0,
            "video_candidate_recognized_frame_count": 0,
            "video_candidate_unrecognized_fraction": 1.0,
            "video_candidate_max_unrecognized_fraction": 0.0,
            "video_legend_operator_invalid_frame_indices": [],
            "video_media_contract_pass": False,
        }, False
    frame_stages: List[Optional[str]] = []
    invalid_frame_indices: List[int] = []
    static_mutation_fractions: List[float] = []
    reference = (
        _align_bgr(_read_bgr(puzzle_path), frames[0].shape)
        if puzzle_path is not None and puzzle_path.exists()
        else frames[0]
    )
    for frame_index, frame in enumerate(frames):
        matrix = _classify_legend_operator_matrix(
            frame,
            grid_bbox=replay["result_grid_bbox"],
            cell_size=replay["result_cell_size"],
            size=replay["size"],
        )
        active_token = _classify_legend_operator_active_token(
            frame,
            replay["program_token_bboxes"],
        )
        stage = None
        if matrix == state_trace[0] and active_token is None:
            stage = "stable_input"
        elif isinstance(active_token, int) and 0 <= active_token < length:
            index = active_token + 1
            transition = _classify_legend_operator_transition_cells(
                frame, grid_bbox=replay["result_grid_bbox"], cell_size=replay["result_cell_size"],
                before=state_trace[index - 1], after=state_trace[index],
            )
            if transition is not None:
                stage = f"amber_transition_{index}"
            elif matrix == state_trace[index]:
                stage = stable_stage_names[index]
        if stage is None:
            invalid_frame_indices.append(frame_index)
        frame_stages.append(stage)
        static_mutation_fractions.append(
            static_mutation_fraction(
                reference,
                frame,
                allowed_change_bboxes=replay["allowed_change_bboxes"],
            )
        )

    frame_count = len(frames)
    compact_groups: List[dict[str, Any]] = []
    for frame_index, stage in enumerate(frame_stages):
        if stage is None:
            continue
        if not compact_groups or compact_groups[-1]["stage"] != stage:
            compact_groups.append({"stage": stage, "frame_indices": [frame_index]})
        else:
            compact_groups[-1]["frame_indices"].append(frame_index)
    observed_stage_sequence = [group["stage"] for group in compact_groups]
    exact_stage_sequence = observed_stage_sequence == expected_stage_sequence
    stable_stage_names = {name: index for index, name in enumerate(stable_stage_names)}
    observed_stable_indices = [
        stable_stage_names[stage]
        for stage in observed_stage_sequence
        if stage in stable_stage_names
    ]
    exact_stable_sequence = observed_stable_indices == list(range(length + 1))
    group_by_stage = {group["stage"]: group for group in compact_groups}
    transition_gap_counts = [
        len(group_by_stage.get(f"amber_transition_{i}", {}).get("frame_indices", []))
        for i in range(1, length + 1)
    ]
    has_visible_transitions = bool(
        exact_stage_sequence and all(count >= 1 for count in transition_gap_counts)
    )
    has_initial_hold = len(group_by_stage.get("stable_input", {}).get("frame_indices", [])) >= 2
    has_final_hold = len(group_by_stage.get("stable_final", {}).get("frame_indices", [])) >= 2
    recognized_count = frame_count - len(invalid_frame_indices)
    unreadable_fraction = (
        len(invalid_frame_indices) / float(frame_count) if frame_count else 1.0
    )
    max_static_mutation_fraction = max(static_mutation_fractions, default=1.0)
    observed_fps = _video_fps(video_path)
    media_contract_ok = bool(
        abs(observed_fps - 16.0) <= 0.05
        and frame_count >= 9
        and frame_count % 4 == 1
        and frame_count <= 157
        and frame_count / max(observed_fps, 1e-9) <= 10.0
    )
    stable_state_score = 1.0 if exact_stable_sequence else min(1.0, len(set(observed_stable_indices)) / (length + 1))
    transition_cell_score = 1.0 if has_visible_transitions else 0.0
    state_order_score = 1.0 if exact_stage_sequence else 0.0
    preservation_score = max(0.0, 1.0 - max_static_mutation_fraction / 0.01)
    artifact_readability_score = recognized_count / float(frame_count) if frame_count else 0.0
    final_state_score = 1.0 if (
        observed_stage_sequence[-1:] == ["stable_final"] and has_final_hold
    ) else 0.0
    passed = bool(
        semantic_pass
        and not invalid_frame_indices
        and exact_stage_sequence
        and has_initial_hold
        and has_final_hold
        and has_visible_transitions
        and max_static_mutation_fraction <= 0.003
        and media_contract_ok
    )
    if invalid_frame_indices:
        reason = "legend-operator video contains an unreadable or semantically illegal matrix, token, or transition frame"
        hard_gate_failures = ["whole_video_semantic_parse"]
    elif max_static_mutation_fraction > 0.003:
        reason = "legend-operator static legend, target, or program content mutates"
        hard_gate_failures = ["static_preservation"]
    elif not exact_stage_sequence:
        reason = (
            "legend-operator video uses hard cuts, missing/incorrect amber transitions, or an invalid stage order; "
            "required order is input, then each token transition and stable state, then final"
        )
        hard_gate_failures = ["state_order"]
    elif not has_initial_hold or not has_final_hold:
        reason = "legend-operator initial or final hold is missing"
        hard_gate_failures = ["readable_holds"]
    elif not has_visible_transitions:
        reason = "legend-operator amber transitions do not mark exactly the independently changed cells"
        hard_gate_failures = ["exact_transition_cells"]
    elif not media_contract_ok:
        reason = "legend-operator video must be 16fps, use 4N+1 frames, and remain at most 10 seconds"
        hard_gate_failures = ["media_contract"]
    else:
        reason = "ok"
        hard_gate_failures = []
    component_scores = {
        "legend_inference_score": 1.0,
        "program_replay_score": 1.0,
        "stable_state_score": stable_state_score,
        "transition_cell_score": transition_cell_score,
        "state_order_score": state_order_score,
        "preservation_score": preservation_score,
        "media_contract_score": 1.0 if media_contract_ok else 0.0,
        "artifact_readability_score": artifact_readability_score,
        "final_state_score": final_state_score,
    }
    result = _verifier_contract_metrics(
        verifier_profile=_LEGEND_OPERATOR_VERIFIER_PROFILE,
        artifact_kind="video",
        component_scores=component_scores,
        binary_correct=passed,
        failure_reason=reason,
        started_at=started_at,
        hard_gate_failures=hard_gate_failures,
    )
    return {
        **semantic_metrics,
        **component_scores,
        **result,
        "video_candidate_state_check_pass": passed,
        "video_candidate_state_reason": reason,
        "video_candidate_state_verifier": "legend_operator_exact_amber_program_parser",
        "video_candidate_frame_count": frame_count,
        "video_candidate_recognized_frame_count": recognized_count,
        "video_candidate_unrecognized_fraction": unreadable_fraction,
        "video_candidate_max_unrecognized_fraction": 0.0,
        "video_legend_operator_expected_states": expected_states,
        "video_legend_operator_observed_states": observed_stable_indices,
        "video_legend_operator_expected_state_indices": list(range(length + 1)),
        "video_legend_operator_exact_sequence": exact_stable_sequence,
        "video_legend_operator_expected_stage_sequence": expected_stage_sequence,
        "video_legend_operator_observed_stage_sequence": observed_stage_sequence,
        "video_legend_operator_exact_stage_sequence": exact_stage_sequence,
        "video_legend_operator_has_initial_hold": has_initial_hold,
        "video_legend_operator_has_final_hold": has_final_hold,
        "video_legend_operator_skipped_tokens": not exact_stable_sequence,
        "video_legend_operator_transition_gap_counts": transition_gap_counts,
        "video_legend_operator_has_visible_transitions": has_visible_transitions,
        "video_legend_operator_invalid_frame_indices": invalid_frame_indices,
        "video_legend_operator_max_static_mutation_fraction": max_static_mutation_fraction,
        "video_legend_operator_observed_fps": observed_fps,
        "video_media_contract_pass": media_contract_ok,
    }, passed

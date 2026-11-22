"""Task-owned source replay and visual observations."""

from __future__ import annotations

import math
import numpy as np
from data.common.verifiers.media import _read_bgr, _video_fps, _video_frames_bgr
from data.common.verifiers.metrics import _verifier_contract_metrics
from data.common.verifiers.observations import (
    CANDIDATE_STATE_MAX_UNRECOGNIZED_FRACTION,
    _pixels_near_rgb,
    static_mutation_fraction,
)
from data.common.verifiers.source import _grid_dimensions, _to_rgb_color, _to_rgb_grid
from pathlib import Path
from time import perf_counter
from typing import Any, List, Optional, Sequence


_HUE_ALLOWED_STEP_PAIRS = ((1, 2), (1, 3), (2, 1), (3, 1), (2, 3), (3, 2))


def _hue_visible_replay(structured: dict[str, Any]) -> Optional[dict[str, Any]]:
    raw_palette_names = structured.get("palette_names")
    raw_visible = structured.get("visible_board_names")
    raw_missing = structured.get("missing_cell")
    if (
        not isinstance(raw_palette_names, list)
        or len(raw_palette_names) != 5
        or not isinstance(raw_visible, list)
        or len(raw_visible) != 5
        or any(not isinstance(row, list) or len(row) != 5 for row in raw_visible)
        or not isinstance(raw_missing, (list, tuple))
        or len(raw_missing) != 2
    ):
        return None
    palette_names = [str(name).strip().upper() for name in raw_palette_names]
    if len(set(palette_names)) != 5 or any(name not in _HUE_VIDEO_PALETTE for name in palette_names):
        return None
    try:
        missing = (int(raw_missing[0]), int(raw_missing[1]))
    except Exception:
        return None
    if not all(0 <= coordinate < 5 for coordinate in missing):
        return None
    absent = [
        (row, col)
        for row, values in enumerate(raw_visible)
        for col, value in enumerate(values)
        if value is None
    ]
    if absent != [missing]:
        return None
    visible: list[list[Optional[str]]] = []
    for values in raw_visible:
        parsed_row: list[Optional[str]] = []
        for value in values:
            normalized = None if value is None else str(value).strip().upper()
            if normalized is not None and normalized not in palette_names:
                return None
            parsed_row.append(normalized)
        visible.append(parsed_row)

    matches: list[dict[str, Any]] = []
    for origin_index in range(5):
        for row_step, col_step in _HUE_ALLOWED_STEP_PAIRS:
            board = [
                [
                    palette_names[(origin_index + row * row_step + col * col_step) % 5]
                    for col in range(5)
                ]
                for row in range(5)
            ]
            if all(
                visible[row][col] is None or visible[row][col] == board[row][col]
                for row in range(5)
                for col in range(5)
            ):
                matches.append(
                    {
                        "origin_index": origin_index,
                        "row_step": row_step,
                        "col_step": col_step,
                        "board_names": board,
                        "correct_color_name": board[missing[0]][missing[1]],
                    }
                )
    if len(matches) != 1:
        return None
    replay = matches[0]
    replay["palette_names"] = palette_names
    replay["visible_board_names"] = visible
    replay["missing_cell"] = [missing[0], missing[1]]
    replay["matching_rule_count"] = len(matches)
    return replay


def _validate_hue(structured: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    replay = _hue_visible_replay(structured)
    if replay is None:
        return {
            "video_semantic_process_check_pass": False,
            "video_semantic_process_reason": "hue visible clues do not identify one unique cycle rule",
            "hue_visible_replay_valid": False,
        }, False
    expected_name = replay["correct_color_name"]
    expected_board = replay["board_names"]
    redundant_board = structured.get("board_names")
    expected_board_colors = [
        [_HUE_VIDEO_PALETTE[name] for name in row]
        for row in expected_board
    ]
    expected_palette = {
        name: _HUE_VIDEO_PALETTE[name]
        for name in replay["palette_names"]
    }
    actual_board_colors = _to_rgb_grid(structured.get("board_colors"))
    actual_palette = structured.get("palette")
    normalized_palette = (
        {
            str(name).strip().upper(): _to_rgb_color(value)
            for name, value in actual_palette.items()
        }
        if isinstance(actual_palette, dict)
        else {}
    )
    metadata_consistent = bool(
        redundant_board == expected_board
        and str(structured.get("correct_color_name") or "").strip().upper() == expected_name
        and structured.get("row_step") == replay["row_step"]
        and structured.get("col_step") == replay["col_step"]
        and structured.get("board_size") == 5
        and actual_board_colors == expected_board_colors
        and _to_rgb_color(structured.get("correct_color")) == _HUE_VIDEO_PALETTE[expected_name]
        and normalized_palette == expected_palette
    )
    return {
        "video_semantic_process_check_pass": True,
        "video_semantic_process_reason": "ok",
        "video_hue_expected_color_name": expected_name,
        "hue_visible_replay_valid": True,
        # Duplicate authoring metadata is useful for generator QA, but it is
        # deliberately outside the candidate correctness decision.  The
        # visible board has already established the unique expected hue.
        "hue_metadata_consistency_valid": metadata_consistent,
        "hue_replayed_origin_index": replay["origin_index"],
        "hue_replayed_row_step": replay["row_step"],
        "hue_replayed_col_step": replay["col_step"],
    }, True


_HUE_VIDEO_PALETTE = {
    "RED": (224, 79, 73),
    "ORANGE": (232, 139, 56),
    "YELLOW": (222, 187, 55),
    "GREEN": (77, 166, 104),
    "TEAL": (48, 164, 159),
    "BLUE": (67, 121, 196),
    "PURPLE": (143, 91, 181),
    "PINK": (207, 91, 141),
}


def _classify_hue_cell(
    frame_bgr: np.ndarray,
    *,
    grid_bbox: Sequence[Any],
    cell_size: float,
    row: int,
    col: int,
    allow_blank: bool,
) -> Optional[str]:
    height, width = frame_bgr.shape[:2]
    scale_x = width / 512.0
    scale_y = height / 512.0
    left, top = float(grid_bbox[0]), float(grid_bbox[1])
    x0 = int(round((left + col * cell_size) * scale_x))
    x1 = int(round((left + (col + 1) * cell_size) * scale_x))
    y0 = int(round((top + row * cell_size) * scale_y))
    y1 = int(round((top + (row + 1) * cell_size) * scale_y))
    inset_x = max(4, int(round(cell_size * 0.14 * scale_x)))
    inset_y = max(4, int(round(cell_size * 0.14 * scale_y)))
    patch = frame_bgr[
        max(0, y0 + inset_y):min(height, y1 - inset_y),
        max(0, x0 + inset_x):min(width, x1 - inset_x),
    ]
    if patch.size == 0:
        return None
    scores = sorted(
        (int(_pixels_near_rgb(patch, color, tolerance=46.0).sum()), name)
        for name, color in _HUE_VIDEO_PALETTE.items()
    )
    best_count, best_name = scores[-1]
    second_count = scores[-2][0]
    area = patch.shape[0] * patch.shape[1]
    minimum = max(24, int(round(area * 0.34)))
    if best_count >= minimum and best_count >= second_count * 1.10:
        return best_name
    if allow_blank:
        rgb = patch[..., ::-1].astype(np.int16)
        bright_fraction = float((np.min(rgb, axis=2) >= 220).mean())
        if bright_fraction >= 0.94:
            return "BLANK"
        # A partially faded correct-color tile is intentionally neither blank
        # nor a stable palette cell. The caller treats this as a transition.
        if bright_fraction < 0.94 or best_count >= max(8, int(round(area * 0.04))):
            return "TRANSITION"
    return None


def _measure_hue_complete_footprint_fade(
    frame_bgr: np.ndarray,
    *,
    source_bgr: np.ndarray,
    terminal_bgr: np.ndarray,
    grid_bbox: Sequence[Any],
    cell_size: float,
    row: int,
    col: int,
) -> dict[str, float]:
    """Measure one uniform local fade over the terminal cell footprint."""

    height, width = frame_bgr.shape[:2]
    scale_x = width / 512.0
    scale_y = height / 512.0
    left, top = float(grid_bbox[0]), float(grid_bbox[1])
    x0 = int(round((left + col * cell_size + 5.0) * scale_x))
    x1 = int(round((left + (col + 1) * cell_size - 5.0) * scale_x))
    y0 = int(round((top + row * cell_size + 5.0) * scale_y))
    y1 = int(round((top + (row + 1) * cell_size - 5.0) * scale_y))
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    patch = frame_bgr[y0:y1, x0:x1].astype(np.float64)
    source = source_bgr[y0:y1, x0:x1].astype(np.float64)
    terminal = terminal_bgr[y0:y1, x0:x1].astype(np.float64)
    if patch.size == 0 or patch.shape != source.shape or patch.shape != terminal.shape:
        return {
            "progress": 0.0,
            "linear_fit_error": 1.0,
            "footprint_support": 0.0,
            "pixel_progress_error": 1.0,
            "active_pixels": 0.0,
        }

    direction = terminal - source
    endpoint_magnitude = np.max(np.abs(direction), axis=2)
    active = endpoint_magnitude >= 12.0
    active_pixels = int(np.sum(active))
    denominator = float(np.sum(direction[active] ** 2))
    if active_pixels < 24 or denominator <= 1.0:
        return {
            "progress": 0.0,
            "linear_fit_error": 1.0,
            "footprint_support": 0.0,
            "pixel_progress_error": 1.0,
            "active_pixels": float(active_pixels),
        }

    delta = patch - source
    raw_progress = float(np.sum(delta[active] * direction[active]) / denominator)
    progress = max(0.0, min(1.0, raw_progress))
    residual = delta - progress * direction
    linear_fit_error = math.sqrt(float(np.sum(residual[active] ** 2)) / denominator)
    delta_magnitude = np.max(np.abs(delta), axis=2)
    support_threshold = np.maximum(3.0, endpoint_magnitude * 0.07)
    footprint_support = float(np.mean(delta_magnitude[active] >= support_threshold[active]))
    per_pixel_denominator = np.sum(direction * direction, axis=2)
    per_pixel_progress = np.zeros_like(endpoint_magnitude, dtype=np.float64)
    per_pixel_progress[active] = (
        np.sum(delta * direction, axis=2)[active]
        / np.maximum(1.0, per_pixel_denominator[active])
    )
    pixel_progress_error = float(
        np.median(np.abs(np.clip(per_pixel_progress[active], 0.0, 1.0) - progress))
    )
    return {
        "progress": float(progress),
        "linear_fit_error": float(linear_fit_error),
        "footprint_support": float(footprint_support),
        "pixel_progress_error": float(pixel_progress_error),
        "active_pixels": float(active_pixels),
    }


def _verify_hue_candidate_state_video(
    video_path: Path,
    structured: dict[str, Any],
    puzzle_path: Path,
) -> tuple[dict[str, Any], bool]:
    started_at = perf_counter()
    profile = "hue_fixed_palette_complete_footprint_fade_video_v4"
    replay = _hue_visible_replay(structured)
    semantic_metrics, semantic_pass = _validate_hue(structured)
    if replay is None or not semantic_pass:
        reason = "hue visible clues do not replay to one metadata-consistent rule"
        return {
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": reason,
            **semantic_metrics,
            **_verifier_contract_metrics(
                verifier_profile=profile,
                artifact_kind="video",
                component_scores={
                    "source_replay_score": 0.0,
                    "step_uniqueness_score": 0.0,
                    "media_contract_score": 0.0,
                    "visual_parseability_score": 0.0,
                    "state_transition_score": 0.0,
                    "preservation_score": 0.0,
                    "final_state_score": 0.0,
                },
                binary_correct=False,
                failure_reason=reason,
                started_at=started_at,
                hard_gate_failures=["independent_visible_clue_replay"],
            ),
        }, False
    board_names = replay["board_names"]
    missing = [tuple(replay["missing_cell"])]
    grid_bbox = structured.get("grid_bbox")
    correct_name = replay["correct_color_name"]
    try:
        board_size = int(structured.get("board_size"))
        cell_size = float(structured.get("cell_size"))
    except Exception:
        board_size = -1
        cell_size = 0.0
    normalized_board = (
        [[str(value).upper() for value in row] for row in board_names]
        if isinstance(board_names, list) and all(isinstance(row, list) for row in board_names)
        else None
    )
    if (
        normalized_board is None
        or _grid_dimensions(normalized_board) != (board_size, board_size)
        or missing is None
        or len(missing) != 1
        or correct_name not in _HUE_VIDEO_PALETTE
        or not isinstance(grid_bbox, (list, tuple))
        or len(grid_bbox) != 4
        or cell_size <= 0
    ):
        return {
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": "missing hue candidate-state metadata",
        }, False

    missing_row, missing_col = missing[0]
    frames = _video_frames_bgr(video_path)
    puzzle = _read_bgr(puzzle_path)
    fps = _video_fps(video_path)
    frame_states: List[str] = []
    fixed_mutations: List[List[Any]] = []
    unreadable_fixed_frames: List[int] = []
    fill_observations: List[dict[str, float]] = []
    static_mutation_fractions: List[float] = []
    cell_left = float(grid_bbox[0]) + missing_col * cell_size
    cell_top = float(grid_bbox[1]) + missing_row * cell_size
    source_reference = (
        np.rint(np.median(np.stack(frames[: min(2, len(frames))]), axis=0)).astype(np.uint8)
        if frames
        else puzzle
    )
    terminal_reference = (
        np.rint(np.median(np.stack(frames[-min(2, len(frames)):]), axis=0)).astype(np.uint8)
        if frames
        else puzzle
    )
    for frame_index, frame in enumerate(frames):
        fill_observations.append(
            _measure_hue_complete_footprint_fade(
                frame,
                source_bgr=source_reference,
                terminal_bgr=terminal_reference,
                grid_bbox=grid_bbox,
                cell_size=cell_size,
                row=missing_row,
                col=missing_col,
            )
        )
        static_mutation_fractions.append(
            static_mutation_fraction(
                puzzle,
                frame,
                codec_tolerant=True,
                allowed_change_bboxes=[
                    [cell_left, cell_top, cell_left + cell_size, cell_top + cell_size],
                ],
            )
        )
        fixed_readable = True
        for row in range(board_size):
            for col in range(board_size):
                if (row, col) == (missing_row, missing_col):
                    continue
                observed = _classify_hue_cell(
                    frame,
                    grid_bbox=grid_bbox,
                    cell_size=cell_size,
                    row=row,
                    col=col,
                    allow_blank=False,
                )
                if observed is None:
                    fixed_readable = False
                elif observed != normalized_board[row][col]:
                    fixed_mutations.append([frame_index, row, col, observed, normalized_board[row][col]])
        if not fixed_readable:
            unreadable_fixed_frames.append(frame_index)
        missing_state = _classify_hue_cell(
            frame,
            grid_bbox=grid_bbox,
            cell_size=cell_size,
            row=missing_row,
            col=missing_col,
            allow_blank=True,
        )
        if missing_state is None:
            missing_state = "UNREADABLE"
        elif missing_state not in {"BLANK", "TRANSITION", correct_name}:
            missing_state = f"WRONG:{missing_state}"
        frame_states.append(missing_state)

    compact_states: List[str] = []
    for state in frame_states:
        if not compact_states or compact_states[-1] != state:
            compact_states.append(state)
    frame_count = len(frames)
    media_contract_ok = bool(
        abs(fps - 16.0) <= 0.2
        and frame_count % 4 == 1
        and 1 <= frame_count <= 161
    )
    unreadable_fraction = len(unreadable_fixed_frames) / frame_count if frame_count else 1.0
    has_initial_hold = len(frame_states) >= 2 and frame_states[:2] == ["BLANK", "BLANK"]
    has_final_hold = len(frame_states) >= 2 and frame_states[-2:] == [correct_name, correct_name]
    transition_count = frame_states.count("TRANSITION")
    wrong_states = [state for state in frame_states if state.startswith("WRONG:")]
    allowed_order = {"BLANK": 0, "TRANSITION": 1, correct_name: 2}
    observed_order = [allowed_order[state] for state in frame_states if state in allowed_order]
    monotonic = all(left <= right for left, right in zip(observed_order, observed_order[1:]))
    has_unreadable_missing = "UNREADABLE" in frame_states
    fade_progress = [item["progress"] for item in fill_observations]
    fade_progress_monotonic = all(
        right + 0.06 >= left for left, right in zip(fade_progress, fade_progress[1:])
    )
    fade_witnesses = [
        frame_index
        for frame_index, item in enumerate(fill_observations)
        if (
            0.08 <= item["progress"] <= 0.92
            and item["linear_fit_error"] <= 0.20
            and item["footprint_support"] >= 0.72
            and item["pixel_progress_error"] <= 0.18
        )
    ]
    witness_progresses = [fade_progress[index] for index in fade_witnesses]
    witness_progress_span = (
        max(witness_progresses) - min(witness_progresses)
        if witness_progresses
        else 0.0
    )
    witness_progress_bins = {round(value, 1) for value in witness_progresses}
    fill_fade_ok = bool(
        len(fade_witnesses) >= 3
        and len(witness_progress_bins) >= 3
        and witness_progress_span >= 0.35
        and fade_progress_monotonic
        and max(fade_progress[: min(2, len(fade_progress))], default=1.0) <= 0.08
        and min(fade_progress[-min(2, len(fade_progress)):], default=0.0) >= 0.92
    )
    max_static_mutation_fraction = max(static_mutation_fractions, default=1.0)
    initial_static_mutation_fraction = max(static_mutation_fractions[:2], default=1.0)
    static_mutation_limit = 0.001
    full_scene_preservation_ok = bool(
        puzzle.shape == (512, 512, 3)
        and all(frame.shape == puzzle.shape for frame in frames)
        and initial_static_mutation_fraction <= static_mutation_limit
        and max_static_mutation_fraction <= static_mutation_limit
    )
    passed = (
        frame_count > 0
        and unreadable_fraction <= CANDIDATE_STATE_MAX_UNRECOGNIZED_FRACTION
        and not fixed_mutations
        and not wrong_states
        and not has_unreadable_missing
        and has_initial_hold
        and has_final_hold
        and transition_count >= 2
        and monotonic
        and fill_fade_ok
        and full_scene_preservation_ok
        and media_contract_ok
    )
    if not media_contract_ok:
        reason = "hue video must be 16fps, at most 10 seconds, and use 4N+1 frames"
        failures = ["media_contract"]
    elif unreadable_fraction > CANDIDATE_STATE_MAX_UNRECOGNIZED_FRACTION:
        reason = "hue fixed cells are not visually parseable in enough frames"
        failures = ["visual_parseability"]
    elif fixed_mutations:
        reason = "a non-missing hue cell changes"
        failures = ["fixed_cell_preservation"]
    elif wrong_states:
        reason = "the missing hue cell shows a stable wrong color"
        failures = ["correct_missing_color"]
    elif has_unreadable_missing:
        reason = "the missing hue cell is visually unparseable"
        failures = ["visual_parseability"]
    elif not has_initial_hold or not has_final_hold:
        reason = "hue initial or final hold is missing"
        failures = ["state_hold"]
    elif transition_count < 2:
        reason = "hue fill hard-cuts without a visible transition"
        failures = ["continuous_fill_order"]
    elif not monotonic:
        reason = "hue fill regresses or changes in an illegal order"
        failures = ["continuous_fill_order"]
    elif not fill_fade_ok:
        reason = "hue fill must fade uniformly over its complete fixed terminal footprint"
        failures = ["complete_footprint_local_fade"]
    elif not full_scene_preservation_ok:
        reason = "hue grid, labels, legend, border, background, or camera changed outside the missing cell"
        failures = ["full_scene_preservation"]
    else:
        reason = "ok"
        failures = []
    return {
        "video_candidate_state_check_pass": passed,
        "video_candidate_state_reason": reason,
        "video_candidate_state_verifier": "hue_fixed_palette_complete_footprint_fade_parser",
        "video_candidate_frame_count": frame_count,
        "video_candidate_unrecognized_fraction": unreadable_fraction,
        "video_candidate_max_unrecognized_fraction": CANDIDATE_STATE_MAX_UNRECOGNIZED_FRACTION,
        "video_candidate_unreadable_fixed_fraction": unreadable_fraction,
        "video_hue_frame_states": frame_states,
        "video_hue_compact_states": compact_states,
        "video_hue_transition_frame_count": transition_count,
        "video_hue_has_initial_hold": has_initial_hold,
        "video_hue_has_final_hold": has_final_hold,
        "video_hue_monotonic_fill": monotonic,
        "video_hue_opacity_progress": fade_progress,
        "video_hue_opacity_progress_monotonic": fade_progress_monotonic,
        "video_hue_complete_footprint_fade_pass": fill_fade_ok,
        "video_hue_fade_witness_frames": fade_witnesses,
        "video_hue_fade_witness_progresses": witness_progresses,
        "video_hue_fade_witness_progress_span": witness_progress_span,
        "video_hue_fade_witness_progress_bin_count": len(witness_progress_bins),
        "video_hue_fill_observations": fill_observations,
        "video_hue_initial_static_mutation_fraction": initial_static_mutation_fraction,
        "video_hue_max_static_mutation_fraction": max_static_mutation_fraction,
        "video_hue_max_allowed_static_mutation_fraction": static_mutation_limit,
        "video_hue_full_scene_preservation_pass": full_scene_preservation_ok,
        "video_hue_fixed_mutations": fixed_mutations[:20],
        "video_fps": fps,
        "video_media_contract_pass": media_contract_ok,
        **_verifier_contract_metrics(
            verifier_profile=profile,
            artifact_kind="video",
            component_scores={
                "source_replay_score": 1.0,
                "step_uniqueness_score": 1.0,
                "media_contract_score": 1.0 if media_contract_ok else 0.0,
                "visual_parseability_score": max(0.0, 1.0 - unreadable_fraction),
                "state_transition_score": 1.0 if has_initial_hold and has_final_hold and transition_count >= 2 and monotonic and fill_fade_ok else 0.0,
                "preservation_score": 1.0 if not fixed_mutations and full_scene_preservation_ok else 0.0,
                "final_state_score": 1.0 if has_final_hold and not wrong_states and not has_unreadable_missing else 0.0,
            },
            binary_correct=passed,
            failure_reason=reason,
            started_at=started_at,
            hard_gate_failures=failures,
        ),
    }, passed

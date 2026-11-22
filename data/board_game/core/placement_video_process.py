"""Source-derived process gates for Board Game placement videos.

The generic Board Game verifier proves that source-rendered semantic states
occur in order.  Placement programs need a stricter, local contract: one
complete glyph or stone appears at the named cell, and Go captures happen only
after that placement has committed.  This module checks those observable
properties without reading authored answers, generated reference videos, or
candidate chain-of-thought.

``replay.source_payload`` must contain the source-recomputed fields below::

    {
        "board_size": 7,
        "placement_process_steps": [
            {
                "target": [row, column],       # internal zero-based cell
                "placed_stage": 1,             # index in replay.video_states
                "final_stage": 2,              # same as placed_stage if no capture
                "captured_cells": [[...], ...],
            },
        ],
        "terminal_overlay": {                # optional source-derived final annotation
            "kind": "winning_line",
            "cells": [[row, column], ...],
            "stage": 4,
        },
    }

The explicit stage indices prevent this process gate from inferring semantics
from pixels.  They are rebuilt by ``source_replay`` from the supplied initial
state and rules, not copied from a canonical answer or reference artifact.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from data.common.verifiers.keyframes import visual_delta
from data.common.verifiers.region_gates import STATIC_REGION_FLOOR, static_region_errors


_SUPPORTED_TASKS = {
    "tic_tac_toe_rule_simulator",
    "gomoku_rule_simulator",
    "go_rule_simulator",
    "go_max_capture_search",
}
_STATE_DYNAMIC_TOLERANCE = 0.055
_STATE_GLOBAL_TOLERANCE = 0.018


def _rgb(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        array = np.asarray(value.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("placement video frames must be RGB images")
    return array


def _mae(left: np.ndarray, right: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    difference = visual_delta(left, right) / 255.0
    if mask is not None and bool(np.any(mask)):
        difference = difference[mask]
    return float(np.mean(difference))


def _pixel_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.max(np.abs(left.astype(np.int16) - right.astype(np.int16)), axis=2)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    try:
        import cv2

        width = radius * 2 + 1
        return cv2.dilate(
            mask.astype(np.uint8),
            np.ones((width, width), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
    except Exception:
        padded = np.pad(mask.astype(bool), radius)
        result = np.zeros_like(mask, dtype=bool)
        height, width = mask.shape
        for dy in range(radius * 2 + 1):
            for dx in range(radius * 2 + 1):
                result |= padded[dy : dy + height, dx : dx + width]
        return result


def _dynamic_mask(states: Sequence[np.ndarray]) -> np.ndarray:
    stack = np.stack(states, axis=0).astype(np.int16)
    # Compare states at the same colour channel.  Reducing RGB before taking
    # the temporal difference would incorrectly classify a static coloured
    # background as process motion.
    mask = np.max(np.abs(stack - stack[0:1]), axis=(0, 3)) >= 7
    return _dilate(mask, 3)


def _cell_centre(task_name: str, board_size: int, cell: Tuple[int, int]) -> Tuple[float, float]:
    """Derive the pixel centre from the one authoritative renderer geometry."""

    row, column = cell
    if not 0 <= row < board_size or not 0 <= column < board_size:
        raise ValueError("placement target lies outside the source board")
    if task_name == "tic_tac_toe_rule_simulator":
        from data.board_game.core.tic_tac_toe import TicTacToeRenderer

        if board_size != 3:
            raise ValueError("Tic-Tac-Toe placement replay must use a 3 by 3 board")
        x0, y0, x1, y1 = TicTacToeRenderer().cell_box((row, column))
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    from data.board_game.core.intersection_games import NumericIntersectionRenderer

    if board_size not in {7, 9}:
        raise ValueError("intersection placement replay must use a 7 or 9 board")
    x, y = NumericIntersectionRenderer(board_size).intersection_xy((row, column))
    return float(x), float(y)


def _cell_region(
    shape: Tuple[int, int],
    task_name: str,
    board_size: int,
    cell: Tuple[int, int],
) -> np.ndarray:
    height, width = shape
    cx, cy = _cell_centre(task_name, board_size, cell)
    yy, xx = np.ogrid[:height, :width]
    if task_name == "tic_tac_toe_rule_simulator":
        from data.board_game.core.tic_tac_toe import TicTacToeRenderer

        half = TicTacToeRenderer().cell / 2.0 - 5.0
        return (np.abs(xx - cx) <= half) & (np.abs(yy - cy) <= half)
    from data.board_game.core.intersection_games import NumericIntersectionRenderer

    radius = float(NumericIntersectionRenderer(board_size).stone_radius) + 1.0
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius + 5.0) ** 2


def _spatial_chunks(
    support: np.ndarray,
    centre: Tuple[float, float],
    *,
    angular: bool,
) -> List[np.ndarray]:
    height, width = support.shape
    yy, xx = np.ogrid[:height, :width]
    cx, cy = centre
    chunks: List[np.ndarray] = []
    if angular:
        angles = (np.arctan2(yy - cy, xx - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        for index in range(8):
            low = index * 2.0 * np.pi / 8.0
            high = (index + 1) * 2.0 * np.pi / 8.0
            chunk = support & (angles >= low) & (angles < high)
            if int(np.sum(chunk)) >= 5:
                chunks.append(chunk)
    else:
        for horizontal, vertical in (
            (xx < cx, yy < cy),
            (xx >= cx, yy < cy),
            (xx < cx, yy >= cy),
            (xx >= cx, yy >= cy),
        ):
            chunk = support & horizontal & vertical
            if int(np.sum(chunk)) >= 5:
                chunks.append(chunk)
    return chunks


def _projection(frame: np.ndarray, start: np.ndarray, end: np.ndarray, mask: np.ndarray) -> float:
    if not bool(np.any(mask)):
        return 0.0
    origin = start[mask].astype(np.float32)
    direction = end[mask].astype(np.float32) - origin
    current = frame[mask].astype(np.float32) - origin
    denominator = float(np.sum(direction * direction))
    if denominator <= 1e-6:
        return 0.0
    return float(np.clip(np.sum(current * direction) / denominator, 0.0, 1.0))


def _transition_metrics(
    frames: Sequence[np.ndarray],
    start: np.ndarray,
    end: np.ndarray,
    frame_indices: Sequence[int],
    cell_masks: Sequence[np.ndarray],
    cell_centres: Sequence[Tuple[float, float]],
    *,
    angular: bool,
    sync_cells: bool,
) -> Tuple[Dict[str, Any], bool]:
    endpoint_support = _pixel_delta(start, end) >= 7
    semantic_region = np.zeros(endpoint_support.shape, dtype=bool)
    per_cell_support: List[np.ndarray] = []
    per_cell_chunks: List[List[np.ndarray]] = []
    for region, centre in zip(cell_masks, cell_centres):
        support = endpoint_support & region
        if int(np.sum(support)) < 20:
            return {"error": "a source-declared cell has no rendered endpoint support"}, False
        semantic_region |= region
        per_cell_support.append(support)
        per_cell_chunks.append(_spatial_chunks(support, centre, angular=angular))
    allowed = _dilate(endpoint_support & semantic_region, 4)
    unexpected_endpoint = int(np.sum(endpoint_support & ~allowed))

    progress_rows: List[List[float]] = []
    chunk_spreads: List[List[float]] = []
    outside_errors: List[float] = []
    outside_changed: List[int] = []
    outside_region_errors: List[float] = []
    path_residuals: List[float] = []
    for frame_index in frame_indices:
        frame = frames[frame_index]
        cell_progress = [
            _projection(frame, start, end, support) for support in per_cell_support
        ]
        progress_rows.append(cell_progress)
        spread_row: List[float] = []
        for chunks in per_cell_chunks:
            values = [_projection(frame, start, end, chunk) for chunk in chunks]
            spread_row.append(max(values) - min(values) if values else 1.0)
        chunk_spreads.append(spread_row)
        outside_errors.append(_mae(frame, start, ~allowed))
        outside_changed.append(int(np.sum((_pixel_delta(frame, start) >= 15) & ~allowed)))
        outside_region_errors.append(static_region_errors(frame, start, ~allowed)[0])

        # A legal reveal/removal follows the source-derived local opacity
        # segment.  A foreign shape, sliding object, or temporary mutation has
        # a large residual even when its global mean is small.
        union_support = np.logical_or.reduce(per_cell_support)
        depth = _projection(frame, start, end, union_support)
        predicted = start.astype(np.float32).copy()
        predicted[union_support] = (
            start[union_support].astype(np.float32) * (1.0 - depth)
            + end[union_support].astype(np.float32) * depth
        )
        residual = visual_delta(frame, predicted) / 255.0
        path_residuals.append(float(np.mean(residual[union_support])))

    mean_progress = [float(np.mean(row)) for row in progress_rows]
    intermediate = [value for value in mean_progress if 0.035 < value < 0.965]
    monotonic = all(
        current + 0.045 >= previous
        for previous, current in zip(mean_progress, mean_progress[1:])
    )
    geometry_sync = bool(
        per_cell_chunks
        and all(len(chunks) >= (6 if angular else 4) for chunks in per_cell_chunks)
        and all(max(row, default=1.0) <= 0.20 for row in chunk_spreads)
    )
    cross_cell_sync = bool(
        not sync_cells
        or all(max(row) - min(row) <= 0.16 for row in progress_rows if row)
    )
    local_only = bool(
        unexpected_endpoint == 0
        and max(outside_errors, default=1.0) <= 0.0045
        and max(outside_region_errors, default=1.0) <= STATIC_REGION_FLOOR
    )
    continuous = bool(
        len(intermediate) >= 3
        and len({round(value, 2) for value in intermediate}) >= 3
        and monotonic
        and mean_progress
        and mean_progress[-1] >= 0.94
    )
    follows_local_path = max(path_residuals, default=1.0) <= 0.030
    passed = bool(geometry_sync and cross_cell_sync and local_only and continuous and follows_local_path)
    return {
        "frame_range": [int(frame_indices[0]), int(frame_indices[-1])] if frame_indices else None,
        "cell_progress": progress_rows,
        "cell_chunk_progress_spreads": chunk_spreads,
        "progress_monotonic_score": float(monotonic),
        "intermediate_progress_count": len(intermediate),
        "distinct_intermediate_progress_count": len({round(value, 2) for value in intermediate}),
        "fixed_complete_geometry_score": float(geometry_sync),
        "cross_cell_sync_score": float(cross_cell_sync),
        "outside_region_errors": outside_errors,
        "outside_region_changed_pixels": outside_changed,
        "outside_region_local_errors": outside_region_errors,
        "local_only_score": float(local_only),
        "local_path_residuals": path_residuals,
        "local_opacity_path_score": float(follows_local_path),
        "transition_continuity_score": float(continuous),
        "pass": passed,
    }, passed


def _terminal_overlay_metrics(
    frames: Sequence[np.ndarray],
    start: np.ndarray,
    end: np.ndarray,
    frame_indices: Sequence[int],
    *,
    task_name: str,
    board_size: int,
    overlay: Mapping[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    cells = list(overlay["cells"])
    support = _pixel_delta(start, end) >= 7
    cell_supports = [
        support & _cell_region(start.shape[:2], task_name, board_size, cell)
        for cell in cells
    ]
    if not bool(np.any(support)) or any(int(np.sum(mask)) < 20 for mask in cell_supports):
        return {"error": "terminal overlay lacks source-rendered support for every winning cell"}, False

    allowed = _dilate(support, 4)
    progress_rows: List[List[float]] = []
    overall_progress: List[float] = []
    outside_errors: List[float] = []
    outside_changed: List[int] = []
    outside_region_errors: List[float] = []
    residuals: List[float] = []
    for frame_index in frame_indices:
        frame = frames[frame_index]
        row = [_projection(frame, start, end, mask) for mask in cell_supports]
        progress_rows.append(row)
        depth = _projection(frame, start, end, support)
        overall_progress.append(depth)
        outside_errors.append(_mae(frame, start, ~allowed))
        outside_changed.append(int(np.sum((_pixel_delta(frame, start) >= 15) & ~allowed)))
        outside_region_errors.append(static_region_errors(frame, start, ~allowed)[0])
        predicted = start.astype(np.float32).copy()
        predicted[support] = (
            start[support].astype(np.float32) * (1.0 - depth)
            + end[support].astype(np.float32) * depth
        )
        residual = visual_delta(frame, predicted) / 255.0
        residuals.append(float(np.mean(residual[support])))

    intermediate = [value for value in overall_progress if 0.035 < value < 0.965]
    monotonic = all(
        current + 0.045 >= previous
        for previous, current in zip(overall_progress, overall_progress[1:])
    )
    synchronous = bool(
        progress_rows
        and all(max(row) - min(row) <= 0.18 for row in progress_rows if row)
    )
    local_only = bool(
        max(outside_errors, default=1.0) <= 0.0045
        and max(outside_region_errors, default=1.0) <= STATIC_REGION_FLOOR
    )
    continuous = bool(
        len(intermediate) >= 3
        and len({round(value, 2) for value in intermediate}) >= 3
        and monotonic
        and overall_progress[-1] >= 0.94
    )
    follows_local_path = max(residuals, default=1.0) <= 0.030
    passed = bool(synchronous and local_only and continuous and follows_local_path)
    return {
        "kind": str(overlay["kind"]),
        "cells": [list(cell) for cell in cells],
        "source_stage_transition": [int(overlay["stage"]) - 1, int(overlay["stage"])],
        "frame_range": [int(frame_indices[0]), int(frame_indices[-1])] if frame_indices else None,
        "cell_progress": progress_rows,
        "overall_progress": overall_progress,
        "winning_cells_sync_score": float(synchronous),
        "progress_monotonic_score": float(monotonic),
        "intermediate_progress_count": len(intermediate),
        "distinct_intermediate_progress_count": len({round(value, 2) for value in intermediate}),
        "outside_region_errors": outside_errors,
        "outside_region_changed_pixels": outside_changed,
        "outside_region_local_errors": outside_region_errors,
        "local_only_score": float(local_only),
        "local_path_residuals": residuals,
        "local_opacity_path_score": float(follows_local_path),
        "transition_continuity_score": float(continuous),
        "pass": passed,
    }, passed


def _parse_cell(raw: Any, *, board_size: int, name: str) -> Tuple[int, int]:
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 2
        or isinstance(raw[0], bool)
        or isinstance(raw[1], bool)
    ):
        raise ValueError(f"{name} must be an internal row,column pair")
    cell = int(raw[0]), int(raw[1])
    if not 0 <= cell[0] < board_size or not 0 <= cell[1] < board_size:
        raise ValueError(f"{name} lies outside the source board")
    return cell


def _parse_steps(
    payload: Mapping[str, Any],
    state_count: int,
) -> Tuple[int, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    board_size = int(payload.get("board_size", 0))
    raw_steps = payload.get("placement_process_steps")
    if board_size <= 0 or not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("source replay lacks placement_process_steps or board_size")
    steps: List[Dict[str, Any]] = []
    previous_final = 0
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            raise ValueError("every placement process step must be a mapping")
        target = _parse_cell(raw.get("target"), board_size=board_size, name="placement target")
        placed_stage = int(raw.get("placed_stage", -1))
        final_stage = int(raw.get("final_stage", -1))
        captures_raw = raw.get("captured_cells", [])
        if not isinstance(captures_raw, list):
            raise ValueError("captured_cells must be a list")
        captures = [
            _parse_cell(cell, board_size=board_size, name="captured cell")
            for cell in captures_raw
        ]
        if placed_stage != previous_final + 1:
            raise ValueError("placement stages must cover source states without gaps")
        expected_final = placed_stage + (1 if captures else 0)
        if final_stage != expected_final or not 0 < placed_stage < state_count:
            raise ValueError("capture and placement stage indices disagree with source semantics")
        if len(set(captures)) != len(captures) or target in captures:
            raise ValueError("captured cells must be distinct from the placement target")
        steps.append(
            {
                "index": index,
                "target": target,
                "placed_stage": placed_stage,
                "final_stage": final_stage,
                "captured_cells": captures,
                "previous_final_stage": previous_final,
            }
        )
        previous_final = final_stage
    terminal_overlay: Optional[Dict[str, Any]] = None
    raw_overlay = payload.get("terminal_overlay")
    if raw_overlay is not None:
        if not isinstance(raw_overlay, Mapping) or set(raw_overlay) != {"kind", "cells", "stage"}:
            raise ValueError("terminal_overlay must contain only kind, cells, and stage")
        kind = str(raw_overlay["kind"])
        expected_kind = {
            "tic_tac_toe_rule_simulator": "winning_cell_borders",
            "gomoku_rule_simulator": "winning_line",
        }.get(str(payload.get("task_name", "")))
        # The replay task name is not duplicated in source_payload.  Infer the
        # only two legal annotation forms from the board size.
        if expected_kind is None:
            expected_kind = {3: "winning_cell_borders", 9: "winning_line"}.get(board_size)
        expected_count = {"winning_cell_borders": 3, "winning_line": 5}.get(kind)
        if kind != expected_kind or expected_count is None:
            raise ValueError("terminal_overlay kind is not valid for this placement board")
        cells_raw = raw_overlay["cells"]
        if not isinstance(cells_raw, list):
            raise ValueError("terminal_overlay cells must be a list")
        cells = [
            _parse_cell(cell, board_size=board_size, name="terminal overlay cell")
            for cell in cells_raw
        ]
        stage = int(raw_overlay["stage"])
        if len(cells) != expected_count or len(set(cells)) != expected_count:
            raise ValueError("terminal_overlay must identify every winning cell exactly once")
        if stage != state_count - 1 or previous_final != stage - 1:
            raise ValueError("terminal_overlay must be the sole state after the final placement commit")
        terminal_overlay = {"kind": kind, "cells": cells, "stage": stage}
    elif previous_final != state_count - 1:
        raise ValueError("placement steps do not terminate at the source terminal state")
    return board_size, steps, terminal_overlay


def _stage_matches(
    frames: Sequence[np.ndarray],
    states: Sequence[np.ndarray],
) -> Tuple[List[List[bool]], np.ndarray]:
    mask = _dynamic_mask(states)
    matches: List[List[bool]] = [[False] * len(frames) for _ in states]
    for frame_index, frame in enumerate(frames):
        distances = [(_mae(frame, state, mask), _mae(frame, state)) for state in states]
        label = min(
            range(len(states)),
            key=lambda index: distances[index][0] / _STATE_DYNAMIC_TOLERANCE
            + distances[index][1] / _STATE_GLOBAL_TOLERANCE,
        )
        dynamic, global_distance = distances[label]
        if dynamic <= _STATE_DYNAMIC_TOLERANCE and global_distance <= _STATE_GLOBAL_TOLERANCE:
            matches[label][frame_index] = True
    return matches, mask


def verify_placement_process(frames: Sequence[Any], replay: Any) -> Tuple[Dict[str, Any], bool]:
    """Verify source-derived placement and capture process semantics.

    The function is deliberately independent of ``verifier.py`` so the main
    route can call it without an import cycle.  ``frames`` are decoded RGB
    candidate-video frames.  ``replay.video_states`` and the process-step
    payload must have been independently rebuilt from visible source input.
    """

    task_name = str(getattr(replay, "task_name", ""))
    base_metrics: Dict[str, Any] = {
        "video_placement_process_verifier": "source_local_commit_v1",
        "video_placement_reference_media_used": False,
        "video_placement_task": task_name,
    }
    if task_name not in _SUPPORTED_TASKS:
        return {**base_metrics, "video_placement_error": "unsupported placement task"}, False
    try:
        candidate = [_rgb(frame) for frame in frames]
        states = [_rgb(image) for image in getattr(replay, "video_states")]
        if len(candidate) < 5 or len(states) < 2:
            raise ValueError("placement process needs candidate frames and at least two source states")
        if any(frame.shape != candidate[0].shape for frame in candidate + states):
            raise ValueError("candidate frames and source states must share one RGB shape")
        payload = getattr(replay, "source_payload", {})
        if not isinstance(payload, Mapping):
            raise ValueError("source_payload must be a mapping")
        board_size, steps, terminal_overlay = _parse_steps(payload, len(states))
        matches, dynamic_mask = _stage_matches(candidate, states)

        # Anchor the first stable occurrence of each source state in strict
        # order.  Every frame between anchors is then inspected by the exact
        # source-declared transition rather than silently ignored.
        anchors: List[int] = []
        cursor = 0
        for stage_index, stage_matches in enumerate(matches):
            # A nearly finished fade can look stable when averaged over all
            # earlier placements. Test cadence on this commit's own footprint.
            local_states = states[max(0,stage_index-1):stage_index+1] if stage_index else states[:2]
            commit_mask = _dynamic_mask(local_states)
            anchor = next(
                (
                    index
                    for index in range(cursor, len(candidate) - 1)
                    if stage_matches[index]
                    and stage_matches[index + 1]
                    and (
                        _projection(candidate[index], states[stage_index - 1], states[stage_index], commit_mask) >= 0.94
                        if stage_index else
                        _projection(candidate[index], states[0], states[1], commit_mask) <= 0.06
                    )
                    and _mae(candidate[index], candidate[index + 1], commit_mask) <= 0.003
                    and _mae(candidate[index], candidate[index + 1]) <= 0.003
                ),
                None,
            )
            if anchor is None:
                raise ValueError(f"candidate never commits source stage {stage_index}")
            anchors.append(anchor)
            cursor = anchor + 1
        if any(left >= right for left, right in zip(anchors, anchors[1:])):
            raise ValueError("source placement stages are not committed in strict order")

        placement_metrics: List[Dict[str, Any]] = []
        capture_metrics: List[Dict[str, Any]] = []
        all_pass = True
        for step in steps:
            previous_stage = int(step["previous_final_stage"])
            placed_stage = int(step["placed_stage"])
            target = step["target"]
            target_mask = _cell_region(candidate[0].shape[:2], task_name, board_size, target)
            target_centre = _cell_centre(task_name, board_size, target)
            transition_indices = list(range(anchors[previous_stage], anchors[placed_stage] + 1))
            metrics, passed = _transition_metrics(
                candidate,
                states[previous_stage],
                states[placed_stage],
                transition_indices,
                [target_mask],
                [target_centre],
                angular=task_name != "tic_tac_toe_rule_simulator",
                sync_cells=False,
            )
            metrics.update(
                {
                    "step": int(step["index"]) + 1,
                    "target": list(target),
                    "source_stage_transition": [previous_stage, placed_stage],
                }
            )
            placement_metrics.append(metrics)
            all_pass = bool(all_pass and passed)

            captures = step["captured_cells"]
            final_stage = int(step["final_stage"])
            if captures:
                capture_indices = list(range(anchors[placed_stage], anchors[final_stage] + 1))
                capture_masks = [
                    _cell_region(candidate[0].shape[:2], task_name, board_size, cell)
                    for cell in captures
                ]
                capture_centres = [
                    _cell_centre(task_name, board_size, cell) for cell in captures
                ]
                metrics, passed = _transition_metrics(
                    candidate,
                    states[placed_stage],
                    states[final_stage],
                    capture_indices,
                    capture_masks,
                    capture_centres,
                    angular=True,
                    sync_cells=True,
                )
                metrics.update(
                    {
                        "step": int(step["index"]) + 1,
                        "captured_cells": [list(cell) for cell in captures],
                        "source_stage_transition": [placed_stage, final_stage],
                    }
                )
                capture_metrics.append(metrics)
                all_pass = bool(all_pass and passed)

        overlay_metrics: Dict[str, Any] = {}
        overlay_pass = terminal_overlay is None
        if terminal_overlay is not None:
            overlay_stage = int(terminal_overlay["stage"])
            overlay_indices = list(range(anchors[overlay_stage - 1], anchors[overlay_stage] + 1))
            overlay_metrics, overlay_pass = _terminal_overlay_metrics(
                candidate,
                states[overlay_stage - 1],
                states[overlay_stage],
                overlay_indices,
                task_name=task_name,
                board_size=board_size,
                overlay=terminal_overlay,
            )
            all_pass = bool(all_pass and overlay_pass)

        # A suffix mutation after the first terminal commit is illegal even if
        # a later frame repairs the board.  This is the observable first-stop
        # guarantee for Tic-Tac-Toe and Gomoku and a general anti-flicker gate.
        terminal_anchor = anchors[-1]
        terminal = states[-1]
        terminal_errors = [_mae(frame, terminal) for frame in candidate[terminal_anchor:]]
        terminal_dynamic_errors = [
            _mae(frame, terminal, dynamic_mask) for frame in candidate[terminal_anchor:]
        ]
        static_mask = ~_dilate(dynamic_mask, 2)
        terminal_static_errors = [
            _mae(frame, terminal, static_mask) for frame in candidate[terminal_anchor:]
        ]
        terminal_static_changed = [
            int(np.sum((_pixel_delta(frame, terminal) >= 15) & static_mask))
            for frame in candidate[terminal_anchor:]
        ]
        terminal_static_region_errors = [
            static_region_errors(frame, terminal, static_mask)[0]
            for frame in candidate[terminal_anchor:]
        ]
        terminal_stable = bool(
            len(terminal_errors) >= 2
            and max(terminal_errors, default=1.0) <= _STATE_GLOBAL_TOLERANCE
            and max(terminal_dynamic_errors, default=1.0) <= _STATE_DYNAMIC_TOLERANCE
            and max(terminal_static_errors, default=1.0) <= 0.0045
            and max(terminal_static_region_errors, default=1.0) <= STATIC_REGION_FLOOR
        )
        all_pass = bool(all_pass and terminal_stable)
        metrics = {
            **base_metrics,
            "video_placement_board_size": board_size,
            "video_placement_source_stage_count": len(states),
            "video_placement_stage_anchors": anchors,
            "video_placement_steps": placement_metrics,
            "video_placement_capture_steps": capture_metrics,
            "video_placement_terminal_overlay": overlay_metrics,
            "video_placement_terminal_overlay_score": float(overlay_pass),
            "video_placement_all_local_reveals_score": float(
                bool(placement_metrics) and all(item.get("pass") for item in placement_metrics)
            ),
            "video_placement_all_capture_commits_score": float(
                all(item.get("pass") for item in capture_metrics)
            ),
            "video_placement_terminal_tail_errors": terminal_errors,
            "video_placement_terminal_tail_dynamic_errors": terminal_dynamic_errors,
            "video_placement_terminal_tail_static_errors": terminal_static_errors,
            "video_placement_terminal_tail_static_changed_pixels": terminal_static_changed,
            "video_placement_terminal_tail_static_region_errors": terminal_static_region_errors,
            "video_placement_no_suffix_mutation_score": float(terminal_stable),
            "video_placement_process_pass": all_pass,
        }
        return metrics, all_pass
    except Exception as exc:
        return {**base_metrics, "video_placement_error": str(exc), "video_placement_process_pass": False}, False


__all__ = ["verify_placement_process"]

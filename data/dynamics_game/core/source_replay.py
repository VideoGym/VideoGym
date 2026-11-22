"""Independent semantic replay and source rendering for the original Dynamics tasks.

The canonical record deliberately contains authored traces and reference media
for data construction.  None of those derived fields may define candidate
correctness.  This module starts from the visible initial state and explicit
operation parameters, replays the rule, and renders the resulting states with
the task's renderer.  The renderer is only a projection of independently
replayed semantics; saved solution images and video keyframes are never read.
"""

from __future__ import annotations

import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


Grid = List[List[int]]
Cell = Tuple[int, int]
_RENDER_ROOT = Path(tempfile.gettempdir()) / "videogym-dynamics-source-render"


@dataclass(frozen=True)
class SourceReplay:
    """Source-derived formal answer and native semantic process states."""

    task_name: str
    text_kind: str
    expected_payload: Any
    scenes: Tuple[Image.Image, ...]
    metrics: Dict[str, Any]
    # TI2V transition model imported from the task's
    # generator module so prompt timing and verifier gating share one source.
    timing: str = "continuous"
    # Source-rendered transient poses bound motion between semantic states.
    motion_scenes: Tuple[Image.Image, ...] = ()


def _grid(value: Any) -> Optional[Grid]:
    if not isinstance(value, list) or not value:
        return None
    try:
        result = [[int(cell) for cell in row] for row in value]
    except Exception:
        return None
    width = len(result[0]) if result else 0
    if width <= 0 or any(len(row) != width for row in result):
        return None
    return result


def _cell(value: Any) -> Cell:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"invalid cell: {value!r}")
    return int(value[0]), int(value[1])


def _candidate_option(
    source: Dict[str, Any],
    expected_grid: Grid,
    *,
    expected_direction: Optional[str] = None,
) -> Tuple[str, List[int]]:
    raw = source.get("candidate_state_grids")
    candidates = [_grid(item) for item in raw] if isinstance(raw, list) else []
    directions = source.get("candidate_directions")
    matches: List[int] = []
    for index, candidate in enumerate(candidates):
        if candidate != expected_grid:
            continue
        if expected_direction is not None:
            if not isinstance(directions, list) or index >= len(directions):
                continue
            if str(directions[index]) != expected_direction:
                continue
        matches.append(index)
    if len(candidates) != 4 or any(item is None for item in candidates) or len(matches) != 1:
        raise ValueError(
            "visible candidate semantics do not contain exactly one source-replayed result"
        )
    return "ABCD"[matches[0]], matches


def _block(source: Dict[str, Any]) -> SourceReplay:
    from data.dynamics_game.block_blast_clear_predict.generator import (
        BlockBlastClearPredictGenerator,
    )

    renderer = BlockBlastClearPredictGenerator(_RENDER_ROOT, seed=0, video=False)
    initial = _grid(source.get("board") or source.get("initial_grid"))
    shape = [_cell(item) for item in source.get("shape", [])]
    anchor = _cell(source.get("anchor"))
    if initial is None or len(initial) != 8 or any(len(row) != 8 for row in initial) or not shape:
        raise ValueError("invalid Block Blast source state")
    placed_piece = [row[:] for row in initial]
    locked = [row[:] for row in initial]
    for dr, dc in shape:
        row, col = anchor[0] + dr, anchor[1] + dc
        if not (0 <= row < 8 and 0 <= col < 8) or initial[row][col] != 0:
            raise ValueError("Block Blast piece does not fit the visible empty landing cells")
        placed_piece[row][col] = 2
        locked[row][col] = 1
    full_rows = [row for row in range(8) if all(locked[row][col] == 1 for col in range(8))]
    full_cols = [col for col in range(8) if all(locked[row][col] == 1 for row in range(8))]
    if not full_rows and not full_cols:
        raise ValueError("Block Blast source placement clears no line")
    final = [row[:] for row in locked]
    clearing = {(row, col) for row in full_rows for col in range(8)} | {
        (row, col) for col in full_cols for row in range(8)
    }
    for row, col in clearing:
        final[row][col] = 0
    option, matches = _candidate_option(source, final)
    scenes = (
        renderer._render(
            initial,
            shape=shape,
            anchor=anchor,
            show_reference_piece=True,
            show_target_guide=True,
        ),
        renderer._render(placed_piece, shape=shape, anchor=anchor, show_reference_piece=False),
        renderer._render(
            locked,
            shape=shape,
            anchor=anchor,
            show_reference_piece=False,
            clearing_cells=clearing,
            clearing_scale=0.68,
        ),
        renderer._render(final, shape=shape, anchor=anchor, show_reference_piece=False),
    )
    preview_cell, left, top, _, _ = renderer._preview_geometry(shape)
    target_left = renderer.layout.origin_x + anchor[1] * renderer.layout.cell
    target_top = renderer.layout.origin_y + anchor[0] * renderer.layout.cell
    motion_scenes = tuple(
        renderer._render_moving_piece(
            initial, shape, anchor,
            (left + (target_left - left) * t, top + (target_top - top) * t),
            preview_cell + (renderer.layout.cell - preview_cell) * t,
        )
        for t in (0.0, 0.5, 1.0)
    )
    return SourceReplay(
        "block_blast_clear_predict",
        "option",
        option,
        scenes,
        {
            "source_replay_kind": "piece_place_then_simultaneous_line_clear_v1",
            "source_piece_card_box": renderer._piece_card_box(),
            "source_replayed_full_rows": full_rows,
            "source_replayed_full_cols": full_cols,
            "source_candidate_match_indices": matches,
        },
        motion_scenes=motion_scenes,
    )


def _breakout(source: Dict[str, Any]) -> SourceReplay:
    from data.dynamics_game.breakout_brick_clear_predict.generator import (
        BreakoutBrickClearPredictGenerator,
        _encode_bricks,
    )

    renderer = BreakoutBrickClearPredictGenerator(_RENDER_ROOT, seed=0, video=False)
    raw_bricks = source.get("bricks")
    if not isinstance(raw_bricks, list):
        raise ValueError("invalid Breakout brick source")
    bricks: Dict[Cell, int] = {}
    for item in raw_bricks:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError("invalid Breakout brick entry")
        bricks[(int(item[0]), int(item[1]))] = int(item[2])
    start_col = int(source.get("start_col"))
    wall = str(source.get("wall"))
    bounce = _cell(source.get("bounce"))
    route_end = _cell(source.get("hit"))
    start = (renderer.rows - 1, start_col)
    points = [
        renderer._start_point(start_col),
        renderer._bounce_point(wall, bounce),
        renderer._hit_point(route_end),
    ]
    logical_hit = renderer._first_brick_on_path(bricks, [(start, bounce), (bounce, route_end)])
    visual_hit = renderer._first_brick_on_pixel_path(bricks, points)
    if logical_hit is None or logical_hit != visual_hit:
        raise ValueError("Breakout logical and visible first-contact replay disagree")
    final = renderer._apply(bricks, logical_hit)
    encoded_final = _encode_bricks(final, renderer.rows, renderer.cols)
    option, matches = _candidate_option(source, encoded_final)
    start_point, bounce_point, hit_point = points
    scenes = (
        renderer._render_image(
            bricks,
            start_col=start_col,
            wall=wall,
            bounce=bounce,
            hit=route_end,
            ball_position=start_point,
        ),
        renderer._render_image(
            bricks,
            start_col=start_col,
            wall=wall,
            bounce=bounce,
            hit=route_end,
            ball_position=bounce_point,
            travelled_points=[start_point, bounce_point],
        ),
        renderer._render_image(
            bricks,
            start_col=start_col,
            wall=wall,
            bounce=bounce,
            hit=route_end,
            ball_position=hit_point,
            travelled_points=points,
        ),
        renderer._render_image(
            final,
            start_col=start_col,
            wall=wall,
            bounce=bounce,
            hit=route_end,
            ball_position=hit_point,
            travelled_points=points,
        ),
    )
    return SourceReplay(
        "breakout_brick_clear_predict",
        "option",
        option,
        scenes,
        {
            "source_replay_kind": "wall_bounce_first_visible_brick_contact_v1",
            "source_replayed_hit": list(logical_hit),
            "source_candidate_match_indices": matches,
        },
    )


def _bubble(source: Dict[str, Any]) -> SourceReplay:
    from data.dynamics_game.bubble_shooter_chain_predict.generator import (
        BubbleShooterChainPredictGenerator,
        _encode_grid,
    )

    renderer = BubbleShooterChainPredictGenerator(_RENDER_ROOT, seed=0, video=False)
    raw_grid = source.get("grid")
    if not isinstance(raw_grid, list) or len(raw_grid) != renderer.rows:
        raise ValueError("invalid Bubble Shooter source grid")
    initial = [[None if cell is None else str(cell) for cell in row] for row in raw_grid]
    target = _cell(source.get("target_cell"))
    colour = str(source.get("shooter_color"))
    if initial[target[0]][target[1]] is not None:
        raise ValueError("Bubble Shooter target is not empty")
    landed = [row[:] for row in initial]
    landed[target[0]][target[1]] = colour
    cluster = renderer._cluster(landed, *target)
    if len(cluster) < 3:
        raise ValueError("Bubble Shooter source shot creates no poppable cluster")
    final = renderer._apply(landed, cluster)
    encoded_final = _encode_grid(final)
    option, matches = _candidate_option(source, encoded_final)
    scenes = (
        renderer._render(initial, shooter_color=colour, target=target),
        renderer._render(landed, shooter_color=None, target=target),
        renderer._render(
            landed,
            shooter_color=None,
            target=target,
            popping_cells=cluster,
            pop_progress=0.62,
        ),
        renderer._render(final, shooter_color=None, target=target, show_arrow=True),
    )
    return SourceReplay(
        "bubble_shooter_chain_predict",
        "option",
        option,
        scenes,
        {
            "source_replay_kind": "hex_landing_connected_cluster_pop_v1",
            "source_replayed_cluster": [list(cell) for cell in sorted(cluster)],
            "source_candidate_match_indices": matches,
        },
    )


def _candy(source: Dict[str, Any]) -> SourceReplay:
    from data.dynamics_game.candy_crush_special_predict.generator import (
        CandyCrushSpecialPredictGenerator,
        _encode_grid,
    )

    renderer = CandyCrushSpecialPredictGenerator(_RENDER_ROOT, seed=0, video=False)
    raw = source.get("board")
    if not isinstance(raw, list) or len(raw) != renderer.rows:
        raise ValueError("invalid Candy source grid")
    initial = [[str(cell) for cell in row] for row in raw]
    special = _cell(source.get("special_cell"))
    kind = str(source.get("kind"))
    row, col = special
    if kind == "row":
        cleared = {(row, index) for index in range(renderer.cols)}
    elif kind == "col":
        cleared = {(index, col) for index in range(renderer.rows)}
    elif kind == "wrap":
        cleared = {
            (r, c)
            for r in range(max(0, row - 1), min(renderer.rows, row + 2))
            for c in range(max(0, col - 1), min(renderer.cols, col + 2))
        }
    elif kind == "bomb":
        colour = initial[row][col]
        cleared = {
            (r, c)
            for r in range(renderer.rows)
            for c in range(renderer.cols)
            if initial[r][c] == colour
        }
    else:
        raise ValueError(f"unknown Candy special kind: {kind!r}")
    final = renderer._apply(initial, cleared)
    encoded_final = _encode_grid(final)
    option, matches = _candidate_option(source, encoded_final)
    scenes = (
        renderer._render(initial, special=special, kind=kind),
        renderer._render(
            initial,
            special=special,
            kind=kind,
            affected=cleared,
            mark_affected=True,
        ),
        renderer._render(
            initial,
            special=special,
            kind=kind,
            affected=cleared,
            clear_progress=0.62,
            mark_affected=True,
        ),
        renderer._render(final),
    )
    return SourceReplay(
        "candy_crush_special_predict",
        "option",
        option,
        scenes,
        {
            "source_replay_kind": "visible_special_glyph_clear_no_gravity_v1",
            "source_replayed_cleared_cells": [list(cell) for cell in sorted(cleared)],
            "source_candidate_match_indices": matches,
        },
    )


def _lane(source: Dict[str, Any]) -> SourceReplay:
    from data.dynamics_game.lane_tap_step.generator import RhythmGameTapStepGenerator

    renderer = RhythmGameTapStepGenerator(_RENDER_ROOT, seed=0, video=False)
    notes = source.get("initial_note_rows")
    speeds = source.get("lane_speeds")
    ticks = int(source.get("ticks", renderer.TICKS))
    if (
        not isinstance(notes, list)
        or not isinstance(speeds, list)
        or len(notes) != renderer.COLS
        or len(speeds) != renderer.COLS
        or ticks != renderer.TICKS
    ):
        raise ValueError("invalid lane rollout source")
    notes = [[int(row) for row in lane] for lane in notes]
    speeds = [int(speed) for speed in speeds]
    states = [renderer._state_at(notes, speeds, tick) for tick in range(ticks + 1)]
    option, matches = _candidate_option(source, states[-1])
    scenes = (
        renderer._render(notes, speeds, time_value=0.0),
        renderer._render(notes, speeds, time_value=1.0),
        renderer._render(notes, speeds, time_value=2.0, final_grid=states[-1]),
    )
    return SourceReplay(
        "lane_tap_step",
        "option",
        option,
        scenes,
        {
            "source_replay_kind": "simultaneous_lane_speed_two_tick_v1",
            "source_candidate_match_indices": matches,
        },
        timing="discrete",
    )


def _lemmings(source: Dict[str, Any]) -> SourceReplay:
    from data.dynamics_game.lemmings_path_predict.generator import (
        LemmingsPathPredictGenerator,
        _encode_state,
    )

    renderer = LemmingsPathPredictGenerator(_RENDER_ROOT, seed=0, video=False)
    terrain = _grid(source.get("grid"))
    start = _cell(source.get("start"))
    ticks = int(source.get("ticks", renderer.ticks))
    if terrain is None or len(terrain) != renderer.size or ticks != renderer.ticks:
        raise ValueError("invalid Lemmings source")
    trace = renderer._walk(terrain, start, ticks, "R")
    final = trace[-1]
    encoded = _encode_state(terrain, walker=final[:2], direction=final[2])
    option, matches = _candidate_option(source, encoded)
    scenes = tuple(renderer._render_tick(terrain, state) for state in trace)
    return SourceReplay(
        "lemmings_path_predict",
        "option",
        option,
        scenes,
        {
            "source_replay_kind": "walk_turn_fall_eight_tick_v1",
            "source_replayed_trace": [list(state) for state in trace],
            "source_candidate_match_indices": matches,
        },
        timing="discrete",
    )


def _driller(source: Dict[str, Any]) -> SourceReplay:
    from data.dynamics_game.mr_driller_drop_predict.generator import (
        MrDrillerDropPredictGenerator,
        _encode_grid,
    )

    renderer = MrDrillerDropPredictGenerator(_RENDER_ROOT, seed=0, video=False)
    raw = source.get("grid")
    if not isinstance(raw, list) or len(raw) != renderer.rows:
        raise ValueError("invalid Mr Driller source")
    initial = [[None if cell is None else str(cell) for cell in row] for row in raw]
    drill_col = int(source.get("drill_col"))
    steps = int(source.get("k_steps", renderer.ticks))
    if steps != renderer.ticks:
        raise ValueError("Mr Driller source must request exactly four ticks")
    trace, log = renderer._simulate_trace(initial, drill_col, steps)
    option, matches = _candidate_option(source, _encode_grid(trace[-1]))
    scenes: List[Image.Image] = [
        renderer._render(trace[0], drill_col=drill_col, drill_row=-1)
    ]
    scenes.extend(
        renderer._render(
            trace[index + 1],
            drill_col=drill_col,
            drill_row=float(index),
        )
        for index in range(steps)
    )
    return SourceReplay(
        "mr_driller_drop_predict",
        "option",
        option,
        tuple(scenes),
        {
            "source_replay_kind": "entered_cell_connected_colour_clear_four_tick_v1",
            "source_replayed_clear_log": log,
            "source_candidate_match_indices": matches,
        },
        timing="discrete",
    )


def _pegboard(source: Dict[str, Any]) -> SourceReplay:
    from data.dynamics_game.pegboard_marble_drop_predict.generator import (
        PegboardMarbleDropPredictGenerator,
        _simulate_path,
    )

    renderer = PegboardMarbleDropPredictGenerator(_RENDER_ROOT, seed=0, video=False)
    raw_pegs = source.get("pegs")
    candidate_slots = source.get("candidate_slots")
    if not isinstance(raw_pegs, dict) or not isinstance(candidate_slots, list) or len(candidate_slots) != 4:
        raise ValueError("invalid Pegboard source")
    pegs: Dict[Cell, str] = {}
    for raw_cell, kind in raw_pegs.items():
        parts = str(raw_cell).split(",")
        if len(parts) != 2:
            raise ValueError("invalid Pegboard peg key")
        pegs[(int(parts[0]), int(parts[1]))] = str(kind)
    path, landing = _simulate_path(pegs, rows=renderer.rows)
    slots = [int(item) for item in candidate_slots]
    matches = [index for index, slot in enumerate(slots) if slot == landing]
    if len(matches) != 1:
        raise ValueError("Pegboard candidates do not uniquely contain the replayed slot")
    vertices = renderer._route_points(path, landing)
    slot_width = renderer.cols_w - 6
    slot_center_x = renderer._slot_x(landing)
    badge_center_y = renderer.bottom_y + 94
    scenes: List[Image.Image] = [
        renderer._render(pegs, candidate_slots=slots, marble_xy=vertices[0])
    ]
    for index in range(1, len(vertices) - 1):
        scenes.append(
            renderer._render(
                pegs,
                candidate_slots=slots,
                marble_xy=vertices[index],
                trail_points=vertices[: index + 1],
            )
        )
    scenes.append(
        renderer._render(
            pegs,
            candidate_slots=slots,
            marble_xy=vertices[-1],
            trail_points=vertices,
        )
    )
    scenes.append(
        renderer._render(
            pegs,
            candidate_slots=slots,
            marble_xy=vertices[-1],
            trail_points=vertices,
            selected_slot=landing,
        )
    )
    return SourceReplay(
        "pegboard_marble_drop_predict",
        "option",
        "ABCD"[matches[0]],
        tuple(scenes),
        {
            "source_replay_kind": "six_peg_left_right_route_v1",
            "source_replayed_contact_path": [list(cell) for cell in path],
            "source_replayed_landing_slot": landing,
            "source_replayed_candidate_slots": slots,
            "source_selection_geometry": {
                "slot_box": [
                    slot_center_x - slot_width // 2,
                    renderer.bottom_y,
                    slot_center_x + slot_width // 2,
                    renderer.bottom_y + 50,
                ],
                "badge_center": [slot_center_x, badge_center_y],
                "badge_radius": 13,
            },
            "source_candidate_match_indices": matches,
        },
    )


def _tetris(source: Dict[str, Any]) -> SourceReplay:
    from data.dynamics_game.tetris_future_board.generator import TetrisFutureBoardGenerator

    renderer = TetrisFutureBoardGenerator(_RENDER_ROOT, seed=0, video=False)
    initial = _grid(source.get("initial_grid"))
    if initial is None or len(initial) != renderer.rows or any(len(row) != renderer.cols for row in initial):
        raise ValueError("invalid Tetris source grid")
    active = [(row, col) for row in range(renderer.rows) for col in range(renderer.cols) if initial[row][col] == 2]
    if len(active) != 4:
        raise ValueError("Tetris source must contain exactly four active cells")
    min_row = min(row for row, _ in active)
    min_col = min(col for _, col in active)
    max_row = max(row for row, _ in active)
    max_col = max(col for _, col in active)
    shape = np.zeros((max_row - min_row + 1, max_col - min_col + 1), dtype=np.int8)
    for row, col in active:
        shape[row - min_row, col - min_col] = 1
    settled = [[0 if cell == 2 else cell for cell in row] for row in initial]
    landing_row = min_row
    while True:
        next_row = landing_row + 1
        legal = True
        for shape_row, shape_col in np.argwhere(shape == 1):
            row = next_row + int(shape_row)
            col = min_col + int(shape_col)
            if row >= renderer.rows or settled[row][col] != 0:
                legal = False
                break
        if not legal:
            break
        landing_row = next_row
    landing = renderer._with_active(settled, shape, landing_row, min_col)
    locked = [[0 if cell == 2 else int(cell) for cell in row] for row in landing]
    for shape_row, shape_col in np.argwhere(shape == 1):
        locked[landing_row + int(shape_row)][min_col + int(shape_col)] = 1
    final, cleared_rows = renderer._clear_full_rows(locked)
    scenes = (
        renderer._render_grid(initial),
        renderer._render_grid(landing),
        renderer._render_grid(locked),
        renderer._render_grid(final),
    )
    return SourceReplay(
        "tetris_future_board",
        "grid",
        final,
        scenes,
        {
            "source_replay_kind": "rigid_drop_lock_simultaneous_row_clear_v1",
            "source_replayed_landing_row": landing_row,
            "source_replayed_cleared_rows": cleared_rows,
        },
    )


def _track(source: Dict[str, Any]) -> SourceReplay:
    from data.dynamics_game.track_match_shot.generator import ZumaTrackShotGenerator

    renderer = ZumaTrackShotGenerator(_RENDER_ROOT, seed=0, video=False)
    grid = _grid(source.get("initial_grid"))
    if grid is None or len(grid) != 1:
        raise ValueError("invalid track source grid")
    initial = grid[0]
    gap = int(source.get("gap_slot"))
    projectile_id = int(source.get("projectile_color_id"))
    projectile = str(source.get("projectile_color"))
    if not (0 <= gap < len(initial)) or initial[gap] != 0:
        raise ValueError("track source gap is invalid")
    filled = initial[:]
    filled[gap] = projectile_id
    left = gap
    right = gap + 1
    while left > 0 and filled[left - 1] == projectile_id:
        left -= 1
    while right < len(filled) and filled[right] == projectile_id:
        right += 1
    final = filled[:]
    cleared = right - left >= 3
    if cleared:
        for index in range(left, right):
            final[index] = 0
    scenes = [
        renderer._render_track(initial, gap_slot=gap, projectile=projectile),
        renderer._render_track(filled, gap_slot=None, projectile=projectile, launcher_loaded=False),
    ]
    if cleared:
        scenes.append(renderer._render_track(final, gap_slot=None, projectile=projectile, launcher_loaded=False))
    return SourceReplay(
        "track_match_shot",
        "grid",
        [final],
        tuple(scenes),
        {
            "source_replay_kind": "insert_then_single_contiguous_run_clear_v1",
            "source_replayed_clear_run": [left, right] if cleared else [],
        },
    )



def _jewel(source: Dict[str, Any]) -> SourceReplay:
    """Replay the visible swap, union clear, and column gravity."""

    from data.dynamics_game.jewel2_future_board.generator import Jewel2FutureBoardGenerator

    initial = _grid(source.get("initial_grid"))
    if initial is None:
        raise ValueError("invalid Jewel2 source grid")
    renderer = Jewel2FutureBoardGenerator(_RENDER_ROOT, seed=0, video=False, rows=len(initial), cols=len(initial[0]))
    raw_swap = source.get("swap_cells") or [source.get("swap_a"), source.get("swap_b")]
    if initial is None or len(initial) != renderer.rows or any(len(row) != renderer.cols for row in initial):
        raise ValueError("invalid Jewel2 source grid")
    if not isinstance(raw_swap, (list, tuple)) or len(raw_swap) != 2:
        raise ValueError("Jewel2 source must expose two swap cells")
    first, second = _cell(raw_swap[0]), _cell(raw_swap[1])
    rows, cols = len(initial), len(initial[0])
    if not (
        0 <= first[0] < rows
        and 0 <= first[1] < cols
        and 0 <= second[0] < rows
        and 0 <= second[1] < cols
        and abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
    ):
        raise ValueError("Jewel2 source swap is not an adjacent in-bounds pair")

    swapped = [row[:] for row in initial]
    swapped[first[0]][first[1]], swapped[second[0]][second[1]] = (
        swapped[second[0]][second[1]],
        swapped[first[0]][first[1]],
    )
    marked: set[Cell] = set()
    for row in range(rows):
        start = 0
        while start < cols:
            end = start + 1
            while end < cols and swapped[row][end] == swapped[row][start]:
                end += 1
            if swapped[row][start] != 0 and end - start >= 3:
                marked.update((row, col) for col in range(start, end))
            start = end
    for col in range(cols):
        start = 0
        while start < rows:
            end = start + 1
            while end < rows and swapped[end][col] == swapped[start][col]:
                end += 1
            if swapped[start][col] != 0 and end - start >= 3:
                marked.update((row, col) for row in range(start, end))
            start = end
    if not marked:
        raise ValueError("Jewel2 source swap creates no clearable run")

    cleared = [row[:] for row in swapped]
    for row, col in marked:
        cleared[row][col] = 0
    final = [[0 for _ in range(cols)] for _ in range(rows)]
    for col in range(cols):
        values = [cleared[row][col] for row in range(rows) if cleared[row][col] != 0]
        for offset, value in enumerate(reversed(values)):
            final[rows - 1 - offset][col] = value

    option, matches = _candidate_option(source, final)
    scenes = (
        renderer._render_grid(initial, highlight=[first, second]),
        renderer._render_grid(swapped),
        renderer._render_grid(cleared),
        renderer._render_grid(final),
    )
    stored_trace = source.get("state_trace")
    return SourceReplay(
        "jewel2_future_board",
        "option",
        option,
        scenes,
        {
            "source_replay_kind": "jewel_swap_union_clear_gravity_v1",
            "source_replayed_clear_cells": [list(cell) for cell in sorted(marked)],
            "source_candidate_match_indices": matches,
            "source_saved_trace_consistent_diagnostic": stored_trace == [initial, swapped, cleared, final],
        },
    )


def _sokoban(source: Dict[str, Any]) -> SourceReplay:
    """Replay a Sokoban source and derive one deterministic shortest route."""

    from data.dynamics_game.sokoban.generator import SokobanPlanGenerator

    raw_grid = source.get("initial_grid")
    initial = _grid(raw_grid)
    if initial is None:
        raise ValueError("invalid Sokoban source grid")
    rows, cols = len(initial), len(initial[0])
    renderer = SokobanPlanGenerator(_RENDER_ROOT, seed=0, video=False, height=rows, width=cols)
    player = next(
        ((row, col) for row, values in enumerate(initial) for col, value in enumerate(values) if int(value) == 3),
        None,
    )
    box = next(
        ((row, col) for row, values in enumerate(initial) for col, value in enumerate(values) if int(value) in {2, 5}),
        None,
    )
    target = next(
        ((row, col) for row, values in enumerate(initial) for col, value in enumerate(values) if int(value) == 4),
        None,
    )
    if player is None or box is None or target is None:
        raise ValueError("Sokoban source must contain one player, box, and target")
    walls = {
        (row, col)
        for row, values in enumerate(initial)
        for col, value in enumerate(values)
        if int(value) == 9
    }
    directions = {
        "up": (-1, 0),
        "right": (0, 1),
        "down": (1, 0),
        "left": (0, -1),
    }
    queue = deque([((player, box), [])])
    seen = {(player, box)}
    route: Optional[List[str]] = None
    while queue:
        (current_player, current_box), path = queue.popleft()
        if current_box == target:
            route = path
            break
        for move, (dr, dc) in directions.items():
            next_player = (current_player[0] + dr, current_player[1] + dc)
            if (
                not (0 <= next_player[0] < rows and 0 <= next_player[1] < cols)
                or next_player in walls
            ):
                continue
            next_box = current_box
            if next_player == current_box:
                next_box = (current_box[0] + dr, current_box[1] + dc)
                if (
                    not (0 <= next_box[0] < rows and 0 <= next_box[1] < cols)
                    or next_box in walls
                ):
                    continue
            state = (next_player, next_box)
            if state in seen:
                continue
            seen.add(state)
            queue.append((state, path + [move]))
    if route is None:
        raise ValueError("Sokoban source has no legal solution")
    board = [["#" if int(value) == 9 else "." for value in row] for row in initial]
    state_trace = [{"player": player, "box": box}]
    current_player, current_box = player, box
    for move in route:
        dr, dc = directions[move]
        next_player = (current_player[0] + dr, current_player[1] + dc)
        next_box = current_box
        if next_player == current_box:
            next_box = (current_box[0] + dr, current_box[1] + dc)
        current_player, current_box = next_player, next_box
        state_trace.append({"player": current_player, "box": current_box})
    scenes = (
        renderer._render_board(board, player, box, target),
        renderer._render_board(board, current_player, current_box, target),
    )
    return SourceReplay(
        "sokoban",
        "cell_path",
        [step["player"] for step in state_trace],
        scenes,
        {
            "source_replay_kind": "sokoban_shortest_push_route_v1",
            "source_replayed_shortest_length": len(route),
            "source_replayed_final_player": list(current_player),
            "source_replayed_final_box": list(current_box),
        },
    )


_REPLAYERS = {
    "block_blast_clear_predict": _block,
    "breakout_brick_clear_predict": _breakout,
    "bubble_shooter_chain_predict": _bubble,
    "candy_crush_special_predict": _candy,
    "jewel2_future_board": _jewel,
    "lane_tap_step": _lane,
    "lemmings_path_predict": _lemmings,
    "mr_driller_drop_predict": _driller,
    "pegboard_marble_drop_predict": _pegboard,
    "sokoban": _sokoban,
    "tetris_future_board": _tetris,
    "track_match_shot": _track,
}



def replay_source(task_name: str, source: Dict[str, Any]) -> SourceReplay:
    """Replay one supported task without consulting any saved answer field."""

    try:
        replayer = _REPLAYERS[task_name]
    except KeyError as exc:
        raise ValueError(f"unsupported source replay task: {task_name}") from exc
    return replayer(source)


__all__ = ["SourceReplay", "replay_source"]

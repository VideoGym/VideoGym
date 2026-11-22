"""Opposed one-step movers that must pass by load-bearing captures.

A 5x5 chessboard holds white pawns marching UP (toward row 0) and black
pawns marching DOWN (toward row N-1). Each turn the solver moves ONE pawn
following pawn-like rules:

* WHITE pawns advance one cell straight UP onto an empty cell.
* BLACK pawns advance one cell straight DOWN onto an empty cell.
* A WHITE pawn captures one BLACK pawn by stepping diagonally up-left or
  up-right onto a cell occupied by a black pawn (the captured pawn is
  removed from the board).
* A BLACK pawn captures one WHITE pawn by stepping diagonally down-left
  or down-right onto a cell occupied by a white pawn.
* No two-square first move; no en-passant.

Goal: the starting board has 2–3 WHITE pawns on row 4 and the same number
of BLACK pawns on row 0.  The displayed target is one exact board on which
every surviving WHITE pawn has reached row 0 and every surviving BLACK pawn
has reached row 4.  The solver must match that displayed board exactly.

Move notation: each move is ``((x1, y1), (x2, y2))``, naming the source and
destination grid positions under the shared one-based convention. Solving,
replay and rendering retain flat zero-based cells; conversion happens only at
the answer boundary. The board after each move is implied.

Mechanic load-bearing check: the displayed target contains fewer objects than
the input, and replay must include one or more legal diagonal captures.  A
straight-only sequence therefore cannot match that exact target.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    cell_to_xy,
    format_xy_position,
)
from data.common.video_encoding import pad_frames_to_wan_compatible_length
from data.rearrangement_game.core.answer_indexing import to_visible_move

PANEL_BG = (246, 244, 239)
PANEL_BORDER = (205, 201, 192)
DARK_SQUARE = (154, 166, 174)
LIGHT_SQUARE = (232, 235, 232)
WHITE_PAWN = (252, 248, 244)
WHITE_OUTLINE = (75, 82, 88)
BLACK_PAWN = (55, 64, 72)
BLACK_OUTLINE = (30, 36, 42)
EMPTY = "."
ROWS = 5
COLS = 5
MIN_OPTIMAL_MOVES = 8
MAX_OPTIMAL_MOVES = 13
PAWN_COUNT_RANGE = (2, 3)

State = Tuple[str, ...]
Move = Tuple[int, int]


def _idx(r: int, c: int) -> int:
    return r * COLS + c


def _rc(idx: int) -> Tuple[int, int]:
    return (idx // COLS, idx % COLS)


def _validate_state(state: Any) -> State:
    if not isinstance(state, (list, tuple)) or len(state) != ROWS * COLS:
        raise ValueError("state must contain exactly 25 cells")
    if any(not isinstance(cell, str) or cell not in {EMPTY, "W", "B"} for cell in state):
        raise ValueError("cells must be exactly '.', 'W', or 'B'")
    return tuple(state)


def _validate_move(move: Any) -> Move:
    if not isinstance(move, (list, tuple)) or len(move) != 2:
        raise ValueError("move must be a pair of integer cell indices")
    src, dst = move
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (src, dst)):
        raise ValueError("move indices must be integers, not booleans")
    if not (0 <= src < ROWS * COLS and 0 <= dst < ROWS * COLS):
        raise ValueError("move indices must lie in 0..24")
    return int(src), int(dst)


def _moves_for_pawn(state: State, idx: int) -> List[int]:
    """Return list of valid destination indices for the pawn at ``idx``."""
    p = state[idx]
    if p == EMPTY:
        return []
    r, c = _rc(idx)
    out = []
    if p == "W":
        # straight up
        if r - 1 >= 0 and state[_idx(r - 1, c)] == EMPTY:
            out.append(_idx(r - 1, c))
        # capture up-left
        if r - 1 >= 0 and c - 1 >= 0 and state[_idx(r - 1, c - 1)] == "B":
            out.append(_idx(r - 1, c - 1))
        # capture up-right
        if r - 1 >= 0 and c + 1 < COLS and state[_idx(r - 1, c + 1)] == "B":
            out.append(_idx(r - 1, c + 1))
    elif p == "B":
        if r + 1 < ROWS and state[_idx(r + 1, c)] == EMPTY:
            out.append(_idx(r + 1, c))
        if r + 1 < ROWS and c - 1 >= 0 and state[_idx(r + 1, c - 1)] == "W":
            out.append(_idx(r + 1, c - 1))
        if r + 1 < ROWS and c + 1 < COLS and state[_idx(r + 1, c + 1)] == "W":
            out.append(_idx(r + 1, c + 1))
    return out


def _all_moves(state: State) -> List[Move]:
    out: List[Move] = []
    for idx, p in enumerate(state):
        if p == EMPTY:
            continue
        for dst in _moves_for_pawn(state, idx):
            out.append((idx, dst))
    return out


def _apply(state: State, move: Move) -> State:
    src, dst = move
    new = list(state)
    new[dst] = new[src]
    new[src] = EMPTY
    return tuple(new)


def _is_target(state: State) -> bool:
    """Every survivor is on the opposite home rank, with both colors present."""
    whites = [(_rc(i)[0], i) for i, p in enumerate(state) if p == "W"]
    blacks = [(_rc(i)[0], i) for i, p in enumerate(state) if p == "B"]
    if not whites or not blacks:
        return False
    for r, _ in whites:
        if r != 0:
            return False
    for r, _ in blacks:
        if r != ROWS - 1:
            return False
    return True


def _bfs(initial: State, max_moves: int) -> Optional[List[Move]]:
    if _is_target(initial):
        return []
    queue = deque([(initial, [])])
    seen = {initial}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_moves:
            continue
        for mv in _all_moves(state):
            ns = _apply(state, mv)
            if ns in seen:
                continue
            seen.add(ns)
            np = path + [mv]
            if _is_target(ns):
                return np
            queue.append((ns, np))
    return None


def _no_capture_moves(state: State) -> List[Move]:
    """Restrict to straight advances only (no diagonal captures)."""
    out = []
    for idx, p in enumerate(state):
        if p == EMPTY:
            continue
        r, c = _rc(idx)
        if p == "W" and r - 1 >= 0 and state[_idx(r - 1, c)] == EMPTY:
            out.append((idx, _idx(r - 1, c)))
        if p == "B" and r + 1 < ROWS and state[_idx(r + 1, c)] == EMPTY:
            out.append((idx, _idx(r + 1, c)))
    return out


def _bfs_no_capture(
    initial: State,
    target: State,
    max_moves: int = MAX_OPTIMAL_MOVES,
) -> Optional[List[Move]]:
    """Return a shortest straight-only path to the exact target, if one exists."""

    if initial == target:
        return []
    if sum(cell != EMPTY for cell in initial) != sum(cell != EMPTY for cell in target):
        return None
    queue = deque([(initial, [])])
    seen = {initial}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_moves:
            continue
        for mv in _no_capture_moves(state):
            ns = _apply(state, mv)
            if ns in seen:
                continue
            seen.add(ns)
            np = path + [mv]
            if ns == target:
                return np
            queue.append((ns, np))
    return None


class ChessPawnSwapGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "chess_pawn_swap"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/chess_pawn_swap"
    MOTION_VIDEO_MOVE_LIMIT = MAX_OPTIMAL_MOVES
    SCENE_OUTER_MARGIN = 8
    SCENE_CHEVRON_BAND = 26
    SCENE_CHEVRON_SCALE = 0.72

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        pawn_count_range: Tuple[int, int] = PAWN_COUNT_RANGE,
        min_optimal_moves: int = MIN_OPTIMAL_MOVES,
        max_optimal_moves: int = MAX_OPTIMAL_MOVES,
        move_frames: int = 8,
    ) -> None:
        if (
            not isinstance(pawn_count_range, tuple)
            or len(pawn_count_range) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in pawn_count_range)
            or tuple(pawn_count_range) != PAWN_COUNT_RANGE
        ):
            raise ValueError("pawn_count_range must be exactly (2, 3)")
        if isinstance(min_optimal_moves, bool) or not isinstance(min_optimal_moves, int):
            raise ValueError("min_optimal_moves must be an integer")
        if isinstance(max_optimal_moves, bool) or not isinstance(max_optimal_moves, int):
            raise ValueError("max_optimal_moves must be an integer")
        if (min_optimal_moves, max_optimal_moves) != (MIN_OPTIMAL_MOVES, MAX_OPTIMAL_MOVES):
            raise ValueError("optimal-move bounds must be exactly 8..13")
        if isinstance(move_frames, bool) or not isinstance(move_frames, int) or move_frames != 8:
            raise ValueError("move_frames must be exactly 8")
        super().__init__(
            output_dir=output_dir,
            seed=seed,
            video=video,
            move_frames=move_frames,
        )
        self.pawn_count_range = PAWN_COUNT_RANGE
        self.min_optimal_moves = MIN_OPTIMAL_MOVES
        self.max_optimal_moves = MAX_OPTIMAL_MOVES

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(200):
            board = [EMPTY] * (ROWS * COLS)
            n_each = self.rng.randint(*self.pawn_count_range)
            white_cols = self.rng.sample(range(COLS), n_each)
            black_cols = self.rng.sample(range(COLS), n_each)
            for c in white_cols:
                board[_idx(ROWS - 1, c)] = "W"
            for c in black_cols:
                board[_idx(0, c)] = "B"
            initial = tuple(board)
            plan = _bfs(initial, self.max_optimal_moves)
            if plan is None:
                continue
            if not self.min_optimal_moves <= len(plan) <= self.max_optimal_moves:
                continue
            replay = initial
            capture_count = 0
            for mv in plan:
                if replay[mv[1]] != EMPTY:
                    capture_count += 1
                replay = _apply(replay, mv)
            if capture_count < 1 or not _is_target(replay):
                continue
            if _bfs_no_capture(initial, replay) is not None:
                continue
            return list(initial), list(replay), list(plan)
        raise RuntimeError(
            "chess_pawn_swap rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state):
        return tuple(state)

    def _serialize_state(self, state):
        return list(state)

    def _apply_move(self, state, move):
        validated_state = _validate_state(state)
        validated_move = _validate_move(move)
        src, dst = validated_move
        if dst not in _moves_for_pawn(validated_state, src):
            raise ValueError("illegal pawn move")
        return list(_apply(validated_state, validated_move))

    def _layout(self, panel_w, panel_h):
        outer = max(8, int(min(panel_w, panel_h) * 0.05))
        usable = min(panel_w, panel_h) - 2 * outer
        cell = usable // max(ROWS, COLS)
        cell -= cell % 2
        bw = cell * COLS
        bh = cell * ROWS
        x0 = (panel_w - bw) // 2
        y0 = (panel_h - bh) // 2
        return {
            "panel_w": panel_w, "panel_h": panel_h,
            "cell": cell, "board_x0": x0, "board_y0": y0,
            "board_w": bw, "board_h": bh,
        }

    def _cell_center(self, layout, r, c):
        return (
            layout["board_x0"] + (c + 0.5) * layout["cell"],
            layout["board_y0"] + (r + 0.5) * layout["cell"],
        )

    def _draw_board(self, draw, layout):
        for r in range(ROWS):
            for c in range(COLS):
                x0 = layout["board_x0"] + c * layout["cell"]
                y0 = layout["board_y0"] + r * layout["cell"]
                x1 = x0 + layout["cell"]
                y1 = y0 + layout["cell"]
                fill = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE
                draw.rectangle((x0, y0, x1, y1), fill=fill)
        line_width = max(2, int(round(layout["cell"] * 0.035)))
        x0 = layout["board_x0"]
        y0 = layout["board_y0"]
        x1 = x0 + layout["board_w"]
        y1 = y0 + layout["board_h"]
        draw.rectangle((x0, y0, x1, y1), outline=(88, 96, 102), width=line_width)

    def _draw_pawn(self, draw, cx: float, cy: float, cell: int, color_key: str):
        if color_key == "W":
            body = WHITE_PAWN
            outline = WHITE_OUTLINE
        else:
            body = BLACK_PAWN
            outline = BLACK_OUTLINE
        h = cell * 0.70
        line_width = max(2, int(round(cell * 0.035)))
        head_r = h * 0.16
        head_cy = cy - h * 0.22
        draw.ellipse(
            (cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r),
            fill=body,
            outline=outline,
            width=line_width,
        )
        skirt_y0 = cy - h * 0.04
        skirt_y1 = cy + h * 0.24
        skirt = [
            (cx - h * 0.13, skirt_y0),
            (cx + h * 0.13, skirt_y0),
            (cx + h * 0.27, skirt_y1),
            (cx - h * 0.27, skirt_y1),
        ]
        draw.polygon(skirt, fill=body, outline=outline)
        base_w = h * 0.62
        base_h = h * 0.14
        draw.rounded_rectangle(
            (cx - base_w / 2, cy + h * 0.22, cx + base_w / 2, cy + h * 0.22 + base_h),
            radius=max(2, int(base_h / 2)),
            fill=body,
            outline=outline,
            width=line_width,
        )

    def _render_panel(self, state, panel_w, panel_h, *, moving=None):
        layout = self._layout(panel_w, panel_h)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle(
            (0, 0, panel_w - 1, panel_h - 1),
            outline=PANEL_BORDER,
            width=2,
        )
        self._draw_board(draw, layout)
        cell = layout["cell"]
        for idx, p in enumerate(state):
            if p == EMPTY:
                continue
            if moving is not None and moving.get("hide_idx") == idx:
                continue
            r, c = _rc(idx)
            cx, cy = self._cell_center(layout, r, c)
            self._draw_pawn(draw, cx, cy, cell, p)
        if moving is not None and "color" in moving and "center" in moving:
            cx, cy = moving["center"]
            self._draw_pawn(draw, cx, cy, cell, moving["color"])
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _animate_move(self, state, move, target_state):
        src, dst = int(move[0]), int(move[1])
        sr, sc = _rc(src)
        dr, dc = _rc(dst)
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        layout = self._layout(panel_w, panel_h)
        s_pix = self._cell_center(layout, sr, sc)
        d_pix = self._cell_center(layout, dr, dc)
        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)
        color = state[src]
        frames = [self._render_static_scene(state, target_state)]
        after = self._apply_move(state, move)
        for step in range(1, self.move_frames - 1):
            t = step / (self.move_frames - 2)
            cx, cy = self._interpolate_pixel_linear(s_pix, d_pix, t)
            anim = {"hide_idx": src, "color": color, "center": (cx, cy)}
            left = self._render_panel(state, panel_w, panel_h, moving=anim)
            frames.append(self._compose_scene(left, right_image))
        frames.append(self._render_static_scene(after, target_state))
        return frames

    def _build_video_frames(self, initial_state, target_state, moves):
        """Preserve every raw motion endpoint instead of uniformly subsampling it."""

        initial_image = self._render_static_scene(initial_state, target_state)
        frames = [np.array(initial_image, dtype=np.uint8) for _ in range(self.hold_frames)]
        current = initial_state
        for move in moves:
            for frame in self._animate_move(current, move, target_state):
                frames.append(np.array(frame, dtype=np.uint8))
            current = self._apply_move(current, move)
            after_image = self._render_static_scene(current, target_state)
            for _ in range(self.pause_frames):
                frames.append(np.array(after_image, dtype=np.uint8))
        final_image = self._render_static_scene(current, target_state)
        for _ in range(self.hold_frames):
            frames.append(np.array(final_image, dtype=np.uint8))
        return pad_frames_to_wan_compatible_length(frames)

    def _describe_input(self) -> str:
        return (
            "The input image shows two side-by-side 5x5 boards with identical square geometry, and the small blue "
            "chevron between them points from the current board on the left to the exact target board on the right. "
            "Light flat silhouettes are WHITE movers and dark flat silhouettes are BLACK movers, and empty cells "
            "contain no object. WHITE starts on the bottom row and moves toward the top row, while BLACK starts on "
            f"the top row and moves toward the bottom row. {GRID_XY_CONVENTION_SENTENCE} "
            "The board carries no hidden state."
        )

    def _describe_rules(self) -> str:
        return (
            "Each turn moves exactly one currently visible object by exactly one row. WHITE may move straight one "
            "row upward in the same column only when that destination is empty, or diagonally one row upward and one "
            "column left/right only when the destination contains BLACK; that BLACK object is then removed. BLACK "
            "uses the mirror rule one row downward and may diagonally remove only WHITE. Straight movement into an "
            "occupied cell, diagonal movement into an empty or same-color cell, leaving the board, moving sideways, "
            "moving backward, remaining in place, moving two rows, moving two objects, and any other capture are "
            "illegal. Removed objects never fall, return, compact, refill, or change identity; all unselected objects "
            "remain fixed after each move."
        )

    def _describe_output(self) -> str:
        return (
            "Return any shortest legal sequence that transforms the entire left board into the displayed right "
            "board exactly, including the exact survivor colors, survivor squares, empty squares, and captured-object "
            "absences. Each move names the source `(x, y)` and destination `(x, y)`. Use one space after every comma. "
            "As a syntax-only example: `Answer: [((5, 1), (4, 1)), ((1, 3), (2, 3)), ...].`"
        )

    def _format_move(self, move):
        # Solver and replay squares are flat 0-based; the formal payload uses (x, y).
        source, destination = to_visible_move(self.TASK_NAME, (int(move[0]), int(move[1])))
        return f"({format_xy_position(source)}, {format_xy_position(destination)})"

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _color_squares(state, color: str) -> str:
        positions = [
            format_xy_position(cell_to_xy(divmod(idx, COLS)))
            for idx, cell in enumerate(state)
            if cell == color
        ]
        if not positions:
            return "none left"
        if len(positions) == 1:
            return f"position {positions[0]}"
        return "positions " + ", ".join(positions)

    def _author_observation(self, initial_state, target_state, moves) -> str:
        return (
            f"White pawns stand on {self._color_squares(initial_state, 'W')} along the bottom row and black "
            f"pawns on {self._color_squares(initial_state, 'B')} along the top row. The target board puts every "
            "white survivor on the top row and every black survivor on the bottom row. Pawns only step straight "
            "into an empty square or capture one square diagonally forward, and straight steps can never pass "
            "through an occupied square, so this position cannot be resolved without at least one capture."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        src, dst = int(move[0]), int(move[1])
        visible_src, visible_dst = to_visible_move(self.TASK_NAME, (src, dst))
        source_position = format_xy_position(visible_src)
        destination_position = format_xy_position(visible_dst)
        color = "white" if previous_state[src] == "W" else "black"
        capture = previous_state[dst] != EMPTY
        dst_row = dst // COLS
        reaches_goal = (previous_state[src] == "W" and dst_row == 0) or (
            previous_state[src] == "B" and dst_row == ROWS - 1
        )
        if capture:
            victim = "black" if previous_state[dst] == "B" else "white"
            action = (
                f"captures diagonally onto {destination_position}, removing the {victim} pawn there"
            )
        else:
            action = f"steps straight ahead to the empty position {destination_position}"
        if reaches_goal:
            landing = " and arrives on its far rank"
        else:
            landing = ""
        return (
            f"The {color} pawn at {source_position} {action}{landing}. "
            f"White now holds {self._color_squares(next_state, 'W')}; black holds "
            f"{self._color_squares(next_state, 'B')}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"The board now matches the target exactly: every white survivor sits on the top row, every "
            f"black survivor on the bottom row, and the {len(moves)} moves used only legal steps and "
            "diagonal captures."
        )

    def _extra_payload(self, initial_state, target_state, moves):
        return {
            "rows": ROWS, "cols": COLS,
            "pawn_count_per_color": sum(cell == "W" for cell in initial_state),
            "capture_count": sum(
                1 for before, move in zip(self._trace_states(initial_state, moves), moves)
                if before[move[1]] != EMPTY
            ),
            "optimal_moves": [(int(m[0]), int(m[1])) for m in moves],
        }

    def _trace_states(self, initial_state, moves):
        state = _validate_state(initial_state)
        states = []
        for move in moves:
            states.append(state)
            state = _apply(state, _validate_move(move))
        return states


__all__ = ["ChessPawnSwapGenerator"]

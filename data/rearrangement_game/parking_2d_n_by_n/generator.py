"""2D parking lot with TWO empty slots — route cars through twin gaps.

A 3x3 grid holds 7 numbered cars and 2 EMPTY slots. Each move drives one
car into ANY empty slot orthogonally adjacent to it, leaving its old slot
empty. The crucial twist that makes this distinct from a single-empty slide
puzzle (and from the existing 1xN ``parking_lot_swap_minimal_moves``) is
the 2 empties: each step, the solver chooses BOTH which car to move AND
which empty slot to fill, since two empties can be reached.

Move notation: ``(car_id, (x, y))``. The car id is a printed token identity
and is never renumbered. The destination uses the shared one-based grid
convention at the answer boundary, while the solver, semantic state, and
renderer keep a zero-based flat cell index.

Mechanic load-bearing check: label the two initial empty holes internally and
propagate each label when a car fills that hole. Every shortest solution must
move both labelled holes at least once; samples admitting a shortest route
that leaves either original hole unused are rejected.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import ImageDraw

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    cell_to_xy,
    format_xy_position,
)
from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
    _shade,
)
from data.rearrangement_game.core.answer_indexing import to_visible_plan

PANEL_BG = (244, 246, 246)
PANEL_BORDER = (202, 207, 210)
PAVEMENT = (207, 214, 216)
LANE_LINE = (250, 250, 247)
EMPTY_SLOT = (232, 236, 237)


def _bfs(initial: Tuple[Optional[int], ...], target: Tuple[Optional[int], ...], rows: int, cols: int, max_moves: int) -> Optional[List[Tuple[int, int]]]:
    if initial == target:
        return []
    queue = deque([(initial, [])])
    seen = {initial}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_moves:
            continue
        empties = [i for i, v in enumerate(state) if v is None]
        for ei in empties:
            er, ec = ei // cols, ei % cols
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = er + dr, ec + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                nb = nr * cols + nc
                car_id = state[nb]
                if car_id is None:
                    continue
                new_state = list(state)
                new_state[ei], new_state[nb] = car_id, None
                new_state_t = tuple(new_state)
                if new_state_t in seen:
                    continue
                seen.add(new_state_t)
                np_path = path + [(int(car_id), int(ei))]
                if new_state_t == target:
                    return np_path
                queue.append((new_state_t, np_path))
    return None


def _shortest_empty_usage_masks(
    initial: Tuple[Optional[int], ...],
    target: Tuple[Optional[int], ...],
    rows: int,
    cols: int,
    max_moves: int,
) -> Tuple[Optional[int], set[int]]:
    """Return shortest distance and the hole-usage masks of all shortest plans."""

    empty_positions = [index for index, value in enumerate(initial) if value is None]
    if len(empty_positions) != 2:
        return None, set()
    labelled = list(initial)
    for label, position in enumerate(empty_positions):
        labelled[position] = -(label + 1)
    start = (tuple(int(value) for value in labelled), 0)
    queue = deque([start])
    distance = {start: 0}
    best_distance: Optional[int] = None
    goal_masks: set[int] = set()

    while queue:
        state, used_mask = queue.popleft()
        current_distance = distance[(state, used_mask)]
        if best_distance is not None and current_distance >= best_distance:
            continue
        for empty_index, value in enumerate(state):
            if value >= 0:
                continue
            empty_label = -value - 1
            empty_row, empty_col = divmod(empty_index, cols)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                row, col = empty_row + dr, empty_col + dc
                if not (0 <= row < rows and 0 <= col < cols):
                    continue
                car_index = row * cols + col
                car_id = state[car_index]
                if car_id <= 0:
                    continue
                next_values = list(state)
                next_values[empty_index], next_values[car_index] = car_id, value
                next_mask = used_mask | (1 << empty_label)
                next_state = tuple(next_values)
                next_distance = current_distance + 1
                if next_distance > max_moves:
                    continue
                key = (next_state, next_mask)
                if key not in distance:
                    distance[key] = next_distance
                    queue.append(key)
                unlabelled = tuple(None if item < 0 else item for item in next_state)
                if unlabelled == target:
                    if best_distance is None:
                        best_distance = next_distance
                    if next_distance == best_distance:
                        goal_masks.add(next_mask)
    return best_distance, goal_masks


class Parking2DNByNGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "parking_2d_n_by_n"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/parking_2d_n_by_n"
    SCENE_OUTER_MARGIN = 8
    SCENE_CHEVRON_BAND = 26
    SCENE_CHEVRON_SCALE = 0.72

    ROWS = 3
    COLS = 3
    EMPTIES = 2

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        max_optimal_moves: int = 7,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)
        self.max_optimal_moves = int(max_optimal_moves)

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        rows, cols, ne = self.ROWS, self.COLS, self.EMPTIES
        n = rows * cols
        car_count = n - ne
        for _ in range(2000):
            target = list(range(1, car_count + 1)) + [None] * ne
            self.rng.shuffle(target)
            initial = list(target)
            # Apply 4..max random valid moves to derive initial.
            k = self.rng.randint(4, self.max_optimal_moves)
            seen_empties_used = set()
            for _step in range(k):
                empties = [i for i, v in enumerate(initial) if v is None]
                self.rng.shuffle(empties)
                made_move = False
                for ei in empties:
                    er, ec = ei // cols, ei % cols
                    nbrs = []
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = er + dr, ec + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            nb = nr * cols + nc
                            if initial[nb] is not None:
                                nbrs.append(nb)
                    if nbrs:
                        pick = self.rng.choice(nbrs)
                        initial[ei], initial[pick] = initial[pick], None
                        seen_empties_used.add(ei)
                        made_move = True
                        break
                if not made_move:
                    break
            if tuple(initial) == tuple(target):
                continue
            plan = _bfs(tuple(initial), tuple(target), rows, cols, self.max_optimal_moves)
            if plan is None:
                continue
            if len(plan) < 4:
                continue
            shortest_distance, usage_masks = _shortest_empty_usage_masks(
                tuple(initial), tuple(target), rows, cols, self.max_optimal_moves
            )
            if shortest_distance != len(plan) or usage_masks != {0b11}:
                continue
            return initial, target, plan
        raise RuntimeError(
            "parking_2d_n_by_n rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state: Any) -> Any:
        return tuple(state)

    def _serialize_state(self, state: Any) -> Any:
        return [int(c) if c is not None else None for c in state]

    def _apply_move(self, state: Any, move: Any) -> Any:
        car_id, target_idx = int(move[0]), int(move[1])
        cur = list(state)
        if car_id not in cur:
            raise ValueError(f"car {car_id} not on lot")
        ci = cur.index(car_id)
        if not (0 <= target_idx < len(cur)):
            raise ValueError(f"target slot {target_idx} out of range")
        if cur[target_idx] is not None:
            raise ValueError(f"target slot {target_idx} not empty")
        cols = self.COLS
        cr, cc = ci // cols, ci % cols
        tr, tc = target_idx // cols, target_idx % cols
        if abs(cr - tr) + abs(cc - tc) != 1:
            raise ValueError(f"car {car_id} not adjacent to slot {target_idx}")
        cur[target_idx], cur[ci] = car_id, None
        return cur

    def _car_color(self, car_id: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(car_id - 1) % len(ITEM_PALETTE)]

    def _layout(self, panel_w: int, panel_h: int) -> Dict[str, Any]:
        rows, cols = self.ROWS, self.COLS
        outer = max(8, int(min(panel_w, panel_h) * 0.04))
        usable = min(panel_w, panel_h) - 2 * outer
        cell = usable // max(rows, cols)
        cell -= cell % 2
        board_w = cell * cols
        board_h = cell * rows
        x0 = (panel_w - board_w) // 2
        y0 = (panel_h - board_h) // 2
        return {
            "panel_w": panel_w,
            "panel_h": panel_h,
            "cell": cell,
            "board_x0": x0,
            "board_y0": y0,
            "board_w": board_w,
            "board_h": board_h,
        }

    def _cell_center(self, layout: Dict[str, Any], r: int, c: int) -> Tuple[float, float]:
        cx = layout["board_x0"] + (c + 0.5) * layout["cell"]
        cy = layout["board_y0"] + (r + 0.5) * layout["cell"]
        return (cx, cy)

    def _draw_pavement(self, draw: ImageDraw.ImageDraw, layout: Dict[str, Any]) -> None:
        x0 = layout["board_x0"]
        y0 = layout["board_y0"]
        x1 = x0 + layout["board_w"]
        y1 = y0 + layout["board_h"]
        draw.rectangle((x0, y0, x1, y1), fill=PAVEMENT)
        # Lane lines between rows and columns.
        cell = layout["cell"]
        for c in range(1, self.COLS):
            xs = x0 + cell * c
            draw.line((xs, y0 + 4, xs, y1 - 4), fill=LANE_LINE, width=2)
        for r in range(1, self.ROWS):
            ys = y0 + cell * r
            draw.line((x0 + 4, ys, x1 - 4, ys), fill=LANE_LINE, width=2)
        # Border.
        draw.rectangle((x0, y0, x1, y1), outline=LANE_LINE, width=2)

    def _draw_car(self, draw: ImageDraw.ImageDraw, cx: float, cy: float, cell: int, car_id: int) -> None:
        color = self._car_color(car_id)
        outline = _shade(color, 0.55)
        car_w = cell * 0.68
        car_h = cell * 0.68
        x0 = cx - car_w / 2
        x1 = cx + car_w / 2
        y0 = cy - car_h / 2
        y1 = cy + car_h / 2
        draw.rounded_rectangle((x0, y0, x1, y1), radius=max(7, int(car_w * 0.24)), fill=color, outline=outline, width=2)
        font_size = max(10, int(car_w * 0.42))
        font = self._draw_font(draw, font_size)
        text_color = (32, 30, 26) if (sum(color) > 380) else (250, 248, 244)
        draw.text((cx, cy), str(car_id), font=font, fill=text_color, anchor="mm")

    def _draw_empty(self, draw: ImageDraw.ImageDraw, cx: float, cy: float, cell: int) -> None:
        size = cell * 0.74
        x0 = cx - size / 2
        x1 = cx + size / 2
        y0 = cy - size / 2
        y1 = cy + size / 2
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=max(7, int(size * 0.18)),
            fill=EMPTY_SLOT,
            outline=(176, 184, 188),
            width=2,
        )

    def _render_panel(self, state, panel_w, panel_h, *, moving=None):
        layout = self._layout(panel_w, panel_h)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        self._draw_pavement(draw, layout)
        cols = self.COLS
        cell = layout["cell"]
        for idx, val in enumerate(state):
            r, c = idx // cols, idx % cols
            cx, cy = self._cell_center(layout, r, c)
            if moving is not None and moving.get("hide_idx") == idx:
                self._draw_empty(draw, cx, cy, cell)
                continue
            if val is None:
                self._draw_empty(draw, cx, cy, cell)
            else:
                self._draw_car(draw, cx, cy, cell, int(val))
        if moving is not None and "car_id" in moving and "center" in moving:
            cx, cy = moving["center"]
            self._draw_car(draw, cx, cy, cell, int(moving["car_id"]))
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _animate_move(self, state, move, target_state):
        car_id = int(move[0])
        target_idx = int(move[1])
        cols = self.COLS
        ci = list(state).index(car_id)
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        layout = self._layout(panel_w, panel_h)
        sr, sc = ci // cols, ci % cols
        tr, tc = target_idx // cols, target_idx % cols
        src = self._cell_center(layout, sr, sc)
        dst = self._cell_center(layout, tr, tc)
        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)
        frames = []
        after = self._apply_move(state, move)
        for step in range(1, self.move_frames + 1):
            t = step / self.move_frames
            if step == self.move_frames:
                frames.append(self._render_static_scene(after, target_state))
                continue
            cx, cy = self._interpolate_pixel_linear(src, dst, t)
            anim = {"hide_idx": ci, "car_id": car_id, "center": (cx, cy)}
            left = self._render_panel(state, panel_w, panel_h, moving=anim)
            frames.append(self._compose_scene(left, right_image))
        return frames

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side 3x3 parking grids on a pavement-grey canvas. Each grid has "
            "7 numbered flat car tokens and 2 EMPTY slots (rendered as pale outlined bays). The left grid shows the "
            "starting arrangement and the right grid shows the desired final arrangement. A blue chevron arrow "
            "between the grids points from left to right. "
            f"{GRID_XY_CONVENTION_SENTENCE} Decorative vehicle details and shadows are omitted."
        )

    def _describe_rules(self) -> str:
        return (
            "Each move drives ONE car into ONE chosen empty slot. The chosen empty slot must be orthogonally adjacent "
            "to the car (sharing an edge — up, down, left, or right). The car's previous slot becomes the new empty. "
            "Cars cannot move diagonally and cannot jump over other cars. When 2 empty slots are both adjacent, you "
            "decide which one to use."
        )

    def _describe_output(self) -> str:
        return (
            "Return the shortest sequence of moves that turns the left grid into the right grid. Each move is the "
            "tuple `(car_id, (x, y))`, where `(x, y)` is the empty grid position the car drives into. "
            "Use one space after every comma. As a syntax-only example: "
            "`Answer: [(4, (1, 2)), (7, (2, 2)), ...].`"
        )

    def _format_move(self, move):
        """Render one internal 0-based move for diagnostics and metadata."""

        car_id, destination = to_visible_plan(self.TASK_NAME, [move])[0]
        return f"({int(car_id)}, {format_xy_position(destination)})"

    def _format_answer(self, moves):
        """Serialize the formal payload, numbering destination cells from 1."""

        visible = to_visible_plan(self.TASK_NAME, moves)
        items = ", ".join(
            f"({int(car_id)}, {format_xy_position(destination)})"
            for car_id, destination in visible
        )
        return f"[{items}]"

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    def _grid_reading(self, state) -> str:
        cols = self.COLS
        rows = []
        for r in range(self.ROWS):
            row_cells = state[r * cols : (r + 1) * cols]
            rows.append(
                ", ".join("an empty bay" if v is None else f"car {v}" for v in row_cells)
            )
        return "; next row: ".join(rows)

    def _author_observation(self, initial_state, target_state, moves) -> str:
        misplaced = sorted(
            int(v)
            for idx, v in enumerate(initial_state)
            if v is not None and target_state[idx] != v
        )
        cars = ", ".join(f"car {c}" for c in misplaced)
        return (
            f"Reading the lot row by row from the top left, it holds {self._grid_reading(initial_state)}. "
            f"Compared with the target grid, {cars} still sit in the wrong cells. A car may only drive "
            "into an empty bay that shares an edge with it, so the two bays have to be steered around the "
            "grid to pull each car into place."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        car_id, dest = int(move[0]), int(move[1])
        source = list(previous_state).index(car_id)
        source_position = format_xy_position(cell_to_xy(divmod(source, self.COLS)))
        destination_position = format_xy_position(cell_to_xy(divmod(dest, self.COLS)))
        settled = target_state[dest] == car_id
        if settled:
            purpose = "which is exactly the cell the target assigns it"
        elif move_index < len(moves) and int(moves[move_index][1]) == source:
            purpose = (
                f"which opens the bay at {source_position} that car "
                f"{int(moves[move_index][0])} needs next"
            )
        else:
            purpose = f"which opens a fresh bay at {source_position}"
        return (
            f"Car {car_id} at {source_position} borders the empty bay at {destination_position}, so it drives "
            f"there, {purpose}. The lot now reads {self._grid_reading(next_state)}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"Every car now occupies its target cell with both empty bays where the target shows them, "
            f"after {len(moves)} single-cell drives."
        )

    def _extra_payload(self, initial_state, target_state, moves):
        return {
            "rows": self.ROWS,
            "cols": self.COLS,
            "empty_count": self.EMPTIES,
            "optimal_moves": [(int(m[0]), int(m[1])) for m in moves],
        }


__all__ = ["Parking2DNByNGenerator"]

"""Layered maze generator — two independent floors shown side-by-side.

Layer 1 (left) and Layer 2 (right) are both standard DFS square-grid mazes.
Staircase cells (yellow, ↕) appear at the same (row, col) in both layers.
Stepping onto a staircase cell switches the agent to the other layer at the
same position.  The start is on Layer 1 and the goal is on Layer 2, so the
solution must cross through at least one staircase.

Layout guarantee (Core Rule #7, **Tier A**): with every staircase transition
disabled (the agent restricted to its current layer), no path from the L1
start to the L2 goal exists; the staircase mechanic is **required**, not
optional. Tier A is satisfied by construction since start_layer=1 and
goal_layer=2 are different layers, but ``create_puzzle`` re-verifies it via
a fail-loud assert after sampling.

Model-facing position convention
--------------------------------
Both floors are drawn side by side under one continuous column axis, so a
position is the plain 1-based ``(x, y)`` matrix position of the whole picture:
``x`` is the row counted from the top and ``y`` is the column counted from the
left, with L1 owning columns ``1..cols`` and L2 continuing at ``cols + 1``.
Internal solver state, replay and pixel geometry stay 0-based ``(row, col,
layer)``; the conversion happens only when the answer is exported and when a
candidate answer is parsed.
"""

from __future__ import annotations

from data.common.assets import render_font

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from data.common.grid_coordinates import GRID_PATH_STAY_ENCODING_SENTENCE
from data.common.pipeline.path_tasks import GRID_AXIS_LABEL_COLOR, GRID_XY_CONVENTION_SENTENCE, PathTaskGenerator, PathTaskRecord, draw_path_line

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
from data.spatial_navigation.core.maze_style import (
    MAZE_CELL_OUTLINE,
    MAZE_OPEN_FILL,
    MAZE_WALL_FILL,
    MAZE_WALL_OUTLINE,
    begin_maze_scene,
    finish_maze_scene,
    strip_outer_ring,
)

WALL = 1
PATH = 0

START_COLOR = (220, 30, 30)
GOAL_COLOR = (220, 30, 30)
LINE_COLOR = (220, 30, 30)
STAIRCASE_COLOR = (255, 220, 100)
LABEL_COLOR = (20, 20, 20)

# Canvas layout
CANVAS_SIZE = 512
GAP = 6            # pixel gap between the two maze halves
LABEL_HEIGHT = 18  # pixels reserved above each maze half for "L1" / "L2"
ROW_LABEL_GUTTER = 26  # left gutter of each floor, holding the 1-based row number x
COL_LABEL_GUTTER = 22  # band above each floor, holding the 1-based column number y

Cell = Tuple[int, int]


class MazeLayeredGenerator(PathTaskGenerator):
    """Generate layered-maze puzzles with two side-by-side floors.

    The agent starts on Layer 1 and must reach the goal on Layer 2 by using
    at least one staircase (yellow cell) that connects the two floors.
    """

    DEFAULT_OUTPUT_DIR = "outputs/spatial_navigation/maze_layered"
    # Two layers share one 512px canvas.  Larger grids squeeze the edge row and
    # column numbers, so the supported range is intentionally narrower than
    # single-panel mazes.
    DEFAULT_ROWS = 9
    DEFAULT_COLS = 9
    DEFAULT_STAIRCASE_COUNT = 3
    ANSWER_COORDINATE_SPACE = True

    LAYERED_RULE = (
        "Two maze floors (L1 left, L2 right) share yellow staircase cells at matching positions. Move one cell "
        "at a time between side-adjacent walkable cells of the current floor; dark slate cells are impassable. "
        "Yellow staircases are walkable like white cells, and on one the route may optionally climb to the cell "
        "at the same position on the other floor. The start is on L1 and the goal on L2, so a valid route must "
        "climb at least once; exactly one staircase offers a workable climb, while every other staircase is "
        "unreachable, walled off from the goal, or costs a far longer way around. You may wait for a step without "
        f"changing floors. {GRID_PATH_STAY_ENCODING_SENTENCE}"
    )
    DEFAULT_TI2V_PROMPT = (
        f"Rule: {LAYERED_RULE} Required Output: Starting from the unchanged input scene, draw one thick red "
        "square-cornered route step by step on L1 up to the working staircase, then continue it from the "
        "matching staircase on L2 to the goal, with no stroke across the gap between the floors. A small "
        "white-ringed red disc rides on the tip of the route and comes to rest on the goal cell."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        f"Rule: {LAYERED_RULE} Required Output: Draw the complete thick red square-cornered route on both "
        "floors, broken only at the staircase climb, with a small white-ringed red disc resting on the goal "
        "cell where the route ends."
    )

    PUZZLE_INPUT_DESCRIPTION = (
        "The input image shows two maze floors (L1 on the left, L2 on the right) side by side, with white "
        "walkable cells, dark slate wall cells, and yellow staircase cells at matching positions on both floors. "
        "The red start cell is on L1 and the red goal cell is on L2."
    )
    REASONING_INPUT_DESCRIPTION = (
        "The input image shows the two-floor layered maze (L1 left, L2 right) with the yellow staircases, red "
        "start on L1, and red goal on L2. Blue numbers run along the left and top edges of each floor: both "
        "floors repeat the same row numbers, while the column numbers continue across the picture, so the lower "
        "column numbers belong to L1 and the higher ones to L2. "
        + GRID_XY_CONVENTION_SENTENCE
    )

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        canvas_width: Optional[int] = None,
        aspect: Optional[float] = None,
        size: Optional[int] = None,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
        staircase_count: int = DEFAULT_STAIRCASE_COUNT,
        seed: Optional[int] = None,
        ti2v_prompt: Optional[str] = None,
        video: bool = False,
    ) -> None:
        if rows < 5 or cols < 5:
            raise ValueError("rows and cols must be at least 5")
        if rows > 9 or cols > 9:
            raise ValueError(
                "rows and cols must be at most 9 so both floors and every edge row/column number stay readable"
            )
        adjusted_rows = rows if rows % 2 == 1 else rows + 1
        adjusted_cols = cols if cols % 2 == 1 else cols + 1
        if staircase_count < 1:
            raise ValueError("staircase_count must be at least 1")

        # Square canvas with side-by-side L1 / L2 (matching the prompt wording).
        # Vertically center the mazes so the previously empty lower band is
        # split evenly top/bottom instead of all below the mazes.  Each floor
        # reserves a left gutter for its row numbers, so the two gutters and the
        # inter-floor gap come off the width budget before the cells do.
        effective_canvas = int(canvas_width) if canvas_width is not None else CANVAS_SIZE
        half_w = (effective_canvas - GAP - 2 * ROW_LABEL_GUTTER) // 2
        # Force cell_size to an even number so cell_center = origin + c*cs + cs/2
        # lands on integer pixels. Combined with an even thickness (cell_size//3
        # rounded down to even below) this guarantees every segment rect has
        # integer left/right/top/bottom and PIL rasterizes every frame on the
        # exact same pixel grid — no subpixel drift at the animating tip (which
        # was showing up as a lopsided "distorted pixel block" at the stroke's
        # leading edge because tip y from _clip_subpath is int-rounded while
        # vertex centers were at .5). maze_square gets this for free because
        # its cell_size=30 is already even.
        cell_size = max(4, half_w // adjusted_cols)
        if cell_size % 2:
            cell_size -= 1
        maze_pixel_w = adjusted_cols * cell_size
        maze_pixel_h = adjusted_rows * cell_size
        canvas_w = effective_canvas
        canvas_h = effective_canvas

        resolved_output = output_dir if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        super().__init__(
            resolved_output,
            canvas_width=canvas_w,
            aspect=canvas_w / canvas_h,
            size=cell_size,
            seed=seed,
            ti2v_prompt=ti2v_prompt,
            video=video,
        )
        self.canvas_dimensions = (canvas_w, canvas_h)

        self.rows = adjusted_rows
        self.cols = adjusted_cols
        self.cell_size = cell_size
        self.staircase_count = staircase_count

        # Center both mazes horizontally + vertically inside the 512x512 canvas.
        # Above each floor sits its "L1"/"L2" caption and then the band holding
        # that floor's column numbers; the row numbers sit in the gutter left of
        # each floor.  Nothing is drawn inside the grid body.
        pair_width = (ROW_LABEL_GUTTER + maze_pixel_w) * 2 + GAP
        h_margin = (canvas_w - pair_width) // 2
        used_h = LABEL_HEIGHT + COL_LABEL_GUTTER + maze_pixel_h
        top_pad = (canvas_h - used_h) // 2
        self._l1_origin_x: int = h_margin + ROW_LABEL_GUTTER
        self._l1_origin_y: int = top_pad + LABEL_HEIGHT + COL_LABEL_GUTTER
        self._l2_origin_x: int = h_margin + ROW_LABEL_GUTTER + maze_pixel_w + GAP + ROW_LABEL_GUTTER
        self._l2_origin_y: int = self._l1_origin_y
        self._l1_label_y: int = top_pad + LABEL_HEIGHT // 2
        self._l2_label_y: int = self._l1_label_y
        self._col_label_y: int = top_pad + LABEL_HEIGHT + COL_LABEL_GUTTER // 2

    # ------------------------------------------------------------------
    # AbstractPuzzleGenerator interface
    # ------------------------------------------------------------------

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> PathTaskRecord:
        puzzle_uuid = puzzle_id or self.next_id()

        # Start and goal positions (fixed corners to ensure diversity)
        start_cell: Cell = (0, 0)
        goal_cell: Cell = (self.rows - 1, self.cols - 1)

        # Quality gate: regenerate until a valid layered path is found
        staircase_positions: List[Cell] = []
        solution_path: List[Tuple[int, int, int]] = []  # (r, c, layer)
        staircase_facts: List[Dict[str, Any]] = []

        for _attempt in range(200):
            g1 = self._generate_maze()
            g2 = self._generate_maze()
            g1[start_cell[0]][start_cell[1]] = PATH
            g2[goal_cell[0]][goal_cell[1]] = PATH

            candidates = [
                (r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if g1[r][c] == PATH and g2[r][c] == PATH
                and (r, c) != start_cell
                and (r, c) != goal_cell
            ]
            if len(candidates) < self.staircase_count:
                continue

            chosen = self.rng.sample(candidates, min(self.staircase_count, len(candidates)))
            stairs = set(chosen)

            path = self._bfs_layered(g1, g2, stairs, start_cell, 1, goal_cell, 2)
            if path is None:
                continue

            # Confirm at least one staircase is used
            layers_visited = {layer for _, _, layer in path}
            if len(layers_visited) < 2:
                continue

            # Tier A load-bearing check (Core Rule #7): with all staircase
            # transitions disabled, the agent is restricted to its starting
            # layer (L1) and cannot enter the goal cell on L2. Plain baseline
            # = same BFS but with `stairs=set()` so layer transitions never
            # fire. This is satisfied by construction since start_layer != goal_layer.
            plain_baseline = self._bfs_layered(g1, g2, set(), start_cell, 1, goal_cell, 2)
            if plain_baseline is not None:
                # Some route reaches the L2 goal without ever taking a staircase.
                continue

            # ---- Unique-staircase load-bearing gate (Core Rule #6: no shortcut leaks). ----
            # The mechanic is: yellow staircase cells are walkable on both layers, climbing is
            # optional. So the cost of using staircase S is:
            #   d1[S] (start→S on L1, ALL stairs treated as walkable) +
            #   1 (the climb) +
            #   d2[S] (S→goal on L2, ALL stairs treated as walkable).
            # We require that exactly ONE staircase yields a cost within
            # `LOAD_BEARING_DETOUR_MIN` of the optimum. The others must be either unreachable
            # on one of the legs OR strictly longer by ≥ LOAD_BEARING_DETOUR_MIN steps.
            d1_from_start = self._bfs_single_layer(g1, start_cell)
            d2_to_goal = self._bfs_single_layer(g2, goal_cell)
            stair_costs: Dict[Cell, int] = {}
            for s in stairs:
                if s in d1_from_start and s in d2_to_goal:
                    stair_costs[s] = d1_from_start[s] + 1 + d2_to_goal[s]
            if not stair_costs:
                # No staircase actually reaches goal — should be filtered earlier, but defend.
                continue
            best_cost = min(stair_costs.values())
            # Reject if optimal solution doesn't actually use the cheapest staircase
            # (`path` length should equal best_cost when including the climb step).
            # Note: BFS path length includes the climb node so optimal_len == best_cost + 1
            # in path-cell-count form. Just sanity-check ordering is right.
            LOAD_BEARING_DETOUR_MIN = 6
            within_budget = [c for c in stair_costs.values() if c <= best_cost + LOAD_BEARING_DETOUR_MIN]
            # Reject layouts where multiple staircases are within the detour budget — that's
            # exactly the "any staircase works" failure mode the user flagged.
            if len(within_budget) > 1:
                continue
            # Also require at least one staircase to be either unreachable OR clearly
            # longer (`> best + LOAD_BEARING_DETOUR_MIN`). Otherwise no discrimination.
            unreachable_or_detour = (
                len(stairs) - len(stair_costs)  # unreachable count
                + sum(1 for c in stair_costs.values() if c > best_cost + LOAD_BEARING_DETOUR_MIN)
            )
            if unreachable_or_detour < 1:
                continue

            grid_l1 = g1
            grid_l2 = g2
            staircase_positions = list(stairs)
            solution_path = path
            staircase_facts = [
                {
                    "cell": [s[0], s[1]],
                    "l1_reachable": s in d1_from_start,
                    "l2_reachable": s in d2_to_goal,
                    "detour": (stair_costs[s] - best_cost) if s in stair_costs else None,
                }
                for s in stairs
            ]
            break
        else:
            raise RuntimeError(
                "maze_layered: failed to generate a valid layered maze after 200 attempts"
            )

        # Tier A fail-loud assert: re-verify the staircase-disabled baseline
        # has no solution after the loop succeeds. This is a safety net so a
        # future refactor that lifts the start/goal-on-different-layers
        # invariant cannot silently degrade to Tier B.
        plain_baseline_check = self._bfs_layered(
            grid_l1, grid_l2, set(), start_cell, 1, goal_cell, 2
        )
        assert plain_baseline_check is None, (
            "Tier A violated: plain baseline solved without using any staircase"
        )

        # Build the 1-based contiguous cell-id index covering both floors.
        # Iterate L1 PATH cells in row-major order first, then L2 PATH cells, so
        # L1 ids occupy 1..N1 and L2 ids occupy N1+1..N1+N2.
        cell_order: List[Tuple[int, int, int]] = []
        for layer_grid, layer in ((grid_l1, 1), (grid_l2, 2)):
            for r in range(self.rows):
                for c in range(self.cols):
                    if layer_grid[r][c] == PATH:
                        cell_order.append((r, c, layer))
        self._rebuild_cell_id_index(cell_order)

        # Convert solution path (r, c, layer) to cell IDs
        solution_cell_ids = [
            self._rc_layer_to_id(r, c, layer) for r, c, layer in solution_path
        ]

        # Pixel centers for start and goal
        start_px = self._cell_center_layer(start_cell[0], start_cell[1], 1)
        goal_px = self._cell_center_layer(goal_cell[0], goal_cell[1], 2)

        # The edge row/column numbers are solver-visible input evidence, so the
        # problem scene is identical in every modality; only the drawn route
        # separates the terminal image from the input.
        scene_image = self._render(
            grid_l1,
            grid_l2,
            staircase_positions,
            start_cell,
            goal_cell,
            solution_path=None,
        )
        solution_image = self._render(
            grid_l1,
            grid_l2,
            staircase_positions,
            start_cell,
            goal_cell,
            solution_path=solution_path,
        )

        puzzle_path, solution_path_file = self.save_images(
            puzzle_uuid, scene_image, solution_image
        )
        reasoning_path = self.save_reasoning_image(puzzle_uuid, scene_image)

        # Build video if requested. Uses the exact same base save_video call
        # shape as maze_square (square joint style, same cell_size//3 thickness
        # rounded to an even number so half-width is integer). maze_base
        # _draw_path_polyline now caps only interior vertices for
        # joint_style="square" so corners stay fully square while per-subpath
        # endpoints land as clean rectangular stroke faces.
        video_path = None
        if self.video:
            path_points = self._solution_path_to_pixels(solution_path, staircase_positions)
            thickness = max(4, self.cell_size // 3)
            if thickness % 2:
                thickness -= 1
            video_path = self.save_video(
                puzzle_uuid,
                scene_image,
                path_points,
                thickness=thickness,
                joint_style="square",
            )

        extra_payload: Dict[str, Any] = {
            "grid_size": [self.rows, self.cols],
            "cell_size": self.cell_size,
            "maze_grid_l1": grid_l1,
            "maze_grid_l2": grid_l2,
            "staircase_positions": [list(pos) for pos in staircase_positions],
            "start": list(start_cell),
            "goal": list(goal_cell),
            "start_layer": 1,
            "goal_layer": 2,
            "solution_path_cell_ids": solution_cell_ids,
            "path_joint_style": "square",
            "staircase_facts": staircase_facts,
            # Both floors share one continuous column axis, so the picture is a
            # rows x (2 * cols) matrix and the L1/L2 captions are the only other
            # rendered text.
            "rendered_glyph_inventory": {
                "axis_row_labels": [str(index + 1) for index in range(self.rows)],
                "axis_col_labels": [str(index + 1) for index in range(2 * self.cols)],
                "fixed_labels": ["L1", "L2"],
            },
            "rendered_glyph_schema": {
                "axis_row_labels": "1-based row numbers drawn once per row in the left gutter of each floor, outside the grid body",
                "axis_col_labels": "1-based column numbers drawn once per column above each floor, continuing across L1 and L2",
                "fixed_labels": "the two floor captions only; no per-cell ID, title, rule prose, status sentence, or answer banner",
            },
        }
        min_path = self.rows + self.cols
        self.add_path_length_metadata(extra_payload, solution_path, minimum=min_path)

        return self.build_record(
            puzzle_uuid,
            start_point=start_px,
            goal_point=goal_px,
            puzzle_path=puzzle_path,
            solution_path=solution_path_file,
            reasoning_path=reasoning_path,
            ti2v_prompt=self.ti2v_prompt,
            extra=extra_payload,
            video_path=video_path,
        )

    # ------------------------------------------------------------------
    # Authored reasoning and interleaved checkpoint text

    def _floor_xy(self, step: Tuple[int, int, int]) -> Tuple[int, int]:
        r, c, layer = step
        return (r + 1, c + 1 + (layer - 1) * self.cols)

    def _leg_summary(self, leg: Sequence[Tuple[int, int, int]]) -> str:
        parts: List[str] = []
        current: Optional[str] = None
        count = 0
        for previous, step in zip(leg, leg[1:]):
            if previous[2] != step[2]:
                continue
            dr, dc = step[0] - previous[0], step[1] - previous[1]
            direction = {(1, 0): "down", (-1, 0): "up", (0, 1): "right", (0, -1): "left"}[(dr, dc)]
            if direction == current:
                count += 1
            else:
                if current is not None:
                    parts.append(f"{current} {count}")
                current = direction
                count = 1
        if current is not None:
            parts.append(f"{current} {count}")
        return ", then ".join(parts)

    def _authored_route_reasoning(
        self,
        path_ids: Sequence[object],
        extra_payload: Dict[str, Any],
    ) -> Optional[str]:
        facts = extra_payload.get("staircase_facts")
        if not isinstance(facts, list) or not facts:
            return None
        path = [self._id_to_cell[int(cell_id)] for cell_id in path_ids]
        start_xy = self._floor_xy(path[0])
        goal_xy = self._floor_xy(path[-1])
        working: Optional[Tuple[int, int]] = None
        rejected: List[str] = []
        for fact in facts:
            cell = (int(fact["cell"][0]), int(fact["cell"][1]))
            l1_xy = (cell[0] + 1, cell[1] + 1)
            if fact.get("detour") == 0:
                working = cell
            elif not fact.get("l1_reachable"):
                rejected.append(
                    f"the staircase at {l1_xy} cannot even be reached from the start on L1"
                )
            elif not fact.get("l2_reachable"):
                rejected.append(
                    f"the staircase at {l1_xy} opens into a pocket of L2 that never connects to the goal"
                )
            else:
                detour = int(fact.get("detour") or 0)
                rejected.append(
                    f"climbing at {l1_xy} is possible but costs about {detour} extra steps of winding corridor"
                )
        if working is None:
            return None
        climb_index = next(
            index
            for index, (previous, step) in enumerate(zip(path, path[1:]), start=1)
            if previous[2] != step[2]
        )
        l1_xy = (working[0] + 1, working[1] + 1)
        l2_xy = (working[0] + 1, working[1] + 1 + self.cols)
        sentences: List[str] = [
            f"The start sits on floor L1 at {start_xy} and the goal on floor L2 at {goal_xy}, so the route "
            "must climb at one of the yellow staircases shared by the two floors.",
        ]
        if rejected:
            sentences.append("Checking the staircases, " + "; ".join(rejected) + ".")
        sentences.append(
            f"That leaves the staircase at {l1_xy} as the working climb."
        )
        leg1 = path[:climb_index]
        if len(leg1) > 1:
            sentences.append(
                f"On L1 the route moves {self._leg_summary(leg1)} to reach that staircase."
            )
        sentences.append(
            f"It climbs there, coming out on the matching staircase cell {l2_xy} of L2."
        )
        leg2 = path[climb_index - 1 :]
        if len(leg2) > 1:
            sentences.append(
                f"On L2 it then moves {self._leg_summary(leg2)}, arriving at the goal {goal_xy}."
            )
        sentences.append(
            "Every move stays on side-adjacent walkable cells of the current floor with a single staircase "
            "climb, so the route is complete and legal."
        )
        return " ".join(sentences)

    # ------------------------------------------------------------------
    # Maze generation (DFS)
    # ------------------------------------------------------------------

    def _generate_maze(self) -> List[List[int]]:
        ringed_rows = self.rows + 2
        ringed_cols = self.cols + 2
        grid: List[List[int]] = [
            [WALL for _ in range(ringed_cols)] for _ in range(ringed_rows)
        ]

        def carve(r: int, c: int) -> None:
            grid[r][c] = PATH
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            self.rng.shuffle(directions)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    1 <= nr < ringed_rows - 1
                    and 1 <= nc < ringed_cols - 1
                    and grid[nr][nc] == WALL
                ):
                    grid[r + dr // 2][c + dc // 2] = PATH
                    carve(nr, nc)

        carve(1, 1)
        return strip_outer_ring(grid)

    # ------------------------------------------------------------------
    # Layered BFS  (state = (row, col, layer))
    # ------------------------------------------------------------------

    def _bfs_single_layer(self, grid: List[List[int]], origin: Cell) -> Dict[Cell, int]:
        """BFS over a single layer (treating staircases as walkable cells). Returns
        a {cell: distance} map of every reachable cell from `origin`."""
        dist: Dict[Cell, int] = {origin: 0}
        queue: deque = deque([origin])
        while queue:
            r, c = queue.popleft()
            d = dist[(r, c)]
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr][nc] == PATH:
                    nb = (nr, nc)
                    if nb not in dist:
                        dist[nb] = d + 1
                        queue.append(nb)
        return dist


    def _bfs_layered(
        self,
        g1: List[List[int]],
        g2: List[List[int]],
        stairs: set,
        start: Cell,
        start_layer: int,
        goal: Cell,
        goal_layer: int,
    ) -> Optional[List[Tuple[int, int, int]]]:
        """BFS over (row, col, layer).  Returns None if no path found."""
        grids = {1: g1, 2: g2}
        init = (start[0], start[1], start_layer)
        target = (goal[0], goal[1], goal_layer)
        queue: deque = deque([init])
        parents: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {
            init: None
        }

        while queue:
            r, c, layer = queue.popleft()
            if (r, c, layer) == target:
                break

            grid = grids[layer]

            # 4-directional moves on the same layer
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if grid[nr][nc] == PATH:
                        nb = (nr, nc, layer)
                        if nb not in parents:
                            parents[nb] = (r, c, layer)
                            queue.append(nb)

            # Staircase: switch layer at same (r, c) if current cell is a staircase
            if (r, c) in stairs:
                other_layer = 3 - layer  # toggles between 1 and 2
                other_grid = grids[other_layer]
                if other_grid[r][c] == PATH:
                    nb = (r, c, other_layer)
                    if nb not in parents:
                        parents[nb] = (r, c, layer)
                        queue.append(nb)

        if target not in parents:
            return None

        node: Optional[Tuple[int, int, int]] = target
        path: List[Tuple[int, int, int]] = []
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Cell ID helpers
    # ------------------------------------------------------------------

    def rehydrate_from_record(self, record: Dict[str, Any]) -> None:
        """Rebuild the 1-based per-floor cell-id index from record['maze_grid_l1']
        and record['maze_grid_l2']."""
        g1 = record.get("maze_grid_l1")
        g2 = record.get("maze_grid_l2")
        if g1 is None or g2 is None:
            return
        rows = len(g1)
        cols = len(g1[0]) if rows else 0
        cell_order: List[Tuple[int, int, int]] = []
        for layer_grid, layer in ((g1, 1), (g2, 2)):
            for r in range(rows):
                for c in range(cols):
                    if layer_grid[r][c] == PATH:
                        cell_order.append((r, c, layer))
        self._rebuild_cell_id_index(cell_order)

    def _rc_layer_to_id(self, r: int, c: int, layer: int) -> int:
        return self._cell_to_id.get((r, c, layer), 0)

    def _id_to_rc_layer(self, cell_id: int) -> Tuple[int, int, int]:
        return self._id_to_cell[cell_id]

    def _cell_id_to_xy(self, cell_id: int) -> Optional[Tuple[int, int]]:
        """Map an internal ``(row, col, layer)`` cell to its model-facing label.

        The two floors are drawn side by side under one continuous column axis,
        so the picture is a single ``rows x (2 * cols)`` matrix: ``x`` is the
        row counted from the top and ``y`` is the column counted from the left,
        with L2 continuing where L1's columns end.
        """

        cell = self._id_to_cell.get(int(cell_id))
        if not isinstance(cell, (list, tuple)) or len(cell) != 3:
            return None
        row, col, layer = int(cell[0]), int(cell[1]), int(cell[2])
        if layer not in (1, 2):
            return None
        return (row + 1, col + 1 + (layer - 1) * self.cols)

    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        """Required by PathTaskGenerator — returns pixel center for a cell ID."""
        r, c, layer = self._id_to_rc_layer(cell_id)
        return self._cell_center_layer(r, c, layer)

    def _cell_center_layer(self, r: int, c: int, layer: int) -> Tuple[float, float]:
        ox = self._l1_origin_x if layer == 1 else self._l2_origin_x
        oy = self._l1_origin_y if layer == 1 else self._l2_origin_y
        x = ox + c * self.cell_size + self.cell_size / 2.0
        y = oy + r * self.cell_size + self.cell_size / 2.0
        return (x, y)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        g1: List[List[int]],
        g2: List[List[int]],
        stairs: List[Cell],
        start: Cell,
        goal: Cell,
        *,
        solution_path: Optional[List[Tuple[int, int, int]]],
    ) -> Image.Image:
        w, h = self.canvas_dimensions
        stairs_set = set(map(tuple, stairs))
        scene, sdraw = begin_maze_scene((w, h))

        # Draw both floors in the shared family style.
        for layer, grid, ox, oy in (
            (1, g1, self._l1_origin_x, self._l1_origin_y),
            (2, g2, self._l2_origin_x, self._l2_origin_y),
        ):
            for r in range(self.rows):
                for c in range(self.cols):
                    left = ox + c * self.cell_size
                    top = oy + r * self.cell_size
                    box = (left, top, left + self.cell_size - 1, top + self.cell_size - 1)
                    if (r, c) in stairs_set:
                        sdraw.rectangle(box, fill=STAIRCASE_COLOR, outline=MAZE_CELL_OUTLINE, width=1)
                    elif layer == 1 and (r, c) == start:
                        sdraw.rectangle(box, fill=START_COLOR)
                    elif layer == 2 and (r, c) == goal:
                        sdraw.rectangle(box, fill=GOAL_COLOR)
                    elif grid[r][c] == PATH:
                        sdraw.rectangle(box, fill=MAZE_OPEN_FILL, outline=MAZE_CELL_OUTLINE, width=1)
                    else:
                        sdraw.rectangle(box, fill=MAZE_WALL_FILL, outline=MAZE_WALL_OUTLINE, width=1)

        # Labels "L1" and "L2" sit directly above their respective floors.
        label_font = self._load_font(max(9, LABEL_HEIGHT - 4) * sdraw.factor)
        sdraw.text(
            (self._l1_origin_x + (self.cols * self.cell_size) // 2, self._l1_label_y),
            "L1",
            fill=LABEL_COLOR,
            anchor="mm",
            font=label_font,
        )
        sdraw.text(
            (self._l2_origin_x + (self.cols * self.cell_size) // 2, self._l2_label_y),
            "L2",
            fill=LABEL_COLOR,
            anchor="mm",
            font=label_font,
        )
        self._draw_axis_labels(sdraw)
        canvas = finish_maze_scene(scene, (w, h))

        # Solution path overlay on the downsampled scene.
        if solution_path:
            self._draw_solution(ImageDraw.Draw(canvas), canvas, solution_path, stairs_set)

        return canvas

    def _draw_solution(
        self,
        draw: ImageDraw.ImageDraw,
        canvas: Image.Image,
        path: List[Tuple[int, int, int]],
        stairs_set: set,
    ) -> None:
        thickness = max(4, self.cell_size // 3)
        if thickness % 2:
            thickness -= 1
        ordered_segments, _ = self._split_solution_path(path)

        for segment_index, (_, seg) in enumerate(ordered_segments):
            if seg:
                draw_path_line(
                    canvas,
                    seg,
                    LINE_COLOR,
                    thickness,
                    joint_style="square",
                    # Only the route's true tip carries the position disc, so
                    # the static terminal image matches the video's last frame.
                    tip_marker=segment_index == len(ordered_segments) - 1,
                )

    def _split_solution_path(
        self,
        path: Sequence[Tuple[int, int, int]],
    ) -> Tuple[List[Tuple[int, List[Tuple[float, float]]]], List[Tuple[int, int, int]]]:
        ordered_segments: List[Tuple[int, List[Tuple[float, float]]]] = []
        stair_markers: List[Tuple[int, int, int]] = []
        if not path:
            return ordered_segments, stair_markers

        current_layer = path[0][2]
        current_segment: List[Tuple[float, float]] = [
            self._cell_center_layer(path[0][0], path[0][1], current_layer)
        ]

        for prev_step, step in zip(path, path[1:]):
            r, c, layer = step
            if layer == current_layer:
                current_segment.append(self._cell_center_layer(r, c, layer))
                continue

            ordered_segments.append((current_layer, current_segment))
            stair_markers.extend([prev_step, step])
            current_layer = layer
            current_segment = [self._cell_center_layer(r, c, layer)]

        ordered_segments.append((current_layer, current_segment))
        return ordered_segments, list(dict.fromkeys(stair_markers))

    def _draw_axis_labels(self, draw: ImageDraw.ImageDraw) -> None:
        """Draw the 1-based row and column numbers around both floors.

        Each floor carries its own left gutter of row numbers, while the column
        numbers continue across the picture so L2 picks up where L1's columns
        end.  Nothing is drawn inside either grid body.
        """

        factor = int(getattr(draw, "factor", 1) or 1)
        font = self._load_font(max(9, int(self.cell_size * 0.42)) * factor)
        for layer, ox, oy in (
            (1, self._l1_origin_x, self._l1_origin_y),
            (2, self._l2_origin_x, self._l2_origin_y),
        ):
            for r in range(self.rows):
                center_y = oy + r * self.cell_size + self.cell_size / 2.0
                draw.text(
                    (ox - ROW_LABEL_GUTTER / 2.0, center_y),
                    str(r + 1),
                    fill=GRID_AXIS_LABEL_COLOR,
                    anchor="mm",
                    font=font,
                )
            for c in range(self.cols):
                center_x = ox + c * self.cell_size + self.cell_size / 2.0
                draw.text(
                    (center_x, self._col_label_y),
                    str(c + 1 + (layer - 1) * self.cols),
                    fill=GRID_AXIS_LABEL_COLOR,
                    anchor="mm",
                    font=font,
                )

    @staticmethod
    def _load_font(size: int) -> ImageFont.FreeTypeFont:
        return render_font(size, bold=True)

    # ------------------------------------------------------------------
    # Video helpers
    # ------------------------------------------------------------------

    def _solution_path_to_pixels(
        self,
        path: List[Tuple[int, int, int]],
        stairs: List[Cell],
    ) -> List[Tuple[float, float] | None]:
        """Convert solution path steps into pixel coordinates for video rendering."""
        del stairs
        points: List[Tuple[float, float] | None] = []
        for _, segment in self._split_solution_path(path)[0]:
            if points:
                points.append(None)
            points.extend(segment)
        return points

__all__ = ["MazeLayeredGenerator"]

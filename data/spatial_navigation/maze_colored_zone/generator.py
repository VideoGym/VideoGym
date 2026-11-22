"""Maze puzzle generator for color-zone sequential stepping tasks.

The maze is a square grid with multiple geometric routes from start to goal.
Every walkable cell is blue or green, and actual moves must alternate
blue → green → blue → green. A repeated grid position is a stay and does not
advance the phase.

Generation first finds an ordinary geometry-only shortest path, then constructs
a longer detour and colors only a route-bearing subset of cells with the phase
required by their grid parity. Other walkable cells receive the opposite phase.
The retained constrained shortest route must be strictly longer than the
ordinary geometric shortest distance. Consequently no geometry-only BFS
shortest route can accidentally satisfy the color rule.

Quality gates require:
  - A valid constrained path of length >= max(12, rows + cols)
  - At least 2 unconstrained geometric routes
  - constrained shortest length > geometric shortest length
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from data.common.grid_coordinates import (
    GRID_PATH_STAY_ENCODING_SENTENCE,
)
from data.common.pipeline.path_tasks import (
    GRID_AXIS_LABEL_COLOR,
    GRID_XY_CONVENTION_SENTENCE,
    PathTaskGenerator,
    PathTaskRecord,
    cell_to_xy,
    draw_grid_axis_labels,
    draw_path_line,
    grid_axis_label_layout,
)
from data.spatial_navigation.core.maze_style import (
    MAZE_ENDPOINT_COLOR,
    MAZE_ROUTE_COLOR,
    begin_maze_scene,
    draw_square_cell_grid,
    finish_maze_scene,
    route_run_breakdown,
    strip_outer_ring,
)

WALL = 1
PATH = 0

START_COLOR = MAZE_ENDPOINT_COLOR
GOAL_COLOR = MAZE_ENDPOINT_COLOR
LINE_COLOR = MAZE_ROUTE_COLOR

# Two alternating colors: blue (phase 0) and green (phase 1)
PHASE_COLORS = [
    (170, 200, 240),  # soft blue  — phase 0
    (170, 230, 185),  # soft green — phase 1
]
PHASE_NAMES = ["blue", "green"]

# Fraction of interior walls to remove to create loops
LOOP_FRACTION = 0.25

Cell = Tuple[int, int]
State = Tuple[int, int, int]  # (row, col, phase)


class MazeColoredZoneGenerator(PathTaskGenerator):
    """Generate maze puzzles with a two-color alternating stepping constraint
    over a multi-path maze so that the constraint is non-trivial."""

    DEFAULT_OUTPUT_DIR = "outputs/spatial_navigation/maze_colored_zone"
    DEFAULT_ROWS = 11
    DEFAULT_COLS = 11
    DEFAULT_CELL_SIZE = 32
    ANSWER_COORDINATE_SPACE = True

    COLOR_RULE = (
        "Treat both red-outlined endpoints as blue. Starting at the upper-left endpoint, move one cell at a time "
        "to a side-adjacent walkable cell in strict blue, green, blue, green order; dark slate cells are "
        "impassable, and the lower-right blue goal must therefore be entered on a blue step. You may wait for a "
        f"step; waiting does not advance the colour order. {GRID_PATH_STAY_ENCODING_SENTENCE}"
    )
    DEFAULT_TI2V_PROMPT = (
        f"Rule: {COLOR_RULE} Required Output: Starting from the unchanged input scene, draw one thick red "
        "square-cornered route step by step through the centres of the visited cells until it reaches the goal. "
        "A small white-ringed red disc rides on the tip of the route and comes to rest on the goal cell."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        f"Rule: {COLOR_RULE} Required Output: Draw the complete thick red square-cornered route through the "
        "centres of the visited cells, with a small white-ringed red disc resting on the goal cell where the "
        "route ends."
    )

    PUZZLE_INPUT_DESCRIPTION = (
        "The input image shows a colored-zone maze of soft-blue and soft-green walkable cells separated by dark "
        "slate wall cells, with red inset borders marking the blue start and blue goal cells."
    )
    REASONING_INPUT_DESCRIPTION = (
        "The input image shows the colored-zone maze. The upper-left red-outlined blue cell is the start and the "
        "lower-right red-outlined blue cell is the goal. Blue numbers run along the left and top edges of the "
        "grid, outside the coloured cells. " + GRID_XY_CONVENTION_SENTENCE
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
        cell_size: Optional[int] = None,
        aspect_ratio: Optional[float] = None,
        seed: Optional[int] = None,
        ti2v_prompt: Optional[str] = None,
        video: bool = False,
    ) -> None:
        canvas_width = 512 if canvas_width is None else int(canvas_width)
        if rows < 5 or cols < 5:
            raise ValueError("rows and cols must be at least 5")
        if rows > 11 or cols > 11:
            raise ValueError("rows and cols must be at most 11 so every edge row/column number stays readable")
        adjusted_rows = rows if rows % 2 == 1 else rows + 1
        adjusted_cols = cols if cols % 2 == 1 else cols + 1
        ratio = aspect_ratio if aspect_ratio is not None else aspect
        requested_cell_size = cell_size if cell_size is not None else size
        effective_cell_size, pad_left, pad_top, pad_right, pad_bottom = grid_axis_label_layout(
            adjusted_rows,
            adjusted_cols,
            canvas_size=canvas_width,
            cell_size=requested_cell_size,
        )
        canvas_dims = (canvas_width, canvas_width)

        final_width, final_height = canvas_dims
        aspect_for_super = (final_width / final_height) if final_height else None
        resolved_output = output_dir if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        super().__init__(
            resolved_output,
            canvas_width=final_width,
            aspect=aspect_for_super,
            size=effective_cell_size,
            seed=seed,
            ti2v_prompt=ti2v_prompt,
            video=video,
        )
        self.rows = adjusted_rows
        self.cols = adjusted_cols
        self.cell_size = effective_cell_size
        self.aspect_ratio = ratio
        self.padding = (pad_left, pad_top, pad_right, pad_bottom)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> PathTaskRecord:
        puzzle_uuid = puzzle_id or self.next_id()
        start: Cell = (0, 0)
        goal: Cell = (self.rows - 1, self.cols - 1)
        min_path_length = self._minimum_solution_path_cells()

        maze_grid, cell_colors, path = self._generate_with_quality_gate(
            start, goal, min_path_length
        )
        self._rebuild_cell_id_index(
            [(r, c) for r in range(self.rows) for c in range(self.cols) if maze_grid[r][c] == PATH]
        )

        pad_left, pad_top, pad_right, pad_bottom, canvas_dims = self._compute_padding()
        cell_bboxes = self._compute_cell_bboxes(pad_left, pad_top)

        # The edge row/column numbers are solver-visible input evidence, so the
        # problem scene is identical in every modality; only the drawn route
        # separates the terminal image from the input.
        reasoning_image = self._render(
            maze_grid, cell_colors,
            start=start, goal=goal, path=None,
            padding=(pad_left, pad_top), canvas_dims=canvas_dims,
        )
        solution_image = self._render(
            maze_grid, cell_colors,
            start=start, goal=goal, path=path,
            padding=(pad_left, pad_top), canvas_dims=canvas_dims,
        )

        puzzle_path, solution_path = self.save_images(puzzle_uuid, reasoning_image, solution_image)
        reasoning_path = self.save_reasoning_image(puzzle_uuid, reasoning_image)

        video_path = None
        if self.video:
            path_points = [self._cell_center(self._get_cell_id(*c)) for c in path]
            video_path = self.save_video(
                puzzle_uuid,
                reasoning_image,
                path_points,
                thickness=max(3, self.cell_size // 3),
                joint_style="square",
            )

        start_point = self._cell_center(self._get_cell_id(*start))
        goal_point = self._cell_center(self._get_cell_id(*goal))

        # Serialize cell_colors as {cell_id: color_name}
        color_map: Dict[str, str] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if maze_grid[r][c] == PATH:
                    cid = str(self._get_cell_id(r, c))
                    color_map[cid] = cell_colors[(r, c)]

        extra_payload: Dict[str, object] = {
            "grid_size": [self.rows, self.cols],
            "cell_size": self.cell_size,
            "maze_grid": maze_grid,
            "start": list(start),
            "goal": list(goal),
            "cell_bboxes": cell_bboxes,
            "padding": [pad_left, pad_top, pad_right, pad_bottom],
            "solution_path_cell_ids": [self._get_cell_id(*c) for c in path],
            "path_joint_style": "square",
            "cell_colors": color_map,
            "color_cycle": PHASE_NAMES,
        }
        self.add_path_length_metadata(extra_payload, path, minimum=min_path_length)
        return self.build_record(
            puzzle_uuid,
            start_point=start_point,
            goal_point=goal_point,
            puzzle_path=puzzle_path,
            solution_path=solution_path,
            reasoning_path=reasoning_path,
            ti2v_prompt=self.ti2v_prompt,
            extra=extra_payload,
            video_path=video_path,
        )

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _minimum_solution_path_cells(self) -> int:
        return max(12, self.rows + self.cols)

    def _generate_with_quality_gate(
        self,
        start: Cell,
        goal: Cell,
        min_path_length: int,
    ) -> Tuple[List[List[int]], Dict[Cell, str], List[Cell]]:
        """Generate until color phase invalidates every geometric shortest route."""
        for _ in range(300):
            maze_grid = self._generate_multipath_maze()
            n_geo_paths = self._count_geometric_paths(maze_grid, start, goal, limit=3)
            if n_geo_paths < 2:
                continue
            geometric_path = self._bfs_geometric(maze_grid, start, goal)
            if not geometric_path or geometric_path == [start]:
                continue

            blocked_edges = list(zip(geometric_path, geometric_path[1:]))
            self.rng.shuffle(blocked_edges)
            for blocked_edge in blocked_edges:
                detour = self._bfs_geometric(
                    maze_grid,
                    start,
                    goal,
                    blocked_edge=blocked_edge,
                )
                if len(detour) < min_path_length or len(detour) <= len(geometric_path):
                    continue
                cell_colors = self._assign_colors(
                    maze_grid,
                    start,
                    goal,
                    legal_route=detour,
                )
                path = self._bfs_constrained(maze_grid, cell_colors, start, goal)
                if len(path) < min_path_length:
                    continue
                # The colour mechanic must force a real detour, not a
                # two-cell parity nudge, or the sample reads at a glance.
                if len(path) < len(geometric_path) + 4:
                    continue
                return maze_grid, cell_colors, path
        raise RuntimeError(
            "Failed to generate a colored-zone maze satisfying the quality gate"
        )

    def _generate_multipath_maze(self) -> List[List[int]]:
        """DFS-carve a perfect maze, punch loop passages, and strip the ring."""
        ringed_rows = self.rows + 2
        ringed_cols = self.cols + 2
        grid = [[WALL] * ringed_cols for _ in range(ringed_rows)]

        def carve(r: int, c: int) -> None:
            grid[r][c] = PATH
            dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            self.rng.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 1 <= nr < ringed_rows - 1 and 1 <= nc < ringed_cols - 1 and grid[nr][nc] == WALL:
                    grid[r + dr // 2][c + dc // 2] = PATH
                    carve(nr, nc)

        carve(1, 1)
        grid[ringed_rows - 2][ringed_cols - 2] = PATH

        # Collect interior wall cells that separate two PATH cells, i.e. cells
        # where removing the wall connects two already-reachable walkable
        # cells, then randomly remove LOOP_FRACTION of them.
        candidate_walls: List[Cell] = []
        for r in range(1, ringed_rows - 1):
            for c in range(1, ringed_cols - 1):
                if grid[r][c] == WALL:
                    path_neighbors = sum(
                        1
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        if 0 <= r + dr < ringed_rows
                        and 0 <= c + dc < ringed_cols
                        and grid[r + dr][c + dc] == PATH
                    )
                    if path_neighbors >= 2:
                        candidate_walls.append((r, c))

        self.rng.shuffle(candidate_walls)
        n_remove = max(1, int(len(candidate_walls) * LOOP_FRACTION))
        for r, c in candidate_walls[:n_remove]:
            grid[r][c] = PATH

        return strip_outer_ring(grid)

    def _assign_colors(
        self,
        grid: List[List[int]],
        start: Cell,
        goal: Cell,
        *,
        legal_route: Sequence[Cell],
    ) -> Dict[Cell, str]:
        """Color the retained route legally and invert every off-route cell."""
        if (
            not legal_route
            or legal_route[0] != start
            or legal_route[-1] != goal
            or (len(legal_route) - 1) % 2 != 0
        ):
            raise RuntimeError("colored-zone endpoints must be connected on the same blue phase")

        route_cells = set(legal_route)
        colors: Dict[Cell, str] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] != PATH:
                    continue
                phase = (abs(r - start[0]) + abs(c - start[1])) % 2
                if (r, c) not in route_cells:
                    phase = 1 - phase
                colors[(r, c)] = PHASE_NAMES[phase]

        if colors.get(start) != "blue" or colors.get(goal) != "blue":
            raise RuntimeError("red-outlined colored-zone endpoints must retain visible blue semantics")

        return colors

    def _bfs_geometric(
        self,
        grid: List[List[int]],
        start: Cell,
        goal: Cell,
        *,
        blocked_edge: Optional[Tuple[Cell, Cell]] = None,
    ) -> List[Cell]:
        """Standard BFS ignoring color constraints; returns shortest path."""
        blocked = (
            {blocked_edge, (blocked_edge[1], blocked_edge[0])}
            if blocked_edge is not None
            else set()
        )
        queue: deque[Cell] = deque([start])
        parents: Dict[Cell, Optional[Cell]] = {start: None}
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbor = (nr, nc)
                    if ((r, c), neighbor) in blocked:
                        continue
                    if grid[nr][nc] == PATH and neighbor not in parents:
                        parents[neighbor] = (r, c)
                        queue.append(neighbor)
        if goal not in parents:
            return [start]
        node: Optional[Cell] = goal
        result: List[Cell] = []
        while node is not None:
            result.append(node)
            node = parents[node]
        result.reverse()
        return result

    def _count_geometric_paths(
        self,
        grid: List[List[int]],
        start: Cell,
        goal: Cell,
        limit: int = 3,
    ) -> int:
        """Return 0, 1, or 2+ start-goal geometric paths without enumerating them.

        The quality gate only needs to distinguish unique-path mazes from mazes
        with at least one alternate route. If a second simple path exists, then
        removing at least one edge from a known start-goal path still leaves the
        endpoints connected by that alternate path.
        """
        path = self._bfs_geometric(grid, start, goal)
        if not path or path == [start]:
            return 0 if start != goal else 1
        if limit <= 1:
            return 1

        for a, b in zip(path, path[1:]):
            if self._has_geometric_path_without_edge(grid, start, goal, blocked_edge=(a, b)):
                return min(limit, 2)
        return 1

    def _has_geometric_path_without_edge(
        self,
        grid: List[List[int]],
        start: Cell,
        goal: Cell,
        *,
        blocked_edge: Tuple[Cell, Cell],
    ) -> bool:
        blocked = {blocked_edge, (blocked_edge[1], blocked_edge[0])}
        queue: deque[Cell] = deque([start])
        seen = {start}
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if grid[nr][nc] != PATH or neighbor in seen:
                    continue
                if ((r, c), neighbor) in blocked:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        return False

    def _bfs_constrained(
        self,
        grid: List[List[int]],
        colors: Dict[Cell, str],
        start: Cell,
        goal: Cell,
    ) -> List[Cell]:
        """BFS with state = (row, col, phase).

        Phase 0 expects blue, phase 1 expects green.
        The start cell must be blue (phase 0).  Each move advances to
        (next_phase = (current_phase + 1) % 2), and the target cell must have
        color PHASE_NAMES[next_phase].
        """
        # Start must be blue; if not, no constrained path exists
        if colors.get(start) != PHASE_NAMES[0]:
            return []

        init_state: State = (start[0], start[1], 0)
        queue: deque[State] = deque([init_state])
        parents: Dict[State, Optional[State]] = {init_state: None}

        while queue:
            r, c, phase = queue.popleft()
            if (r, c) == goal:
                # Reconstruct path (cell sequence only)
                node: Optional[State] = (r, c, phase)
                result: List[Cell] = []
                while node is not None:
                    result.append((node[0], node[1]))
                    node = parents[node]
                result.reverse()
                return result
            next_phase = (phase + 1) % 2
            required_color = PHASE_NAMES[next_phase]
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if grid[nr][nc] == PATH and colors.get((nr, nc)) == required_color:
                        ns: State = (nr, nc, next_phase)
                        if ns not in parents:
                            parents[ns] = (r, c, phase)
                            queue.append(ns)
        return []

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        grid: Sequence[Sequence[int]],
        cell_colors: Dict[Cell, str],
        *,
        start: Cell,
        goal: Cell,
        path: Optional[Sequence[Cell]],
        padding: Tuple[int, int],
        canvas_dims: Tuple[int, int],
    ) -> Image.Image:
        pad_left, pad_top = padding
        phase_rgb = {name: PHASE_COLORS[i] for i, name in enumerate(PHASE_NAMES)}
        scene, sdraw = begin_maze_scene(canvas_dims)
        draw_square_cell_grid(
            sdraw,
            rows=self.rows,
            cols=self.cols,
            cell_size=self.cell_size,
            pad_left=pad_left,
            pad_top=pad_top,
            is_wall=lambda r, c: grid[r][c] != PATH,
            open_fill=lambda r, c: phase_rgb[cell_colors[(r, c)]],
        )
        self._draw_endpoint(sdraw, start, pad_left, pad_top)
        self._draw_endpoint(sdraw, goal, pad_left, pad_top)
        draw_grid_axis_labels(
            sdraw,
            rows=self.rows,
            cols=self.cols,
            cell_size=self.cell_size,
            pad_left=pad_left,
            pad_top=pad_top,
            color=GRID_AXIS_LABEL_COLOR,
        )
        canvas = finish_maze_scene(scene, canvas_dims)

        if path:
            thickness = max(2, self.cell_size // 3)
            points = [
                (
                    pad_left + c * self.cell_size + self.cell_size / 2,
                    pad_top + r * self.cell_size + self.cell_size / 2,
                )
                for r, c in path
            ]
            draw_path_line(canvas, points, LINE_COLOR, thickness, joint_style="square")

        return canvas

    def _draw_endpoint(
        self,
        sdraw,
        cell: Cell,
        pad_left: int,
        pad_top: int,
    ) -> None:
        r, c = cell
        left = pad_left + c * self.cell_size
        top = pad_top + r * self.cell_size
        right = left + self.cell_size
        bottom = top + self.cell_size
        inset = max(2, self.cell_size // 9)
        width = max(2, self.cell_size // 10)
        sdraw.rectangle(
            (left + inset, top + inset, right - 1 - inset, bottom - 1 - inset),
            outline=START_COLOR,
            width=width,
        )

    def _authored_route_reasoning(
        self,
        path_ids: Sequence[object],
        extra_payload: Dict[str, object],
    ) -> Optional[str]:
        color_map = extra_payload.get("cell_colors")
        if not isinstance(color_map, dict):
            return None
        path = [self._id_to_cell[int(cell_id)] for cell_id in path_ids]
        colors = [str(color_map.get(str(int(cell_id)), "?")) for cell_id in path_ids]
        start_xy = cell_to_xy(path[0])
        goal_xy = cell_to_xy(path[-1])
        moves = len(path) - 1
        sentences: List[str] = [
            f"Both red-outlined endpoints read as blue: the start at {start_xy} and the goal at {goal_xy}. "
            "Every move has to land on the opposite colour of the current cell, so from the blue start the "
            "route must step onto green, then blue, and so on, which rules out several short corridors whose "
            "colours repeat.",
        ]
        connectors = (
            "It then runs {direction}",
            "From there it continues {direction}",
            "Next it turns {direction}",
            "The route then heads {direction}",
        )
        runs = route_run_breakdown(path)
        cell_index = {cell: idx for idx, cell in enumerate(path)}
        for run_index, (direction, cells) in enumerate(runs):
            exit_xy = cell_to_xy(cells[-1])
            steps = len(cells) - 1
            exit_color = colors[cell_index[cells[-1]]]
            if run_index == 0:
                opener = f"From the start, the route steps {direction}"
            else:
                opener = connectors[(run_index - 1) % len(connectors)].format(direction=direction)
            span = f"one cell to {exit_xy}" if steps == 1 else f"{steps} cells to {exit_xy}"
            if run_index == len(runs) - 1:
                sentences.append(f"{opener} {span}, entering the goal on the required blue step.")
            else:
                sentences.append(
                    f"{opener} {span}, landing on {exit_color} with the alternation intact."
                )
        sentences.append(
            f"The route makes {moves} moves in total, an even count, so it arrives on blue exactly at the "
            "blue goal and every step alternated colours legally."
        )
        return " ".join(sentences)

    # ------------------------------------------------------------------
    # Geometry helpers (mirror maze_square convention)
    # ------------------------------------------------------------------

    def _get_cell_id(self, r: int, c: int) -> int:
        return self._cell_to_id.get((r, c), 0)

    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        row, col = self._id_to_cell[cell_id]
        pad_left, pad_top, _, _, _ = self._compute_padding()
        x = pad_left + col * self.cell_size + self.cell_size / 2.0
        y = pad_top + row * self.cell_size + self.cell_size / 2.0
        return (x, y)

    def _compute_padding(self) -> Tuple[int, int, int, int, Tuple[int, int]]:
        pad_left, pad_top, pad_right, pad_bottom = self.padding
        return pad_left, pad_top, pad_right, pad_bottom, self.canvas_dimensions

    def _compute_cell_bboxes(
        self, pad_left: int, pad_top: int
    ) -> List[List[Tuple[int, int, int, int]]]:
        bboxes: List[List[Tuple[int, int, int, int]]] = []
        for r in range(self.rows):
            row_boxes: List[Tuple[int, int, int, int]] = []
            for c in range(self.cols):
                left = pad_left + c * self.cell_size
                top = pad_top + r * self.cell_size
                row_boxes.append((left, top, left + self.cell_size, top + self.cell_size))
            bboxes.append(row_boxes)
        return bboxes


__all__ = ["MazeColoredZoneGenerator"]

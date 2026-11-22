"""Maze puzzle generator for elevation-constraint pathfinding tasks.

Unlike a DFS-carved maze, this variant generates an open grid where most
cells are walkable but carry elevation levels 0–4. Movement between adjacent
cells is permitted only when their elevation difference is at most 1.

The quality gate compares the elevation-constrained shortest route with the
ordinary geometry-only shortest distance on the same walls. A sample is kept
only when the constrained route is strictly longer, which proves that every
ordinary BFS shortest route contains at least one illegal elevation jump.
Generation constructs a legal rising-and-falling detour while off-route cells
alternate between cliff-separated extremes, then independently resolves the
full elevation graph before retaining the sample.

The five shade meanings are stated in the Prompt so the scene itself carries
elevation shades without numbering the cells. The only glyphs in the frame are
the 1-based row numbers down the left gutter and column numbers across the top
gutter, which name the model-facing ``(x, y)`` grid positions of the answer.
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
)

WALL = 1
PATH = 0

# Elevation → RGB: very light green (low) → very dark green (high)
ELEVATION_COLORS = [
    (230, 255, 230),  # 0 — lowland
    (180, 220, 180),  # 1
    (130, 185, 130),  # 2
    (80,  150, 80),   # 3
    (40,  110, 40),   # 4 — highland
]
START_COLOR = MAZE_ENDPOINT_COLOR
GOAL_COLOR = MAZE_ENDPOINT_COLOR
LINE_COLOR = MAZE_ROUTE_COLOR

MAX_ELEVATION = 4
WALKABLE_FRACTION = 0.70  # target fraction of interior cells that are walkable
MAX_ELEVATION_STEP = 1    # largest allowed elevation difference between neighbours

Cell = Tuple[int, int]


class MazeElevationGenerator(PathTaskGenerator):
    """Generate open-grid maze puzzles with elevation-based movement limits."""

    DEFAULT_OUTPUT_DIR = "outputs/spatial_navigation/maze_elevation"
    DEFAULT_ROWS = 11
    DEFAULT_COLS = 11
    DEFAULT_CELL_SIZE = 32
    ANSWER_COORDINATE_SPACE = True

    ELEVATION_RULE = (
        "Both red-outlined lightest-green endpoints sit at elevation 0. Move one cell at a time to a "
        "side-adjacent walkable cell whose elevation differs from the current cell by at most 1; dark slate "
        f"cells are impassable. You may wait for a step. {GRID_PATH_STAY_ENCODING_SENTENCE} Any complete legal "
        "route from the upper-left start to the lower-right goal is accepted."
    )
    DEFAULT_TI2V_PROMPT = (
        f"Rule: {ELEVATION_RULE} Required Output: Starting from the unchanged input scene, draw one thick red "
        "square-cornered route step by step through the centres of the visited cells until it reaches the goal. "
        "A small white-ringed red disc rides on the tip of the route and comes to rest on the goal cell."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        f"Rule: {ELEVATION_RULE} Required Output: Draw the complete thick red square-cornered route through "
        "the centres of the visited cells, with a small white-ringed red disc resting on the goal cell where "
        "the route ends."
    )

    PUZZLE_INPUT_DESCRIPTION = (
        "The input image shows an elevation maze whose walkable cells are shaded in five tones of green from "
        "lightest (elevation 0) to darkest (elevation 4), with dark slate wall cells, a red-outlined "
        "lightest-green start cell, and a red-outlined lightest-green goal cell."
    )
    REASONING_INPUT_DESCRIPTION = (
        "The input image shows the elevation maze: every walkable cell is shaded in one of five tones of green "
        "from lightest (elevation 0) to darkest (elevation 4). The upper-left red-outlined level-0 cell is the "
        "start and the lower-right red-outlined level-0 cell is the goal. Blue numbers run along the left and "
        "top edges of the grid. " + GRID_XY_CONVENTION_SENTENCE
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
        # Keep odd dimensions for aesthetic symmetry (optional but mirrors square)
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

        maze_grid, elevations, path = self._generate_with_quality_gate(
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
            maze_grid, elevations,
            start=start, goal=goal, path=None,
            padding=(pad_left, pad_top), canvas_dims=canvas_dims,
        )
        solution_image = self._render(
            maze_grid, elevations,
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

        # Serialize elevations as {cell_id: elevation}
        elev_map: Dict[str, int] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if maze_grid[r][c] == PATH:
                    cid = str(self._get_cell_id(r, c))
                    elev_map[cid] = elevations[(r, c)]

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
            "cell_elevations": elev_map,
            "max_elevation_step": MAX_ELEVATION_STEP,
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
        # Keep the bar proportional but not too tall for small grids.
        return max(8, (self.rows + self.cols) // 2)

    def _generate_with_quality_gate(
        self,
        start: Cell,
        goal: Cell,
        min_path_length: int,
    ) -> Tuple[List[List[int]], Dict[Cell, int], List[Cell]]:
        for walkable_fraction, attempts in (
            (WALKABLE_FRACTION, 360),
            (0.90, 180),
        ):
            for _ in range(attempts):
                maze_grid = self._generate_wall_grid(walkable_frac=walkable_fraction)
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
                    elevations = self._assign_load_bearing_elevations(
                        maze_grid,
                        detour,
                        start=start,
                        goal=goal,
                    )
                    path = self._bfs_elevation(maze_grid, elevations, start, goal)
                    # The cliff mechanic must force a real detour, not a
                    # two-cell parity nudge, or the sample reads at a glance.
                    if len(path) < min_path_length or len(path) < len(geometric_path) + 4:
                        continue
                    return maze_grid, elevations, path
        raise RuntimeError(
            "Failed to generate an elevation maze satisfying the quality gate"
        )

    def _generate_wall_grid(self, walkable_frac: float = WALKABLE_FRACTION) -> List[List[int]]:
        """Generate the random wall geometry used by the ablation gate."""
        rows, cols = self.rows, self.cols
        grid = [
            [PATH if self.rng.random() < walkable_frac else WALL for _ in range(cols)]
            for _ in range(rows)
        ]
        grid[0][0] = PATH
        grid[rows - 1][cols - 1] = PATH
        return grid

    def _assign_load_bearing_elevations(
        self,
        grid: List[List[int]],
        legal_route: Sequence[Cell],
        *,
        start: Cell,
        goal: Cell,
    ) -> Dict[Cell, int]:
        """Create a legal detour surrounded by alternating 0↔4 cliff cells."""
        route_index = {cell: index for index, cell in enumerate(legal_route)}
        route_moves = len(legal_route) - 1
        elevations: Dict[Cell, int] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                cell = (r, c)
                if grid[r][c] != PATH:
                    continue
                index = route_index.get(cell)
                if index is not None:
                    elevations[cell] = min(index, route_moves - index, MAX_ELEVATION)
                else:
                    parity = (abs(r - start[0]) + abs(c - start[1])) % 2
                    elevations[cell] = MAX_ELEVATION if parity == 0 else 0
        elevations[start] = 0
        elevations[goal] = 0
        return elevations

    def _bfs_geometric(
        self,
        grid: List[List[int]],
        start: Cell,
        goal: Cell,
        *,
        blocked_edge: Optional[Tuple[Cell, Cell]] = None,
    ) -> List[Cell]:
        """Return an ordinary shortest route that ignores all elevations."""
        if grid[start[0]][start[1]] != PATH or grid[goal[0]][goal[1]] != PATH:
            return []
        blocked = (
            {blocked_edge, (blocked_edge[1], blocked_edge[0])}
            if blocked_edge is not None
            else set()
        )
        queue: deque[Cell] = deque([start])
        parents: Dict[Cell, Optional[Cell]] = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (cell[0] + dr, cell[1] + dc)
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue
                if (cell, neighbor) in blocked:
                    continue
                if grid[neighbor[0]][neighbor[1]] != PATH or neighbor in parents:
                    continue
                parents[neighbor] = cell
                queue.append(neighbor)
        if goal not in parents:
            return []
        node: Optional[Cell] = goal
        result: List[Cell] = []
        while node is not None:
            result.append(node)
            node = parents[node]
        result.reverse()
        return result

    def _bfs_elevation(
        self,
        grid: List[List[int]],
        elevations: Dict[Cell, int],
        start: Cell,
        goal: Cell,
    ) -> List[Cell]:
        """BFS where a move (r,c)→(nr,nc) is valid iff
        abs(elevation[r,c] - elevation[nr,nc]) <= MAX_ELEVATION_STEP."""
        if grid[start[0]][start[1]] != PATH or grid[goal[0]][goal[1]] != PATH:
            return []
        queue: deque[Cell] = deque([start])
        parents: Dict[Cell, Optional[Cell]] = {start: None}
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                break
            e_cur = elevations.get((r, c), 0)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if grid[nr][nc] == PATH and (nr, nc) not in parents:
                        e_nbr = elevations.get((nr, nc), 0)
                        if abs(e_cur - e_nbr) <= MAX_ELEVATION_STEP:
                            parents[(nr, nc)] = (r, c)
                            queue.append((nr, nc))

        if goal not in parents:
            return []
        node: Optional[Cell] = goal
        result: List[Cell] = []
        while node is not None:
            result.append(node)
            node = parents[node]
        result.reverse()
        return result

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        grid: Sequence[Sequence[int]],
        elevations: Dict[Cell, int],
        *,
        start: Cell,
        goal: Cell,
        path: Optional[Sequence[Cell]],
        padding: Tuple[int, int],
        canvas_dims: Tuple[int, int],
    ) -> Image.Image:
        pad_left, pad_top = padding
        scene, sdraw = begin_maze_scene(canvas_dims)
        draw_square_cell_grid(
            sdraw,
            rows=self.rows,
            cols=self.cols,
            cell_size=self.cell_size,
            pad_left=pad_left,
            pad_top=pad_top,
            is_wall=lambda r, c: grid[r][c] != PATH,
            open_fill=lambda r, c: ELEVATION_COLORS[elevations.get((r, c), 0)],
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
        elev_map = extra_payload.get("cell_elevations")
        if not isinstance(elev_map, dict):
            return None
        path = [self._id_to_cell[int(cell_id)] for cell_id in path_ids]
        levels = [int(elev_map.get(str(int(cell_id)), 0)) for cell_id in path_ids]
        start_xy = cell_to_xy(path[0])
        goal_xy = cell_to_xy(path[-1])
        peak = max(levels)
        sentences: List[str] = [
            f"Both red-outlined endpoints sit at elevation 0: the start at {start_xy} and the goal at "
            f"{goal_xy}. A step may change elevation by at most one level, and the cells beside the direct "
            "corridors jump between the lightest and darkest greens, so the route has to follow the gradual "
            "slope instead of cutting across a cliff.",
        ]
        connectors = (
            "It then continues {direction}",
            "From there it runs {direction}",
            "Next it heads {direction}",
            "The slope then leads {direction}",
        )
        runs = route_run_breakdown(path)
        cell_index = {cell: idx for idx, cell in enumerate(path)}
        for run_index, (direction, cells) in enumerate(runs):
            exit_xy = cell_to_xy(cells[-1])
            steps = len(cells) - 1
            entry_level = levels[cell_index[cells[0]]]
            exit_level = levels[cell_index[cells[-1]]]
            if run_index == 0:
                opener = f"From the start, the route climbs {direction}" if exit_level > entry_level else f"From the start, the route moves {direction}"
            else:
                opener = connectors[(run_index - 1) % len(connectors)].format(direction=direction)
            span = f"one cell to {exit_xy}" if steps == 1 else f"{steps} cells to {exit_xy}"
            if exit_level > entry_level:
                slope = f"rising level by level to elevation {exit_level}"
            elif exit_level < entry_level:
                slope = f"descending level by level to elevation {exit_level}"
            else:
                slope = f"staying at elevation {exit_level}"
            if run_index == len(runs) - 1:
                sentences.append(f"{opener} {span}, {slope}, and arrives at the goal.")
            else:
                sentences.append(f"{opener} {span}, {slope}.")
        sentences.append(
            f"The route crests at elevation {peak} and eases back down to 0, so no step ever changes height "
            "by more than one level and the goal is reached legally."
        )
        return " ".join(sentences)

    # ------------------------------------------------------------------
    # Geometry helpers
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


__all__ = ["MazeElevationGenerator"]

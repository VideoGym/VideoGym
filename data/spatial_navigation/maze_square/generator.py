"""Maze puzzle generator for grid-based path drawing tasks."""

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

MAZE_RULE = (
    "Move one cell at a time between side-adjacent white cells: up, down, left, or right, with no diagonal "
    "moves, no moves through dark slate cells, and no moves off the grid. You may wait for a step. "
    f"{GRID_PATH_STAY_ENCODING_SENTENCE} Any complete legal route from the red start to the red goal is accepted."
)


class MazeGenerator(PathTaskGenerator):
    """Generate maze puzzles that require drawing a path from start to goal."""

    DEFAULT_OUTPUT_DIR = "outputs/spatial_navigation/maze_square"
    DEFAULT_ROWS = 11
    DEFAULT_COLS = 11
    DEFAULT_CELL_SIZE = 32
    ANSWER_COORDINATE_SPACE = True
    DEFAULT_TI2V_PROMPT = (
        f"Rule: {MAZE_RULE} Required Output: Starting from the unchanged input scene, draw one thick red "
        "square-cornered route step by step through the centres of the visited white cells until it reaches the "
        "goal. A small white-ringed red disc rides on the tip of the route and marks the current position; it "
        "comes to rest on the goal cell."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        f"Rule: {MAZE_RULE} Required Output: Draw the complete thick red square-cornered route through the "
        "centres of the visited white cells, with a small white-ringed red disc resting on the goal cell where "
        "the route ends."
    )
    PUZZLE_INPUT_DESCRIPTION = (
        "The input image shows a square maze of white walkable cells and dark slate wall cells on a white "
        "canvas, with a red start square and a red goal square."
    )
    REASONING_INPUT_DESCRIPTION = (
        "The input image shows a square maze of white walkable cells and dark slate wall cells. The upper-left "
        "red square is the start and the lower-right red square is the goal. Blue numbers run along the left and "
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
        path_cells_range: Optional[Sequence[int]] = None,
        aspect_ratio: Optional[float] = None,
        seed: Optional[int] = None,
        ti2v_prompt: Optional[str] = None,
        video: bool = False,
    ) -> None:
        canvas_width = 512 if canvas_width is None else int(canvas_width)
        # ``rows``/``cols`` are the playable grid dimensions; the carve helper
        # adds its own temporary boundary ring and strips it before play.
        # A dimension of 5 stays 5 after the parity adjustment below, and a
        # 5-wide perfect maze is a 3x3 lattice of rooms joined by single
        # corridors: the longest possible corner-to-corner route is 9 rooms
        # plus 8 corridors = 17 cells, so it can never reach the 18-cell
        # quality gate in ``_minimum_solution_path_cells``.  6 is the smallest
        # dimension that generates (parity lifts it to 7).
        if rows < 6 or cols < 6:
            raise ValueError(
                "rows and cols must be at least 6; a 5-cell dimension caps the longest route at 17 cells "
                "and can never satisfy the path-length quality gate"
            )
        if rows > 21 or cols > 21:
            raise ValueError(
                "rows and cols must be at most 21 within the 512-pixel scene"
            )
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
        self.path_cells_range = tuple(map(int, path_cells_range)) if path_cells_range is not None else None
        if self.path_cells_range is not None and not (
            len(self.path_cells_range) == 2
            and self._minimum_solution_path_cells() <= self.path_cells_range[0] <= self.path_cells_range[1]
            and self.path_cells_range[1] <= self.rows * self.cols
        ):
            raise ValueError("path_cells_range must respect the task's minimum and grid capacity")

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> PathTaskRecord:
        puzzle_uuid = puzzle_id or self.next_id()
        start = (0, 0)
        goal = (self.rows - 1, self.cols - 1)
        min_path_length = self._minimum_solution_path_cells()
        maze_grid, path = self._generate_maze_with_quality_gate(start, goal, min_path_length)
        self._rebuild_cell_id_index(
            [(r, c) for r in range(self.rows) for c in range(self.cols) if maze_grid[r][c] == PATH]
        )

        pad_left, pad_top, pad_right, pad_bottom, canvas_dims = self._compute_padding()
        cell_bboxes = self._compute_cell_bboxes(pad_left, pad_top)

        # The edge row/column numbers are solver-visible input evidence, so the
        # problem scene is identical in every modality; only the drawn route
        # separates the terminal image from the input.
        reasoning_image = self._render_maze(
            maze_grid,
            start=start,
            goal=goal,
            path=None,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
        )
        solution_image = self._render_maze(
            maze_grid,
            start=start,
            goal=goal,
            path=path,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
        )

        puzzle_path, solution_path = self.save_images(puzzle_uuid, reasoning_image, solution_image)
        reasoning_path = self.save_reasoning_image(puzzle_uuid, reasoning_image)

        video_path = None
        if self.video:
            path_points = [self._cell_center(self._get_cell_id(*cell)) for cell in path]
            video_path = self.save_video(
                puzzle_uuid,
                reasoning_image,
                path_points,
                thickness=max(3, self.cell_size // 3),
                joint_style="square",
            )

        start_point = self._cell_center(self._get_cell_id(*start))
        goal_point = self._cell_center(self._get_cell_id(*goal))
        extra_payload: Dict[str, object] = {
            "grid_size": [self.rows, self.cols],
            "cell_size": self.cell_size,
            "maze_grid": maze_grid,
            "start": list(start),
            "goal": list(goal),
            "cell_bboxes": cell_bboxes,
            "padding": [pad_left, pad_top, pad_right, pad_bottom],
            "solution_path_cell_ids": [self._get_cell_id(*cell) for cell in path],
            "path_joint_style": "square",
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

    def _authored_route_reasoning(
        self,
        path_ids: Sequence[object],
        extra_payload: Dict[str, object],
    ) -> Optional[str]:
        grid = extra_payload.get("maze_grid")
        if not isinstance(grid, list):
            return None
        path = [self._id_to_cell[int(cell_id)] for cell_id in path_ids]
        start_xy = cell_to_xy(path[0])
        goal_xy = cell_to_xy(path[-1])

        def is_open(row: int, col: int) -> bool:
            return 0 <= row < self.rows and 0 <= col < self.cols and grid[row][col] == PATH

        sentences: List[str] = [
            f"The maze is {self.rows} rows by {self.cols} columns; the red start sits at {start_xy} and the "
            f"red goal at {goal_xy}. Dark slate cells are impassable, so the route has to thread the white "
            "corridors between them.",
        ]
        deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        connectors = (
            "It then continues {direction}",
            "Next it runs {direction}",
            "The corridor then leads {direction}",
            "From there it heads {direction}",
        )
        runs = route_run_breakdown(path)
        for run_index, (direction, cells) in enumerate(runs):
            exit_xy = cell_to_xy(cells[-1])
            steps = len(cells) - 1
            if run_index == 0:
                opener = f"From the start, the open corridor leads {direction}"
            else:
                opener = connectors[(run_index - 1) % len(connectors)].format(direction=direction)
            span = f"one cell to {exit_xy}" if steps == 1 else f"{steps} cells to {exit_xy}"
            if run_index == len(runs) - 1:
                sentences.append(f"{opener} {span}, arriving at the goal.")
                continue
            dr, dc = deltas[direction]
            ahead_row, ahead_col = cells[-1][0] + dr, cells[-1][1] + dc
            if not (0 <= ahead_row < self.rows and 0 <= ahead_col < self.cols):
                blocker = "the grid edge"
            elif grid[ahead_row][ahead_col] != PATH:
                blocker = "a slate cell"
            else:
                blocker = None
            next_direction = runs[run_index + 1][0]
            open_sides = sum(
                1
                for side_dr, side_dc in deltas.values()
                if (side_dr, side_dc) != (-dr, -dc) and is_open(cells[-1][0] + side_dr, cells[-1][1] + side_dc)
            )
            if blocker is not None and open_sides <= 1:
                turn = f"where {blocker} forces a turn {next_direction}"
            elif blocker is not None:
                turn = f"where {blocker} blocks the way ahead and the route turns {next_direction}"
            else:
                turn = f"where the corridor branches and the route takes the {next_direction} branch"
            sentences.append(f"{opener} {span}, {turn}.")
        sentences.append(
            f"Every step moved between side-adjacent white cells, so the route from {start_xy} to {goal_xy} "
            "is complete and legal."
        )
        return " ".join(sentences)

    def _minimum_solution_path_cells(self) -> int:
        return max(18, self.rows + self.cols + 4)

    def _generate_maze_with_quality_gate(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        min_path_length: int,
    ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        for _ in range(160):
            maze_grid = self._generate_maze()
            path = self._bfs_path(maze_grid, start, goal)
            if len(path) >= min_path_length and (
                self.path_cells_range is None or self.path_cells_range[0] <= len(path) <= self.path_cells_range[1]
            ):
                return maze_grid, path
        raise RuntimeError("Failed to generate a square maze that satisfies the path-length quality gate")

    def _generate_maze(self) -> List[List[int]]:
        ringed_rows = self.rows + 2
        ringed_cols = self.cols + 2
        grid = [[WALL for _ in range(ringed_cols)] for _ in range(ringed_rows)]

        def carve(r: int, c: int) -> None:
            grid[r][c] = PATH
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            self.rng.shuffle(directions)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < ringed_rows - 1 and 1 <= nc < ringed_cols - 1 and grid[nr][nc] == WALL:
                    grid[r + dr // 2][c + dc // 2] = PATH
                    carve(nr, nc)

        carve(1, 1)
        # Ensure goal cell is reachable
        grid[ringed_rows - 2][ringed_cols - 2] = PATH
        return strip_outer_ring(grid)

    def _bfs_path(
        self,
        grid: List[List[int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        queue: deque[Tuple[int, int]] = deque([start])
        parents = {start: None}
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if grid[nr][nc] == PATH and (nr, nc) not in parents:
                        parents[(nr, nc)] = (r, c)
                        queue.append((nr, nc))

        if goal not in parents:
            return []
        node = goal
        result: List[Tuple[int, int]] = []
        while node is not None:
            result.append(node)
            node = parents[node]
        result.reverse()
        return result

    def _get_cell_id(self, r: int, c: int) -> int:
        return self._cell_to_id.get((r, c), 0)

    def _compute_padding(self) -> Tuple[int, int, int, int, Tuple[int, int]]:
        # Reuse the resolved canvas from __init__ so later renders keep any extra
        # padding added to hit the requested final canvas size.
        pad_left, pad_top, pad_right, pad_bottom = self.padding
        return pad_left, pad_top, pad_right, pad_bottom, self.canvas_dimensions

    def _compute_cell_bboxes(self, pad_left: int, pad_top: int) -> List[List[Tuple[int, int, int, int]]]:
        bboxes: List[List[Tuple[int, int, int, int]]] = []
        for r in range(self.rows):
            row_boxes: List[Tuple[int, int, int, int]] = []
            for c in range(self.cols):
                left = pad_left + c * self.cell_size
                top = pad_top + r * self.cell_size
                row_boxes.append((left, top, left + self.cell_size, top + self.cell_size))
            bboxes.append(row_boxes)
        return bboxes

    def _render_maze(
        self,
        grid: Sequence[Sequence[int]],
        *,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        path: Optional[Sequence[Tuple[int, int]]],
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
        )
        for cell, color in ((start, START_COLOR), (goal, GOAL_COLOR)):
            left = pad_left + cell[1] * self.cell_size
            top = pad_top + cell[0] * self.cell_size
            sdraw.rectangle(
                (left, top, left + self.cell_size - 1, top + self.cell_size - 1),
                fill=color,
            )
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
            thickness = max(3, self.cell_size // 3)
            points = [
                (
                    pad_left + c * self.cell_size + self.cell_size / 2,
                    pad_top + r * self.cell_size + self.cell_size / 2,
                )
                for r, c in path
            ]
            draw_path_line(canvas, points, LINE_COLOR, thickness, joint_style="square")

        return canvas

    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        row, col = self._id_to_cell[cell_id]
        pad_left, pad_top, _, _, _ = self._compute_padding()
        x = pad_left + col * self.cell_size + self.cell_size / 2.0
        y = pad_top + row * self.cell_size + self.cell_size / 2.0
        return (x, y)


__all__ = ["MazeGenerator"]

"""Toroidal Maze generator with wrap-around boundaries."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from data.common.grid_coordinates import (
    GRID_PATH_STAY_ENCODING_SENTENCE,
)
from data.common.pipeline.path_tasks import (
    GRID_AXIS_LABEL_COLOR,
    _draw_pawn_marker,
    cell_to_xy,
    GRID_XY_CONVENTION_SENTENCE,
    PathTaskGenerator,
    PathTaskRecord,
    draw_grid_axis_labels,
    draw_path_line,
    grid_axis_label_layout,
)

WALL = 1
PATH = 0

from data.spatial_navigation.core.maze_style import (
    begin_maze_scene,
    draw_square_cell_grid,
    finish_maze_scene,
    route_run_breakdown,
)

START_COLOR = (220, 30, 30)
GOAL_COLOR = (220, 30, 30)
LINE_COLOR = (220, 30, 30)

Cell = Tuple[int, int]


class MazeToroidalGenerator(PathTaskGenerator):
    """Wrap-around boundaries (Left <-> Right, Top <-> Bottom)."""

    DEFAULT_OUTPUT_DIR = "outputs/spatial_navigation/maze_toroidal"
    DEFAULT_ROWS = 11
    DEFAULT_COLS = 11
    DEFAULT_CELL_SIZE = 32
    ANSWER_COORDINATE_SPACE = True
    TOROIDAL_RULE = (
        "Movement is only between side-adjacent white cells; dark slate cells are impassable. The grid wraps "
        "around: a route that steps off the left edge immediately re-enters from the right edge on the same "
        "numbered row, and one that steps off the top edge re-enters from the bottom edge in the same numbered "
        f"column, and vice versa. You may wait for a step. {GRID_PATH_STAY_ENCODING_SENTENCE} Any complete legal "
        "route from the red start to the red goal is accepted."
    )
    DEFAULT_TI2V_PROMPT = (
        f"Rule: {TOROIDAL_RULE} Required Output: Starting from the unchanged input scene, draw one thick red "
        "square-cornered route step by step through the centres of the visited cells. When the route wraps "
        "across an edge, the stroke stops at the exit cell and continues from the matching cell on the opposite "
        "edge. A small white-ringed red disc rides on the tip of the route and comes to rest on the goal cell."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        f"Rule: {TOROIDAL_RULE} Required Output: Draw the complete thick red square-cornered route through the "
        "centres of the visited cells, breaking the stroke where the route wraps across an edge, with a small "
        "white-ringed red disc resting on the goal cell where the route ends."
    )
    PUZZLE_INPUT_DESCRIPTION = (
        "The input image shows a toroidal maze of white traversable cells and dark slate wall cells, with a red "
        "start square and a red goal square."
    )
    REASONING_INPUT_DESCRIPTION = (
        "The input image shows the toroidal maze. The upper-left red square is the start and the lower-right "
        "red square is the goal. Blue numbers run along the left and top edges of the grid. "
        + GRID_XY_CONVENTION_SENTENCE
    )
    MIN_WRAP_SEGMENT_CELLS = 10
    TARGET_BOUNDARY_OPENINGS = 6

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
        if rows < 9 or cols < 9:
            raise ValueError("rows and cols must be at least 9 for a readable, load-bearing wrap route")
        if rows > 13 or cols > 13:
            raise ValueError("rows and cols must be at most 13 so every edge row/column number stays readable")
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

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> PathTaskRecord:
        puzzle_uuid = puzzle_id or self.next_id()
        min_path_length = self._minimum_solution_path_cells()
        maze_grid, path, start, goal = self._generate_toroidal_maze(min_path_length=min_path_length)
        self._rebuild_cell_id_index(
            [(r, c) for r in range(self.rows) for c in range(self.cols) if maze_grid[r][c] == PATH]
        )

        pad_left, pad_top, _, _, canvas_dims = self._compute_padding()
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
            path_points = self._path_point_segments_with_breaks(path)
            video_path = self.save_video(
                puzzle_uuid,
                reasoning_image,
                path_points,
                thickness=max(3, self.cell_size // 3),
                joint_style="square",
            )

        start_point = self._cell_center(self._get_cell_id(*start))
        goal_point = self._cell_center(self._get_cell_id(*goal))

        extra_payload = {
            "grid_size": [self.rows, self.cols],
            "cell_size": self.cell_size,
            "maze_grid": maze_grid,
            "start": list(start),
            "goal": list(goal),
            "cell_bboxes": cell_bboxes,
            "solution_path_cell_ids": [self._get_cell_id(*cell) for cell in path],
            "wrap_jump_count": self._count_wrap_jumps(path),
            "uses_wraparound": self._count_wrap_jumps(path) > 0,
            "boundary_path_cell_count": self._count_boundary_path_cells(maze_grid),
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

    def _minimum_solution_path_cells(self) -> int:
        # A perfect odd-grid maze has 2*rooms-1 traversable cells.  Small
        # supported grids cannot satisfy the default absolute target plus the
        # ten-cell pre-bridge reserve, so scale the quality floor to reachable
        # topology instead of accepting a constructor configuration that only
        # fails after hundreds of rejections.
        traversable_capacity = (
            2 * ((self.rows - 1) // 2) * ((self.cols - 1) // 2) - 1
        )
        return max(
            12,
            min(self.rows + self.cols + 8, traversable_capacity - 14),
        )

    def _generate_toroidal_maze(self, *, min_path_length: int) -> Tuple[List[List[int]], List[Cell], Cell, Cell]:
        start = (1, 1)
        goal = (self.rows - 2, self.cols - 2)
        min_base_path_length = min_path_length + 10

        for _ in range(220):
            base_grid = self._generate_base_maze()
            base_path = self._bfs_plain_path(base_grid, start, goal)
            if len(base_path) < min_base_path_length:
                continue

            layout = self._inject_wrap_bridge(
                base_grid,
                base_path,
                start=start,
                goal=goal,
                min_path_length=min_path_length,
            )
            if layout is not None:
                grid, path = layout
                # The wrap must be the only way through: reject the rare
                # layout where a wrap-free planar route still exists.
                if self._bfs_plain_path(grid, start, goal):
                    continue
                return grid, path, start, goal

        raise RuntimeError("Failed to generate a toroidal maze that uses wrap-around with a sufficiently long path")

    def _generate_base_maze(self) -> List[List[int]]:
        grid = [[WALL for _ in range(self.cols)] for _ in range(self.rows)]

        def carve(r: int, c: int) -> None:
            grid[r][c] = PATH
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            self.rng.shuffle(directions)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < self.rows - 1 and 1 <= nc < self.cols - 1 and grid[nr][nc] == WALL:
                    grid[r + dr // 2][c + dc // 2] = PATH
                    carve(nr, nc)

        carve(1, 1)
        grid[self.rows - 2][self.cols - 2] = PATH
        return grid

    def _inject_wrap_bridge(
        self,
        grid: List[List[int]],
        base_path: Sequence[Cell],
        *,
        start: Cell,
        goal: Cell,
        min_path_length: int,
    ) -> Optional[Tuple[List[List[int]], List[Cell]]]:
        cut_indices = list(range(2, len(base_path) - 2))
        self.rng.shuffle(cut_indices)
        for cut_idx in cut_indices:
            cut_r, cut_c = base_path[cut_idx]
            if (cut_r, cut_c) in {start, goal}:
                continue

            candidate_grid = [row[:] for row in grid]
            candidate_grid[cut_r][cut_c] = WALL

            if self._bfs_plain_path(candidate_grid, start, goal):
                continue

            start_reachable = self._reachable_cells(candidate_grid, start)
            goal_reachable = self._reachable_cells(candidate_grid, goal)
            if goal in start_reachable or not start_reachable or not goal_reachable:
                continue

            wrap_options = self._candidate_wrap_options(start_reachable, goal_reachable)
            self.rng.shuffle(wrap_options)
            for orientation, index, _start_side in wrap_options:
                bridged_grid = [row[:] for row in candidate_grid]
                if orientation == "horizontal":
                    row = index
                    bridged_grid[row][0] = PATH
                    bridged_grid[row][self.cols - 1] = PATH
                else:
                    col = index
                    bridged_grid[0][col] = PATH
                    bridged_grid[self.rows - 1][col] = PATH

                path = self._bfs_toroidal_path(bridged_grid, start, goal)
                if not path:
                    continue
                if self._count_wrap_jumps(path) == 0:
                    continue
                if len(path) < min_path_length:
                    continue
                leading_cells, trailing_cells = self._wrap_segment_lengths(path)
                if leading_cells < self.MIN_WRAP_SEGMENT_CELLS or trailing_cells < self.MIN_WRAP_SEGMENT_CELLS:
                    continue
                bridged_grid, path = self._add_auxiliary_wrap_openings(
                    bridged_grid,
                    path=path,
                    start=start,
                    goal=goal,
                    orientation=orientation,
                    required_index=index,
                )
                if self._count_boundary_path_cells(bridged_grid) < self.TARGET_BOUNDARY_OPENINGS:
                    continue
                return bridged_grid, path
        return None

    def _add_auxiliary_wrap_openings(
        self,
        grid: List[List[int]],
        *,
        path: List[Cell],
        start: Cell,
        goal: Cell,
        orientation: str,
        required_index: int,
    ) -> Tuple[List[List[int]], List[Cell]]:
        augmented = [row[:] for row in grid]
        candidates: List[int] = []
        if orientation == "horizontal":
            for row in range(1, self.rows - 1):
                if row == required_index:
                    continue
                candidates.append(row)
        else:
            for col in range(1, self.cols - 1):
                if col == required_index:
                    continue
                candidates.append(col)
        self.rng.shuffle(candidates)
        current_openings = self._count_boundary_path_cells(augmented)
        target_openings = max(self.TARGET_BOUNDARY_OPENINGS, current_openings)
        for index in candidates:
            if current_openings >= target_openings:
                break
            trial = [row[:] for row in augmented]
            if orientation == "horizontal":
                trial[index][0] = PATH
                trial[index][self.cols - 1] = PATH
            else:
                trial[0][index] = PATH
                trial[self.rows - 1][index] = PATH
            trial_path = self._bfs_toroidal_path(trial, start, goal)
            if trial_path != path:
                continue
            augmented = trial
            current_openings = self._count_boundary_path_cells(augmented)
        return augmented, path

    def _candidate_wrap_options(
        self,
        start_reachable: set[Cell],
        goal_reachable: set[Cell],
    ) -> List[Tuple[str, int, str]]:
        options: List[Tuple[str, int, str]] = []
        for row in range(1, self.rows - 1):
            left_cell = (row, 1)
            right_cell = (row, self.cols - 2)
            if left_cell in start_reachable and right_cell in goal_reachable:
                options.append(("horizontal", row, "left"))
            if right_cell in start_reachable and left_cell in goal_reachable:
                options.append(("horizontal", row, "right"))
        for col in range(1, self.cols - 1):
            top_cell = (1, col)
            bottom_cell = (self.rows - 2, col)
            if top_cell in start_reachable and bottom_cell in goal_reachable:
                options.append(("vertical", col, "top"))
            if bottom_cell in start_reachable and top_cell in goal_reachable:
                options.append(("vertical", col, "bottom"))
        return options

    def _bfs_plain_path(self, grid: Sequence[Sequence[int]], start: Cell, goal: Cell) -> List[Cell]:
        queue = deque([start])
        parents = {start: None}
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return self._reconstruct_path(parents, goal)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr][nc] == PATH and (nr, nc) not in parents:
                    parents[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
        return []

    def _reachable_cells(self, grid: Sequence[Sequence[int]], start: Cell) -> set[Cell]:
        queue = deque([start])
        visited = {start}
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr][nc] == PATH and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return visited

    def _bfs_toroidal_path(self, grid: Sequence[Sequence[int]], start: Cell, goal: Cell) -> List[Cell]:
        queue = deque([start])
        parents = {start: None}
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return self._reconstruct_path(parents, goal)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = (r + dr) % self.rows, (c + dc) % self.cols
                if grid[nr][nc] == PATH and (nr, nc) not in parents:
                    parents[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
        return []

    @staticmethod
    def _reconstruct_path(parents: Dict[Cell, Optional[Cell]], goal: Cell) -> List[Cell]:
        path: List[Cell] = []
        node: Optional[Cell] = goal
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
        return path

    @staticmethod
    def _count_wrap_jumps(path: Sequence[Cell]) -> int:
        count = 0
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            if abs(r1 - r2) > 1 or abs(c1 - c2) > 1:
                count += 1
        return count

    @staticmethod
    def _wrap_segment_lengths(path: Sequence[Cell]) -> Tuple[int, int]:
        for index, (previous, current) in enumerate(zip(path, path[1:])):
            if abs(previous[0] - current[0]) > 1 or abs(previous[1] - current[1]) > 1:
                leading = index + 1
                trailing = len(path) - leading
                return leading, trailing
        return 0, 0

    def _count_boundary_path_cells(self, grid: Sequence[Sequence[int]]) -> int:
        count = 0
        for row in range(self.rows):
            count += int(grid[row][0] == PATH)
            count += int(grid[row][self.cols - 1] == PATH)
        for col in range(1, self.cols - 1):
            count += int(grid[0][col] == PATH)
            count += int(grid[self.rows - 1][col] == PATH)
        return count

    def _get_cell_id(self, r: int, c: int) -> int:
        return self._cell_to_id.get((r, c), 0)

    def _compute_padding(self):
        return self.padding[0], self.padding[1], self.padding[2], self.padding[3], self.canvas_dimensions

    def _compute_cell_bboxes(self, pad_left, pad_top):
        bboxes = []
        for r in range(self.rows):
            row_boxes = []
            for c in range(self.cols):
                left = pad_left + c * self.cell_size
                top = pad_top + r * self.cell_size
                row_boxes.append((left, top, left + self.cell_size, top + self.cell_size))
            bboxes.append(row_boxes)
        return bboxes

    def _render_maze(
        self,
        grid,
        *,
        start,
        goal,
        path,
        padding,
        canvas_dims,
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
            draw = ImageDraw.Draw(canvas)
            thickness = max(2, self.cell_size // 3)
            for segment in self._path_point_segments(path):
                if len(segment) >= 2:
                    draw_path_line(canvas, segment, LINE_COLOR, thickness, joint_style="square", tip_marker=False)
                elif len(segment) == 1:
                    cx, cy = segment[0]
                    draw.rectangle((cx - thickness / 2, cy - thickness / 2, cx + thickness / 2, cy + thickness / 2), fill=LINE_COLOR)
            goal_center = self._cell_center(self._get_cell_id(*path[-1]))
            _draw_pawn_marker(draw, goal_center, thickness, LINE_COLOR)

        return canvas

    def _authored_route_reasoning(
        self,
        path_ids,
        extra_payload,
    ):
        grid = extra_payload.get("maze_grid")
        if not isinstance(grid, list):
            return None
        path = [self._id_to_cell[int(cell_id)] for cell_id in path_ids]
        start_xy = cell_to_xy(path[0])
        goal_xy = cell_to_xy(path[-1])
        sentences = [
            f"The red start sits at {start_xy} and the red goal at {goal_xy}. This maze wraps around: a step "
            "off one edge re-enters from the opposite edge on the same numbered row or column, and the walls "
            "block enough of the interior that the open boundary cells are the way through.",
        ]
        # Split the route at wrap jumps, narrate each straight stretch, and
        # call out every edge crossing explicitly.
        sub_walk: list = [path[0]]
        edge_names = {"up": ("top", "bottom"), "down": ("bottom", "top"), "left": ("left", "right"), "right": ("right", "left")}
        segment_openers = ("From the start, the route", "After the wrap, the route", "It then", "From there, the route")
        opener_index = 0

        def narrate_walk(cells: list, arrival: str) -> None:
            nonlocal opener_index
            if len(cells) < 2:
                return
            runs = route_run_breakdown(cells)
            parts = []
            for direction, run_cells in runs:
                steps = len(run_cells) - 1
                exit_xy = cell_to_xy(run_cells[-1])
                parts.append(f"{direction} {steps} cell{'s' if steps > 1 else ''} to {exit_xy}")
            opener = segment_openers[min(opener_index, len(segment_openers) - 1)]
            opener_index += 1
            sentences.append(f"{opener} moves {', then '.join(parts)}{arrival}.")

        for previous, cell in zip(path, path[1:]):
            dr = cell[0] - previous[0]
            dc = cell[1] - previous[1]
            if abs(dr) > 1 or abs(dc) > 1:
                narrate_walk(sub_walk, "")
                if abs(dr) > 1:
                    direction = "up" if dr > 0 else "down"
                else:
                    direction = "left" if dc > 0 else "right"
                exit_edge, entry_edge = edge_names[direction]
                exit_xy = cell_to_xy(previous)
                entry_xy = cell_to_xy(cell)
                sentences.append(
                    f"At {exit_xy} it steps {direction} off the {exit_edge} edge and re-enters from the "
                    f"{entry_edge} edge at {entry_xy}, staying on the same numbered "
                    f"{'row' if direction in ('left', 'right') else 'column'}."
                )
                sub_walk = [cell]
            else:
                sub_walk.append(cell)
        narrate_walk(sub_walk, ", arriving at the goal")
        wraps = int(extra_payload.get("wrap_jump_count") or 0)
        sentences.append(
            f"Every ordinary step moved between side-adjacent white cells and the route used the wrap "
            f"{wraps} time{'s' if wraps != 1 else ''}, so it reaches the goal legally."
        )
        return " ".join(sentences)

    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        row, col = self._id_to_cell[cell_id]
        pad_left, pad_top, _, _, _ = self._compute_padding()
        x = pad_left + col * self.cell_size + self.cell_size / 2.0
        y = pad_top + row * self.cell_size + self.cell_size / 2.0
        return (x, y)

    def _path_point_segments_with_breaks(
        self,
        path: Sequence[Cell],
    ) -> List[Optional[Tuple[float, float]]]:
        """Flatten wrap-aware pixel segments into a single point sequence with
        `None` sentinels at each wrap crossing so the base-class ``save_video``
        can animate each straight sub-path with sub-edge distance interpolation."""
        segments = self._path_point_segments(path)
        points: List[Optional[Tuple[float, float]]] = []
        for idx, segment in enumerate(segments):
            if idx > 0:
                points.append(None)
            points.extend(segment)
        return points

    def _path_point_segments(self, path: Sequence[Cell]) -> List[List[Tuple[float, float]]]:
        if not path:
            return []
        segments: List[List[Tuple[float, float]]] = [[self._cell_center(self._get_cell_id(*path[0]))]]
        for previous, current in zip(path, path[1:]):
            current_point = self._cell_center(self._get_cell_id(*current))
            if abs(previous[0] - current[0]) > 1 or abs(previous[1] - current[1]) > 1:
                exit_point, entry_point = self._wrap_boundary_points(previous, current)
                segments[-1].append(exit_point)
                segments.append([entry_point, current_point])
                continue
            segments[-1].append(current_point)
        return [segment for segment in segments if segment]

    def _wrap_boundary_points(self, previous: Cell, current: Cell) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        pad_left, pad_top, _, _, _ = self._compute_padding()
        prev_center = self._cell_center(self._get_cell_id(*previous))
        curr_center = self._cell_center(self._get_cell_id(*current))
        left_x = pad_left
        right_x = pad_left + self.cols * self.cell_size - 1
        top_y = pad_top
        bottom_y = pad_top + self.rows * self.cell_size - 1

        if previous[0] == current[0]:
            if previous[1] == 0 and current[1] == self.cols - 1:
                return (left_x, prev_center[1]), (right_x, curr_center[1])
            if previous[1] == self.cols - 1 and current[1] == 0:
                return (right_x, prev_center[1]), (left_x, curr_center[1])
        if previous[1] == current[1]:
            if previous[0] == 0 and current[0] == self.rows - 1:
                return (prev_center[0], top_y), (curr_center[0], bottom_y)
            if previous[0] == self.rows - 1 and current[0] == 0:
                return (prev_center[0], bottom_y), (curr_center[0], top_y)
        raise ValueError("wrap boundary points requested for a non-wrap transition")

__all__ = ["MazeToroidalGenerator"]

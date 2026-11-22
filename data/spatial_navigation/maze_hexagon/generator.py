"""Hexagonal maze generator using concentric rings on an axial grid."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw

from data.common.grid_coordinates import (
    GRID_PATH_STAY_ENCODING_SENTENCE,
)
from data.common.pipeline.media import load_font
from data.common.pipeline.path_tasks import (
    GRID_AXIS_LABEL_COLOR,
    GRID_AXIS_LABEL_GUTTER,
    PathTaskGenerator,
    PathTaskRecord,
    draw_path_line,
)
from data.common.render_primitives import DOCTRINE_SUPERSAMPLE

# Rendering palette
from data.spatial_navigation.core.maze_style import (
    MAZE_BODY_FILL,
    MAZE_EDGE_WALL,
    begin_maze_scene,
    finish_maze_scene,
)

PATH_COLOR = MAZE_BODY_FILL
WALL_COLOR = MAZE_EDGE_WALL
START_COLOR = (220, 30, 30)
GOAL_COLOR = START_COLOR
LINE_COLOR = (220, 30, 30)

Axial = Tuple[int, int]

# Axial direction vectors for pointy-top hexagons (q, r)
DIRECTIONS: Tuple[Axial, ...] = (
    (1, -1),
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
)

EDGE_CORNER_INDICES: Tuple[Tuple[int, int], ...] = (
    (5, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
)

SQRT_THREE = math.sqrt(3.0)


class MazeHexagonGenerator(PathTaskGenerator):
    """Generate mazes laid out on a finite hexagonal grid."""

    DEFAULT_OUTPUT_DIR = "outputs/spatial_navigation/maze_hexagon"
    ANSWER_COORDINATE_SPACE = True
    HEXAGON_RULE = (
        "A move is legal only between two hexagons that share an open edge; touching at a corner is not "
        "adjacency, and no move may cross a dark wall edge. The boundary red dot is the start and the centre "
        f"red dot is the goal. You may wait for a step. {GRID_PATH_STAY_ENCODING_SENTENCE} Any complete legal "
        "route from start to goal is accepted."
    )
    DEFAULT_TI2V_PROMPT = (
        f"Rule: {HEXAGON_RULE} Required Output: Starting from the unchanged input scene, trace one thick red "
        "route step by step through the centres of the visited hexagons until it reaches the goal, keeping both "
        "red endpoint dots visible. A small white-ringed red disc rides on the tip of the route and comes to "
        "rest on the goal hexagon."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        f"Rule: {HEXAGON_RULE} Required Output: Draw the complete thick red route through the centres of the "
        "visited hexagons, with a small white-ringed red disc resting on the goal hexagon where the route ends."
    )
    PUZZLE_INPUT_DESCRIPTION = (
        "The input image shows a honeycomb maze of light hexagons separated by dark wall edges on a white "
        "canvas, with a red start dot in a boundary hexagon and a red goal dot in the centre hexagon."
    )
    REASONING_INPUT_DESCRIPTION = (
        "The input image shows the hexagonal maze, with a red start dot in a boundary hexagon and a red goal "
        "dot in the centre hexagon. Blue numbers run along the left and top edges of the honeycomb: a left-edge "
        "number marks one horizontal level of hexagon centres counted from the top, and a top-edge number marks "
        "one column of hexagons counted from the left. A hexagon's position is written (x, y), where x is the "
        "left-edge number level its centre sits on and y is the top-edge number of its column, both starting at "
        "1; because the columns are staggered, neighbouring columns sit on alternating levels."
    )

    DEFAULT_RADIUS = 3
    DEFAULT_CELL_RADIUS = 38
    LABEL_GUTTER = GRID_AXIS_LABEL_GUTTER

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        radius: int = DEFAULT_RADIUS,
        cell_radius: Optional[int] = None,
        wall_thickness: Optional[int] = None,
        size: Optional[int] = None,
        seed: Optional[int] = None,
        ti2v_prompt: Optional[str] = None,
        canvas_width: Optional[int] = None,
        aspect: Optional[float] = None,
        video: bool = False,
    ) -> None:
        canvas_width = 512 if canvas_width is None else int(canvas_width)
        if radius < 2:
            raise ValueError("radius must be at least 2")

        self.radius = int(radius)
        self.cells: List[Axial] = []
        for q in range(-self.radius, self.radius + 1):
            for r in range(-self.radius, self.radius + 1):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= self.radius:
                    self.cells.append((q, r))
        if not self.cells:
            raise ValueError("No cells generated for the requested radius")
        self.cell_set: Set[Axial] = set(self.cells)
        self.cell_to_id: Dict[Axial, int] = {cell: i + 1 for i, cell in enumerate(self.cells)}

        # Precompute unit grid spread for dimension calculations.
        min_uq, max_uq, min_ur, max_ur = float("inf"), float("-inf"), float("inf"), float("-inf")
        for q, r in self.cells:
            ux = 1.5 * q
            uy = SQRT_THREE * (r + q / 2.0)
            min_uq = min(min_uq, ux)
            max_uq = max(max_uq, ux)
            min_ur = min(min_ur, uy)
            max_ur = max(max_ur, uy)
        spread_x = max_uq - min_uq
        spread_y = max_ur - min_ur

        target_cr = int(
            cell_radius if cell_radius is not None else (size if size is not None else self.DEFAULT_CELL_RADIUS)
        )
        is_user_set = (cell_radius is not None) or (size is not None)

        def _get_layout_size(cr: int) -> Tuple[int, int]:
            wt = int(wall_thickness if wall_thickness is not None else max(6, cr // 6))
            # Total width = (spread + 2)*cr + 2*(cr+wt)
            w = int(math.ceil((spread_x + 2.0) * cr + 2.0 * wt))
            h = int(math.ceil((spread_y + 2.0) * cr + 2.0 * wt))
            return w, h

        def _check_fits(cr: int, width_limit: Optional[int] = None) -> bool:
            limit = width_limit if width_limit is not None else canvas_width
            if limit is None:
                return True
            bw, bh = _get_layout_size(cr)
            # The left and top gutters carry the 1-based level and column
            # numbers that name every answer position, so the honeycomb body
            # must leave that band free on both axes.
            w_limit = int(limit) - self.LABEL_GUTTER
            if aspect is None:
                return bw <= w_limit and bh <= w_limit
            asp = float(aspect)
            min_w_needed = max(bw, int(math.ceil(bh * asp)))
            return w_limit >= min_w_needed - 1

        if not is_user_set:
            target_w = int(canvas_width) if canvas_width is not None else 512
            low, high = 12, max(12, target_w)
            best = 12
            while low <= high:
                mid = (low + high) // 2
                if _check_fits(mid, target_w):
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            target_cr = best

        if target_cr < 12:
            raise ValueError("cell_radius (or auto-sized) must be at least 12 pixels to preserve visible walls")

        self.cell_radius = target_cr
        self.wall_thickness = int(wall_thickness if wall_thickness is not None else max(6, self.cell_radius // 6))

        if self.wall_thickness <= 0:
            raise ValueError("wall_thickness must be positive")

        self.walk_radius = max(4.0, self.cell_radius - self.wall_thickness * 0.8)
        self.outer_margin = self.cell_radius + self.wall_thickness * 3

        self._centers = {cell: self._axial_to_pixel(cell) for cell in self.cells}

        # We rely on the precomputed formula matching strictly:
        base_width, base_height = _get_layout_size(self.cell_radius)

        final_width, final_height = self._resolve_canvas_dimensions(
            base_width + self.LABEL_GUTTER,
            base_height + self.LABEL_GUTTER,
            canvas_width,
            aspect,
        )
        aspect_for_super = final_width / final_height

        resolved_output = output_dir if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        super().__init__(
            resolved_output,
            canvas_width=final_width,
            aspect=aspect_for_super,
            size=self.cell_radius,
            seed=seed,
            ti2v_prompt=ti2v_prompt,
            video=video,
        )

        self.canvas_width_px = final_width
        self.canvas_height_px = final_height
        # Centre the honeycomb inside the canvas area that is left once the
        # left and top label gutters are reserved.
        self.center = (
            self.LABEL_GUTTER + (final_width - self.LABEL_GUTTER) / 2.0,
            self.LABEL_GUTTER + (final_height - self.LABEL_GUTTER) / 2.0,
        )
        # Model-facing axis extents: one column per hexagon column and one
        # level per distinct hexagon-centre height (columns are staggered, so
        # neighbouring columns occupy alternating levels).
        self.cols = 2 * self.radius + 1
        self.rows = 4 * self.radius + 1
        self.cells_perimeter: List[Axial] = [cell for cell in self.cells if self._distance(cell) == self.radius]

    # ------------------------------------------------------------------
    # Canvas sizing helpers

    def _resolve_canvas_dimensions(
        self,
        natural_width: int,
        natural_height: int,
        canvas_width: Optional[int],
        aspect: Optional[float],
    ) -> Tuple[int, int]:
        if natural_width <= 0 or natural_height <= 0:
            raise ValueError("Natural canvas dimensions must be positive")

        width: int
        height: int

        if aspect is not None:
            aspect_value = float(aspect)
            if aspect_value <= 0:
                raise ValueError("aspect must be positive")
            min_width = max(natural_width, int(math.ceil(natural_height * aspect_value)))
            if canvas_width is not None:
                width = int(canvas_width)
                if width < min_width:
                    raise ValueError("canvas_width is too small for the requested aspect ratio")
            else:
                width = min_width
            height = int(math.ceil(width / aspect_value))
            if height < natural_height:
                height = natural_height
                width = int(math.ceil(height * aspect_value))
        else:
            if canvas_width is not None:
                width = int(canvas_width)
                if width < natural_width or width < natural_height:
                    raise ValueError("canvas_width is too small for the hexagon layout")
            else:
                width = max(natural_width, natural_height)
            height = width
        return max(width, natural_width), max(height, natural_height)

    # ------------------------------------------------------------------
    # Public API

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> PathTaskRecord:
        goal_cell: Axial = (0, 0)
        min_path_length = self._minimum_solution_path_cells()
        passages, start_cell, solution = self._generate_passages_with_quality_gate(goal_cell, min_path_length)

        # The edge level/column numbers are solver-visible input evidence, so
        # the problem scene is identical in every modality; only the drawn
        # route separates the terminal image from the input.
        reasoning_image = self._render_maze(passages, start_cell, goal_cell, path=None)
        solution_image = self._render_maze(passages, start_cell, goal_cell, path=solution)

        record_id = puzzle_id or self.next_id()
        puzzle_path, solution_path = self.save_images(record_id, reasoning_image, solution_image)
        reasoning_path = self.save_reasoning_image(record_id, reasoning_image)

        video_path = None
        if self.video:
            path_points = [self._cell_center_from_cell(cell) for cell in solution]
            video_path = self.save_video(
                record_id,
                reasoning_image,
                path_points,
                thickness=self._route_thickness(),
                duration=4.0,
                supersample=DOCTRINE_SUPERSAMPLE,
            )

        start_point = self._cell_center_from_cell(start_cell)
        goal_point = self._cell_center_from_cell(goal_cell)

        extra_payload = {
            "radius": self.radius,
            "cell_radius": self.cell_radius,
            "wall_thickness": self.wall_thickness,
            "start_cell": list(start_cell),
            "goal_cell": list(goal_cell),
            "start_cell_id": self.cell_to_id[start_cell],
            "goal_cell_id": self.cell_to_id[goal_cell],
            # Source topology, not a solved route.  The verifier rebuilds the
            # shortest-path set from these visible open edges and deliberately
            # ignores the authored solution_path_cell_ids field.
            "open_edges": sorted(
                [
                    sorted((self.cell_to_id[cell], self.cell_to_id[neighbor]))
                    for cell, neighbors in passages.items()
                    for neighbor in neighbors
                    if self.cell_to_id[cell] < self.cell_to_id[neighbor]
                ]
            ),
            "solution_path_cell_ids": [self.cell_to_id[cell] for cell in solution],
        }
        self.add_path_length_metadata(extra_payload, solution, minimum=min_path_length)

        return self.build_record(
            record_id,
            start_point=start_point,
            goal_point=goal_point,
            puzzle_path=puzzle_path,
            solution_path=solution_path,
            reasoning_path=reasoning_path,
            ti2v_prompt=self.ti2v_prompt,
            extra=extra_payload,
            video_path=video_path,
        )

    def _authored_route_reasoning(
        self,
        path_ids,
        extra_payload,
    ):
        cells = []
        for cell_id in path_ids:
            index = int(cell_id) - 1
            if not 0 <= index < len(self.cells):
                return None
            cells.append(self.cells[index])
        labels = [self._cell_id_to_xy(int(cell_id)) for cell_id in path_ids]
        if any(label is None for label in labels):
            return None
        start_xy, goal_xy = labels[0], labels[-1]

        def step_direction(a, b):
            ax, ay = self._cell_center_from_cell(a)
            bx, by = self._cell_center_from_cell(b)
            dx, dy = bx - ax, by - ay
            if abs(dx) < 1e-6:
                return "down" if dy > 0 else "up"
            horizontal = "right" if dx > 0 else "left"
            vertical = "lower" if dy > 0 else "upper"
            return f"{vertical}-{horizontal}"

        runs = []
        current_direction = step_direction(cells[0], cells[1]) if len(cells) > 1 else None
        run_length = 1
        run_end = 1
        for index in range(1, len(cells)):
            direction = step_direction(cells[index - 1], cells[index])
            if direction == current_direction and index > 1:
                run_length += 1
            elif index > 1:
                runs.append((current_direction, run_length, labels[index - 1]))
                current_direction = direction
                run_length = 1
            run_end = index
        runs.append((current_direction, run_length, labels[run_end]))

        sentences = [
            f"The red start dot sits in the boundary hexagon {start_xy} and the red goal dot in the centre "
            f"hexagon {goal_xy}. Moves are only legal across open shared edges, and the dark wall edges close "
            "off the direct sides, so the route follows the open corridor of the honeycomb.",
        ]
        connectors = (
            "It then steps {direction}",
            "Next it moves {direction}",
            "From there it continues {direction}",
            "The corridor then bends {direction}",
        )
        for run_index, (direction, count, end_label) in enumerate(runs):
            hexes = f"{count} hexagon{'s' if count > 1 else ''}"
            if run_index == 0:
                opener = f"From the start, the route crosses its open edge {direction}"
            else:
                opener = connectors[(run_index - 1) % len(connectors)].format(direction=direction)
            if run_index == len(runs) - 1:
                sentences.append(f"{opener} {hexes} to {end_label}, arriving at the goal.")
            else:
                sentences.append(f"{opener} {hexes} to {end_label}.")
        sentences.append(
            "Every step crossed an open shared edge, never a wall edge or a corner, so the route is complete "
            "and legal."
        )
        return " ".join(sentences)

    def _minimum_solution_path_cells(self) -> int:
        return max(10, self.radius * 4)

    def _generate_passages_with_quality_gate(
        self,
        goal_cell: Axial,
        min_path_length: int,
    ) -> Tuple[Dict[Axial, Set[Axial]], Axial, List[Axial]]:
        for _ in range(120):
            passages = self._generate_passages()
            best_start: Optional[Axial] = None
            best_solution: List[Axial] = []
            for candidate in self.cells_perimeter:
                candidate_path = self._shortest_path(passages, candidate, goal_cell)
                if len(candidate_path) > len(best_solution):
                    best_start = candidate
                    best_solution = candidate_path
            if best_start is not None and len(best_solution) >= min_path_length:
                return passages, best_start, best_solution
        raise RuntimeError("Failed to generate a hexagonal maze that satisfies the path-length quality gate")

    # ------------------------------------------------------------------
    # Maze construction

    def _generate_passages(self) -> Dict[Axial, Set[Axial]]:
        passages: Dict[Axial, Set[Axial]] = {cell: set() for cell in self.cells}
        visited: Set[Axial] = set()

        def dfs(cell: Axial) -> None:
            visited.add(cell)
            neighbors = list(self._neighbors(cell))
            self.rng.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    passages[cell].add(neighbor)
                    passages[neighbor].add(cell)
                    dfs(neighbor)

        dfs((0, 0))
        return passages

    def _neighbors(self, cell: Axial) -> Iterable[Axial]:
        q, r = cell
        for dq, dr in DIRECTIONS:
            neighbor = (q + dq, r + dr)
            if neighbor in self.cell_set:
                yield neighbor

    def _shortest_path(
        self,
        passages: Dict[Axial, Set[Axial]],
        start: Axial,
        goal: Axial,
    ) -> List[Axial]:
        queue: Deque[Axial] = deque([start])
        parents: Dict[Axial, Optional[Axial]] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for neighbor in passages[current]:
                if neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)
        if goal not in parents:
            return []
        path: List[Axial] = []
        step: Optional[Axial] = goal
        while step is not None:
            path.append(step)
            step = parents[step]
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Rendering

    def _render_maze(
        self,
        passages: Dict[Axial, Set[Axial]],
        start_cell: Axial,
        goal_cell: Axial,
        *,
        path: Optional[Sequence[Axial]],
    ) -> Image.Image:
        scene, sdraw = begin_maze_scene((self.canvas_width_px, self.canvas_height_px))

        for cell in self.cells:
            center = self._cell_center_from_cell(cell)
            corners = self._hex_corners(center, self.walk_radius * 1.5)
            sdraw.polygon(corners, fill=PATH_COLOR)

        for cell in self.cells:
            self._draw_cell_walls(sdraw, cell, passages)

        self._draw_marker(sdraw, start_cell, START_COLOR)
        self._draw_marker(sdraw, goal_cell, GOAL_COLOR)
        self._draw_axis_labels(sdraw)
        # The route joins hexagon centres along six axes, so its segments are
        # oriented quads with round joints: they have to be laid down while the
        # scene is still supersampled, otherwise the finished canvas carries an
        # aliased stroke that no later downsample can repair.
        if path:
            self._draw_solution(scene, path, factor=sdraw.factor)

        return finish_maze_scene(scene, (self.canvas_width_px, self.canvas_height_px))

    def _draw_cell_walls(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Axial,
        passages: Dict[Axial, Set[Axial]],
    ) -> None:
        center = self._cell_center_from_cell(cell)
        outer_corners = self._hex_corners(center, self.cell_radius)
        for direction_index, delta in enumerate(DIRECTIONS):
            neighbor = (cell[0] + delta[0], cell[1] + delta[1])
            has_neighbor = neighbor in self.cell_set
            connected = has_neighbor and neighbor in passages[cell]
            if has_neighbor and connected:
                continue
            a_idx, b_idx = EDGE_CORNER_INDICES[direction_index]
            a = outer_corners[a_idx]
            b = outer_corners[b_idx]
            self._draw_wall_segment(draw, a, b)

    def _draw_wall_segment(
        self,
        draw: ImageDraw.ImageDraw,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return
        ux = dx / length
        uy = dy / length
        nx = -uy
        ny = ux
        half = self.wall_thickness / 2.0
        extend = half * 0.55
        start_x = start[0] - ux * extend
        start_y = start[1] - uy * extend
        end_x = end[0] + ux * extend
        end_y = end[1] + uy * extend
        draw.polygon(
            [
                (start_x + nx * half, start_y + ny * half),
                (start_x - nx * half, start_y - ny * half),
                (end_x - nx * half, end_y - ny * half),
                (end_x + nx * half, end_y + ny * half),
            ],
            fill=WALL_COLOR,
        )

    def _draw_marker(self, draw: ImageDraw.ImageDraw, cell: Axial, color: Tuple[int, int, int]) -> None:
        x, y = self._cell_center_from_cell(cell)
        radius = max(6, int(self.cell_radius * 0.45))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    def _draw_solution(
        self,
        image: Image.Image,
        path: Sequence[Axial],
        *,
        factor: int = 1,
    ) -> None:
        """Stroke the route on *image*, which may be a ``factor``x scene canvas."""

        if len(path) < 2:
            return
        draw_path_line(
            image,
            [self._cell_center_from_cell(cell) for cell in path],
            LINE_COLOR,
            self._route_thickness(),
            factor=factor,
        )

    def _route_thickness(self) -> int:
        return max(4, int(self.cell_radius * 0.35))

    def _draw_axis_labels(self, draw: ImageDraw.ImageDraw) -> None:
        """Number the honeycomb levels and columns in the outer gutters.

        The left gutter carries ``x`` (the horizontal level of a hexagon
        centre, counted from the top) and the top gutter carries ``y`` (the
        hexagon column, counted from the left), which is exactly the
        model-facing ``(x, y)`` answer grammar.  Nothing is drawn inside the
        honeycomb body.
        """

        factor = int(getattr(draw, "factor", 1) or 1)
        font = load_font(max(9, int(self.cell_radius * 0.42)) * factor)
        center_x, center_y = self.center
        level_step = self.cell_radius * SQRT_THREE / 2.0
        for level in range(self.rows):
            offset = level - 2 * self.radius
            draw.text(
                (self.LABEL_GUTTER / 2.0, center_y + offset * level_step),
                str(level + 1),
                fill=GRID_AXIS_LABEL_COLOR,
                anchor="mm",
                font=font,
            )
        for column in range(self.cols):
            offset = column - self.radius
            draw.text(
                (center_x + offset * 1.5 * self.cell_radius, self.LABEL_GUTTER / 2.0),
                str(column + 1),
                fill=GRID_AXIS_LABEL_COLOR,
                anchor="mm",
                font=font,
            )

    # ------------------------------------------------------------------
    # Geometry helpers

    def _axial_to_pixel(self, cell: Axial) -> Tuple[float, float]:
        q, r = cell
        x = self.cell_radius * (1.5 * q)
        y = self.cell_radius * (SQRT_THREE * (r + q / 2.0))
        return x, y

    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        cell = self.cells[cell_id - 1]
        return self._cell_center_from_cell(cell)

    def _cell_id_to_xy(self, cell_id: int) -> Optional[Tuple[int, int]]:
        """Map an internal axial cell to its model-facing 1-based ``(x, y)``.

        ``y`` is the hexagon column (``q`` counted from the left edge) and
        ``x`` is the horizontal level of the hexagon centre (``r + q / 2``
        counted from the top in half-steps), which is what the left and top
        gutters number.  Internal geometry keeps using axial ``(q, r)``.
        """

        index = int(cell_id) - 1
        if not 0 <= index < len(self.cells):
            return None
        q, r = self.cells[index]
        return (2 * r + q + 2 * self.radius + 1, q + self.radius + 1)

    def _cell_center_from_cell(self, cell: Axial) -> Tuple[float, float]:
        rel_x, rel_y = self._centers[cell]
        return (self.center[0] + rel_x, self.center[1] + rel_y)

    def _hex_corners(self, center: Tuple[float, float], radius: float) -> List[Tuple[float, float]]:
        cx, cy = center
        corners: List[Tuple[float, float]] = []
        for i in range(6):
            angle = math.radians(60 * i)
            corners.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        return corners


    def _distance(self, cell: Axial) -> int:
        q, r = cell
        s = -q - r
        return max(abs(q), abs(r), abs(s))

__all__ = ["MazeHexagonGenerator"]

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.spatial_navigation.core.base import (
    _AXIS_LABEL_COLOR,
    _CANVAS_BG,
    _DIR_TO_DELTA,
    _GRID_LINE,
    _LABEL_COLOR,
    _PATH_COLOR,
    GridScenario,
    SpatialNavigationGeneratorBase,
)
from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    cell_to_xy,
    format_xy_path_answer,
)
from data.spatial_navigation.core.maze_style import route_run_breakdown
from data.common.video_encoding import encode_rgb_frames_to_webm, fit_frames_to_video_budget
from data.common.animation_policy import (
    DISCRETE_STATE_HOLD_FRAMES,
    frame_budget_for_discrete_state_count,
)
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path

Cell = Tuple[int, int]


@dataclass
class CostGridScenario(GridScenario):
    cost_grid: List[List[int]] = field(default_factory=list)
    max_total_cost: int = 0


class GridHighestCostGenerator(SpatialNavigationGeneratorBase):
    TASK_NAME = "grid_highest_cost"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Grid Highest Cost"

    def __init__(
        self,
        output_dir=None,
        *,
        seed: Optional[int] = None,
        size: int = 4,
        video: bool = True,
        video_fps: int = 16,
        cell_size: Optional[int] = None,
        **_ignored: Any,
    ) -> None:
        super().__init__(
            output_dir,
            seed=seed,
            size=size,
            reward_count_range=(0, 0),
            video=video,
            video_fps=video_fps,
            cell_size=cell_size,
        )

    def _get_cell_id(self, row: int, col: int) -> int:
        return row * self.size + col + 1

    def _cell_bboxes(self) -> List[List[List[int]]]:
        rows: List[List[List[int]]] = []
        for row in range(self.size):
            row_boxes: List[List[int]] = []
            for col in range(self.size):
                row_boxes.append([int(value) for value in self._cell_box((row, col))])
            rows.append(row_boxes)
        return rows

    def _solution_path_cell_ids(self, path_cells: Sequence[Cell]) -> List[int]:
        return [self._get_cell_id(row, col) for row, col in path_cells]

    def _enumerate_best_path(
        self,
        cost_grid: Sequence[Sequence[int]],
        start: Cell,
        goal: Cell,
    ) -> tuple[Optional[List[Cell]], int]:
        rows = len(cost_grid)
        cols = len(cost_grid[0]) if rows else 0
        best_cost = -1
        best_path: Optional[List[Cell]] = None

        def dfs(cell: Cell, path: List[Cell], total: int) -> None:
            nonlocal best_cost, best_path
            if cell == goal:
                if total > best_cost:
                    best_cost = total
                    best_path = list(path)
                return

            row, col = cell
            for nxt in ((row, col + 1), (row + 1, col)):
                if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                    continue
                path.append(nxt)
                dfs(nxt, path, total + int(cost_grid[nxt[0]][nxt[1]]))
                path.pop()

        dfs(start, [start], int(cost_grid[start[0]][start[1]]))
        return best_path, best_cost

    def _moves_from_path(self, path_cells: Sequence[Cell]) -> List[str]:
        moves: List[str] = []
        for current, nxt in zip(path_cells, path_cells[1:]):
            dr = nxt[0] - current[0]
            dc = nxt[1] - current[1]
            for move, delta in _DIR_TO_DELTA.items():
                if delta == (dr, dc):
                    moves.append(move)
                    break
            else:  # pragma: no cover
                raise RuntimeError("Non-adjacent path step encountered.")
        return moves

    def _greedy_total(self, grid: Sequence[Sequence[int]], *, prefer_right: bool) -> int:
        """Total collected by always stepping onto the larger neighbour."""

        row = col = 0
        total = int(grid[0][0])
        while (row, col) != (self.size - 1, self.size - 1):
            right = int(grid[row][col + 1]) if col + 1 < self.size else None
            down = int(grid[row + 1][col]) if row + 1 < self.size else None
            if right is None:
                row += 1
                total += down
            elif down is None:
                col += 1
                total += right
            elif right > down or (right == down and prefer_right):
                col += 1
                total += right
            else:
                row += 1
                total += down
        return total

    def _sample_cost_grid(self) -> tuple[List[List[int]], List[Cell], int]:
        start = (0, 0)
        goal = (self.size - 1, self.size - 1)
        for _ in range(500):
            grid = [
                [self.rng.randint(1, 9) for _ in range(self.size)]
                for _ in range(self.size)
            ]
            path_cells, best_cost = self._enumerate_best_path(grid, start, goal)
            if path_cells is None:
                continue
            # Anti-degeneration: a grid solved by locally picking the larger
            # neighbour needs no lookahead.  Keep only grids where both greedy
            # tie-break variants fall short of the true maximum.
            if self._greedy_total(grid, prefer_right=True) == best_cost:
                continue
            if self._greedy_total(grid, prefer_right=False) == best_cost:
                continue
            return grid, path_cells, best_cost
        raise RuntimeError("Failed to generate a highest-cost path on the grid.")

    def sample_scenario(self) -> GridScenario:
        cost_grid, path_cells, best_cost = self._sample_cost_grid()
        moves = self._moves_from_path(path_cells)
        start = (0, 0)
        goal = (self.size - 1, self.size - 1)
        return CostGridScenario(
            task_name=self.TASK_NAME,
            task_family=self.TASK_FAMILY,
            title=self.TASK_TITLE,
            input_desc=(
                "The Grid Highest Cost input image shows a square cost grid on a white canvas. Each cell contains a "
                "visible cost value, the green cell is the start, the red cell is the goal, and an orange circular "
                "agent starts on the green start cell. Row numbers run down the left edge of the grid and column "
                f"numbers run across the top edge. {GRID_XY_CONVENTION_SENTENCE}"
            ),
            rules_desc=(
                "The agent can move right or down to an adjacent cell, or stay on its current cell for one step. "
                "The total route cost is the sum of the start value and each newly entered cell value, including the "
                "goal; a stay does not add the current cell value again."
            ),
            ti2v_output=(
                "Generate a short video that shows any highest-total-cost legal right/down/stay route from the green "
                "start cell to the red goal cell. A stay holds the unchanged agent and cumulative trace for one "
                "additional state without adding cost. Show one exact grid state per route step, hold each state for "
                "the same duration, and switch the orange agent and red cumulative route trace directly at each "
                "cell boundary."
            ),

            size=self.size,
            start=start,
            goal=goal,
            move_sequence=moves,
            path_cells=path_cells,
            blocked_cells=[],
            rewards=[],
            start_color=(61, 181, 99),
            goal_color=(228, 52, 52),
            agent_color=(245, 130, 32),
            cost_grid=cost_grid,
            max_total_cost=best_cost,
        )

    def _draw_path_points(
        self,
        draw: ImageDraw.ImageDraw,
        points: Sequence[Tuple[float, float]],
    ) -> None:
        if len(points) >= 2:
            draw_spatial_navigation_path(
                draw,
                points,
                color=_PATH_COLOR,
                width=max(6, self.cell_size // 6),
            )

    def _render_board(
        self,
        scenario: CostGridScenario,
        *,
        agent_cell: Cell,
        collected_indices: Iterable[int] = (),
        show_path: bool = False,
        path_cells: Sequence[Cell] = (),
        path_points: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Image.Image:
        image, draw = self._begin_scene()
        axis_font = self._load_font(max(12, min(self.cell_size // 3, int(self.margin * 0.72))))
        cost_font = self._load_font(max(20, self.cell_size // 3))

        for row in range(self.size):
            for col in range(self.size):
                left, top, right, bottom = self._cell_box((row, col))
                fill = _CANVAS_BG
                if (row, col) == scenario.start:
                    fill = scenario.start_color
                elif (row, col) == scenario.goal:
                    fill = scenario.goal_color
                draw.rectangle([left, top, right, bottom], fill=fill, outline=_GRID_LINE, width=1)

        if show_path:
            if path_points is not None:
                self._draw_path_points(draw, path_points)
            elif path_cells:
                self._draw_path_overlay(draw, path_cells)

        # 1-based (x, y) guides: x down the left margin, y across the top
        # margin.  Nothing is drawn inside a cell except its cost value.
        for row in range(self.size):
            draw.text(
                (self.margin * 0.5, self.margin + row * self.cell_size + self.cell_size * 0.5),
                str(row + 1),
                fill=_AXIS_LABEL_COLOR,
                font=axis_font,
                anchor="mm",
            )
        for col in range(self.size):
            draw.text(
                (self.margin + col * self.cell_size + self.cell_size * 0.5, self.margin * 0.5),
                str(col + 1),
                fill=_AXIS_LABEL_COLOR,
                font=axis_font,
                anchor="mm",
            )

        for row in range(self.size):
            for col in range(self.size):
                value = int(scenario.cost_grid[row][col])
                color = (255, 255, 255) if (row, col) in {scenario.start, scenario.goal} else _LABEL_COLOR
                draw.text(self._cell_center((row, col)), str(value), fill=color, font=cost_font, anchor="mm")

        agent_inset = max(12, self.cell_size // 4)
        cx, cy = self._cell_center(agent_cell)
        radius = (self.cell_size - 2 * agent_inset) / 2.0
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=scenario.agent_color,
            outline=tuple(max(0, channel - 70) for channel in scenario.agent_color),
            width=2,
        )
        return self._finish_scene(image)

    def _write_video(self, scenario: GridScenario, video_path: Path) -> int:
        cost_scenario = scenario
        if not isinstance(cost_scenario, CostGridScenario):
            raise TypeError("GridHighestCostGenerator expects CostGridScenario")
        path = list(cost_scenario.path_cells)
        if not path:
            return 0

        frames: List[np.ndarray] = []
        protected_indices: List[int] = []
        for step_index, cell in enumerate(path):
            image = self._render_board(
                cost_scenario,
                agent_cell=cell,
                show_path=True,
                path_cells=path[: step_index + 1],
            )
            start_index = len(frames)
            array = np.array(image.convert("RGB"), dtype=np.uint8)
            frames.extend(array.copy() for _ in range(DISCRETE_STATE_HOLD_FRAMES))
            protected_indices.append(start_index)

        frames = fit_frames_to_video_budget(
            frames,
            max_frames=frame_budget_for_discrete_state_count(len(path)),
            final_hold_frames=DISCRETE_STATE_HOLD_FRAMES,
            protected_indices=protected_indices,
        )
        if not encode_rgb_frames_to_webm(frames, video_path, fps=self.video_fps):
            return 0
        return len(frames)

    def _build_answer_payload(self, scenario: CostGridScenario) -> Dict[str, Any]:
        payload = super()._build_answer_payload(scenario)
        payload["cost_grid"] = [[int(cell) for cell in row] for row in scenario.cost_grid]
        payload["max_total_cost"] = int(scenario.max_total_cost)
        return payload

    def _authored_reasoning(self, scenario: GridScenario) -> str:
        assert isinstance(scenario, CostGridScenario)
        path = list(scenario.path_cells)
        costs = scenario.cost_grid
        route_values = [int(costs[row][col]) for row, col in path]
        big_on_route = sorted(
            ((value, cell) for value, cell in zip(route_values[1:], path[1:])),
            key=lambda item: -item[0],
        )[:2]
        big_text = " and ".join(
            f"the {value} at {cell_to_xy(cell)}" for value, cell in big_on_route
        )
        sentences: List[str] = [
            f"Every route runs from the green {cell_to_xy(path[0])} to the red {cell_to_xy(path[-1])} using only "
            f"right and down moves, adding each newly entered cell's value to the start value of "
            f"{route_values[0]}. The richest cells this side of the grid are {big_text}, so the best route is the "
            "one that lines its corners up to collect them.",
        ]
        total = route_values[0]
        for direction, cells in route_run_breakdown(path):
            gained = [int(costs[row][col]) for row, col in cells[1:]]
            total += sum(gained)
            values = " + ".join(str(value) for value in gained)
            sentences.append(
                f"It goes {direction} through {', '.join(str(cell_to_xy(cell)) for cell in cells[1:])}, adding "
                f"{values} for a running total of {total}."
            )
        sentences.append(
            f"The route ends on the red goal with a total of {total}, and no other right/down route collects more."
        )
        return " ".join(sentences)

    def _build_record(self, scenario: GridScenario) -> Dict[str, Any]:
        if not isinstance(scenario, CostGridScenario):
            raise TypeError('GridHighestCostGenerator expects CostGridScenario')
        puzzle_id = self._next_id()
        prompts = self._prompts(scenario)
        solution_path_cell_ids = self._solution_path_cell_ids(scenario.path_cells)
        puzzle_image = self._render_board(scenario, agent_cell=scenario.start)
        reasoning_image = self._render_board(scenario, agent_cell=scenario.start)
        solution_image = self._render_board(scenario, agent_cell=scenario.goal, show_path=True, path_cells=scenario.path_cells)
        image_path = self.output_dir / f'{puzzle_id}_puzzle.png'
        reasoning_path = self.output_dir / f'{puzzle_id}_reasoning.png'
        solution_path = self.output_dir / f'{puzzle_id}_solution.png'
        puzzle_image.save(image_path)
        reasoning_image.save(reasoning_path)
        solution_image.save(solution_path)
        video_path = None
        video_num_frames = None
        if self.video:
            video_path = self.output_dir / f'{puzzle_id}_solution.webm'
            video_num_frames = self._write_video(scenario, video_path)
        answer_line = format_xy_path_answer(scenario.path_cells)
        self._last_solution_image_rel = self.relativize_path(solution_path)
        payload: Dict[str, Any] = {'id': puzzle_id, 'answer': answer_line, 'task_name': scenario.task_name, 'task_family': scenario.task_family, 'image': self.relativize_path(image_path), 'reasoning_image': self.relativize_path(reasoning_path), 'solution_image_path': self.relativize_path(solution_path), 'answer_payload': self._build_answer_payload(scenario), 'seed': self.seed, 'canvas_dimensions': [width for width in self._canvas_size()], 'grid_size': [self.size, self.size], 'cell_size': self.cell_size, 'maze_grid': [[0 for _ in range(self.size)] for _ in range(self.size)], 'start': [scenario.start[0], scenario.start[1]], 'goal': [scenario.goal[0], scenario.goal[1]], 'start_point': [float(v) for v in self._cell_center(scenario.start)], 'goal_point': [float(v) for v in self._cell_center(scenario.goal)], 'cell_bboxes': self._cell_bboxes(), 'solution_path_cell_ids': solution_path_cell_ids, 'path_joint_style': 'round', 'solution_path_length_cells': len(solution_path_cell_ids), 'min_solution_path_length_cells': len(solution_path_cell_ids), **prompts}
        if video_path is not None:
            payload['solution_video_path'] = self.relativize_path(video_path)
            payload['video_fps'] = self.video_fps
            payload['video_num_frames'] = video_num_frames
        return payload

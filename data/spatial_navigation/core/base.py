from __future__ import annotations

from data.common.assets import render_font

import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.common.base import AbstractPuzzleGenerator
from data.common.grid_coordinates import format_xy_path_answer
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.video_encoding import encode_rgb_frames_to_webm, fit_frames_to_video_budget
from data.common.animation_policy import (
    DISCRETE_STATE_HOLD_FRAMES,
    frame_budget_for_discrete_state_count,
)
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path
from data.spatial_navigation.core.prompt_contract import build_spatial_navigation_prompt_bundle

_DIR_TO_DELTA: Dict[str, Tuple[int, int]] = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "stay": (0, 0),
}
_REWARD_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (58, 134, 255),
    (255, 191, 76),
    (233, 103, 186),
    (132, 94, 247),
    (66, 184, 131),
)
_SHAPES = ("triangle", "diamond", "star", "circle")
_CANVAS_BG = (255, 255, 255)
_GRID_LINE = (25, 25, 25)
_PATH_COLOR = (214, 42, 42)
_LABEL_COLOR = (20, 20, 20)
# Family-wide royal-blue 1-based axis numbers (same as the maze scenes).
_AXIS_LABEL_COLOR = (30, 90, 220)

@dataclass(frozen=True)
class GridReward:
    cell: Tuple[int, int]
    color: Tuple[int, int, int]
    color_name: str
    style: str

@dataclass
class GridScenario:
    task_name: str
    task_family: str
    title: str
    input_desc: str
    rules_desc: str
    ti2v_output: str
    size: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    move_sequence: List[str]
    path_cells: List[Tuple[int, int]]
    blocked_cells: List[Tuple[int, int]]
    rewards: List[GridReward]
    start_color: Tuple[int, int, int]
    goal_color: Tuple[int, int, int]
    agent_color: Tuple[int, int, int]
    collect_on_visit: bool = True

class SpatialNavigationGeneratorBase(AbstractPuzzleGenerator[Dict[str, Any]]):
    TASK_NAME = "grid_task"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Grid Task"
    DEFAULT_SIZE = 10
    DEFAULT_REWARD_COUNT_RANGE = (3, 4)

    CANVAS_SIDE = 512

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: Optional[int] = None,
        size: int = DEFAULT_SIZE,
        reward_count_range: Tuple[int, int] = DEFAULT_REWARD_COUNT_RANGE,
        video: bool = True,
        video_fps: int = 16,
        cell_size: Optional[int] = None,
    ) -> None:
        resolved_output_dir = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_spatial_navigation"
        super().__init__(resolved_output_dir)
        self.rng = random.Random(seed)
        self.seed = seed
        self.size = max(4, int(size))
        self.reward_count_range = reward_count_range
        self.video = bool(video)
        self.video_fps = int(video_fps)
        # Auto-fit a 512×512 canvas: derive cell_size so that
        # margin*2 + size*cell_size == CANVAS_SIDE, with margin = cell_size // 2.
        # That gives cell_size = CANVAS_SIDE / (size + 1). Force even-pixel alignment.
        if cell_size is None:
            ideal = self.CANVAS_SIDE / (self.size + 1)
            cell = max(28, int(ideal) // 2 * 2)
            self.cell_size = cell
        else:
            self.cell_size = max(20, int(cell_size) // 2 * 2)
        # Margin chosen to fill the canvas exactly to CANVAS_SIDE when possible.
        remaining = self.CANVAS_SIDE - self.size * self.cell_size
        if remaining >= 24:
            self.margin = remaining // 2
        else:
            self.margin = max(16, self.cell_size // 2)
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def create_puzzle(self) -> Dict[str, Any]:
        return self._build_record(self.sample_scenario())

    def sample_scenario(self) -> GridScenario:
        raise NotImplementedError

    def _next_id(self) -> str:
        puzzle_id = f"{self.TASK_NAME}-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    SCENE_SUPERSAMPLE = DOCTRINE_SUPERSAMPLE

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        return render_font(int(size) * self.SCENE_SUPERSAMPLE, bold=False)

    def _begin_scene(self) -> Tuple[Image.Image, ScaledDraw]:
        width, height = self._canvas_size()
        return new_scaled_canvas(width, height, self.SCENE_SUPERSAMPLE, _CANVAS_BG)

    def _finish_scene(self, image: Image.Image) -> Image.Image:
        return finish_scaled_canvas(image, self._canvas_size())

    def _cell_box(self, cell: Tuple[int, int]) -> Tuple[int, int, int, int]:
        row, col = cell
        left = self.margin + col * self.cell_size
        top = self.margin + row * self.cell_size
        return left, top, left + self.cell_size, top + self.cell_size

    def _cell_center(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        left, top, right, bottom = self._cell_box(cell)
        return (left + right) / 2.0, (top + bottom) / 2.0

    def _canvas_size(self) -> Tuple[int, int]:
        side = self.margin * 2 + self.size * self.cell_size
        return side, side

    def _build_state_trace(
        self,
        scenario: GridScenario,
        path_cells: Sequence[Tuple[int, int]],
    ) -> List[List[List[int]]]:
        reward_lookup = {reward.cell: index for index, reward in enumerate(scenario.rewards)}
        collected: set[int] = set()
        trace: List[List[List[int]]] = []
        for cell in path_cells:
            reward_idx = reward_lookup.get(cell)
            if reward_idx is not None:
                collected.add(reward_idx)
            grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
            for blocked_row, blocked_col in scenario.blocked_cells:
                grid[blocked_row][blocked_col] = 9
            for reward_idx, reward in enumerate(scenario.rewards):
                if reward_idx not in collected:
                    reward_row, reward_col = reward.cell
                    grid[reward_row][reward_col] = 4
            start_row, start_col = scenario.start
            goal_row, goal_col = scenario.goal
            grid[start_row][start_col] = 1
            grid[goal_row][goal_col] = 2
            row, col = cell
            grid[row][col] = 5
            trace.append(grid)
        return trace

    def _draw_star(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
        *,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
    ) -> None:
        cx, cy = self._cell_center(cell)
        outer = self.cell_size * 0.24
        inner = outer * 0.45
        points: List[Tuple[float, float]] = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            radius = outer if index % 2 == 0 else inner
            points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
        draw.polygon(points, fill=fill, outline=outline)

    def _draw_reward(
        self,
        draw: ImageDraw.ImageDraw,
        reward: GridReward,
        *,
        font: ImageFont.ImageFont,
    ) -> None:
        left, top, right, bottom = self._cell_box(reward.cell)
        inset = max(8, self.cell_size // 6)
        fill = reward.color
        outline = tuple(max(0, channel - 70) for channel in fill)
        if reward.style == "block":
            draw.rounded_rectangle(
                [left + inset, top + inset, right - inset, bottom - inset],
                radius=max(6, inset // 2),
                fill=fill,
                outline=outline,
                width=2,
            )
            return
        if reward.style == "circle":
            draw.ellipse([left + inset, top + inset, right - inset, bottom - inset], fill=fill, outline=outline, width=2)
            return
        if reward.style == "triangle":
            cx, cy = self._cell_center(reward.cell)
            radius = self.cell_size * 0.26
            points = [
                (cx, cy - radius),
                (cx - radius * 0.92, cy + radius * 0.8),
                (cx + radius * 0.92, cy + radius * 0.8),
            ]
            draw.polygon(points, fill=fill, outline=outline)
            return
        if reward.style == "diamond":
            cx, cy = self._cell_center(reward.cell)
            radius = self.cell_size * 0.26
            points = [(cx, cy - radius), (cx - radius, cy), (cx, cy + radius), (cx + radius, cy)]
            draw.polygon(points, fill=fill, outline=outline)
            return
        if reward.style == "star":
            self._draw_star(draw, reward.cell, fill=fill, outline=outline)
            return

        draw.text(self._cell_center(reward.cell), reward.color_name[:1].upper(), fill=outline, font=font, anchor="mm")

    def _draw_path_overlay(self, draw: ImageDraw.ImageDraw, path_cells: Sequence[Tuple[int, int]]) -> None:
        points = [self._cell_center(cell) for cell in path_cells]
        if len(points) >= 2:
            draw_spatial_navigation_path(
                draw,
                points,
                color=_PATH_COLOR,
                width=max(6, self.cell_size // 6),
            )


    def _render_board(
        self,
        scenario: GridScenario,
        *,
        agent_cell: Tuple[int, int],
        collected_indices: Iterable[int] = (),
        show_path: bool = False,
        path_cells: Sequence[Tuple[int, int]] = (),
    ) -> Image.Image:
        image, draw = self._begin_scene()
        label_font = self._load_font(max(18, self.cell_size // 3))
        collected = set(collected_indices)

        blocked_set = set(scenario.blocked_cells)
        for row in range(self.size):
            for col in range(self.size):
                left, top, right, bottom = self._cell_box((row, col))
                if (row, col) in blocked_set:
                    fill = (54, 64, 82)  # slate
                    outline = (32, 38, 52)
                else:
                    fill = _CANVAS_BG
                    outline = _GRID_LINE
                draw.rectangle([left, top, right, bottom], fill=fill, outline=outline, width=1)

        for row in range(self.size):
            draw.text(
                (self.margin - self.cell_size * 0.35, self.margin + row * self.cell_size + self.cell_size * 0.5),
                str(row + 1),
                fill=_AXIS_LABEL_COLOR,
                font=label_font,
                anchor="mm",
            )
        for col in range(self.size):
            draw.text(
                (self.margin + col * self.cell_size + self.cell_size * 0.5, self.margin - self.cell_size * 0.35),
                str(col + 1),
                fill=_AXIS_LABEL_COLOR,
                font=label_font,
                anchor="mm",
            )

        if show_path and path_cells:
            self._draw_path_overlay(draw, path_cells)

        for reward_idx, reward in enumerate(scenario.rewards):
            if reward_idx in collected:
                continue
            self._draw_reward(draw, reward, font=label_font)

        start_left, start_top, start_right, start_bottom = self._cell_box(scenario.start)
        goal_left, goal_top, goal_right, goal_bottom = self._cell_box(scenario.goal)
        inset = max(5, self.cell_size // 12)
        draw.rounded_rectangle(
            [start_left + inset, start_top + inset, start_right - inset, start_bottom - inset],
            radius=max(8, inset),
            fill=scenario.start_color,
            outline=tuple(max(0, channel - 60) for channel in scenario.start_color),
            width=2,
        )
        draw.rounded_rectangle(
            [goal_left + inset, goal_top + inset, goal_right - inset, goal_bottom - inset],
            radius=max(8, inset),
            fill=scenario.goal_color,
            outline=tuple(max(0, channel - 60) for channel in scenario.goal_color),
            width=2,
        )

        agent_inset = max(12, self.cell_size // 4)
        agent_radius = (self.cell_size - 2 * agent_inset) / 2.0
        cx, cy = self._cell_center(agent_cell)
        draw.ellipse(
            [cx - agent_radius, cy - agent_radius, cx + agent_radius, cy + agent_radius],
            fill=scenario.agent_color,
            outline=tuple(max(0, channel - 70) for channel in scenario.agent_color),
            width=2,
        )
        return self._finish_scene(image)

    def _write_video(self, scenario: GridScenario, video_path: Path) -> int:
        """Render exact grid states with no sub-cell transition frames."""
        path = list(scenario.path_cells)
        if not path:
            return 0

        reward_lookup = {reward.cell: index for index, reward in enumerate(scenario.rewards)}
        collected: set[int] = set()

        # Pre-collect on starting cell if it is a reward cell.
        if scenario.collect_on_visit:
            start_idx = reward_lookup.get(path[0])
            if start_idx is not None:
                collected.add(start_idx)

        frames: List[np.ndarray] = []
        protected_indices: List[int] = []

        def append_state(
            cell: Tuple[int, int],
            visible_collected: Iterable[int],
            path_prefix: Sequence[Tuple[int, int]],
        ) -> None:
            image = self._render_board(
                scenario,
                agent_cell=cell,
                collected_indices=visible_collected,
                show_path=True,
                path_cells=path_prefix,
            )
            start_index = len(frames)
            frame = np.array(image.convert("RGB"), dtype=np.uint8)
            frames.extend(frame.copy() for _ in range(DISCRETE_STATE_HOLD_FRAMES))
            protected_indices.append(start_index)

        append_state(path[0], collected, path[:1])
        for step_index, cell in enumerate(path[1:], start=1):
            if scenario.collect_on_visit:
                reward_idx = reward_lookup.get(cell)
                if reward_idx is not None:
                    collected.add(reward_idx)
            append_state(cell, collected, path[: step_index + 1])

        budget = frame_budget_for_discrete_state_count(len(path))
        frames = fit_frames_to_video_budget(
            frames,
            max_frames=budget,
            final_hold_frames=DISCRETE_STATE_HOLD_FRAMES,
            protected_indices=protected_indices,
        )
        if not encode_rgb_frames_to_webm(frames, video_path, fps=self.video_fps):
            return 0
        return len(frames)

    def _prompts(self, scenario: GridScenario) -> Dict[str, str]:
        return build_spatial_navigation_prompt_bundle(
            task_name=scenario.task_name,
            input_description=scenario.input_desc,
            rule_description=scenario.rules_desc,
            ti2v_output=scenario.ti2v_output,
        )

    def _build_answer_payload(self, scenario: GridScenario) -> Dict[str, Any]:
        return {
            "grid_size": scenario.size,
            "start": [scenario.start[0], scenario.start[1]],
            "goal": [scenario.goal[0], scenario.goal[1]],
            "blocked_cells": [[row, col] for row, col in scenario.blocked_cells],
            "required_cells": [[reward.cell[0], reward.cell[1]] for reward in scenario.rewards],
            "required_items": [
                {
                    "cell": [reward.cell[0], reward.cell[1]],
                    "color": reward.color_name,
                    "style": reward.style,
                }
                for reward in scenario.rewards
            ],
            "move_sequence": list(scenario.move_sequence),
            "path_cells": [[row, col] for row, col in scenario.path_cells],
            "path_length": len(scenario.move_sequence),
            "state_trace": self._build_state_trace(scenario, scenario.path_cells),
        }

    def _build_record(self, scenario: GridScenario) -> Dict[str, Any]:
        puzzle_id = self._next_id()
        prompts = self._prompts(scenario)
        puzzle_image = self._render_board(scenario, agent_cell=scenario.start)
        solution_image = self._render_board(
            scenario,
            agent_cell=scenario.goal,
            collected_indices=range(len(scenario.rewards)),
            show_path=True,
            path_cells=scenario.path_cells,
        )
        image_path = self.output_dir / f"{puzzle_id}_puzzle.png"
        solution_path = self.output_dir / f"{puzzle_id}_solution.png"
        puzzle_image.save(image_path)
        solution_image.save(solution_path)

        video_path = None
        video_num_frames = None
        if self.video:
            video_path = self.output_dir / f"{puzzle_id}_solution.webm"
            video_num_frames = self._write_video(scenario, video_path)

        self._last_solution_image_rel = self.relativize_path(solution_path)
        payload: Dict[str, Any] = {
            "id": puzzle_id,
            "task_name": scenario.task_name,
            "task_family": scenario.task_family,
            "image": self.relativize_path(image_path),
            "reasoning_image": self.relativize_path(image_path),
            "solution_image_path": self.relativize_path(solution_path),
            "answer": format_xy_path_answer(scenario.path_cells),
            "answer_payload": self._build_answer_payload(scenario),
            "seed": self.seed,
            **prompts,
        }
        if video_path is not None:
            payload["solution_video_path"] = self.relativize_path(video_path)
            payload["video_fps"] = self.video_fps
            payload["video_num_frames"] = video_num_frames
        return payload

"""Four-tick vertical drill and connected-color clear prediction."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import DEFAULT_FPS
from data.dynamics_game.core.scene import finish_scene, new_scene, scene_font
from data.common.pipeline.discrete_dynamics import (
    encode_video,
    make_record_dataclass,
    shuffle_with_correct_semantics,
    write_artifacts,
)
from data.common.multiple_choice import option_label
from data.dynamics_game.core.choice_composite import (
    attach_choice_input_fields,
    build_composite_mcq_prompts,
    finalize_choice_artifacts,
)
from data.dynamics_game.core.coordinate_format import with_grid_xy_convention


Cell = Tuple[int, int]
Grid = List[List[Optional[str]]]

_INPUT_DESC = with_grid_xy_convention(
    "The single problem image shows a 12-row by 6-column board with rows 1 through 12 and columns 1 through 6 labeled around the wells. A well may contain a red, blue, "
    "green, or yellow block, or it may be pale and empty. An orange drill is centered above one column. A four-dot "
    "badge means that exactly four ticks must be applied."
)
_RULE_SUMMARY = (
    "Let y be the printed number of the column beneath the drill. On tick 1 the drill enters (1, y); on each later "
    "tick N it enters (N, y). When the entered well contains a colored block, clear that block and every block of the same color "
    "connected to it through shared sides. Diagonal contact does not connect blocks. All cleared wells become pale "
    "empty wells and remain empty. The other blocks do not fall or shift. Apply exactly four ticks."
)
_TI2V_OUTPUT = (
    "Hold the input, then show the complete board after each of the four ticks in order as exact states: in "
    "each state the drill sits in the newly entered well of its column and the shared-side same-color "
    "component of that well's block has vanished into pale empty wells together with it. An already empty "
    "entered well changes nothing. Hold the tick-4 state at the end."
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render the board after tick 4 with every cleared well pale and empty and the orange drill centered at (4, y)."
)

BG = (247, 249, 253)
PANEL = (255, 255, 255)
WELL = (237, 242, 249)
WELL_EDGE = (207, 217, 232)
TEXT = (48, 59, 80)
DRILL = (244, 139, 61)
DRILL_EDGE = (166, 80, 28)
BIT = (109, 124, 150)
COLORS = {
    "R": ((225, 82, 88), (151, 42, 49)),
    "B": ((75, 139, 218), (36, 84, 157)),
    "G": ((75, 177, 112), (35, 111, 68)),
    "Y": ((245, 199, 62), (170, 124, 24)),
}
_LAYER_CODES = {None: 0, "R": 1, "B": 2, "G": 3, "Y": 4}
_COLOR_NAMES = {"R": "red", "B": "blue", "G": "green", "Y": "yellow"}

_BD2Record = make_record_dataclass("mr_driller_drop_predict", "dynamics_game")


def _encode_grid(grid: Sequence[Sequence[Optional[str]]]) -> List[List[int]]:
    return [[_LAYER_CODES[cell] for cell in row] for row in grid]


def _connected_color(grid: Sequence[Sequence[Optional[str]]], row: int, col: int, rows: int, cols: int) -> set[Cell]:
    target = grid[row][col]
    if target is None:
        return set()
    seen = {(row, col)}
    queue = deque([(row, col)])
    while queue:
        current_row, current_col = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row, next_col = current_row + dr, current_col + dc
            if (next_row, next_col) in seen or not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            if grid[next_row][next_col] == target:
                seen.add((next_row, next_col))
                queue.append((next_row, next_col))
    return seen


class MrDrillerDropPredictGenerator(AbstractPuzzleGenerator):
    rows = 12
    cols = 6
    ticks = 4

    def __init__(self, output_dir, *, seed=None, video_fps=DEFAULT_FPS, video=False):
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.video_fps = int(video_fps)
        self.video = bool(video)
        self._counter = 0
        self.cell = 34
        self.left = 154
        self.top = 78

    def _random_cavern(self, drill_col: int) -> Grid:
        grid: Grid = [
            [self.rng.choice(("R", "B", "G", "Y")) if self.rng.random() < 0.82 else None for _ in range(self.cols)]
            for _ in range(self.rows)
        ]
        colors = list(("R", "B", "G", "Y"))
        self.rng.shuffle(colors)
        for row in range(self.ticks):
            grid[row][drill_col] = colors[row]
            side = drill_col + self.rng.choice((-1, 1))
            if 0 <= side < self.cols and self.rng.random() < 0.8:
                grid[row][side] = colors[row]
        return grid

    def _simulate_trace(self, grid: Sequence[Sequence[Optional[str]]], drill_col: int, steps: int = 4) -> Tuple[List[Grid], List[Dict[str, Any]]]:
        state = [list(row) for row in grid]
        trace: List[Grid] = [[list(row) for row in state]]
        log: List[Dict[str, Any]] = []
        for step in range(min(steps, self.rows)):
            cluster = _connected_color(state, step, drill_col, self.rows, self.cols)
            for row, col in cluster:
                state[row][col] = None
            log.append({"step": step, "entered": [step, drill_col], "cleared": sorted(cluster)})
            trace.append([list(row) for row in state])
        return trace, log

    def _simulate(self, grid, drill_col, k_steps):
        trace, log = self._simulate_trace(grid, drill_col, k_steps)
        return trace[-1], min(k_steps, self.rows), log

    def _build_state(self):
        for _ in range(500):
            drill_col = self.rng.randrange(self.cols)
            grid = self._random_cavern(drill_col)
            trace, log = self._simulate_trace(grid, drill_col, self.ticks)
            if all(entry["cleared"] for entry in log) and any(len(entry["cleared"]) >= 2 for entry in log):
                return grid, drill_col, trace, log
        raise RuntimeError("Unable to build a four-tick drill board with a load-bearing connected clear.")

    def _cell_center(self, row: float, col: float) -> Tuple[float, float]:
        return self.left + (col + 0.5) * self.cell, self.top + (row + 0.5) * self.cell

    def _draw_drill(self, draw: ImageDraw.ImageDraw, center: Tuple[float, float]) -> None:
        cx, cy = center
        draw.rounded_rectangle((cx - 13, cy - 14, cx + 13, cy + 9), radius=8, fill=DRILL, outline=DRILL_EDGE, width=2)
        draw.arc((cx - 10, cy - 18, cx + 10, cy + 2), 180, 360, fill=(255, 194, 118), width=3)
        draw.polygon(
            [(cx - 8, cy + 8), (cx + 8, cy + 8), (cx, cy + 18)],
            fill=BIT,
        )

    def _render(
        self,
        grid: Sequence[Sequence[Optional[str]]],
        *,
        drill_col: int,
        drill_row: float,
        pop_cells: Optional[set[Cell]] = None,
        pop_scale: float = 1.0,
    ) -> Image.Image:
        image, draw = new_scene(BG)
        draw.rounded_rectangle((126, 48, 386, 498), radius=24, fill=PANEL, outline=(210, 220, 234), width=2)
        label_font = scene_font(11)
        for col in range(self.cols):
            x = self.left + (col + 0.5) * self.cell
            draw.text((x, self.top - 13), str(col + 1), fill=TEXT, font=label_font, anchor="mm")
        for row in range(self.rows):
            y = self.top + (row + 0.5) * self.cell
            draw.text((self.left - 10, y), str(row + 1), fill=TEXT, font=label_font, anchor="rm")
        for row in range(self.rows):
            for col in range(self.cols):
                cx, cy = self._cell_center(row, col)
                draw.rounded_rectangle((cx - 15, cy - 15, cx + 15, cy + 15), radius=7, fill=WELL, outline=WELL_EDGE, width=1)
                value = grid[row][col]
                if value is None:
                    continue
                scale = pop_scale if pop_cells and (row, col) in pop_cells else 1.0
                radius = max(1.0, 12.0 * scale)
                fill, edge = COLORS[value]
                draw.rounded_rectangle(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    radius=max(1, int(6 * scale)),
                    fill=fill,
                    outline=edge,
                    width=2,
                )
        # Static four-dot tick badge: the horizon is input data, not a counter.
        draw.rounded_rectangle((154, 12, 286, 42), radius=14, fill=(240, 245, 252), outline=(102, 131, 187), width=2)
        for index in range(self.ticks):
            cx = 177 + index * 29
            draw.ellipse((cx - 6, 27 - 6, cx + 6, 27 + 6), fill=(255, 255, 255), outline=(71, 112, 187), width=2)
        drill_y = self.top - 26 if drill_row < 0 else self._cell_center(drill_row, drill_col)[1]
        self._draw_drill(draw, (self._cell_center(0, drill_col)[0], drill_y))
        return finish_scene(image)

    def _write_video(self, trace: Sequence[Grid], drill_col: int, log: Sequence[Dict[str, Any]], video_path: Path) -> int:
        """Discrete world: each tick commits the drill entry and its component clear together."""

        _ = log
        frames: List[np.ndarray] = []
        states = [self._render(trace[0], drill_col=drill_col, drill_row=-1)]
        states.extend(
            self._render(trace[step + 1], drill_col=drill_col, drill_row=step)
            for step in range(self.ticks)
        )
        for state in states:
            array = np.asarray(state.convert("RGB"), dtype=np.uint8)
            frames.extend(array.copy() for _ in range(8))
        frames.extend(frames[-1].copy() for _ in range(1))
        assert len(frames) == 8 * len(states) + 1 == 41
        return encode_video(frames, video_path, fps=self.video_fps)

    def _build_distractors(self, grid, drill_col, trace):
        no_chain = [list(row) for row in grid]
        for row in range(self.ticks):
            no_chain[row][drill_col] = None
        global_color = [list(row) for row in grid]
        for row in range(self.ticks):
            color = global_color[row][drill_col]
            if color is None:
                continue
            for rr in range(self.rows):
                for cc in range(self.cols):
                    if global_color[rr][cc] == color:
                        global_color[rr][cc] = None
        three_tick = trace[3]
        return [no_chain, global_color, three_tick]

    def create_puzzle(self):
        sample_id = f"mr_driller_drop_predict-{self._counter:05d}"
        self._counter += 1
        grid, drill_col, trace, log = self._build_state()
        puzzle_image = self._render(grid, drill_col=drill_col, drill_row=-1)
        solution_image = self._render(trace[-1], drill_col=drill_col, drill_row=3)
        distractor_states = self._build_distractors(grid, drill_col, trace)
        option_tiles, candidate_state_grids, correct_index = shuffle_with_correct_semantics(
            solution_image,
            _encode_grid(trace[-1]),
            [self._render(item, drill_col=drill_col, drill_row=3) for item in distractor_states],
            [_encode_grid(item) for item in distractor_states],
            self.rng,
        )
        initial_grid = _encode_grid(trace[0])
        state_trace = [_encode_grid(state) for state in trace]
        answer_label = option_label(correct_index)
        payload: Dict[str, Any] = {
            "mechanic": "four_tick_vertical_connected_clear",
            "grid": [list(row) for row in grid],
            "drill_col": drill_col,
            "k_steps": self.ticks,
            "final_grid": [list(row) for row in trace[-1]],
            "chain_log": log,
            "initial_grid": initial_grid,
            "state_grid": state_trace[-1],
            "state_trace": state_trace,
            "process_state_trace": state_trace,
            "candidate_state_grids": candidate_state_grids,
            "correct_option_index": correct_index,
            "correct_option": answer_label,
        }
        prompts = build_composite_mcq_prompts(
            input_desc=_INPUT_DESC,
            rule_summary=_RULE_SUMMARY,
            ti2v_output=_TI2V_OUTPUT,
            solution_image_output=_SOLUTION_IMAGE_OUTPUT,
            continuity_timing="discrete",
            continuity_moving="the drill and every cleared block",
            continuity_static=(
                "every surviving block, the board wells, the tick badge, the row and column labels, "
                "the background, and the camera"
            ),
            continuity_hold="the final board after the fourth tick",
        )
        artifacts = write_artifacts(
            output_dir=self.output_dir,
            sample_id=sample_id,
            puzzle_image=puzzle_image,
            solution_image=solution_image,
            option_tiles=option_tiles,
            correct_index=correct_index,
            payload=payload,
            write_video=(lambda video_path: self._write_video(trace, drill_col, log, video_path)) if self.video else None,
        )
        finalize_choice_artifacts(
            output_dir=self.output_dir,
            artifacts=artifacts,
            option_tiles=option_tiles,
            correct_index=correct_index,
            payload=payload,
            video_fps=self.video_fps,
        )
        solution_path = self.relativize_path(artifacts["solution_path"])
        reasoning_images = [self.relativize_path(artifacts["puzzle_path"])]
        record = _BD2Record(
            id=sample_id,
            image=self.relativize_path(artifacts["puzzle_path"]),
            reasoning_image=self.relativize_path(artifacts["puzzle_path"]),
            reasoning_images=reasoning_images,
            solution_image_path=solution_path,
            solution_video_path=self.relativize_path(artifacts["video_path"]) if artifacts["frame_count"] else None,
            video_fps=self.video_fps if artifacts["frame_count"] else None,
            video_num_frames=artifacts["frame_count"] or None,
            ti2v_prompt=prompts["ti2v_prompt"],


            answer=answer_label,
            correct_option=answer_label,


            question="Which candidate shows the board after exactly four drill ticks?",
            structured_answer=payload,
            answer_payload=payload,
        )

        return attach_choice_input_fields(
            record,
            artifacts=artifacts,
            relativize=self.relativize_path,
        )


__all__ = ["MrDrillerDropPredictGenerator"]

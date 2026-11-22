"""Eight-tick walk, turn, and fall prediction on discrete terrain."""

from __future__ import annotations

import random
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


State = Tuple[int, int, str]

_INPUT_DESC = with_grid_xy_convention(
    "The single problem image shows a 10-column by 10-row field with rows and columns labeled 1 through 10. Pale cells "
    "are sky and brown cells are solid terrain. "
    "A green circular walker occupies one sky cell directly above terrain. Its pointed beak shows whether it is "
    "facing left or right. A badge containing eight dots means that exactly 8 ticks must be applied."
)
_RULE_SUMMARY = (
    "At each tick, inspect the horizontally adjacent cell in the walker's facing direction. If that cell is outside "
    "the field or is solid terrain, the walker stays in its current cell and reverses its facing direction; that "
    "uses the whole tick. Otherwise it first moves horizontally into the adjacent sky cell. It then falls straight "
    "down within the same column until the cell immediately below it is solid terrain, keeping the same facing. "
    "Apply exactly 8 ticks. Terrain never changes."
)
_TI2V_OUTPUT = (
    "Hold the input, then show the complete board state after each of the 8 ticks in order. On a blocked "
    "tick, the walker stays in its cell with only its beak reversed; on every other tick it appears one cell "
    "over in its facing direction and settled straight down onto the nearest support in that column. Hold the "
    "tick-8 state at the end."
)
_SOLUTION_IMAGE_OUTPUT = "Render the unchanged terrain with the walker at its position and facing after tick 8."

BG = (246, 249, 253)
SKY = (229, 241, 250)
SKY_EDGE = (204, 221, 235)
EARTH = (166, 112, 70)
EARTH_EDGE = (111, 70, 43)
EARTH_TOP = (213, 162, 104)
WALKER = (82, 184, 111)
WALKER_EDGE = (37, 111, 64)
TEXT = (42, 54, 76)

_BD2Record = make_record_dataclass("lemmings_path_predict", "dynamics_game")


def _encode_state(grid: Sequence[Sequence[int]], *, walker: Tuple[int, int], direction: str) -> List[List[int]]:
    encoded = [[1 if int(cell) == 1 else 0 for cell in row] for row in grid]
    encoded[walker[0]][walker[1]] = 4 if direction == "R" else 5
    return encoded


class LemmingsPathPredictGenerator(AbstractPuzzleGenerator):
    size = 10
    ticks = 8

    def __init__(self, output_dir, *, seed=None, video_fps=DEFAULT_FPS, video=False):
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.video_fps = int(video_fps)
        self.video = bool(video)
        self._counter = 0
        self.cell = 38
        self.left = 66
        self.top = 82

    def _terrain(self) -> List[List[int]]:
        heights = [self.rng.randint(2, 7) for _ in range(self.size)]
        # Limit adjacent cliffs so falls are readable but still consequential.
        for col in range(1, self.size):
            heights[col] = max(2, min(7, heights[col], heights[col - 1] + 4))
        grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for col, height in enumerate(heights):
            for row in range(self.size - height, self.size):
                grid[row][col] = 1
        return grid

    def _walk(self, grid: Sequence[Sequence[int]], start: Tuple[int, int], steps: int = 8, direction: str = "R") -> List[State]:
        row, col = start
        trace: List[State] = [(row, col, direction)]
        for _ in range(steps):
            delta = 1 if direction == "R" else -1
            next_col = col + delta
            if not (0 <= next_col < self.size) or int(grid[row][next_col]) == 1:
                direction = "L" if direction == "R" else "R"
                trace.append((row, col, direction))
                continue
            col = next_col
            while row + 1 < self.size and int(grid[row + 1][col]) == 0:
                row += 1
            trace.append((row, col, direction))
        return trace

    def _build_state(self):
        for _ in range(500):
            grid = self._terrain()
            starts = []
            for col in range(1, self.size - 1):
                first_solid = next((row for row in range(self.size) if grid[row][col] == 1), None)
                if first_solid is not None and first_solid > 0:
                    starts.append((first_solid - 1, col))
            self.rng.shuffle(starts)
            for start in starts:
                trace = self._walk(grid, start, self.ticks)
                moved = sum(left[:2] != right[:2] for left, right in zip(trace, trace[1:]))
                turned = sum(left[:2] == right[:2] and left[2] != right[2] for left, right in zip(trace, trace[1:]))
                fell = sum(right[0] > left[0] for left, right in zip(trace, trace[1:]))
                positions = {state[:2] for state in trace}
                if (
                    moved >= 4
                    and turned >= 1
                    and fell >= 1
                    and turned + fell <= 5
                    and len(positions) >= 4
                    and len(set(trace)) >= 6
                    and trace[-1] not in trace[:-1]
                ):
                    return grid, start, trace
        raise RuntimeError("Unable to generate an eight-tick walk with both a fall and a turn.")

    def _cell_center(self, row: float, col: float) -> Tuple[float, float]:
        return self.left + (col + 0.5) * self.cell, self.top + (row + 0.5) * self.cell

    def _draw_walker(self, draw: ImageDraw.ImageDraw, center: Tuple[float, float], direction: str) -> None:
        cx, cy = center
        radius = 11
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=WALKER, outline=WALKER_EDGE, width=3)
        sign = 1 if direction == "R" else -1
        draw.polygon(
            [
                (cx + sign * 14, cy),
                (cx + sign * 7, cy - 6),
                (cx + sign * 7, cy + 6),
            ],
            fill=WALKER_EDGE,
        )

    def _render(self, grid, *, walker: Optional[Tuple[int, int]] = None, direction: str = "R") -> Image.Image:
        image, draw = new_scene(BG)
        draw.rounded_rectangle((34, 48, 478, 490), radius=24, fill=(255, 255, 255), outline=(211, 220, 233), width=2)
        label_font = scene_font(13)
        for col in range(self.size):
            x = self.left + (col + 0.5) * self.cell
            draw.text((x, self.top - 11), str(col + 1), fill=TEXT, font=label_font, anchor="mm")
        for row in range(self.size):
            y = self.top + (row + 0.5) * self.cell
            draw.text((self.left - 9, y), str(row + 1), fill=TEXT, font=label_font, anchor="rm")
        for row in range(self.size):
            for col in range(self.size):
                x0 = self.left + col * self.cell
                y0 = self.top + row * self.cell
                box = (x0 + 1.5, y0 + 1.5, x0 + self.cell - 1.5, y0 + self.cell - 1.5)
                if int(grid[row][col]) == 1:
                    draw.rounded_rectangle(box, radius=5, fill=EARTH, outline=EARTH_EDGE, width=1)
                    if row == 0 or int(grid[row - 1][col]) == 0:
                        draw.line([(x0 + 5, y0 + 6), (x0 + self.cell - 5, y0 + 6)], fill=EARTH_TOP, width=3)
                else:
                    draw.rounded_rectangle(box, radius=5, fill=SKY, outline=SKY_EDGE, width=1)
        # Eight-dot tick badge; dots are rules, not decoration.
        draw.rounded_rectangle((42, 14, 210, 44), radius=14, fill=(239, 245, 253), outline=(104, 135, 192), width=2)
        for index in range(self.ticks):
            cx = 61 + index * 18
            draw.ellipse((cx - 4, 29 - 4, cx + 4, 29 + 4), fill=(73, 111, 181))
        if walker is not None:
            self._draw_walker(draw, self._cell_center(*walker), direction)
        return finish_scene(image)

    def _render_tick(self, grid, state: State) -> Image.Image:
        return self._render(grid, walker=state[:2], direction=state[2])

    def _write_video(self, grid, trace: Sequence[State], video_path: Path) -> int:
        """Discrete world: hold each committed tick state, then switch directly."""

        frames: List[np.ndarray] = []
        for state in trace:
            array = np.asarray(self._render_tick(grid, state).convert("RGB"), dtype=np.uint8)
            frames.extend(array.copy() for _ in range(4))
        frames.extend(frames[-1].copy() for _ in range(1))
        assert len(frames) == 4 * len(trace) + 1 == 37
        return encode_video(frames, video_path, fps=self.video_fps)

    def _distractors(self, grid, trace: Sequence[State]) -> List[State]:
        correct = trace[-1]
        candidates: List[State] = []
        preferred = [4, 6, 2, 7, 1, 5, 3, 0]
        for index in preferred:
            candidate = trace[index]
            if candidate[:2] != correct[:2] and candidate[:2] not in {item[:2] for item in candidates}:
                candidates.append(candidate)
            if len(candidates) == 3:
                return candidates
        raise RuntimeError("Generated trace does not provide three distinct distractor positions.")

    def create_puzzle(self):
        sample_id = f"lemmings_path_predict-{self._counter:05d}"
        self._counter += 1
        grid, start, trace = self._build_state()
        final = trace[-1]
        puzzle_image = self._render_tick(grid, trace[0])
        solution_image = self._render_tick(grid, final)
        distractor_states = self._distractors(grid, trace)
        option_tiles, candidate_state_grids, correct_index = shuffle_with_correct_semantics(
            solution_image,
            _encode_state(grid, walker=final[:2], direction=final[2]),
            [self._render_tick(grid, state) for state in distractor_states],
            [_encode_state(grid, walker=state[:2], direction=state[2]) for state in distractor_states],
            self.rng,
        )
        # Event-based key states: every committed turn and
        # every landing that ends a fall is a nameable causal commit.  Plain
        # walking ticks are narrated in text between images.
        event_ticks = [
            tick
            for tick in range(1, self.ticks + 1)
            if trace[tick][:2] == trace[tick - 1][:2] or trace[tick][0] > trace[tick - 1][0]
        ]
        intermediate_event_ticks = [tick for tick in event_ticks if tick < self.ticks]
        selected_ticks = (*intermediate_event_ticks, self.ticks)
        initial_grid = _encode_state(grid, walker=start, direction="R")
        full_state_trace = [_encode_state(grid, walker=state[:2], direction=state[2]) for state in trace]
        state_grid = full_state_trace[-1]
        answer_label = option_label(correct_index)
        payload: Dict[str, Any] = {
            "mechanic": "eight_tick_walk_turn_fall",
            "grid": [list(row) for row in grid],
            "start": list(start),
            "ticks": self.ticks,
            "path": [[row, col, direction] for row, col, direction in trace],
            "final": list(final[:2]),
            "final_direction": final[2],
            "initial_grid": initial_grid,
            "state_grid": state_grid,
            "state_trace": full_state_trace,
            "process_state_trace": [full_state_trace[0], *(full_state_trace[tick] for tick in selected_ticks)],
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
            continuity_moving="the walker",
            continuity_static=(
                "every terrain cell, the tick badge, the row and column labels, the sky background, and the camera"
            ),
            continuity_hold="the walker's state after the eighth tick",
        )
        artifacts = write_artifacts(
            output_dir=self.output_dir,
            sample_id=sample_id,
            puzzle_image=puzzle_image,
            solution_image=solution_image,
            option_tiles=option_tiles,
            correct_index=correct_index,
            payload=payload,
            write_video=(lambda video_path: self._write_video(grid, trace, video_path)) if self.video else None,
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


            question="Which candidate shows the walker after exactly eight ticks?",
            structured_answer=payload,
            answer_payload=payload,
        )

        return attach_choice_input_fields(
            record,
            artifacts=artifacts,
            relativize=self.relativize_path,
        )


__all__ = ["LemmingsPathPredictGenerator"]

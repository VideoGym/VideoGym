"""Simultaneous multi-lane falling-note prediction.

Four lanes each have an explicit integer fall speed.  Every visible note moves
at its lane's speed for two synchronous ticks.  Samples guarantee that exactly
one lane's lowest note reaches the strike line on tick two and that no note
crosses the line earlier.  The task therefore requires multi-object state
rollout rather than selecting a visually lowest stationary marker.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import DEFAULT_FPS
from data.dynamics_game.core.scene import finish_scene, new_scene, scene_font
from data.common.multiple_choice import (
    option_label,
    save_labeled_option_images,
)
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.dynamics_game.core.choice_composite import (
    attach_choice_input_fields,
    build_composite_mcq_prompts,
)
from data.dynamics_game.core.coordinate_format import display_indices_text


_INPUT_DESC = (
    "The single input image contains four vertical lanes labeled 1 through 4 and note rows labeled 1 through 7. "
    "Red circular notes sit at integer row positions above a magenta strike line, and a badge above each lane "
    "shows that lane's downward speed as 1 or 2 rows per tick."
)
_RULE_SUMMARY = (
    "Apply exactly two synchronous ticks. During each tick, every note in every lane moves straight downward "
    "by exactly the speed printed above its lane; all lanes update together, and notes never change lanes or "
    "pass one another. In every instance, exactly one lowest note reaches the strike line at the end of tick "
    "2; it is removed and replaced by a green hit marker in the same lane. Every other note remains visible at "
    "its computed position, and no note reaches or crosses the strike line earlier."
)
_TI2V_OUTPUT = (
    "Hold the input, then show the complete board after tick 1 and the complete board after tick 2 as exact "
    "states: every note sits at its speed-derived row, and in the tick-2 state the unique note that reached "
    "the strike line has been replaced by one green hit marker in its lane. Hold that final state."
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render the complete state after tick 2: all surviving notes at their computed positions and one green hit "
    "marker on the unique lane whose lowest note reached the strike line. Preserve the lane geometry and the "
    "four speed badges."
)
BG = (246, 248, 252)
PANEL = (255, 255, 255)
LANES = ((237, 242, 250), (229, 236, 247))
GRID = (188, 198, 216)
TEXT = (36, 46, 66)
NOTE = (239, 86, 78)
NOTE_EDGE = (151, 44, 43)
HIT = (74, 181, 118)
HIT_EDGE = (35, 116, 70)
STRIKE = (190, 55, 143)
SPEED = (70, 103, 180)


@dataclass
class RhythmGameTapStepRecord:
    id: str
    task_name: str = "lane_tap_step"
    task_family: str = "dynamics_game"
    image: str = ""
    reasoning_image: str = ""
    reasoning_images: List[str] = field(default_factory=list)
    solution_image_path: str = ""
    solution_video_path: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None
    ti2v_prompt: str = ""
    solution_image_prompt: str = ""
    answer: str = ""
    correct_option: str = ""
    question: str = ""
    question_description: str = "lane_tap_step"
    question_id: int = 0
    qa_type: str = "State Prediction"
    qa_level: str = "Medium"
    plot_level: str = "Medium"
    answer_grid: List[List[int]] = field(default_factory=list)
    structured_answer: Dict[str, Any] = field(default_factory=dict)
    answer_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class RhythmGameTapStepGenerator(AbstractPuzzleGenerator[RhythmGameTapStepRecord]):
    ROWS = 7
    COLS = 4
    TICKS = 2

    def __init__(self, output_dir: str | Path, *, seed: Optional[int] = None, video_fps: int = DEFAULT_FPS,
                 video: bool = False, **_: Any) -> None:
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.video_fps = video_fps
        self.video = video
        self._counter = 0
        self.left = 78
        self.top = 100
        self.lane_w = 89
        self.row_h = 49
        self.strike_y = self.top + self.ROWS * self.row_h
        self.lane_label_y = self.strike_y + 26
        self.panel_bottom = self.lane_label_y + 23
        self.panel_top = 58

    def record_to_dict(self, record: RhythmGameTapStepRecord | Dict[str, Any]) -> Dict[str, Any]:
        return record.to_dict() if hasattr(record, "to_dict") else dict(record)

    def _build_state(self) -> Tuple[List[List[int]], List[int], int]:
        target = self.rng.randrange(self.COLS)
        speeds = [self.rng.choice((1, 2)) for _ in range(self.COLS)]
        notes: List[List[int]] = []
        for lane, speed in enumerate(speeds):
            if lane == target:
                nearest = self.ROWS - self.TICKS * speed
            else:
                max_nearest = self.ROWS - self.TICKS * speed - 1
                nearest = self.rng.randint(max(2, max_nearest - 1), max_nearest)
            candidates = list(range(0, max(0, nearest - 1)))
            self.rng.shuffle(candidates)
            lane_notes = sorted((candidates[:2] + [nearest]))
            notes.append(lane_notes)
        assert sum(row + self.TICKS * speeds[lane] == self.ROWS for lane, rows in enumerate(notes) for row in rows) == 1
        return notes, speeds, target

    def _state_at(self, notes: Sequence[Sequence[int]], speeds: Sequence[int], tick: int) -> List[List[int]]:
        grid = [[0 for _ in range(self.COLS)] for _ in range(self.ROWS + 1)]
        for lane, lane_notes in enumerate(notes):
            for row in lane_notes:
                moved = row + tick * speeds[lane]
                if moved == self.ROWS:
                    grid[self.ROWS][lane] = 2
                elif moved < self.ROWS:
                    grid[moved][lane] = 1
        return grid

    def _distractors(self, notes: Sequence[Sequence[int]], speeds: Sequence[int], target: int) -> List[List[List[int]]]:
        correct = self._state_at(notes, speeds, self.TICKS)
        one_tick = self._state_at(notes, speeds, 1)
        all_speed_one = self._state_at(notes, [1] * self.COLS, self.TICKS)
        frozen = [row[:] for row in correct]
        lane = (target + 1) % self.COLS
        for row in range(self.ROWS + 1):
            frozen[row][lane] = 0
        for row in notes[lane]:
            frozen[row][lane] = 1
        out: List[List[List[int]]] = []
        seen = {self._grid_key(correct)}
        for candidate in (one_tick, all_speed_one, frozen):
            key = self._grid_key(candidate)
            if key not in seen:
                seen.add(key)
                out.append(candidate)
        for lane in range(self.COLS):
            if len(out) == 3:
                break
            candidate = [row[:] for row in correct]
            candidate[self.ROWS][target] = 0
            candidate[self.ROWS][lane] = 2
            key = self._grid_key(candidate)
            if key not in seen:
                seen.add(key)
                out.append(candidate)
        if len(out) != 3:
            raise RuntimeError("Unable to build three unique lane-rollout distractors.")
        return out

    @staticmethod
    def _grid_key(grid: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(int(value) for value in row) for row in grid)

    def _draw_note(self, draw: ImageDraw.ImageDraw, cx: float, cy: float) -> None:
        radius = 15
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=NOTE, outline=NOTE_EDGE, width=2)

    def _draw_hit(self, draw: ImageDraw.ImageDraw, cx: float, cy: float) -> None:
        import math

        r = 16
        points = []
        for index in range(16):
            angle = -math.pi / 2 + index * math.pi / 8
            radius = r if index % 2 == 0 else r * 0.48
            points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
        draw.polygon(points, fill=HIT, outline=HIT_EDGE)

    def _render(self, notes: Sequence[Sequence[int]], speeds: Sequence[int], *, time_value: float = 0.0,
                final_grid: Optional[Sequence[Sequence[int]]] = None) -> Image.Image:
        image, draw = new_scene(BG)
        font = scene_font(18)
        small = scene_font(15)
        left, top, lane_w = self.left, self.top, self.lane_w
        strike_y = self.strike_y
        draw.rounded_rectangle(
            [54, self.panel_top, 458, self.panel_bottom],
            radius=24,
            fill=PANEL,
            outline=(218, 224, 235),
            width=2,
        )
        for lane in range(self.COLS):
            x0 = left + lane * lane_w
            x1 = x0 + lane_w
            draw.rectangle([x0, top, x1, strike_y], fill=LANES[lane % 2])
            draw.line([(x0, top), (x0, strike_y)], fill=GRID, width=1)
            cx = x0 + lane_w / 2
            draw.rounded_rectangle([cx - 25, 68, cx + 25, 94], radius=10, fill=(235, 241, 253), outline=SPEED, width=2)
            draw.text((cx, 81), f"↓{speeds[lane]}", font=small, fill=SPEED, anchor="mm")
            draw.text((cx, self.lane_label_y), str(lane + 1), font=font, fill=TEXT, anchor="mm")
        draw.line([(left + self.COLS * lane_w, top), (left + self.COLS * lane_w, strike_y)], fill=GRID, width=1)
        for row in range(self.ROWS + 1):
            y = self.top + row * self.row_h
            draw.line([(left, y), (left + self.COLS * lane_w, y)], fill=GRID, width=1)
            if row < self.ROWS:
                draw.text((74, self.top + (row + 0.5) * self.row_h), str(row + 1), font=font, fill=TEXT, anchor="rm")
        draw.rounded_rectangle([left - 6, strike_y - 5, left + self.COLS * lane_w + 6, strike_y + 7], radius=5, fill=STRIKE)
        for lane, lane_notes in enumerate(notes):
            cx = self.left + lane * self.lane_w + self.lane_w / 2
            for row in lane_notes:
                moved = row + time_value * speeds[lane]
                if moved >= self.ROWS:
                    continue
                cy = self.top + (moved + 0.5) * self.row_h
                self._draw_note(draw, cx, cy)
        if final_grid is not None:
            for lane in range(self.COLS):
                if int(final_grid[self.ROWS][lane]) == 2:
                    cx = self.left + lane * self.lane_w + self.lane_w / 2
                    self._draw_hit(draw, cx, strike_y)
        return finish_scene(image)

    def _render_grid(self, grid: Sequence[Sequence[int]], speeds: Sequence[int]) -> Image.Image:
        notes = [[] for _ in range(self.COLS)]
        for row in range(self.ROWS):
            for lane in range(self.COLS):
                if int(grid[row][lane]) == 1:
                    notes[lane].append(row)
        return self._render(notes, speeds, final_grid=grid)

    def _write_video(self, notes: Sequence[Sequence[int]], speeds: Sequence[int], final_grid: Sequence[Sequence[int]], path: Path) -> int:
        """Discrete world: hold each complete tick state, then switch directly."""

        frames: List[np.ndarray] = []
        states = (
            self._render(notes, speeds),
            self._render(notes, speeds, time_value=1.0),
            self._render(notes, speeds, time_value=2.0, final_grid=final_grid),
        )
        for state in states:
            array = np.asarray(state, dtype=np.uint8)
            frames.extend(array.copy() for _ in range(13))
        frames.extend(frames[-1].copy() for _ in range(2))
        assert len(frames) == 41
        return len(frames) if encode_rgb_frames_to_webm(frames, path, fps=self.video_fps, extra_args=["-g", "1", "-bf", "0"]) else 0

    def _prompts(self) -> Dict[str, str]:
        return build_composite_mcq_prompts(
            input_desc=_INPUT_DESC,
            rule_summary=_RULE_SUMMARY,
            ti2v_output=_TI2V_OUTPUT,
            solution_image_output=_SOLUTION_IMAGE_OUTPUT,
            continuity_timing="discrete",
            continuity_moving="the note discs and the green hit marker",
            continuity_static=(
                "the four lanes, the strike line, the lane and row labels, the speed badges, "
                "the background, and the camera"
            ),
            continuity_hold="the complete state after the second tick",
        )

    def create_puzzle(self) -> RhythmGameTapStepRecord:
        sample_id = f"lane_tap_step-{self._counter:05d}"
        self._counter += 1
        notes, speeds, target = self._build_state()
        tick_one = self._state_at(notes, speeds, 1)
        final_grid = self._state_at(notes, speeds, self.TICKS)
        distractors = self._distractors(notes, speeds, target)
        option_states = [final_grid, *distractors]
        order = list(range(4))
        self.rng.shuffle(order)
        answer_index = order.index(0)
        option_states = [option_states[index] for index in order]

        dirs = {name: self.output_dir / name for name in ("puzzles", "solutions", "reasoning_options", "videos", "states")}
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        puzzle_path = dirs["puzzles"] / f"{sample_id}.png"
        solution_path = dirs["solutions"] / f"{sample_id}.png"
        video_path = dirs["videos"] / f"{sample_id}.webm"

        self._render(notes, speeds).save(puzzle_path)
        self._render_grid(final_grid, speeds).save(solution_path)
        option_tiles = [self._render_grid(grid, speeds) for grid in option_states]
        option_paths = save_labeled_option_images(option_tiles, dirs["reasoning_options"], stem=sample_id, labels=("A", "B", "C", "D"))
        frame_count = self._write_video(notes, speeds, final_grid, video_path) if self.video else 0
        answer_label = option_label(answer_index)
        payload = {
            "mechanic": "simultaneous_lane_fall_two_ticks",
            "rows": self.ROWS,
            "cols": self.COLS,
            "ticks": self.TICKS,
            "lane_speeds": speeds,
            "initial_note_rows": notes,
            "target_lane": target,
            "initial_grid": self._state_at(notes, speeds, 0),
            "state_grid": final_grid,
            "state_trace": [self._state_at(notes, speeds, 0), tick_one, final_grid],
            "process_state_trace": [self._state_at(notes, speeds, 0), tick_one, final_grid],
            "candidate_state_grids": option_states,
            "correct_option_index": answer_index,
            "correct_option": answer_label,
        }
        artifacts = {
            "puzzle_path": puzzle_path,
            "solution_path": solution_path,
            "options_paths": option_paths,
            "video_path": video_path,
            "frame_count": frame_count,
        }
        prompts = self._prompts()
        reasoning_images = [self.relativize_path(puzzle_path)]
        "; ".join(
            f"lane {lane + 1} has speed {speeds[lane]} and rows {display_indices_text(lane_notes)} move to "
            f"{display_indices_text([row + speeds[lane] for row in lane_notes])}"
            for lane, lane_notes in enumerate(notes)
        )
        "; ".join(
            f"lane {lane + 1} survivors finish at rows "
            f"{display_indices_text([row + 2 * speeds[lane] for row in lane_notes if row + 2 * speeds[lane] < self.ROWS])}"
            for lane, lane_notes in enumerate(notes)
        )
        record = RhythmGameTapStepRecord(
            id=sample_id,
            image=self.relativize_path(puzzle_path),
            reasoning_image=self.relativize_path(puzzle_path),
            reasoning_images=reasoning_images,
            solution_image_path=self.relativize_path(solution_path),
            solution_video_path=self.relativize_path(video_path) if frame_count else None,
            video_fps=self.video_fps if frame_count else None,
            video_num_frames=frame_count or None,
            ti2v_prompt=prompts["ti2v_prompt"],

            answer=answer_label,
            correct_option=answer_label,
            question="Which candidate shows the complete lane state after two simultaneous ticks?",
            answer_grid=final_grid,
            structured_answer=payload,
            answer_payload=payload,
        )

        return attach_choice_input_fields(
            record,
            artifacts=artifacts,
            relativize=self.relativize_path,
        )


__all__ = ["RhythmGameTapStepGenerator", "RhythmGameTapStepRecord"]

"""Deterministic pegboard marble landing prediction.

The board has six triangular rows.  A red peg sends the marble to the child
down-left, while a blue peg sends it to the child down-right.  The six local
decisions determine exactly one of the seven bottom slots.  Boundary outcomes
0 and 6 are legal (all-left and all-right paths respectively).

The same simulated route drives the answer, final image, structured metadata,
and continuous TI2V motion.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.prompts import build_sectioned_prompt
from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import CANVAS_SIZE, DEFAULT_FPS
from data.dynamics_game.core.scene import finish_scene, new_scene, scene_font
from data.common.pipeline.discrete_dynamics import (
    LABEL_COLOR,
    encode_video,
    make_record_dataclass,
    write_artifacts,
)
from data.common.multiple_choice import option_label
from data.dynamics_game.core.coordinate_format import with_grid_xy_convention
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="continuous",
    moving_nouns="the dropping marble",
    static_nouns="everything the rules leave unchanged, including the board, labels, background, and camera",
    hold_noun="the final board state",
)

_INPUT_DESC = with_grid_xy_convention(
    "The single input image shows a triangular pegboard with rows labeled 1 through 6: the top row has one peg, the next row has "
    "two, and so on through six pegs in the bottom row. Red pegs contain left arrows and blue pegs contain right "
    "arrows. Seven landing slots numbered 1 through 7 sit below the board, and exactly four slots carry candidate "
    "badges A through D. A silver marble starts above the top peg."
)
_RULE_SUMMARY = (
    "The marble strikes exactly one peg in each row. At a red left-arrow peg it continues to the child peg or exit "
    "position down-left; at a blue right-arrow peg it continues to the child peg or exit position down-right. "
    "After the sixth peg, the same rule sends it into one numbered slot from 1 through 7, including boundary slots "
    "1 and 7 when the route stays entirely on one side. Choose the unique candidate badge attached to that slot."
)
_TI2V_OUTPUT = (
    "Hold the input, then move the same silver marble continuously through one peg in each row, pausing "
    "briefly at every contact and changing direction only afterward according to that arrow. Extend one "
    "cumulative green dashed trail behind it without erasing earlier segments. After the sixth peg, move it "
    "into the resulting slot; only then change the complete outlines of that slot and its attached badge from "
    "gray to green together, then hold. Keep all pegs, arrows, unselected slots and badges, labels, fills, "
    "and spacing fixed; do not skip a row or select early."
)
PEG_L = (228, 80, 80)
PEG_L_DARK = (160, 30, 30)
PEG_R = (60, 130, 220)
PEG_R_DARK = (30, 70, 160)
MARBLE = (220, 225, 240)
MARBLE_DARK = (60, 70, 90)
SLOT_BG = (245, 245, 248)
PATH_COLOR = (58, 190, 94)
CANDIDATE_BG = (255, 255, 255)
SELECTION_NEUTRAL = (120, 130, 150)
SELECTION_GREEN = (55, 155, 75)
CANDIDATE_EDGE = SELECTION_NEUTRAL
_BD2Record = make_record_dataclass("pegboard_marble_drop_predict", "dynamics_game")


def _encode_state(
    pegs: Dict[Tuple[int, int], str],
    landing_slot: Optional[int] = None,
    rows: int = 6,
) -> List[List[int]]:
    grid = [[0 for _ in range(rows + 1)] for _ in range(rows + 1)]
    for (row, col), kind in pegs.items():
        grid[row][col] = 1 if kind == "L" else 2
    if landing_slot is not None:
        grid[rows][landing_slot] = 3
    return grid


def _simulate_path(
    pegs: Dict[Tuple[int, int], str],
    *,
    rows: int = 6,
) -> Tuple[List[Tuple[int, int]], int]:
    """Replay all row decisions and return struck pegs plus unique slot."""

    path: List[Tuple[int, int]] = []
    col = 0
    for row in range(rows):
        cell = (row, col)
        kind = pegs.get(cell)
        if kind not in {"L", "R"}:
            raise ValueError(f"missing or invalid peg at {cell}: {kind!r}")
        path.append(cell)
        if kind == "R":
            col += 1
        if not 0 <= col <= row + 1:
            raise ValueError(f"peg transition left triangular boundary after row {row}")
    return path, col


class PegboardMarbleDropPredictGenerator(AbstractPuzzleGenerator):
    def __init__(self, output_dir, *, seed=None, video_fps=DEFAULT_FPS, video=False):
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.rows = 6
        self.video_fps = video_fps
        self.video = video
        self._counter = 0
        self.peg_radius = 16
        self.row_h = 50
        self.cols_w = 56
        # Keep the starting marble fully inside the 512x512 safe area while
        # retaining room for slot numbers and candidate badges below.
        self.top_y = 54
        self.bottom_y = self.top_y + self.rows * self.row_h + 20

    def _build_state(self):
        pegs = {
            (row, col): self.rng.choice(("L", "R"))
            for row in range(self.rows)
            for col in range(row + 1)
        }
        path, landing_slot = _simulate_path(pegs, rows=self.rows)
        return pegs, path, landing_slot

    def _peg_xy(self, row: int, col: int) -> Tuple[float, float]:
        x = (CANVAS_SIZE // 2) - (row * self.cols_w) // 2 + col * self.cols_w
        y = self.top_y + row * self.row_h
        return float(x), float(y)

    def _slot_x(self, slot: int) -> float:
        return float((CANVAS_SIZE // 2) - (self.rows * self.cols_w) // 2 + slot * self.cols_w)

    def _slot_center(self, slot: int) -> Tuple[float, float]:
        return self._slot_x(slot), float(self.bottom_y + 24)

    def _start_point(self) -> Tuple[float, float]:
        return self._peg_xy(0, 0)[0], float(self.top_y - 30)

    def _route_points(
        self,
        path: Sequence[Tuple[int, int]],
        landing_slot: int,
    ) -> List[Tuple[float, float]]:
        return [self._start_point(), *(self._peg_xy(row, col) for row, col in path), self._slot_center(landing_slot)]

    def _draw_dotted_segment(
        self,
        draw: ImageDraw.ImageDraw,
        start: Tuple[float, float],
        end: Tuple[float, float],
        *,
        color: Tuple[int, int, int] = PATH_COLOR,
        dash: float = 9.0,
        gap: float = 6.0,
    ) -> None:
        ax, ay = start
        bx, by = end
        dx, dy = bx - ax, by - ay
        length = float(np.hypot(dx, dy))
        if length < 1.0:
            return
        ux, uy = dx / length, dy / length
        offset = 0.0
        while offset < length:
            finish = min(length, offset + dash)
            draw.line(
                [(ax + ux * offset, ay + uy * offset), (ax + ux * finish, ay + uy * finish)],
                fill=color,
                width=4,
            )
            offset += dash + gap

    def _draw_marble(self, draw: ImageDraw.ImageDraw, point: Tuple[float, float]) -> None:
        x, y = point
        radius = 14
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=MARBLE, outline=MARBLE_DARK, width=2)

    @staticmethod
    def _selection_color(progress: float) -> Tuple[int, int, int]:
        value = max(0.0, min(1.0, float(progress)))
        eased = value * value * (3.0 - 2.0 * value)
        return tuple(
            int(round(start + (end - start) * eased))
            for start, end in zip(SELECTION_NEUTRAL, SELECTION_GREEN)
        )

    def _render(
        self,
        pegs: Dict[Tuple[int, int], str],
        *,
        candidate_slots: Sequence[int],
        marble_xy: Optional[Tuple[float, float]],
        trail_points: Sequence[Tuple[float, float]] = (),
        selected_slot: Optional[int] = None,
        selection_progress: float = 1.0,
    ) -> Image.Image:
        image, draw = new_scene((247, 249, 253))
        glyph_font = scene_font(18)
        candidate_font = scene_font(16)
        row_font = scene_font(13)
        slot_w = self.cols_w - 6

        draw.rounded_rectangle((22, 18, 490, 498), radius=26, fill=(255, 255, 255), outline=(210, 220, 234), width=2)

        for slot in range(self.rows + 1):
            center_x = self._slot_x(slot)
            x0 = center_x - slot_w // 2
            selected = slot == selected_slot
            edge = self._selection_color(selection_progress) if selected else SELECTION_NEUTRAL
            draw.rounded_rectangle(
                (x0, self.bottom_y, x0 + slot_w, self.bottom_y + 50),
                radius=7,
                fill=SLOT_BG,
                outline=edge,
                width=2,
            )
            draw.text((center_x, self.bottom_y + 65), str(slot + 1), fill=LABEL_COLOR, font=glyph_font, anchor="mm")

        for index, slot in enumerate(candidate_slots):
            center_x = self._slot_x(slot)
            center_y = self.bottom_y + 94
            selected = slot == selected_slot
            edge = self._selection_color(selection_progress) if selected else CANDIDATE_EDGE
            draw.ellipse(
                (center_x - 13, center_y - 13, center_x + 13, center_y + 13),
                fill=CANDIDATE_BG,
                outline=edge,
                width=2,
            )
            draw.text((center_x, center_y), option_label(index), fill=LABEL_COLOR, font=candidate_font, anchor="mm")

        for start, end in zip(trail_points, trail_points[1:]):
            self._draw_dotted_segment(draw, start, end)

        for row in range(self.rows):
            _, y = self._peg_xy(row, 0)
            draw.text((72, y), str(row + 1), fill=LABEL_COLOR, font=row_font, anchor="rm")

        for (row, col), kind in pegs.items():
            x, y = self._peg_xy(row, col)
            if kind == "L":
                fill, edge, sign = PEG_L, PEG_L_DARK, -1
            else:
                fill, edge, sign = PEG_R, PEG_R_DARK, 1
            draw.ellipse(
                (x - self.peg_radius, y - self.peg_radius, x + self.peg_radius, y + self.peg_radius),
                fill=fill,
                outline=edge,
                width=3,
            )
            points = [
                (x + sign * 8, y),
                (x - sign * 2, y - 7),
                (x - sign * 2, y - 3),
                (x - sign * 8, y - 3),
                (x - sign * 8, y + 3),
                (x - sign * 2, y + 3),
                (x - sign * 2, y + 7),
            ]
            draw.polygon(points, fill=(255, 255, 255))

        if marble_xy is not None:
            self._draw_marble(draw, marble_xy)
        return finish_scene(image)

    def _write_video(
        self,
        pegs: Dict[Tuple[int, int], str],
        path: Sequence[Tuple[int, int]],
        landing_slot: int,
        candidate_slots: Sequence[int],
        video_path: Path,
    ) -> int:
        vertices = self._route_points(path, landing_slot)
        frames: List[np.ndarray] = []

        initial = self._render(
            pegs,
            candidate_slots=candidate_slots,
            marble_xy=vertices[0],
        )
        frames.extend(np.asarray(initial, dtype=np.uint8).copy() for _ in range(5))

        substeps = 4
        for segment_index, (start, end) in enumerate(zip(vertices, vertices[1:])):
            for substep in range(1, substeps + 1):
                ratio = substep / substeps
                point = (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
                trail = [*vertices[: segment_index + 1], point]
                frame = self._render(
                    pegs,
                    candidate_slots=candidate_slots,
                    marble_xy=point,
                    trail_points=trail,
                )
                frames.append(np.asarray(frame, dtype=np.uint8))
            if segment_index < self.rows:
                frames.append(frames[-1].copy())

        landing = self._render(
            pegs,
            candidate_slots=candidate_slots,
            marble_xy=vertices[-1],
            trail_points=vertices,
        )
        frames.extend(np.asarray(landing, dtype=np.uint8).copy() for _ in range(4))
        for step in range(1, 9):
            selection = self._render(
                pegs,
                candidate_slots=candidate_slots,
                marble_xy=vertices[-1],
                trail_points=vertices,
                selected_slot=landing_slot,
                selection_progress=step / 8.0,
            )
            frames.append(np.asarray(selection, dtype=np.uint8))
        final = self._render(
            pegs,
            candidate_slots=candidate_slots,
            marble_xy=vertices[-1],
            trail_points=vertices,
            selected_slot=landing_slot,
        )
        frames.extend(np.asarray(final, dtype=np.uint8).copy() for _ in range(14))
        assert len(frames) == 65
        return encode_video(frames, video_path, fps=self.video_fps)

    def _build_prompts(self) -> Dict[str, str]:
        return {
            "ti2v_prompt": build_sectioned_prompt(
                _INPUT_DESC,
                _TI2V_OUTPUT,
                mode="ti2v",
                rule_description=_RULE_SUMMARY,
                continuity_paragraph=_TI2V_CONTINUITY,
            ),
        }

    def create_puzzle(self):
        sample_id = f"pegboard_marble_drop_predict-{self._counter:05d}"
        self._counter += 1
        pegs, path, landing_slot = self._build_state()

        wrong_slots = [slot for slot in range(self.rows + 1) if slot != landing_slot]
        candidate_slots = [landing_slot, *self.rng.sample(wrong_slots, 3)]
        self.rng.shuffle(candidate_slots)
        correct_index = candidate_slots.index(landing_slot)

        vertices = self._route_points(path, landing_slot)
        puzzle_image = self._render(
            pegs,
            candidate_slots=candidate_slots,
            marble_xy=vertices[0],
        )
        solution_image = self._render(
            pegs,
            candidate_slots=candidate_slots,
            marble_xy=vertices[-1],
            trail_points=vertices,
            selected_slot=landing_slot,
        )
        option_tiles = [
            self._render(
                pegs,
                candidate_slots=candidate_slots,
                marble_xy=self._slot_center(slot),
                selected_slot=slot,
            )
            for slot in candidate_slots
        ]

        prompts = self._build_prompts()
        initial_grid = _encode_state(pegs, rows=self.rows)
        state_grid = _encode_state(pegs, landing_slot=landing_slot, rows=self.rows)
        process_state_trace = [
            [[int(value) for value in grid_row] for grid_row in initial_grid]
            for _ in range(self.rows + 1)
        ]
        process_state_trace.append(
            [[int(value) for value in grid_row] for grid_row in state_grid]
        )
        directions = [pegs[cell] for cell in path]
        payload = {
            "mechanic_family": "pegboard_marble",
            "mechanic": "six_row_red_left_blue_right_marble_route",
            "pegs": {f"{row},{col}": kind for (row, col), kind in pegs.items()},
            "path": [list(cell) for cell in path],
            "directions": directions,
            "landing_slot": landing_slot,
            "candidate_slots": list(candidate_slots),
            "initial_grid": initial_grid,
            "state_grid": state_grid,
            "state_trace": [initial_grid, state_grid],
            "process_state_trace": process_state_trace,
            "correct_option_index": correct_index,
            "correct_option": option_label(correct_index),
        }
        write_video_fn = (
            lambda video_path: self._write_video(
                pegs,
                path,
                landing_slot,
                candidate_slots,
                video_path,
            )
        ) if self.video else None
        artifacts = write_artifacts(
            output_dir=self.output_dir,
            sample_id=sample_id,
            puzzle_image=puzzle_image,
            solution_image=solution_image,
            option_tiles=option_tiles,
            correct_index=correct_index,
            payload=payload,
            write_video=write_video_fn,
        )

        answer_label = option_label(correct_index)
        puzzle_path = self.relativize_path(artifacts["puzzle_path"])
        solution_path = self.relativize_path(artifacts["solution_path"])
        return _BD2Record(
            id=sample_id,
            image=puzzle_path,
            reasoning_image=puzzle_path,
            reasoning_images=[puzzle_path],
            solution_image_path=solution_path,
            solution_video_path=self.relativize_path(artifacts["video_path"]) if artifacts["frame_count"] > 0 else None,
            video_fps=self.video_fps if artifacts["frame_count"] > 0 else None,
            video_num_frames=artifacts["frame_count"] if artifacts["frame_count"] > 0 else None,
            ti2v_prompt=prompts["ti2v_prompt"],


            answer=answer_label,
            correct_option=answer_label,


            question="Which candidate badge marks the slot reached by the marble after all six peg deflections?",
            structured_answer=payload,
            answer_payload=payload,
        )


__all__ = ["PegboardMarbleDropPredictGenerator"]

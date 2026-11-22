"""Unique bounded-press Lights Out with a visible press cue in TI2V."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    format_xy_position_list_answer,
    format_xy_position_list_payload,
)
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.common.prompts import build_sectioned_prompt
from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import CANVAS_SIZE, load_font
from data.common.pipeline.board_records import BoardChoiceRecord, write_board_choice_record
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.video_encoding import encode_rgb_frames_to_webm

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="continuous",
    moving_nouns="the pressed-cell shade and the switching cells",
    static_nouns="the grid lines, the row and column numbers, background, and camera",
    hold_noun="the final all-OFF board",
)

TASK_NAME = "lightsout_single_press"

_ON_COLOR = (255, 230, 100)
_OFF_COLOR = (74, 79, 88)
_PRESS_SHADE = 0.5  # blend fraction toward black on the pressed cell

_INPUT_DESC = (
    "The single input image shows a 5 by 5 grid of square cells. Column numbers 1-5 run along the top and row numbers "
    f"1-5 run down the left. {GRID_XY_CONVENTION_SENTENCE} Yellow "
    "cells are ON and dark gray cells are OFF."
)
_RULE_SUMMARY = (
    "Press between one and three distinct cells, each exactly once. A press switches the pressed cell and every "
    "horizontally or vertically adjacent cell: ON becomes OFF and OFF becomes ON. Diagonal cells never switch, and "
    "neighbor positions outside the board are ignored. Press order does not change the result. Exactly one set of one "
    "to three presses turns every cell OFF."
)
_TI2V_OUTPUT = (
    "For each press of the unique set, first darken the pressed cell alone to a clearly pressed shade. While it stays "
    "shaded, switch it and its in-bounds orthogonal neighbors together to their opposite colors, then lift the shade to "
    "leave the committed colors before the next press. Any order of the set is valid. Do not switch a diagonal or "
    "distant cell, shade more than one cell at a time, or add any other marker."
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render the final board after the unique press set with every cell OFF and the grid, numbers, framing, and camera "
    "unchanged; add no ring, badge, or shading."
)


def lights_layout(size: int) -> Tuple[int, int, int]:
    """Edge-to-edge cell layout shared by the renderer and the verifier.

    The grid fills the canvas except for the family-wide 44-pixel top/left
    label strip; the right/bottom pad is only the residue of an even integer
    cell size.
    """

    strip = 44
    cell = (CANVAS_SIZE - strip - 8) // size
    cell -= cell % 2
    return cell, strip, strip


class LightsOutSinglePressGenerator(AbstractPuzzleGenerator[BoardChoiceRecord]):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        size: int = 5,
        press_count_range: Tuple[int, int] = (1, 3),
        video_fps: int = 16,
        video: bool = False,
    ) -> None:
        super().__init__(output_dir)
        self.seed = seed
        self.rng = random.Random(seed)
        self.size = max(4, size)
        self.video_fps = video_fps
        self.video = video
        self._counter = 0
        self.press_count_range = tuple(press_count_range)

    def record_to_dict(self, record: BoardChoiceRecord | Dict[str, Any]) -> Dict[str, Any]:
        return record.to_dict() if hasattr(record, "to_dict") else dict(record)

    def _layout(self) -> Tuple[int, int, int]:
        return lights_layout(self.size)

    def _toggle(self, grid: List[List[int]], row: int, col: int) -> None:
        for rr, cc in self._toggle_cells(row, col):
            grid[rr][cc] = 1 - grid[rr][cc]

    def _toggle_cells(self, row: int, col: int) -> List[Tuple[int, int]]:
        cells = []
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = row + dr
            cc = col + dc
            if 0 <= rr < self.size and 0 <= cc < self.size:
                cells.append((rr, cc))
        return cells

    def _solutions(self, puzzle: Sequence[Sequence[int]]):
        mask = (1 << self.size) - 1
        rows = [sum(int(v) << c for c, v in enumerate(row)) for row in puzzle]
        solutions = []
        for first in range(1 << self.size):
            presses = [first]
            for row in range(1, self.size):
                previous = presses[-1]
                above = presses[-2] if row > 1 else 0
                presses.append(
                    rows[row - 1]
                    ^ previous
                    ^ ((previous << 1) & mask)
                    ^ (previous >> 1)
                    ^ above
                )
            last = presses[-1]
            if rows[-1] ^ last ^ ((last << 1) & mask) ^ (last >> 1) ^ presses[-2]:
                continue
            cells = tuple(
                (r, c)
                for r, p in enumerate(presses)
                for c in range(self.size)
                if p & (1 << c)
            )
            if 1 <= len(cells) <= self.press_count_range[1]:
                solutions.append(cells)
        return sorted(solutions)

    def _build_board(
        self,
    ) -> tuple[
        List[List[int]],
        List[List[int]],
        Tuple[Tuple[int, int], ...],
        Tuple[Tuple[int, int], ...],
    ]:
        solution = [[0 for _ in range(self.size)] for _ in range(self.size)]
        cells = [(row, col) for row in range(self.size) for col in range(self.size)]
        for _ in range(200):
            count = self.rng.randint(*self.press_count_range)
            press_order = tuple(self.rng.sample(cells, count))
            answer_set = tuple(sorted(press_order))
            puzzle = [row[:] for row in solution]
            for row, col in press_order:
                self._toggle(puzzle, row, col)
            if self._solutions(puzzle) == [answer_set]:
                return puzzle, solution, answer_set, press_order
        raise RuntimeError("Could not construct a unique bounded-press board.")

    def _render_board(
        self,
        grid: Sequence[Sequence[int]],
        *,
        cell_alpha: Optional[Dict[Tuple[int, int], float]] = None,
        press_cue: Optional[Tuple[Tuple[int, int], float]] = None,
        analysis_presses: Sequence[Tuple[int, int]] = (),
        parity_badges: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> Image.Image:
        """Render the board.

        ``cell_alpha`` maps cells to a partial-toggle alpha used during the
        switch dissolve.  ``press_cue`` is the single pressed cell with its
        shade depth: the cell fill is blended toward black so the video shows
        which cell is pressed before and while its plus-neighborhood switches.
        """

        cell, ox, oy = self._layout()
        supersampled, draw = new_scaled_canvas(
            CANVAS_SIZE, CANVAS_SIZE, DOCTRINE_SUPERSAMPLE, (255, 255, 255)
        )
        factor = draw.factor
        font = load_font(max(11, int(cell * 0.30)) * factor)
        for r in range(self.size):
            for c in range(self.size):
                x0 = ox + c * cell
                y0 = oy + r * cell
                if cell_alpha is not None and (r, c) in cell_alpha:
                    a = cell_alpha[(r, c)]
                    if int(grid[r][c]) == 1:
                        # Was on, fading toward off.
                        fill = tuple(int(_ON_COLOR[i] * (1 - a) + _OFF_COLOR[i] * a) for i in range(3))
                    else:
                        # Was off, fading toward on.
                        fill = tuple(int(_OFF_COLOR[i] * (1 - a) + _ON_COLOR[i] * a) for i in range(3))
                else:
                    fill = _ON_COLOR if int(grid[r][c]) == 1 else _OFF_COLOR
                if press_cue is not None and press_cue[0] == (r, c):
                    depth = max(0.0, min(1.0, float(press_cue[1]))) * _PRESS_SHADE
                    fill = tuple(int(round(value * (1.0 - depth))) for value in fill)
                draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=fill, outline=(176, 182, 194), width=2)
        for c in range(self.size):
            x = ox + c * cell + cell / 2
            draw.text((x, oy - 22), str(c + 1), fill=(60, 70, 90), font=font, anchor="mm")
        for r in range(self.size):
            y = oy + r * cell + cell / 2
            draw.text((ox - 22, y), str(r + 1), fill=(60, 70, 90), font=font, anchor="mm")
        if analysis_presses:
            press_font = load_font(max(14, int(cell * 0.24)) * factor)
            press_colour = (28, 151, 177)
            for row, col in sorted(analysis_presses):
                cx = ox + col * cell + cell / 2
                cy = oy + row * cell + cell / 2
                radius = cell * 0.29
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    outline=press_colour,
                    width=5,
                )
                draw.text((cx, cy), "P", fill=press_colour, font=press_font, anchor="mm")
        if parity_badges is not None:
            badge_font = load_font(max(12, int(cell * 0.19)) * factor)
            for row in range(self.size):
                for col in range(self.size):
                    parity = int(parity_badges[(row, col)])
                    cx = ox + col * cell + cell * 0.79
                    cy = oy + row * cell + cell * 0.21
                    radius = max(10, int(cell * 0.14))
                    body = (35, 117, 137) if parity else (224, 229, 235)
                    edge = (21, 91, 108) if parity else (111, 121, 134)
                    text_colour = (255, 255, 255) if parity else (63, 72, 84)
                    draw.ellipse(
                        (cx - radius, cy - radius, cx + radius, cy + radius),
                        fill=body,
                        outline=edge,
                        width=2,
                    )
                    draw.text((cx, cy), str(parity), fill=text_colour, font=badge_font, anchor="mm")
        return finish_scaled_canvas(supersampled, (CANVAS_SIZE, CANVAS_SIZE))

    def _write_video(
        self,
        puzzle_grid: Sequence[Sequence[int]],
        solution_grid: Sequence[Sequence[int]],
        press_order: Sequence[Tuple[int, int]],
        video_path: Path,
    ) -> int:
        """Write the press-cue video.

        Per press: the pressed cell alone darkens (2 frames in, 1 hold), the
        plus-neighborhood dissolves to its toggled colors while the shade is
        held (4 frames), the shade lifts (2 frames), and the committed board
        holds (3 frames).  That is 12 frames per press, so every press count
        keeps the 4N+1 total.
        """

        frames: List[np.ndarray] = []

        def emit(image: Image.Image, repeat: int = 1) -> None:
            array = np.array(image, dtype=np.uint8)
            for _ in range(repeat):
                frames.append(array.copy())

        current = [[int(value) for value in row] for row in puzzle_grid]
        emit(self._render_board(current), 5)
        for press_row, press_col in press_order:
            press = (press_row, press_col)
            toggled = self._toggle_cells(press_row, press_col)
            for depth in (0.5, 1.0, 1.0):
                emit(self._render_board(current, press_cue=(press, depth)))
            for k in range(1, 5):
                emit(
                    self._render_board(
                        current,
                        cell_alpha={cell: k / 4.0 for cell in toggled},
                        press_cue=(press, 1.0),
                    )
                )
            self._toggle(current, press_row, press_col)
            for depth in (2.0 / 3.0, 1.0 / 3.0):
                emit(self._render_board(current, press_cue=(press, depth)))
            emit(self._render_board(current), 3)
        if current != [list(row) for row in solution_grid]:
            raise RuntimeError("Press sequence does not reach the all-OFF state.")
        emit(self._render_board(solution_grid), 8)
        if len(frames) % 4 != 1:
            raise RuntimeError("Lights Out frame count must satisfy 4N+1.")
        if not encode_rgb_frames_to_webm(
            frames, video_path, fps=self.video_fps,
            extra_args=["-g", "1", "-bf", "0"],
        ):
            return 0
        return len(frames)

    def _prompts(self) -> Dict[str, str]:
        return {
            "ti2v_prompt": build_sectioned_prompt(
                _INPUT_DESC.replace("5 by 5", f"{self.size} by {self.size}").replace(
                    "1-5", f"1-{self.size}"
                ),
                _TI2V_OUTPUT,
                mode="ti2v",
                rule_description=_RULE_SUMMARY
                if self.press_count_range[1] == 3
                else _RULE_SUMMARY.replace(
                    "one and three", f"one and {self.press_count_range[1]}"
                ).replace("to three", f"to {self.press_count_range[1]}"),
                continuity_paragraph=_TI2V_CONTINUITY,
            ),
        }

    def create_puzzle(self) -> BoardChoiceRecord:
        sample_id = f"{TASK_NAME}-{self._counter:05d}"
        self._counter += 1
        puzzle_grid, solution_grid, press_cells, press_order = self._build_board()
        puzzle_image = self._render_board(puzzle_grid)
        solution_image = self._render_board(solution_grid)
        parity = {
            (row, col): sum(
                (row, col) in self._toggle_cells(press_row, press_col)
                for press_row, press_col in press_cells
            ) % 2
            for row in range(self.size)
            for col in range(self.size)
        }
        if any(parity[(row, col)] != int(puzzle_grid[row][col]) for row in range(self.size) for col in range(self.size)):
            raise RuntimeError("The unique press-set parity must equal the visible ON mask.")

        state_trace = [[row[:] for row in puzzle_grid]]
        current = [row[:] for row in puzzle_grid]
        commit_images = [puzzle_image]
        for press_row, press_col in press_order:
            self._toggle(current, press_row, press_col)
            state_trace.append([row[:] for row in current])
            commit_images.append(self._render_board(current))
        if current != solution_grid:
            raise RuntimeError("Authored press sequence does not reach the all-OFF state.")

        visible_press_cells = tuple((row + 1, col + 1) for row, col in press_cells)
        visible_press_order = tuple((row + 1, col + 1) for row, col in press_order)
        answer_payload = format_xy_position_list_payload(press_cells)
        answer_text = format_xy_position_list_answer(press_cells)
        payload = {
            "semantic_state": {
                "initial_grid": [row[:] for row in puzzle_grid],
                "board_size": self.size,
                "maximum_press_count": self.press_count_range[1],
            },
            "state_grid": [row[:] for row in solution_grid],
            "initial_grid": [row[:] for row in puzzle_grid],
            "state_trace": state_trace,
            "canonical_action": "; ".join(
                f"press {row} {col}" for row, col in visible_press_order
            ),
            "press_cells": [list(cell) for cell in visible_press_cells],
            "press_sequence": [list(cell) for cell in visible_press_order],
            "process_state_trace": state_trace,
            "board_size": self.size,
            "maximum_press_count": self.press_count_range[1],
        }
        return write_board_choice_record(
            generator=self,
            sample_id=sample_id,
            task_name=TASK_NAME,
            task_family="board_game",
            question_description="bounded_press_set",
            puzzle_image=puzzle_image,
            solution_image=solution_image,
            option_tiles=(),
            correct_option_index=None,
            prompts=self._prompts(),
            structured_payload=payload,
            question_text=f"Return the unique set of at most {self.press_count_range[1]} presses that switches every cell OFF.",
            video_states=None,
            video_verifier_states=commit_images,
            video_writer=lambda path: self._write_video(
                puzzle_grid, solution_grid, press_order, path
            ),
            single_input=True,
            direct_answer_text=answer_text,
            direct_answer_value=answer_payload,
            qa_type="Search",
            qa_level="Medium",
        )


__all__ = ["LightsOutSinglePressGenerator", "lights_layout"]

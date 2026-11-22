"""Candy Crush special-candy detonation prediction.

Mechanic: a 7x7 board contains regular colored candies plus exactly ONE special
candy marked with a white-and-gold double rim:
- Horizontal/vertical white lines: clears the marked row or column.
- Gold square frame: clears the in-bounds 3x3 neighbourhood.
- Silver star: clears every token matching the special token's own visible color.

Predict the after-clear board.

Mechanic load-bearing: the kind-of-special is identifiable from its inner glyph.
A baseline that just clears the tapped cell yields a different answer in every
case (special clears >=3 cells), so the visual cue is required.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import CANVAS_SIZE, DEFAULT_FPS, BoardLayout
from data.dynamics_game.core.scene import finish_scene, new_scene, scene_font
from data.common.pipeline.discrete_dynamics import (
    BG, GRID_LINE, LABEL_COLOR,
    encode_video, make_record_dataclass,
    shuffle_with_correct_semantics, write_artifacts,
)
from data.common.multiple_choice import option_label
from data.dynamics_game.core.choice_composite import (
    attach_choice_input_fields,
    build_composite_mcq_prompts,
    finalize_choice_artifacts,
)
from data.dynamics_game.core.coordinate_format import with_grid_xy_convention

_INPUT_DESC = with_grid_xy_convention(
    "The input image shows a 7 by 7 board with rows and columns labeled 1 through 7. Each cell holds one round token "
    "colored red, blue, green, orange, or purple. Exactly one token has a white-and-gold double rim; that rim marks "
    "the token to activate. Its inner glyph defines the complete clearing behavior—horizontal white lines clear every "
    "cell in the marked token's row; vertical white lines clear every cell in its column; a gold square clears every "
    "in-bounds cell in the 3 by 3 neighborhood centered on it; and a silver star clears every token whose visible "
    "fill color matches the marked token's fill color. The marked cell itself is included in every clearing rule. "
    "A neighborhood is clipped at the board boundary."
)
_RULE_SUMMARY = (
    "Apply exactly the clearing rule encoded by the marked token's inner glyph. Every cleared location becomes the "
    "same pale empty board well. Do not move any surviving token, do not apply gravity, and do not create, recolor, "
    "or remove any other token."
)
_TI2V_OUTPUT = (
    "Hold the input, then add one gold rounded frame inside the marked token's cell and inside the cell of "
    "every token its glyph rule affects. Next shrink exactly those framed tokens together into their own "
    "pale empty wells, let the gold frames vanish with the final shrink, and hold the cleared board."
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render the exact final board after applying the marked token's glyph rule once, with only the affected cells "
    "changed to pale empty wells and every other visible element preserved."
)

CANDY_PALETTE = [
    ("R", (228, 80, 80), (160, 30, 30)),
    ("B", (60, 120, 220), (30, 60, 160)),
    ("G", (80, 175, 105), (40, 110, 60)),
    ("O", (245, 165, 65), (170, 100, 30)),
    ("P", (165, 90, 200), (110, 50, 150)),
]
EMPTY_FILL = (241, 245, 251)
CANDY_CODES = {"": 0, None: 0, "R": 1, "B": 2, "G": 3, "O": 4, "P": 5}

_BD2Record = make_record_dataclass("candy_crush_special_predict", "dynamics_game")


def _encode_grid(grid):
    return [[CANDY_CODES[cell] for cell in row] for row in grid]


class CandyCrushSpecialPredictGenerator(AbstractPuzzleGenerator):
    def __init__(self, output_dir, *, seed=None, video_fps=DEFAULT_FPS, video=False):
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.rows = 7
        self.cols = 7
        self.video_fps = video_fps
        self.video = video
        self._counter = 0
        self.layout = BoardLayout.for_grid(self.rows, self.cols, board_max=CANVAS_SIZE - 64, reserve_top=28)

    def _random_grid(self):
        return [[self.rng.choice(CANDY_PALETTE)[0] for _ in range(self.cols)] for _ in range(self.rows)]

    def _build_state(self):
        grid = self._random_grid()
        kind = self.rng.choice(["row", "col", "wrap", "bomb"])
        sr = self.rng.randrange(self.rows)
        sc = self.rng.randrange(self.cols)
        cleared = set()
        if kind == "row":
            cleared = {(sr, c) for c in range(self.cols)}
        elif kind == "col":
            cleared = {(r, sc) for r in range(self.rows)}
        elif kind == "wrap":
            cleared = {
                (r, c)
                for r in range(max(0, sr - 1), min(self.rows, sr + 2))
                for c in range(max(0, sc - 1), min(self.cols, sc + 2))
            }
        else:  # bomb — clear every token matching the special token's own visible color.
            target_color = grid[sr][sc]
            cleared = {(r, c) for r in range(self.rows) for c in range(self.cols) if grid[r][c] == target_color}
            cleared.add((sr, sc))
        return grid, kind, (sr, sc), cleared

    def _apply(self, grid, cleared):
        out = [list(row) for row in grid]
        for r, c in cleared:
            out[r][c] = None
        return out

    def _build_distractors(self, grid, kind, special, correct_cleared):
        sr, sc = special
        # Distractor 1: only the special cell cleared.
        d1 = self._apply(grid, {special})
        # Distractor 2: wrong dimension (row vs col swap), or wrong neighborhood (3x3 vs 5x5).
        if kind == "row":
            alt = {(r, sc) for r in range(self.rows)}
        elif kind == "col":
            alt = {(sr, c) for c in range(self.cols)}
        elif kind == "wrap":
            alt = {(r, c) for r in range(max(0, sr - 2), min(self.rows, sr + 3)) for c in range(max(0, sc - 2), min(self.cols, sc + 3))}
        else:  # bomb -> clear cells with second-most-common color
            counter = Counter(grid[r][c] for r in range(self.rows) for c in range(self.cols))
            target_color = grid[sr][sc]
            ranked = [color for color, _ in counter.most_common() if color != target_color]
            if not ranked:
                raise RuntimeError("Bomb distractor requires a visible non-target color.")
            target = ranked[0]
            alt = {(r, c) for r in range(self.rows) for c in range(self.cols) if grid[r][c] == target}
            alt.add(special)
        d2 = self._apply(grid, alt)
        # Distractor 3: cleared correctly but with one extra random cell.
        d3_set = set(correct_cleared)
        candidates = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in d3_set]
        if candidates:
            d3_set.add(self.rng.choice(candidates))
        d3 = self._apply(grid, d3_set)
        return [d1, d2, d3]

    def _draw_candy(self, draw, code, center, cell, *, scale=1.0):
        cx, cy = center
        size = cell * 0.36 * float(scale)
        for k, fill, outline in CANDY_PALETTE:
            if k == code:
                draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=fill, outline=outline, width=2)
                return

    def _draw_special_overlay(self, draw, kind, center, cell):
        cx, cy = center
        size = cell * 0.42
        # Glowing white rim.
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], outline=(255, 255, 255), width=4)
        draw.ellipse([cx - size - 3, cy - size - 3, cx + size + 3, cy + size + 3], outline=(255, 230, 100), width=2)
        # Inner glyph.
        if kind == "row":
            for k in range(-2, 3, 2):
                draw.line([(cx - size * 0.7, cy + k * 4), (cx + size * 0.7, cy + k * 4)],
                          fill=(255, 255, 255), width=2)
        elif kind == "col":
            for k in range(-2, 3, 2):
                draw.line([(cx + k * 4, cy - size * 0.7), (cx + k * 4, cy + size * 0.7)],
                          fill=(255, 255, 255), width=2)
        elif kind == "wrap":
            s = size * 0.55
            draw.rectangle([cx - s, cy - s, cx + s, cy + s], outline=(245, 200, 60), width=4)
        else:  # bomb — silver star
            import math
            outer = size * 0.6
            inner = size * 0.28
            pts = []
            for i in range(10):
                ang = -math.pi / 2 + i * math.pi / 5
                r = outer if i % 2 == 0 else inner
                pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
            draw.polygon(pts, fill=(220, 220, 230), outline=(150, 150, 165))


    def _render(self, grid, *, special=None, kind=None, affected=(), clear_progress=0.0, mark_affected=False, moving_centres=None):
        image, draw = new_scene(BG)
        cell = self.layout.cell
        font = scene_font(max(11, int(cell * 0.32)))
        for r in range(self.rows):
            for c in range(self.cols):
                x0, y0, x1, y1 = self.layout.cell_box(r, c)
                draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=GRID_LINE, width=1)
        label_y = self.layout.origin_y - 14
        for c in range(self.cols):
            x = self.layout.origin_x + c * cell + cell / 2
            draw.text((x, label_y), str(c + 1), fill=LABEL_COLOR, font=font, anchor="mm")
        label_x = self.layout.origin_x - 12
        for r in range(self.rows):
            y = self.layout.origin_y + r * cell + cell / 2
            draw.text((label_x, y), str(r + 1), fill=LABEL_COLOR, font=font, anchor="rm")
        for r in range(self.rows):
            for c in range(self.cols):
                v = grid[r][c]
                if v is None:
                    x0, y0, x1, y1 = self.layout.cell_box(r, c)
                    draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], fill=EMPTY_FILL)
        for r in range(self.rows):
            for c in range(self.cols):
                v = grid[r][c]
                if v is not None:
                    scale = max(0.04, 1.0 - float(clear_progress)) if (r, c) in set(affected) else 1.0
                    centre = (moving_centres or {}).get((r,c), self.layout.cell_center(r,c))
                    self._draw_candy(draw, v, centre, cell, scale=scale)
                if mark_affected and (r, c) in set(affected):
                    x0, y0, x1, y1 = self.layout.cell_box(r, c)
                    draw.rounded_rectangle([x0 + 4, y0 + 4, x1 - 4, y1 - 4], radius=8, outline=(245, 183, 55), width=4)
        if special and kind:
            cx, cy = (moving_centres or {}).get(tuple(special), self.layout.cell_center(*special))
            self._draw_special_overlay(draw, kind, (cx, cy), cell)
        return finish_scene(image)

    def _write_video(self, video_path: Path, *, grid, kind, special, cleared, final) -> int:
        frames: List[np.ndarray] = []

        def push(image: Image.Image, count: int = 1) -> None:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
            frames.extend(frame.copy() for _ in range(count))

        initial = self._render(grid, special=special, kind=kind)
        push(initial, 8)
        for index in range(8):
            push(self._render(grid, special=special, kind=kind, affected=cleared, mark_affected=index >= 3))
        for index in range(1, 9):
            push(self._render(grid, special=special, kind=kind, affected=cleared, clear_progress=index / 8.0, mark_affected=True))
        push(self._render(final), 17)
        assert len(frames) == 41
        return encode_video(frames, video_path, fps=self.video_fps)

    def create_puzzle(self):
        sample_id = f"candy_crush_special_predict-{self._counter:05d}"
        self._counter += 1
        grid, kind, special, cleared = self._build_state()
        # Mechanic load-bearing: special clears >= 3 cells, so plain "clear tapped cell" yields different answer.
        assert len(cleared) >= 3, "special must clear >=3 cells"
        final = self._apply(grid, cleared)
        puzzle_image = self._render(grid, special=special, kind=kind)
        solution_image = self._render(final)
        distractors = self._build_distractors(grid, kind, special, cleared)
        keys = {
            tuple(tuple(0 if cell is None else cell for cell in row) for row in candidate)
            for candidate in (final, *distractors)
        }
        if len(keys) != 4:
            raise RuntimeError("Candy candidates must contain four distinct terminal boards.")
        distractor_tiles = [self._render(d) for d in distractors]
        option_tiles, candidate_state_grids, correct_index = shuffle_with_correct_semantics(
            solution_image,
            _encode_grid(final),
            distractor_tiles,
            [_encode_grid(distractor) for distractor in distractors],
            self.rng,
        )
        prompts = build_composite_mcq_prompts(
            input_desc=_INPUT_DESC,
            rule_summary=_RULE_SUMMARY,
            ti2v_output=_TI2V_OUTPUT,
            solution_image_output=_SOLUTION_IMAGE_OUTPUT,
            continuity_timing="continuous",
            continuity_moving="the marked token and every cleared token",
            continuity_static=(
                "every surviving token, the board wells, the row and column labels, the background, and the camera"
            ),
            continuity_hold="the final cleared board",
        )
        initial_grid = _encode_grid(grid)
        state_grid = _encode_grid(final)
        payload = {
            "mechanic": "special_token_area_clear_no_gravity",
            "board": [list(row) for row in grid],
            "kind": kind,
            "special_cell": list(special),
            "cleared_cells": sorted(list(cleared)),
            "initial_grid": initial_grid,
            "state_grid": state_grid,
            "state_trace": [initial_grid, state_grid],
            "process_state_trace": [initial_grid, initial_grid, state_grid],
            "candidate_state_grids": candidate_state_grids,
            "correct_option_index": correct_index,
            "correct_option": option_label(correct_index),
        }
        artifacts = write_artifacts(
            output_dir=self.output_dir, sample_id=sample_id,
            puzzle_image=puzzle_image, solution_image=solution_image,
            option_tiles=option_tiles, correct_index=correct_index, payload=payload,
            write_video=(lambda video_path: self._write_video(video_path, grid=grid, kind=kind, special=special, cleared=cleared, final=final)) if self.video else None,
        )
        finalize_choice_artifacts(
            output_dir=self.output_dir,
            artifacts=artifacts,
            option_tiles=option_tiles,
            correct_index=correct_index,
            payload=payload,
            video_fps=self.video_fps,
        )
        answer_label = option_label(correct_index)
        reasoning_images = [self.relativize_path(artifacts["puzzle_path"])]
        record = _BD2Record(
            id=sample_id,
            image=self.relativize_path(artifacts["puzzle_path"]),
            reasoning_image=self.relativize_path(artifacts["puzzle_path"]),
            reasoning_images=reasoning_images,
            solution_image_path=self.relativize_path(artifacts["solution_path"]),
            solution_video_path=self.relativize_path(artifacts["video_path"]) if artifacts["frame_count"] > 0 else None,
            video_fps=self.video_fps if artifacts["frame_count"] > 0 else None,
            video_num_frames=artifacts["frame_count"] if artifacts["frame_count"] > 0 else None,
            ti2v_prompt=prompts["ti2v_prompt"],

            answer=answer_label, correct_option=answer_label,


            question="Which candidate after-board correctly applies the marked token's clearing rule?",
            structured_answer=payload, answer_payload=payload,
        )

        return attach_choice_input_fields(
            record,
            artifacts=artifacts,
            relativize=self.relativize_path,
        )


__all__ = ["CandyCrushSpecialPredictGenerator"]

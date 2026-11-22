from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    cell_to_xy,
    format_xy_path_payload,
)
from data.common.prompts import build_sectioned_prompt
from data.common.base import AbstractPuzzleGenerator
from data.common.video_encoding import (
    MOTION_VIDEO_FRAMES,
    encode_rgb_frames_to_webm,
    fit_frames_to_video_budget,
)

# Frame budget and the solution length it admits. The video holds the scrambled
# board for 8 frames, spends 4 frames sliding each tile, and holds the solved
# board for 8, so a solution of M moves needs 4*M + 16 frames padded up to the
# next 4N+1. MOTION_VIDEO_FRAMES = 65 therefore admits M <= 12 (65 frames =
# 4.06 s at 16 fps, inside the  cap of 157 frames / 10 s), and a
# shorter solution still yields a shorter video because the shared encoder only
# pads up to the next 4N+1.
VIDEO_FRAME_BUDGET = MOTION_VIDEO_FRAMES
MAX_SOLUTION_MOVES = (VIDEO_FRAME_BUDGET - 1 - 16) // 4

from data.common.pipeline.sliding_tiles import (
    _apply_blank_move,
    _bfs_unique_solution,
    _goal_state,
    _load_font,
    _valid_blank_moves,
)
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)

class FifteenPuzzleBoardStateGenerator(AbstractPuzzleGenerator[Dict[str, Any]]):
    TASK_NAME = "sliding_tile_15"
    TASK_FAMILY = "rearrangement_game"

    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        size: int = 4,
        video: bool = False,
        video_fps: int = 16,
        min_solution_moves: int = 4,
        max_solution_moves: int = 8,
    ) -> None:
        super().__init__(output_dir)
        self.seed = seed
        self.rng = random.Random(seed)
        self.size = max(2, int(size))
        self.video = video
        self.video_fps = video_fps
        self.min_solution_moves = int(min_solution_moves)
        self.max_solution_moves = int(max_solution_moves)
        if self.min_solution_moves < 1 or self.max_solution_moves < self.min_solution_moves:
            raise ValueError(
                "Require 1 <= min_solution_moves <= max_solution_moves for sliding_tile_15."
            )
        if self.max_solution_moves > MAX_SOLUTION_MOVES:
            raise ValueError(
                f"max_solution_moves must be at most {MAX_SOLUTION_MOVES} for sliding_tile_15; the reference "
                f"video renders 8 hold + 4 per move + 8 hold frames, so a longer solution cannot fit the "
                f"{VIDEO_FRAME_BUDGET}-frame budget."
            )
        if self.size == 2 and self.max_solution_moves > 6:
            raise ValueError("A 2x2 sliding puzzle cannot use max_solution_moves > 6.")
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _scramble(self, steps: int) -> Tuple[List[List[int]], List[str]]:
        state = _goal_state(self.size)
        moves: List[str] = []
        last_move: Optional[str] = None
        for _ in range(steps):
            valid = _valid_blank_moves(state)
            if last_move is not None and len(valid) > 1:
                reverse = {"up": "down", "down": "up", "left": "right", "right": "left"}[last_move]
                if reverse in valid:
                    valid.remove(reverse)
            move = self.rng.choice(valid)
            state = _apply_blank_move(state, move)
            moves.append(move)
            last_move = move
        return state, moves

    def _generate_instance(self) -> Tuple[List[List[int]], List[str]]:
        goal = _goal_state(self.size)
        target_depth = self.rng.randint(self.min_solution_moves, self.max_solution_moves)
        for _ in range(512):
            puzzle, scramble = self._scramble(target_depth)
            solution = _bfs_unique_solution(puzzle, goal)
            if not solution or len(solution) != target_depth:
                continue
            reverse = [
                {"up": "down", "down": "up", "left": "right", "right": "left"}[move]
                for move in reversed(scramble)
            ]
            if solution == reverse:
                return puzzle, solution
        raise RuntimeError(
            "Failed to generate a sliding puzzle with a unique shortest solution "
            f"of length {target_depth}."
        )

    def _layout(self, size: int) -> Tuple[int, int, int]:
        cell = (512 - 96) // size
        cell -= cell % 2
        ox = (512 - size * cell) // 2
        oy = (512 - size * cell) // 2
        return cell, ox - (ox % 2), oy - (oy % 2)

    def _render_board(
        self,
        state: Sequence[Sequence[int]],
        *,
        moving_tile: Optional[Tuple[int, float, float]] = None,
        suppress_value: Optional[int] = None,
    ) -> Image.Image:
        size = len(state)
        cell, ox, oy = self._layout(size)
        image, draw = new_scaled_canvas(512, 512, DOCTRINE_SUPERSAMPLE, (247, 249, 251))
        font = _load_font(max(18, int(cell * 0.27)) * DOCTRINE_SUPERSAMPLE)
        radius = max(8, cell // 9)
        board_pad = 7
        draw.rounded_rectangle(
            [
                ox - board_pad,
                oy - board_pad,
                ox + size * cell + board_pad,
                oy + size * cell + board_pad,
            ],
            radius=radius + 5,
            fill=(224, 229, 234),
            outline=(174, 182, 190),
            width=2,
        )

        def draw_tile(value: int, x0: float, y0: float) -> None:
            draw.rounded_rectangle(
                [x0 + 5, y0 + 5, x0 + cell - 5, y0 + cell - 5],
                radius=radius,
                fill=(238, 244, 249),
                outline=(105, 126, 146),
                width=2,
            )
            draw.text(
                (x0 + cell / 2, y0 + cell / 2 - 1),
                str(value), fill=(42, 54, 66), font=font, anchor="mm",
            )

        for r in range(size):
            for c in range(size):
                v = int(state[r][c])
                x0 = ox + c * cell
                y0 = oy + r * cell
                if v == 0:
                    draw.rounded_rectangle(
                        [x0 + 5, y0 + 5, x0 + cell - 5, y0 + cell - 5],
                        radius=radius, fill=(249, 251, 252),
                        outline=(202, 209, 216), width=1,
                    )
                    continue
                if suppress_value is not None and v == suppress_value:
                    # Empty cell appearance for the moving tile.
                    draw.rounded_rectangle(
                        [x0 + 5, y0 + 5, x0 + cell - 5, y0 + cell - 5],
                        radius=radius, fill=(249, 251, 252),
                        outline=(202, 209, 216), width=1,
                    )
                    continue
                draw_tile(v, x0, y0)

        if moving_tile is not None:
            v, mx, my = moving_tile
            draw_tile(v, mx, my)
        return finish_scaled_canvas(image, (512, 512))

    def _write_video(self, states: Sequence[Sequence[Sequence[int]]], video_path: Path) -> int:
        if not states:
            return 0
        size = len(states[0])
        cell, ox, oy = self._layout(size)
        frames: List[np.ndarray] = []
        # Initial hold.
        first = self._render_board(states[0])
        for _ in range(8):
            frames.append(np.array(first, dtype=np.uint8))
        # For each transition, sub-pixel slide the tile that moved into the blank.
        for i in range(len(states) - 1):
            a = states[i]
            b = states[i + 1]
            # Find the cell whose value changed: the blank moves; the other changed cell's value is the moving tile.
            moved_value: Optional[int] = None
            from_cell: Optional[Tuple[int, int]] = None
            to_cell: Optional[Tuple[int, int]] = None
            for r in range(size):
                for c in range(size):
                    if int(a[r][c]) != int(b[r][c]):
                        if int(a[r][c]) == 0:
                            to_cell = (r, c)
                            moved_value = int(b[r][c])
                        else:
                            from_cell = (r, c)
            if from_cell is None or to_cell is None or moved_value is None:
                # No change; just hold.
                frames.append(np.array(self._render_board(b), dtype=np.uint8))
                continue
            sx = ox + from_cell[1] * cell
            sy = oy + from_cell[0] * cell
            ex = ox + to_cell[1] * cell
            ey = oy + to_cell[0] * cell
            # Four in-between samples keep every tile slide continuous while
            # allowing the declared maximum solution to fit the 4N+1 video
            # contract: 8 initial + 4 per move + 8 final, then the shared
            # encoder pads one final hold frame.  At MAX_SOLUTION_MOVES that
            # is 8 + 48 + 8 = 64 frames padded to 65 (4.06 s at 16 fps).
            substeps = 4
            # During the slide, hide the moving tile from its origin cell.
            display_state = [list(row) for row in a]
            display_state[from_cell[0]][from_cell[1]] = 0
            for k in range(1, substeps + 1):
                t = k / substeps
                mx = sx + (ex - sx) * t
                my = sy + (ey - sy) * t
                f = self._render_board(display_state, moving_tile=(moved_value, mx, my))
                frames.append(np.array(f, dtype=np.uint8))
        # Final hold.
        last = self._render_board(states[-1])
        for _ in range(8):
            frames.append(np.array(last, dtype=np.uint8))
        frames = fit_frames_to_video_budget(
            frames,
            max_frames=VIDEO_FRAME_BUDGET,
            final_hold_frames=8,
        )
        if not encode_rgb_frames_to_webm(
            frames, video_path, fps=self.video_fps,
            extra_args=["-g", "1", "-bf", "0"],
        ):
            return 0
        return len(frames)

    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f'{self.TASK_NAME}-{self._counter:05d}'
        self._counter += 1
        puzzle, solution_moves = self._generate_instance()
        state_trace = [puzzle]
        current = [row[:] for row in puzzle]
        for move in solution_moves:
            current = _apply_blank_move(current, move)
            state_trace.append(current)
        final_grid = [row[:] for row in current]
        blank_path = [next(((row, col) for row in range(self.size) for col in range(self.size) if int(state[row][col]) == 0)) for state in state_trace]
        prompts = self._prompts()
        puzzle_dir = self.output_dir / 'puzzles'
        solution_dir = self.output_dir / 'solutions'
        video_dir = self.output_dir / 'videos'
        for directory in (puzzle_dir, solution_dir, video_dir):
            directory.mkdir(parents=True, exist_ok=True)
        puzzle_path = puzzle_dir / f'{sample_id}.png'
        solution_path = solution_dir / f'{sample_id}.png'
        video_path = video_dir / f'{sample_id}.webm'
        puzzle_image = self._render_board(puzzle)
        solution_image = self._render_board(final_grid)
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)
        frame_count = self._write_video(state_trace, video_path) if self.video else 0
        payload = {'state_grid': final_grid, 'state_trace': state_trace, 'blank_path': [list(cell_to_xy(cell)) for cell in blank_path], 'board_size': len(final_grid), 'optimal_step_count': len(solution_moves), 'shortest_plan_count': 1}
        return {'id': sample_id, 'task_name': self.TASK_NAME, 'task_family': self.TASK_FAMILY, 'initial_grid': [list(row) for row in puzzle], 'board_size': int(self.size), 'image': self.relativize_path(puzzle_path), 'reasoning_image': self.relativize_path(puzzle_path), 'solution_image_path': self.relativize_path(solution_path), 'solution_video_path': self.relativize_path(video_path) if frame_count > 0 else None, 'video_fps': self.video_fps if frame_count > 0 else None, 'video_num_frames': frame_count if frame_count > 0 else None, 'ti2v_prompt': prompts['ti2v_prompt'],  'answer': self._answer_value(blank_path), 'question': "Return the blank's complete shortest grid-position path that solves the puzzle.", 'question_description': self.TASK_NAME, 'structured_answer': payload, 'answer_payload': payload, 'answer_grid': final_grid}

    def _prompts(self) -> Dict[str, str]:
        input_desc = f'The input image shows a clean {self.size}x{self.size} sliding-tile board on a pale square canvas. Every occupied slot contains one flat, equal-size tile with only its required number; one slot is blank. There are no decorative icons, shadows, gradients, headings, or repeated labels. The task is to restore the ascending row-major order. {GRID_XY_CONVENTION_SENTENCE}'
        rules_desc = f'Move the blank only up, right, down, or left by swapping it with one adjacent tile. The goal is the solved board with tiles 1 through {self.size * self.size - 1} in order and the blank in the bottom-right corner.'
        ti2v_output = 'Generate a short static-camera video that begins on the unchanged input image and replays the unique shortest blank-move sequence. For every move, slide exactly one adjacent tile directly into the blank slot along a straight horizontal or vertical path. Keep its number, width, height, corner radius, colour, and orientation fixed; do not pop, pulse, grow, shrink, rotate, bounce, overshoot, or teleport it. Land exactly in the next static board state without a snap. Keep all other tiles and the camera fixed, and end by holding the solved board.'
        solution_image_output = 'Render the solved still image with the tiles in ascending order and the blank in the bottom-right corner.'
        ti2v = build_sectioned_prompt(input_desc, f'{rules_desc} {ti2v_output}', mode='ti2v')
        solution_image = build_sectioned_prompt(input_desc, f'{rules_desc} {solution_image_output}', mode='solution_image')
        return {'ti2v_prompt': ti2v, 'solution_image_prompt': solution_image}

    def _answer_value(self, blank_path: Sequence[Sequence[int]]) -> str:
        return format_xy_path_payload(blank_path)

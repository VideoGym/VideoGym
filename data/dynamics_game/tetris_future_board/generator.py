"""Rigid falling-block landing, locking, and full-row clearing."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.prompts import build_sectioned_prompt
from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import DEFAULT_FPS
from data.dynamics_game.core.scene import finish_scene, new_scene, scene_font
from data.common.pipeline.discrete_dynamics import encode_video
from data.dynamics_game.core.coordinate_format import with_grid_xy_convention
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="continuous",
    moving_nouns="the falling piece and the cleared rows",
    static_nouns="everything the rules leave unchanged, including the board, labels, background, and camera",
    hold_noun="the final board state",
)


_INPUT_DESC = with_grid_xy_convention(
    "The single problem image shows a 10-row by 10-column board. Pale rounded wells are empty, dark-blue blocks are "
    "fixed from earlier completed drops and row clears, and four coral blocks form one connected active piece near the "
    "top. Row and column labels run from 1 to 10."
)
_RULE_SUMMARY = (
    "Move the coral piece straight downward as one rigid shape. It does not rotate and does not move horizontally. "
    "Stop at the lowest row where all four cells remain inside the board and none overlaps a fixed block; moving the "
    "piece one more row down would be illegal. Change its four coral cells into fixed dark-blue blocks. Then remove "
    "every row whose ten wells are all filled. After removing those rows simultaneously, shift every surviving row "
    "above them downward by the number of removed rows and insert the same number of empty rows at the top."
)
_TI2V_OUTPUT = (
    "Hold the input, then move the same rigid coral piece straight down to its lowest legal row without "
    "rotating or passing through fixed blocks. After it lands, change its four cells to dark blue, then "
    "shrink every complete row simultaneously into pale wells. Only after the clear, move all surviving rows "
    "above it down by the required amount while preserving each row pattern and adding pale rows at the top, "
    "then hold. Keep the board, labels, wells, and spacing fixed; do not clear rows one at a time or leave "
    "black rows."
)


@dataclass
class TetrisFutureBoardRecord:
    id: str
    task_name: str = "tetris_future_board"
    task_family: str = "dynamics_game"
    image: str = ""
    reasoning_image: str = ""
    solution_image_path: str = ""
    solution_video_path: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None
    ti2v_prompt: str = ""
    solution_image_prompt: str = ""
    answer: str = ""
    question: str = ""
    question_description: str = "future_board"
    question_id: int = 0
    qa_type: str = "State Prediction"
    qa_level: str = "Medium"
    plot_level: str = "Medium"
    answer_grid: List[List[int]] = field(default_factory=list)
    structured_answer: Dict[str, Any] = field(default_factory=dict)
    answer_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "image": self.image,
            "reasoning_image": self.reasoning_image,
            "solution_image_path": self.solution_image_path,
            "solution_video_path": self.solution_video_path,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
            "ti2v_prompt": self.ti2v_prompt,
            "answer": self.answer,
            "question": self.question,
            "question_description": self.question_description,
            "question_id": self.question_id,
            "qa_type": self.qa_type,
            "qa_level": self.qa_level,
            "plot_level": self.plot_level,
            "answer_grid": self.answer_grid,
            "structured_answer": self.structured_answer,
            "answer_payload": self.answer_payload,
        }


class TetrisFutureBoardGenerator(AbstractPuzzleGenerator[TetrisFutureBoardRecord]):
    SHAPES = {
        "I": np.array([[1, 1, 1, 1]], dtype=int),
        "O": np.array([[1, 1], [1, 1]], dtype=int),
        "T": np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
        "L": np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    }

    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        rows: int = 10,
        cols: int = 10,
        setup_moves_range: Tuple[int, int] = (8, 14),
        video_fps: int = DEFAULT_FPS,
        video: bool = False,
    ) -> None:
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.seed = seed
        self.rows = rows
        self.cols = cols
        self.setup_moves_range = setup_moves_range
        if (self.rows, self.cols) != (10, 10):
            raise ValueError("tetris_future_board requires the contracted 10 by 10 board")
        if (
            len(self.setup_moves_range) != 2
            or self.setup_moves_range[0] < 1
            or self.setup_moves_range[0] > self.setup_moves_range[1]
        ):
            raise ValueError("setup_moves_range must be a positive inclusive (minimum, maximum) pair")
        self.video_fps = int(video_fps)
        self.video = bool(video)
        self._counter = 0
        self.cell = 39
        self.left = 65
        self.top = 78

    def record_to_dict(self, record: TetrisFutureBoardRecord | Dict[str, Any]) -> Dict[str, Any]:
        return record.to_dict() if hasattr(record, "to_dict") else dict(record)

    @staticmethod
    def _position_is_legal(
        settled: Sequence[Sequence[int]],
        shape: np.ndarray,
        row: int,
        col: int,
    ) -> bool:
        rows = len(settled)
        cols = len(settled[0]) if rows else 0
        if row < 0 or col < 0 or row + shape.shape[0] > rows or col + shape.shape[1] > cols:
            return False
        return all(
            int(settled[row + shape_row][col + shape_col]) == 0
            for shape_row, shape_col in np.argwhere(shape == 1)
        )

    def _find_landing_row(
        self,
        settled: Sequence[Sequence[int]],
        shape: np.ndarray,
        start_row: int,
        col: int,
    ) -> int:
        if not self._position_is_legal(settled, shape, start_row, col):
            raise ValueError("piece does not have a legal starting position")
        row = int(start_row)
        while self._position_is_legal(settled, shape, row + 1, col):
            row += 1
        return row

    @staticmethod
    def _lock_piece(
        settled: Sequence[Sequence[int]],
        shape: np.ndarray,
        row: int,
        col: int,
    ) -> List[List[int]]:
        locked = [[int(cell) for cell in line] for line in settled]
        for shape_row, shape_col in np.argwhere(shape == 1):
            board_row = row + int(shape_row)
            board_col = col + int(shape_col)
            if locked[board_row][board_col] != 0:
                raise ValueError("cannot lock a piece over an occupied cell")
            locked[board_row][board_col] = 1
        return locked

    def _initial_state(self) -> Tuple[List[List[int]], np.ndarray, int, int, str]:
        shape_name = self.rng.choice(tuple(self.SHAPES))
        shape = self.SHAPES[shape_name].copy()
        col = self.rng.randint(0, self.cols - shape.shape[1])
        shape_height = int(shape.shape[0])

        # Columns the active piece will fall through must stay clear to the floor
        # so its next drop completes the bottom row (guaranteeing at least one
        # clear). Every SHAPES entry has a full-width bottom row, so the whole
        # active footprint is an open shaft regardless of the fixed stack height.
        gap_columns = {
            col + local_col
            for local_col, value in enumerate(shape[-1])
            if int(value) == 1
        }
        fixed_columns = [board_col for board_col in range(self.cols) if board_col not in gap_columns]
        minimum_moves, maximum_moves = (int(value) for value in self.setup_moves_range)

        # Build a compact but genuinely irregular mid-game stack: uneven column
        # surfaces and a few covered holes/overhangs, consistent with legal prior
        # play. A larger move budget yields a fuller, taller board. One solid
        # spine column keeps every occupied row supported, so no wholly empty row
        # is ever trapped below fixed cells and at most one blank row is displayed.
        for _attempt in range(400):
            settled = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            peak_top = shape_height + self.rng.choice((0, 1))
            moves = self.rng.randint(minimum_moves, maximum_moves)
            move_span = max(1, maximum_moves - minimum_moves)
            fullness = (moves - minimum_moves) / move_span
            rough = max(1, int(round((self.rows - 1 - peak_top) * (0.55 - 0.35 * fullness))))

            spine = self.rng.choice(fixed_columns)
            tops: Dict[int, int] = {}
            for board_col in fixed_columns:
                if board_col == spine:
                    tops[board_col] = peak_top
                else:
                    tops[board_col] = min(self.rows - 1, peak_top + self.rng.randint(0, rough))
            for board_col, top in tops.items():
                for row in range(top, self.rows):
                    settled[row][board_col] = 1

            # Punch covered holes (an empty cell with a fixed block above it),
            # never on the spine and never on the bottom row that must clear.
            hole_candidates = [
                (row, board_col)
                for board_col in fixed_columns
                if board_col != spine
                for row in range(tops[board_col] + 1, self.rows - 1)
            ]
            self.rng.shuffle(hole_candidates)
            target_holes = min(len(hole_candidates), 2 + int(round(3 * fullness)))
            holes = 0
            punched_columns: set[int] = set()
            for row, board_col in hole_candidates:
                if holes >= target_holes:
                    break
                if board_col in punched_columns:
                    continue
                settled[row][board_col] = 0
                punched_columns.add(board_col)
                holes += 1

            heights = [self.rows - tops[board_col] for board_col in fixed_columns]
            initial = self._with_active(settled, shape, 0, col)
            blank_rows = sum(not any(int(cell) != 0 for cell in row) for row in initial)
            if blank_rows > 1:
                continue
            if holes < 1 or max(heights) - min(heights) < 2:
                continue
            if any(int(settled[self.rows - 1][board_col]) != 1 for board_col in fixed_columns):
                continue
            if any(int(settled[row][board_col]) != 0 for board_col in gap_columns for row in range(self.rows)):
                continue
            return settled, shape, 0, col, shape_name

        raise RuntimeError("failed to sample an irregular legal tetris setup board")

    def _with_active(self, settled: Sequence[Sequence[int]], shape: np.ndarray, row: int, col: int) -> List[List[int]]:
        grid = [[int(cell) for cell in line] for line in settled]
        for sr in range(shape.shape[0]):
            for sc in range(shape.shape[1]):
                if int(shape[sr, sc]) == 1:
                    grid[row + sr][col + sc] = 2
        return grid

    @staticmethod
    def _clear_full_rows(grid: Sequence[Sequence[int]]) -> Tuple[List[List[int]], List[int]]:
        full_rows = [index for index, row in enumerate(grid) if all(int(cell) == 1 for cell in row)]
        remaining = [[int(cell) for cell in row] for index, row in enumerate(grid) if index not in full_rows]
        empty = [[0 for _ in range(len(grid[0]))] for _ in full_rows]
        return empty + remaining, full_rows

    def _simulate_drop(
        self,
        settled: Sequence[Sequence[int]],
        shape: np.ndarray,
        start_row: int,
        col: int,
    ) -> Tuple[int, List[List[int]], List[List[int]], List[List[int]], List[int]]:
        landing_row = self._find_landing_row(settled, shape, start_row, col)
        landing = self._with_active(settled, shape, landing_row, col)
        locked = self._lock_piece(settled, shape, landing_row, col)
        final, cleared_rows = self._clear_full_rows(locked)
        return landing_row, landing, locked, final, cleared_rows

    def _cell_box(self, row: float, col: float) -> Tuple[float, float, float, float]:
        x0 = self.left + col * self.cell
        y0 = self.top + row * self.cell
        return x0, y0, x0 + self.cell, y0 + self.cell

    def _render_grid(
        self,
        grid: Sequence[Sequence[int]],
        *,
        active_overlay: Optional[Tuple[np.ndarray, int, float]] = None,
        clear_rows: Sequence[int] = (),
        clear_scale: float = 1.0,
        piece_lock_progress: float = 0.0,
        suppress_cells: Sequence[Tuple[int, int]] = (),
        moving_blocks: Sequence[Tuple[float, int, int]] = (),
    ) -> Image.Image:
        image, draw = new_scene((247, 249, 253))
        draw.rounded_rectangle((42, 50, 470, 492), radius=26, fill=(255, 255, 255), outline=(210, 220, 234), width=2)
        label_font = scene_font(13)
        for col in range(self.cols):
            x0, _, x1, _ = self._cell_box(0, col)
            draw.text(((x0 + x1) / 2, 66), str(col + 1), fill=(74, 87, 111), font=label_font, anchor="mm")
        for row in range(self.rows):
            _, y0, _, y1 = self._cell_box(row, 0)
            draw.text((54, (y0 + y1) / 2), str(row + 1), fill=(74, 87, 111), font=label_font, anchor="mm")

        suppress_set = set(suppress_cells)
        for row in range(self.rows):
            for col in range(self.cols):
                x0, y0, x1, y1 = self._cell_box(row, col)
                draw.rounded_rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), radius=7, fill=(236, 242, 249), outline=(207, 218, 232), width=1)
                value = int(grid[row][col])
                if value == 0 or (row, col) in suppress_set:
                    continue
                block_scale = clear_scale if row in clear_rows else 1.0
                inset = 5 + (1.0 - block_scale) * self.cell * 0.36
                if value == 2:
                    progress = max(0.0, min(1.0, piece_lock_progress))
                    fill = tuple(int((1.0 - progress) * a + progress * b) for a, b in zip((229, 91, 85), (54, 82, 145)))
                    edge = tuple(int((1.0 - progress) * a + progress * b) for a, b in zip((153, 45, 43), (29, 49, 101)))
                else:
                    fill = (54, 82, 145)
                    edge = (29, 49, 101)
                draw.rounded_rectangle((x0 + inset, y0 + inset, x1 - inset, y1 - inset), radius=max(1, int(7 * block_scale)), fill=fill, outline=edge, width=2)

        if active_overlay is not None:
            shape, base_col, sub_row = active_overlay
            for sr in range(shape.shape[0]):
                for sc in range(shape.shape[1]):
                    if int(shape[sr, sc]) != 1:
                        continue
                    x0, y0, x1, y1 = self._cell_box(sub_row + sr, base_col + sc)
                    draw.rounded_rectangle((x0 + 5, y0 + 5, x1 - 5, y1 - 5), radius=7, fill=(229, 91, 85), outline=(153, 45, 43), width=2)
        for sub_row, block_col, value in moving_blocks:
            x0, y0, x1, y1 = self._cell_box(sub_row, block_col)
            fill = (229, 91, 85) if int(value) == 2 else (54, 82, 145)
            edge = (153, 45, 43) if int(value) == 2 else (29, 49, 101)
            draw.rounded_rectangle(
                (x0 + 5, y0 + 5, x1 - 5, y1 - 5),
                radius=7,
                fill=fill,
                outline=edge,
                width=2,
            )
        return finish_scene(image)

    def _write_video(
        self,
        settled: Sequence[Sequence[int]],
        shape: np.ndarray,
        col: int,
        landing_row: int,
        landing: Sequence[Sequence[int]],
        locked: Sequence[Sequence[int]],
        final: Sequence[Sequence[int]],
        cleared_rows: Sequence[int],
        video_path: Path,
    ) -> int:
        frames: List[np.ndarray] = []

        def push(image: Image.Image, count: int = 1) -> None:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
            frames.extend(frame.copy() for _ in range(count))

        initial = self._with_active(settled, shape, 0, col)
        push(self._render_grid(initial), 8)
        for index in range(1, 25):
            row = landing_row * (index / 24.0)
            push(self._render_grid(settled, active_overlay=(shape, col, row)))
        push(self._render_grid(landing), 4)
        for index in range(1, 5):
            push(self._render_grid(landing, piece_lock_progress=index / 4.0))
        for block_scale in (0.86, 0.72, 0.58, 0.44, 0.30, 0.18, 0.08, 0.08):
            push(self._render_grid(locked, clear_rows=cleared_rows, clear_scale=block_scale))
        after_clear = [[0 if row in cleared_rows else int(cell) for cell in locked[row]] for row in range(self.rows)]
        falling: List[Tuple[Tuple[int, int], Tuple[int, int], int]] = []
        for row in range(self.rows):
            if row in cleared_rows:
                continue
            drop = sum(1 for cleared_row in cleared_rows if cleared_row > row)
            for block_col in range(self.cols):
                value = int(locked[row][block_col])
                if value and drop:
                    falling.append(((row, block_col), (row + drop, block_col), value))
        suppress = [source for source, _, _ in falling]
        for index in range(1, 9):
            t = index / 8.0
            moving = [
                (source[0] + (target[0] - source[0]) * t, source[1], value)
                for source, target, value in falling
            ]
            push(self._render_grid(after_clear, suppress_cells=suppress, moving_blocks=moving))
        push(self._render_grid(final), 9)
        assert len(frames) == 65
        return encode_video(frames, video_path, fps=self.video_fps)

    def _prompts(self) -> Dict[str, str]:
        return {
            "ti2v_prompt": build_sectioned_prompt(
                _INPUT_DESC,
                _TI2V_OUTPUT,
                mode="ti2v",
                rule_description=_RULE_SUMMARY,
                continuity_paragraph=_TI2V_CONTINUITY,
            ),
        }

    def create_puzzle(self) -> TetrisFutureBoardRecord:
        sample_id = f"tetris_future_board-{self._counter:05d}"
        self._counter += 1
        settled, shape, start_row, col, shape_name = self._initial_state()
        landing_row, landing, locked, final_grid, cleared_rows = self._simulate_drop(settled, shape, start_row, col)
        initial_grid = self._with_active(settled, shape, start_row, col)

        puzzles_dir = self.output_dir / "puzzles"
        solutions_dir = self.output_dir / "solutions"
        videos_dir = self.output_dir / "videos"
        for directory in (puzzles_dir, solutions_dir, videos_dir):
            directory.mkdir(parents=True, exist_ok=True)
        puzzle_path = puzzles_dir / f"{sample_id}.png"
        solution_path = solutions_dir / f"{sample_id}.png"
        video_path = videos_dir / f"{sample_id}.webm"
        self._render_grid(initial_grid).save(puzzle_path)
        self._render_grid(final_grid).save(solution_path)
        frame_count = (
            self._write_video(settled, shape, col, landing_row, landing, locked, final_grid, cleared_rows, video_path)
            if self.video
            else 0
        )
        prompts = self._prompts()
        state_trace = [initial_grid, landing, final_grid]
        payload: Dict[str, Any] = {
            "mechanic": "rigid_vertical_drop_lock_clear_shift",
            "state_grid": final_grid,
            "initial_grid": initial_grid,
            "state_trace": state_trace,
            "process_state_trace": state_trace,
            "settled_grid": settled,
            "landing_grid": landing,
            "locked_grid": locked,
            "rows": self.rows,
            "cols": self.cols,
            "active_shape_name": shape_name,
            "active_tetromino_shape": shape.astype(int).tolist(),
            "active_tetromino_column": col,
            "landing_row": landing_row,
            "cleared_rows": cleared_rows,
        }
        puzzle_rel = self.relativize_path(puzzle_path)
        solution_rel = self.relativize_path(solution_path)
        return TetrisFutureBoardRecord(
            id=sample_id,
            image=puzzle_rel,
            reasoning_image=puzzle_rel,
            solution_image_path=solution_rel,
            solution_video_path=self.relativize_path(video_path) if frame_count else None,
            video_fps=self.video_fps if frame_count else None,
            video_num_frames=frame_count or None,
            ti2v_prompt=prompts["ti2v_prompt"],

            answer=json.dumps(final_grid),
            question="Return the exact final board after the active piece drops, locks, clears complete rows, and shifts the survivors.",
            answer_grid=final_grid,
            structured_answer=payload,
            answer_payload=payload,
        )


__all__ = ["TetrisFutureBoardGenerator", "TetrisFutureBoardRecord"]

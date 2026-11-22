from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import CANVAS_SIZE
from data.dynamics_game.core.scene import finish_scene, new_scene, scene_font
from data.common.multiple_choice import (
    option_label,
    save_labeled_option_images,
)
from data.common.video_encoding import SIMPLE_VIDEO_FRAMES, encode_rgb_frames_to_webm, fit_frames_to_video_budget
from data.dynamics_game.core.choice_composite import (
    attach_choice_input_fields,
    build_composite_mcq_prompts,
)
from data.dynamics_game.core.coordinate_format import with_grid_xy_convention

_INPUT_DESC = with_grid_xy_convention(
    "The single problem image shows a rectangular grid of white wells with one-based row and column coordinate labels. Every "
    "nonempty well contains one fixed-size colored circular jewel. Exactly two orthogonally adjacent wells have gold "
    "borders; the jewels currently inside those two wells are the only pair that may be swapped."
)
_RULE_SUMMARY = (
    "Execute exactly one simultaneous swap between the two gold-bordered adjacent jewels. On the post-swap board, find "
    "every maximal horizontal or vertical run of at least three jewels of the same color and take the union of all cells "
    "in those runs, including intersections. Remove that whole union simultaneously. Only after removal is complete, "
    "apply gravity independently in each column: surviving jewels keep their vertical order and fall straight downward "
    "into the lowest available wells. Do not move jewels sideways, refill empty wells, or perform a second clear."
)
_TI2V_OUTPUT = (
    "Hold the input, then slide the two gold-bordered jewels simultaneously into each other's cells with "
    "their identities unchanged. After they settle, shrink the complete union of qualifying runs together "
    "into white wells. Only after that clear finishes, move every survivor straight down within its original "
    "column while preserving column order, then hold. Keep the grid, labels, well geometry, and spacing "
    "fixed; do not refill, cascade, or move sideways."
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render the solved still image that shows the final board after the highlighted swap, all resulting clears, and "
    "gravity."
)


@dataclass
class Jewel2FutureBoardRecord:
    id: str
    task_name: str = "jewel2_future_board"
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
    question_description: str = "swap_clear_gravity"
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
            "reasoning_images": list(self.reasoning_images),
            "solution_image_path": self.solution_image_path,
            "solution_video_path": self.solution_video_path,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
            "ti2v_prompt": self.ti2v_prompt,
            "answer": self.answer,
            "correct_option": self.correct_option,
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


class Jewel2FutureBoardGenerator(AbstractPuzzleGenerator[Jewel2FutureBoardRecord]):
    COLORS = {
        1: (221, 87, 70),
        2: (74, 169, 108),
        3: (65, 124, 233),
        4: (244, 194, 66),
        5: (164, 102, 221),
    }

    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        rows: int = 6,
        cols: int = 6,
        paired_runs: bool = False,
        video_fps: int = 16,
        video: bool = False,
    ) -> None:
        super().__init__(output_dir)
        self.seed = seed
        self.rng = random.Random(seed)
        self.paired_runs = bool(paired_runs)
        self.rows = rows
        self.cols = cols
        self.video_fps = video_fps
        self.video = video
        self._counter = 0

    def record_to_dict(self, record: Jewel2FutureBoardRecord | Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(record, "to_dict"):
            return record.to_dict()
        return dict(record)

    def _layout(self) -> Tuple[int, int, int]:
        max_dim = max(self.rows, self.cols)
        cell = (CANVAS_SIZE - 80) // max_dim
        cell -= cell % 2
        ox = (CANVAS_SIZE - self.cols * cell) // 2
        oy = (CANVAS_SIZE - self.rows * cell) // 2
        return cell, ox - (ox % 2), oy - (oy % 2)

    def _build_base_grid(self) -> List[List[int]]:
        palette = list(self.COLORS)
        grid: List[List[int]] = []
        offset = self.rng.randrange(len(palette))
        for row in range(self.rows):
            values: List[int] = []
            for col in range(self.cols):
                values.append(palette[(row + 2 * col + offset) % len(palette)])
            grid.append(values)
        return grid

    def _initial_state(
        self,
    ) -> Tuple[List[List[int]], Tuple[int, int], Tuple[int, int]]:
        """Plant one of four broken-run motifs completed by the marked swap.

        The gold-bordered pair always exchanges a blocking jewel inside the
        planned run with a matching jewel just outside it.  Motifs vary the
        run orientation, the run length, and whether two runs share the
        swapped-in cell, so the post-swap union ranges over 3-5 cells and the
        swap direction is not constant.  Every candidate layout is validated:
        the initial board may contain no run, the swap must clear exactly the
        planned union, and gravity must move at least one survivor.
        """

        for _attempt in range(400):
            grid = self._build_base_grid()
            match_color = self.rng.choice(list(self.COLORS))
            blocker_color = self.rng.choice(
                [color for color in self.COLORS if color != match_color]
            )
            motif = self.rng.choice(("h3", "v3", "h4", "corner"))
            if self.paired_runs:
                row = self.rng.randrange(1, self.rows - 2)
                col = self.rng.randrange(self.cols - 4)
                run = [(r, col + i) for r in (row, row + 1) for i in range(5)]
                for r, c in run:
                    grid[r][c] = match_color if r == row else blocker_color
                hole, source = (row, col + 2), (row + 1, col + 2)
                grid[hole[0]][hole[1]], grid[source[0]][source[1]] = (
                    blocker_color,
                    match_color,
                )
            elif motif == "h3":
                row = self.rng.randrange(1, self.rows - 1)
                col = self.rng.randrange(0, self.cols - 2)
                run = [(row, col), (row, col + 1), (row, col + 2)]
                hole = (row, col + 1)
                source = (row + 1, col + 1)
            elif motif == "h4":
                row = self.rng.randrange(1, self.rows - 1)
                col = self.rng.randrange(0, self.cols - 3)
                run = [(row, col + offset) for offset in range(4)]
                hole = (row, col + self.rng.choice((1, 2)))
                source = (row + 1, hole[1])
            elif motif == "v3":
                row = self.rng.randrange(0, self.rows - 3)
                col = self.rng.randrange(1, self.cols - 1)
                run = [(row, col), (row + 1, col), (row + 2, col)]
                hole = (row + 1, col)
                source = (row + 1, col + self.rng.choice((-1, 1)))
            else:  # corner: a horizontal and a vertical run share the swapped-in cell.
                row = self.rng.randrange(2, self.rows - 1)
                col = self.rng.randrange(2, self.cols)
                run = [
                    (row, col - 2),
                    (row, col - 1),
                    (row, col),
                    (row - 2, col),
                    (row - 1, col),
                ]
                hole = (row, col)
                source = (row + 1, col)
            if not self.paired_runs:
                for cell_row, cell_col in run:
                    grid[cell_row][cell_col] = match_color
                grid[hole[0]][hole[1]] = blocker_color
                grid[source[0]][source[1]] = match_color

            if self._clear_matches(grid) != grid:
                continue
            swapped = self._swap(grid, hole, source)
            cleared = self._clear_matches(swapped)
            marked = {
                (cell_row, cell_col)
                for cell_row in range(self.rows)
                for cell_col in range(self.cols)
                if int(swapped[cell_row][cell_col]) != 0
                and int(cleared[cell_row][cell_col]) == 0
            }
            if marked != set(run):
                continue
            if self._apply_gravity(cleared) == cleared:
                continue
            return grid, hole, source
        raise RuntimeError("Unable to plant a validated jewel swap motif in 400 attempts.")

    def _swap(self, grid: Sequence[Sequence[int]], first: Tuple[int, int], second: Tuple[int, int]) -> List[List[int]]:
        swapped = [list(row) for row in grid]
        r1, c1 = first
        r2, c2 = second
        swapped[r1][c1], swapped[r2][c2] = swapped[r2][c2], swapped[r1][c1]
        return swapped

    def _clear_matches(self, grid: Sequence[Sequence[int]]) -> List[List[int]]:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        marks = [[False for _ in range(cols)] for _ in range(rows)]

        for row in range(rows):
            start = 0
            while start < cols:
                value = int(grid[row][start])
                end = start + 1
                while end < cols and int(grid[row][end]) == value:
                    end += 1
                if value != 0 and end - start >= 3:
                    for col in range(start, end):
                        marks[row][col] = True
                start = end

        for col in range(cols):
            start = 0
            while start < rows:
                value = int(grid[start][col])
                end = start + 1
                while end < rows and int(grid[end][col]) == value:
                    end += 1
                if value != 0 and end - start >= 3:
                    for row in range(start, end):
                        marks[row][col] = True
                start = end

        cleared = [list(row) for row in grid]
        for row in range(rows):
            for col in range(cols):
                if marks[row][col]:
                    cleared[row][col] = 0
        return cleared

    def _apply_gravity(self, grid: Sequence[Sequence[int]]) -> List[List[int]]:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        for col in range(cols):
            values = [int(grid[row][col]) for row in range(rows) if int(grid[row][col]) != 0]
            row_ptr = rows - 1
            for value in reversed(values):
                result[row_ptr][col] = value
                row_ptr -= 1
        return result

    def _simulate_swap(self, grid: Sequence[Sequence[int]], first: Tuple[int, int], second: Tuple[int, int]) -> Tuple[List[List[List[int]]], List[List[int]]]:
        swapped = self._swap(grid, first, second)
        cleared = self._clear_matches(swapped)
        final_grid = self._apply_gravity(cleared)
        return [[list(row) for row in grid], swapped, cleared, final_grid], final_grid

    def _mutate_final_grid(self, final_grid: Sequence[Sequence[int]]) -> List[List[int]]:
        mutated = [list(row) for row in final_grid]
        nonzero_cells = [(row, col) for row in range(self.rows) for col in range(self.cols) if mutated[row][col] != 0]
        # FIX: swap two cells with DIFFERENT colors so the mutation actually
        # changes the grid. The previous implementation swapped nonzero[0] with
        # nonzero[-1] unconditionally, which was a no-op when both cells shared
        # the same color — yielding an MCQ distractor identical to the correct
        # option.
        for idx, (r1, c1) in enumerate(nonzero_cells):
            for r2, c2 in nonzero_cells[idx + 1 :]:
                if mutated[r1][c1] != mutated[r2][c2]:
                    mutated[r1][c1], mutated[r2][c2] = mutated[r2][c2], mutated[r1][c1]
                    return mutated
        # Fallback: if every non-zero cell shares a single color, zero one out.
        if nonzero_cells:
            row, col = nonzero_cells[0]
            mutated[row][col] = 0
        return mutated

    def _build_choice_states(self, state_trace: Sequence[Sequence[Sequence[int]]]) -> tuple[list[list[list[int]]], int]:
        candidate_states = [
            [list(row) for row in state_trace[-1]],
            [list(row) for row in state_trace[1]],
            [list(row) for row in state_trace[2]],
            self._mutate_final_grid(state_trace[-1]),
        ]
        ordered = list(range(len(candidate_states)))
        self.rng.shuffle(ordered)
        shuffled_states = [candidate_states[index] for index in ordered]
        answer_index = ordered.index(0)
        return shuffled_states, answer_index

    def _render_grid(
        self,
        grid: Sequence[Sequence[int]],
        *,
        highlight: Optional[Sequence[Tuple[int, int]]] = None,
        suppress_cells: Optional[Sequence[Tuple[int, int]]] = None,
        moving_jewels: Optional[Sequence[Tuple[Tuple[float, float], int]]] = None,
        clear_alpha: Optional[Tuple[Sequence[Tuple[int, int]], float]] = None,
    ) -> Image.Image:
        cell, ox, oy = self._layout()
        image, draw = new_scene((255, 255, 255))
        font = scene_font(max(11, int(cell * 0.32)))
        highlight_set = set(highlight or [])
        suppress_set = set(suppress_cells or [])
        clear_set, clear_a = (set(clear_alpha[0]), clear_alpha[1]) if clear_alpha else (set(), 0.0)

        for r in range(self.rows):
            for c in range(self.cols):
                x0 = ox + c * cell
                y0 = oy + r * cell
                draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=(255, 255, 255), outline=(180, 180, 180), width=1)
        # Labels.
        for r in range(self.rows):
            y = oy + r * cell + cell // 2
            draw.text((ox - 8, y), str(r + 1), fill=(60, 70, 90), font=font, anchor="rm")
        for c in range(self.cols):
            x = ox + c * cell + cell // 2
            draw.text((x, oy - 12), str(c + 1), fill=(60, 70, 90), font=font, anchor="mm")
        # Jewels.
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in suppress_set:
                    continue
                v = int(grid[r][c])
                if v == 0:
                    continue
                x0 = ox + c * cell
                y0 = oy + r * cell
                fill = self.COLORS[v]
                if (r, c) in clear_set:
                    if clear_a >= 0.995:
                        continue
                    fill = tuple(int(fill[i] * (1 - clear_a) + 255 * clear_a) for i in range(3))
                inset = max(3, cell // 8)
                if (r, c) in clear_set:
                    inset = int(inset * (1.0 - clear_a) + (cell / 2.0) * clear_a)
                outline = (40, 40, 50)
                if (r, c) in clear_set:
                    outline = tuple(int(outline[i] * (1 - clear_a) + 255 * clear_a) for i in range(3))
                draw.ellipse([x0 + inset, y0 + inset, x0 + cell - inset, y0 + cell - inset],
                             fill=fill, outline=outline, width=2)
        # Moving jewel overlays at sub-pixel positions.
        if moving_jewels:
            for (cx, cy), v in moving_jewels:
                fill = self.COLORS[v]
                inset = max(3, cell // 8)
                draw.ellipse([cx - cell / 2 + inset, cy - cell / 2 + inset,
                               cx + cell / 2 - inset, cy + cell / 2 - inset],
                             fill=fill, outline=(40, 40, 50), width=2)
        # Highlight rings drawn last.
        for r, c in highlight_set:
            x0 = ox + c * cell
            y0 = oy + r * cell
            draw.rectangle([x0 + 2, y0 + 2, x0 + cell - 2, y0 + cell - 2],
                           outline=(212, 162, 32), width=3)
        return finish_scene(image)

    def _write_video(
        self,
        grid_initial: Sequence[Sequence[int]],
        swap_first: Tuple[int, int],
        swap_second: Tuple[int, int],
        states: Sequence[Sequence[Sequence[int]]],
        video_path: Path,
    ) -> int:
        """Smooth: hold -> direct swap -> clear -> falling gravity -> hold."""
        cell, ox, oy = self._layout()
        # State indices: 0 = initial, 1 = post-swap, 2 = post-clear, 3 = post-gravity.
        if len(states) < 4:
            return 0
        frames: List[np.ndarray] = []

        def cell_xy(r: int, c: int) -> Tuple[float, float]:
            return ox + c * cell + cell / 2.0, oy + r * cell + cell / 2.0

        # 0 - initial hold (with highlight).
        before = self._render_grid(grid_initial, highlight=[swap_first, swap_second])
        for _ in range(4):
            frames.append(np.array(before, dtype=np.uint8))

        # 1 - sub-pixel swap. The two highlighted jewels slide directly and
        # simultaneously into each other's original cells.
        v1 = int(grid_initial[swap_first[0]][swap_first[1]])
        v2 = int(grid_initial[swap_second[0]][swap_second[1]])
        x1, y1 = cell_xy(*swap_first)
        x2, y2 = cell_xy(*swap_second)
        for k in range(1, 9):
            t = k / 8.0
            cx1 = x1 + (x2 - x1) * t
            cy1 = y1 + (y2 - y1) * t
            cx2 = x2 + (x1 - x2) * t
            cy2 = y2 + (y1 - y2) * t
            frame = self._render_grid(
                grid_initial,
                suppress_cells=[swap_first, swap_second],
                moving_jewels=[((cx1, cy1), v1), ((cx2, cy2), v2)],
            )
            frames.append(np.array(frame, dtype=np.uint8))

        # 2 - hold post-swap.
        post_swap = self._render_grid(states[1])
        for _ in range(2):
            frames.append(np.array(post_swap, dtype=np.uint8))

        # 3 - dissolve clear: identify cleared cells (in states[1] but not states[2]).
        cleared = []
        for r in range(self.rows):
            for c in range(self.cols):
                if int(states[1][r][c]) != 0 and int(states[2][r][c]) == 0:
                    cleared.append((r, c))
        for k in range(1, 5):
            a = k / 4.0
            frame = self._render_grid(states[1], clear_alpha=(cleared, a))
            frames.append(np.array(frame, dtype=np.uint8))

        # 4 - gravity: surviving jewels fall straight down within their columns.
        falling: List[Tuple[Tuple[int, int], Tuple[int, int], int]] = []
        for c in range(self.cols):
            survivors = [(r, int(states[2][r][c])) for r in range(self.rows) if int(states[2][r][c]) != 0]
            dst_start = self.rows - len(survivors)
            for offset, (src_r, value) in enumerate(survivors):
                dst_r = dst_start + offset
                if src_r != dst_r:
                    falling.append(((src_r, c), (dst_r, c), value))

        post_gravity = self._render_grid(states[3])
        if falling:
            suppress_falling = [src for src, _, _ in falling]
            for k in range(1, 9):
                t = k / 8.0
                eased = t * t * (3.0 - 2.0 * t)
                moving = []
                for (src_r, src_c), (dst_r, dst_c), value in falling:
                    x0, y0 = cell_xy(src_r, src_c)
                    x1f, y1f = cell_xy(dst_r, dst_c)
                    moving.append(((x0 + (x1f - x0) * eased, y0 + (y1f - y0) * eased), value))
                frame = self._render_grid(
                    states[2],
                    suppress_cells=suppress_falling,
                    moving_jewels=moving,
                )
                frames.append(np.array(frame, dtype=np.uint8))
        else:
            for _ in range(2):
                frames.append(np.array(post_gravity, dtype=np.uint8))

        # 5 - final hold.
        for _ in range(8):
            frames.append(np.array(post_gravity, dtype=np.uint8))

        frames = fit_frames_to_video_budget(
            frames,
            max_frames=SIMPLE_VIDEO_FRAMES,
            final_hold_frames=8,
        )
        if not encode_rgb_frames_to_webm(
            frames, video_path, fps=self.video_fps,
            extra_args=["-g", "1", "-bf", "0"],
        ):
            return 0
        return len(frames)

    def _prompts(self) -> Dict[str, str]:
        return build_composite_mcq_prompts(
            input_desc=_INPUT_DESC,
            rule_summary=_RULE_SUMMARY,
            ti2v_output=_TI2V_OUTPUT,
            solution_image_output=_SOLUTION_IMAGE_OUTPUT,
            continuity_timing="continuous",
            continuity_moving="the two swapped jewels, the cleared jewels, and every falling survivor",
            continuity_static=(
                "the board grid, the row and column labels, every jewel the rules leave in place, "
                "the background, and the camera"
            ),
            continuity_hold="the final compacted board",
        )

    def create_puzzle(self) -> Jewel2FutureBoardRecord:
        sample_id = f"jewel2_future_board-{self._counter:05d}"
        self._counter += 1

        grid, swap_first, swap_second = self._initial_state()
        state_trace, final_grid = self._simulate_swap(grid, swap_first, swap_second)
        candidate_states, answer_index = self._build_choice_states(state_trace)

        puzzles_dir = self.output_dir / "puzzles"
        solutions_dir = self.output_dir / "solutions"
        options_dir = self.output_dir / "reasoning_options"
        videos_dir = self.output_dir / "videos"
        steps_dir = self.output_dir / "reasoning_steps"
        for directory in (puzzles_dir, solutions_dir, options_dir, videos_dir, steps_dir):
            directory.mkdir(parents=True, exist_ok=True)

        puzzle_path = puzzles_dir / f"{sample_id}.png"
        solution_path = solutions_dir / f"{sample_id}.png"
        video_path = videos_dir / f"{sample_id}.webm"
        highlight = [swap_first, swap_second]
        puzzle_image = self._render_grid(grid, highlight=highlight)
        solution_image = self._render_grid(final_grid)
        swapped_path = steps_dir / f"{sample_id}__swapped.png"
        cleared_path = steps_dir / f"{sample_id}__cleared.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)
        self._render_grid(state_trace[1]).save(swapped_path)
        self._render_grid(state_trace[2]).save(cleared_path)

        option_tiles = [self._render_grid(state) for state in candidate_states]
        option_paths = save_labeled_option_images(
            option_tiles,
            options_dir,
            stem=sample_id,
            labels=("A", "B", "C", "D"),
        )
        reasoning_images = [self.relativize_path(puzzle_path)]

        frame_count = self._write_video(grid, swap_first, swap_second, state_trace, video_path) if self.video else 0

        prompts = self._prompts()
        answer_label = option_label(answer_index)
        payload = {
            "state_grid": [list(row) for row in final_grid],
            "initial_grid": [list(row) for row in grid],
            "state_trace": state_trace,
            "process_state_trace": [state_trace[0], state_trace[-1]],
            "swap_cells": [list(swap_first), list(swap_second)],
            "swap_a": list(swap_first),
            "swap_b": list(swap_second),
            "cleared_cells": [
                [row, col]
                for row in range(self.rows)
                for col in range(self.cols)
                if int(state_trace[1][row][col]) != 0 and int(state_trace[2][row][col]) == 0
            ],
            "candidate_state_grids": candidate_states,
            "correct_option_index": answer_index,
            "correct_option": answer_label,
            "rows": self.rows,
            "cols": self.cols,
        }
        artifacts = {
            "puzzle_path": puzzle_path,
            "solution_path": solution_path,
            "options_paths": option_paths,
            "video_path": video_path,
            "frame_count": frame_count,
        }
        record = Jewel2FutureBoardRecord(
            id=sample_id,
            image=self.relativize_path(puzzle_path),
            reasoning_image=self.relativize_path(puzzle_path),
            reasoning_images=reasoning_images,
            solution_image_path=self.relativize_path(solution_path),
            solution_video_path=self.relativize_path(video_path) if frame_count > 0 else None,
            video_fps=self.video_fps if frame_count > 0 else None,
            video_num_frames=frame_count if frame_count > 0 else None,
            ti2v_prompt=prompts["ti2v_prompt"],

            answer=answer_label,
            correct_option=answer_label,
            question="Choose the candidate result-board that matches the outcome of the highlighted swap after clears and gravity.",
            answer_grid=[list(row) for row in final_grid],
            structured_answer=payload,
            answer_payload=payload,
        )

        return attach_choice_input_fields(
            record,
            artifacts=artifacts,
            relativize=self.relativize_path,
        )

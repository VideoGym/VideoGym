from __future__ import annotations

import random
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from data.common.base import AbstractPuzzleGenerator
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.scalar_lists import format_integer_answer
from data.next_state_prediction.core.rendering import (
    discrete_video_frames,
    finish_state_canvas,
    new_state_canvas,
)

pattern_instructions = {
    "rotten_oranges": (
        "The input image shows a grid of empty cells, fresh oranges, and rotten oranges. Determine the minimum number "
        "of minutes until no fresh orange remains, or -1 if that is impossible."
    )
}

_INPUT_DESC = (
    "The input image shows the initial state of one square grid. A pale cell is empty (0), an orange cell is a "
    "fresh orange (1), and a brown cell is a rotten orange (2)."
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render only the final reachable grid state in the same fixed layout; put the exact integer result in the "
    "typed text delivery rather than inside the image."
)
_TI2V_OUTPUT = (
    "Generate a video starting at minute 0. Commit every simultaneous one-minute propagation state in order as one "
    "complete grid: all fresh cells reached from an orthogonally adjacent rotten cell turn brown together. Do not "
    "skip minutes, do not use diagonal spread, and do not change an empty cell."
)
_CANVAS = (512, 512)
_BG = (247, 248, 246)
_CARD = (255, 255, 253)
_INK = (38, 42, 46)
_MUTED = (91, 96, 101)
_EMPTY = (244, 246, 244)
_FRESH = (248, 155, 48)
_ROTTEN = (126, 80, 36)

def _minutes_to_rot(grid: List[List[int]]) -> tuple[int, List[List[int]]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    minutes = [[-1 for _ in range(cols)] for _ in range(rows)]
    queue: deque[Tuple[int, int]] = deque()
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 2:
                minutes[row][col] = 0
                queue.append((row, col))
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = row + dr
            nc = col + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1 and minutes[nr][nc] == -1:
                minutes[nr][nc] = minutes[row][col] + 1
                queue.append((nr, nc))
    fresh_cells = [minutes[row][col] for row in range(rows) for col in range(cols) if grid[row][col] == 1]
    if any(value == -1 for value in fresh_cells):
        return -1, minutes
    return max(fresh_cells, default=0), minutes



def _rollout_states(grid: List[List[int]], minute_grid: List[List[int]]) -> List[List[List[int]]]:
    max_reachable = max(0, max((value for row in minute_grid for value in row), default=0))
    states: List[List[List[int]]] = []
    for minute in range(max_reachable + 1):
        states.append(
            [
                [
                    2
                    if int(grid[row][col]) == 1 and 0 <= int(minute_grid[row][col]) <= minute
                    else int(grid[row][col])
                    for col in range(len(grid[row]))
                ]
                for row in range(len(grid))
            ]
        )
    return states

class RottenOrangesPattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = "rotten_oranges"
    task_family = "next_state_prediction"

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        size_range: tuple[int, int] = (4, 6),
        video: bool = True,
        video_fps: int = 16,
    ) -> None:
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_rotten_oranges"
        super().__init__(resolved)
        self.rng = random.Random(seed)
        self.size_range = size_range
        self.video = bool(video)
        self.video_fps = int(video_fps)
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _sample_grid(self) -> List[List[int]]:
        for _attempt in range(256):
            size = self.rng.randint(*self.size_range)
            grid = [
                [self.rng.choices([0, 1, 2], weights=(3, 7, 2), k=1)[0] for _ in range(size)]
                for _ in range(size)
            ]
            if not any(cell == 2 for row in grid for cell in row):
                grid[self.rng.randrange(size)][self.rng.randrange(size)] = 2
            if not any(cell == 1 for row in grid for cell in row):
                grid[self.rng.randrange(size)][self.rng.randrange(size)] = 1
            _minutes, minute_grid = _minutes_to_rot(grid)
            reachable_fresh = [
                minute_grid[row][col]
                for row in range(size)
                for col in range(size)
                if grid[row][col] == 1 and minute_grid[row][col] >= 1
            ]
            if reachable_fresh and 2 <= max(reachable_fresh) <= 7:
                return grid
        return [
            [2, 1, 1, 0, 1],
            [0, 0, 1, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]

    @staticmethod
    def _grid_geometry(size: int) -> tuple[int, int, int, int]:
        cell_size = min(64, 324 // max(1, size))
        grid_size = cell_size * size
        left = (512 - grid_size) // 2
        top = (512 - grid_size) // 2
        return left, top, cell_size, grid_size

    def _render_state(self, grid: List[List[int]], *, status: str) -> Image.Image:
        del status
        image, draw = new_state_canvas(_BG, _CARD, (219, 222, 218))

        size = len(grid)
        left, top, cell_size, grid_size = self._grid_geometry(size)
        palette = {0: _EMPTY, 1: _FRESH, 2: _ROTTEN}
        for row in range(size):
            for col in range(size):
                # Each cell's state lives in its colour alone.
                value = int(grid[row][col])
                x0 = left + col * cell_size
                y0 = top + row * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                draw.rounded_rectangle(
                    (x0 + 2, y0 + 2, x1 - 2, y1 - 2),
                    radius=9,
                    fill=palette[value],
                    outline=(204, 207, 203),
                    width=2,
                )
        draw.rounded_rectangle(
            (left - 5, top - 5, left + grid_size + 5, top + grid_size + 5),
            radius=12,
            outline=(122, 128, 132),
            width=2,
        )

        return finish_state_canvas(image)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=_INPUT_DESC,
            symbol_legend=(
                "A pale cell = empty cell (0); it never changes.",
                "An orange cell = fresh orange (1).",
                "A brown cell = rotten orange (2).",
            ),
            rules=(
                "At each minute, all rotten oranges act simultaneously.",
                "A rotten orange changes each orthogonally adjacent fresh orange to rotten.",
                "Diagonal cells do not interact, and empty cells never change.",
                "The answer is the first minute with no fresh orange, or -1 if any fresh orange remains unreachable.",
            ),
            modality_outputs={
                "ti2v": _TI2V_OUTPUT,
                "solution_image": _SOLUTION_IMAGE_OUTPUT,
            },
            preservation_constraints=(
                "Keep the grid size, cell positions, empty cells, and grid geometry unchanged.",
                "Change only fresh oranges, and only from fresh to rotten.",
            ),
            transition_constraints=(
                "Start from the supplied minute-0 grid.",
                "Commit every integer minute in order as one complete state with simultaneous orthogonal propagation and no skipped minute.",
                "Do not change an empty or already rotten cell.",
                "The exact integer result belongs in the typed text delivery, not inside the image.",
            ),
            ti2v_continuity={
                "timing": "discrete",
                "moving_nouns": "the fresh cells that turn rotten",
                "static_nouns": "the empty cells, the grid frame, the background, and the camera",
                "hold_noun": "the exact final reachable grid",
            },
            answer=AnswerContract(
                kind="integer",
                format_template="Answer: <integer>.",
                parser_profile="integer",
                verifier_profile="rotten_oranges_bfs_rollout",
                format_example="Answer: 99.",
                accepts_multiple_valid_answers=False,
            ),
            render_profile="rotten_oranges_rollout_v2",
        )

    def _prompts(self) -> tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {f"{mode}_prompt": contract.build_prompt(mode) for mode in contract.prompt_modes()}
        # These prompts are constructed only from the static contract. Small integer answers
        # also occur legitimately in the 0/1/2 legend and format examples, so substring-based
        # answer leakage checks would produce false positives for this task.
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid rotten-oranges prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _render_video(self, states: List[List[List[int]]], *, answer: int, output_path: Path) -> int:
        del answer
        renders = [self._render_state(state, status="") for state in states]
        frames = [np.asarray(image, dtype=np.uint8) for image in discrete_video_frames(renders)]
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)


    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1

        grid = self._sample_grid()
        minutes, minute_grid = _minutes_to_rot(grid)
        states = _rollout_states(grid, minute_grid)
        final_grid = states[-1]
        contract, prompts = self._prompts()

        sample_dir = self.output_dir / sample_id
        puzzle_dir = sample_dir / "puzzles"
        solution_dir = sample_dir / "solutions"
        video_dir = sample_dir / "videos"
        for directory in (puzzle_dir, solution_dir, video_dir):
            directory.mkdir(parents=True, exist_ok=True)

        final_status = (
            f"Final · {minutes} minute{'s' if minutes != 1 else ''}"
            if minutes >= 0
            else "Final · impossible (-1)"
        )
        puzzle_image = self._render_state(grid, status="Minute 0")
        solution_image = self._render_state(final_grid, status=final_status)
        puzzle_path = puzzle_dir / "puzzle.png"
        solution_path = solution_dir / "solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        solution_video_path = None
        video_num_frames = None
        if self.video:
            video_path = video_dir / "solution.webm"
            video_num_frames = self._render_video(states, answer=minutes, output_path=video_path)
            if video_num_frames > 0:
                solution_video_path = video_path

        answer_text = format_integer_answer(minutes)
        left, top, cell_size, grid_size = self._grid_geometry(len(grid))
        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": "Compute the minimum number of minutes until no fresh orange remains, or -1 if impossible.",
            "answer": answer_text,
            "structured_answer": {
                "input_grid": grid,
                "minutes": minutes,
                "minute_grid": minute_grid,
                "final_grid": final_grid,
                "grid_bbox": [left, top, left + grid_size, top + grid_size],
                "cell_size": cell_size,
            },
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path.as_posix() if solution_video_path is not None else None,
            "video_fps": self.video_fps if video_num_frames else None,
            "video_num_frames": video_num_frames,
        }

    def make_sample(self) -> tuple[Dict[str, Any], Image.Image, Image.Image]:
        record = self.create_puzzle()
        puzzle = Image.open(record["image"]).convert("RGB").copy()
        solution = Image.open(record["solution_image_path"]).convert("RGB").copy()
        return record, puzzle, solution

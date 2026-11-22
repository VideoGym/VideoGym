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
from data.next_state_prediction.core.rendering import (
    discrete_video_frames,
    finish_state_canvas,
    new_state_canvas,
)


pattern_instructions = {
    "grid_component": (
        "The input image shows a binary grid of 0s and 1s. Count the size of the largest 4-directionally connected "
        "component of 1s and return that integer."
    )
}

_INPUT_DESC = (
    "The input image shows one square binary grid. Pale cells are empty, and dark cells are active cells that "
    "may belong to connected components."
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render the same grid with the unique largest component highlighted in green, every other 1 cell muted, every "
    "0 cell unchanged, and no title, status line, or answer banner."
)
_TI2V_OUTPUT = (
    "Generate a video that keeps the input grid readable, then commits the complete terminal partition as one "
    "simultaneous global update: the unique largest component turns green and every other active cell turns muted "
    "gray together."
)

_BG = (247, 248, 246)
_CARD = (255, 255, 253)
_ZERO = (246, 247, 244)
_ONE = (72, 83, 103)
_ACTIVE = (240, 164, 57)
_LABELED = (139, 151, 174)
_LARGEST = (57, 158, 112)
_OTHER = (205, 210, 216)


def _component_cells(grid: List[List[int]]) -> List[List[Tuple[int, int]]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    seen = [[False for _ in range(cols)] for _ in range(rows)]
    components: List[List[Tuple[int, int]]] = []
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] != 1 or seen[row][col]:
                continue
            queue: deque[Tuple[int, int]] = deque([(row, col)])
            seen[row][col] = True
            component: List[Tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                component.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and grid[nr][nc] == 1
                        and not seen[nr][nc]
                    ):
                        seen[nr][nc] = True
                        queue.append((nr, nc))
            components.append(component)
    return components


def _largest_component(grid: List[List[int]]) -> tuple[int, int, List[List[int]]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    labels = [[0 for _ in range(cols)] for _ in range(rows)]
    components = _component_cells(grid)
    best_label = 0
    best_size = 0
    for label, component in enumerate(components, start=1):
        for row, col in component:
            labels[row][col] = label
        if len(component) > best_size:
            best_size = len(component)
            best_label = label
    return best_size, best_label, labels


def _component_visual_states(
    grid: List[List[int]],
    components: List[List[Tuple[int, int]]],
) -> List[tuple[str, int | None, List[List[str]]]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    states: List[tuple[str, int | None, List[List[str]]]] = [
        (
            "input",
            None,
            [["one" if grid[row][col] == 1 else "zero" for col in range(cols)] for row in range(rows)],
        )
    ]
    sizes = [len(component) for component in components]
    best_index = sizes.index(max(sizes)) if sizes else -1
    largest = set(components[best_index]) if best_index >= 0 else set()
    final_state = [
        [
            "zero"
            if grid[row][col] == 0
            else "largest"
            if (row, col) in largest
            else "other"
            for col in range(cols)
        ]
        for row in range(rows)
    ]
    states.append(("final", best_index, final_state))
    return states


class GridComponentPattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = "grid_component"
    task_family = "next_state_prediction"

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        size_range: tuple[int, int] = (5, 7),
        video: bool = True,
        video_fps: int = 16,
    ) -> None:
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_grid_component"
        super().__init__(resolved)
        self.rng = random.Random(seed)
        self.size_range = size_range
        self.video = bool(video)
        self.video_fps = int(video_fps)
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _sample_grid(self) -> List[List[int]]:
        size = self.rng.randint(*self.size_range)
        for _attempt in range(1024):
            grid = [
                [self.rng.choices([0, 1], weights=(6, 4), k=1)[0] for _ in range(size)]
                for _ in range(size)
            ]
            components = _component_cells(grid)
            sizes = sorted((len(component) for component in components), reverse=True)
            if (
                3 <= len(components) <= 5
                and sizes[0] >= 4
                and sizes[1] >= 2
                and sizes.count(sizes[0]) == 1
            ):
                return grid

        fallback = [[0 for _ in range(size)] for _ in range(size)]
        for row, col in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)):
            fallback[row][col] = 1
        for row, col in ((0, size - 1), (1, size - 1), (1, size - 2)):
            fallback[row][col] = 1
        for row, col in ((size - 1, 0), (size - 1, 1)):
            fallback[row][col] = 1
        fallback[size - 2][size - 1] = 1
        return fallback

    @staticmethod
    def _grid_geometry(size: int) -> tuple[int, int, int, int]:
        cell_size = min(46, 322 // max(1, size))
        grid_size = cell_size * size
        left = (512 - grid_size) // 2
        top = 105 + (322 - grid_size) // 2
        return left, top, cell_size, grid_size

    def _render_state(
        self,
        grid: List[List[int]],
        components: List[List[Tuple[int, int]]],
        *,
        phase: str,
        component_index: int | None = None,
    ) -> Image.Image:
        image, draw = new_state_canvas(_BG, _CARD, (218, 221, 217))

        sizes = [len(component) for component in components]
        best_size = max(sizes, default=0)
        best_index = sizes.index(best_size) if sizes else -1

        size = len(grid)
        left, top, cell_size, grid_size = self._grid_geometry(size)
        active = set(components[component_index]) if phase == "component" and component_index is not None else set()
        labeled = (
            {cell for component in components[: int(component_index or 0)] for cell in component}
            if phase == "component"
            else set()
        )
        largest = set(components[best_index]) if phase == "final" and best_index >= 0 else set()
        for row in range(size):
            for col in range(size):
                cell = (row, col)
                # Each cell's state lives in its colour alone.
                if grid[row][col] == 0:
                    category = "zero"
                elif phase == "input":
                    category = "one"
                elif phase == "component":
                    if cell in active:
                        category = "active"
                    elif cell in labeled:
                        category = "labeled"
                    else:
                        category = "one"
                elif cell in largest:
                    category = "largest"
                else:
                    category = "other"
                palette = {
                    "zero": _ZERO,
                    "one": _ONE,
                    "active": _ACTIVE,
                    "labeled": _LABELED,
                    "largest": _LARGEST,
                    "other": _OTHER,
                }
                x0 = left + col * cell_size
                y0 = top + row * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                draw.rounded_rectangle(
                    (x0 + 2, y0 + 2, x1 - 2, y1 - 2),
                    radius=7,
                    fill=palette[category],
                    outline=(176, 116, 27) if category == "active" else (196, 201, 198),
                    width=3 if category == "active" else 2,
                )
        draw.rounded_rectangle(
            (left - 5, top - 5, left + grid_size + 5, top + grid_size + 5),
            radius=12,
            outline=(112, 119, 124),
            width=2,
        )

        return finish_state_canvas(image)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=_INPUT_DESC,
            symbol_legend=(
                "A pale cell = empty cell; it never belongs to a component.",
                "A dark cell = active cell that belongs to exactly one connected component.",
            ),
            rules=(
                "Two cells labeled 1 are connected only through a chain of orthogonally adjacent cells labeled 1.",
                "Diagonal contact alone does not connect two cells, and a path may never cross a 0 cell.",
                "A connected component is a maximal set of mutually connected cells labeled 1.",
                "Count every component and return the number of cells in the largest one.",
                "The visible grid has between three and five components and exactly one largest component.",
            ),
            modality_outputs={
                "ti2v": _TI2V_OUTPUT,
                "solution_image": _SOLUTION_IMAGE_OUTPUT,
            },
            preservation_constraints=(
                "Keep the grid size, every cell's occupancy, and the grid geometry unchanged.",
                "Change only the component-state colors; no cell may move, appear, or disappear.",
            ),
            transition_constraints=(
                "Start from the supplied binary grid.",
                "Commit the complete partition as one simultaneous global update: the unique largest component becomes green and all other components become muted gray together.",
                "Component order is not semantic, so never show a partial partition in which only some components have changed color.",
                "Do not change which cells are active or empty.",
            ),
            ti2v_continuity={
                "timing": "discrete",
                "moving_nouns": "the active cells' state colors",
                "static_nouns": "the empty cells, the grid frame, the background, and the camera",
                "hold_noun": "the exact final partition",
            },
            answer=AnswerContract(
                kind="nonnegative_integer",
                format_template="Answer: <positive integer>.",
                parser_profile="integer",
                verifier_profile="grid_component_partition_replay",
                format_example="Answer: 99.",
                accepts_multiple_valid_answers=False,
            ),
            render_profile="grid_component_synchronized_global_commit_v3",
        )

    def _prompts(self) -> tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {f"{mode}_prompt": contract.build_prompt(mode) for mode in contract.prompt_modes()}
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid grid-component prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _render_video(
        self,
        grid: List[List[int]],
        components: List[List[Tuple[int, int]]],
        *,
        output_path: Path,
    ) -> int:
        states = _component_visual_states(grid, components)
        input_phase, input_index, _input_state = states[0]
        final_phase, final_index, _final_state = states[-1]
        renders = [
            self._render_state(grid, components, phase=input_phase, component_index=input_index),
            self._render_state(grid, components, phase=final_phase, component_index=final_index),
        ]
        frames = [np.asarray(image, dtype=np.uint8) for image in discrete_video_frames(renders)]
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)


    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1

        grid = self._sample_grid()
        components = _component_cells(grid)
        best_size, best_label, labels = _largest_component(grid)
        sizes = [len(component) for component in components]
        answer_raw = str(best_size)
        contract, prompts = self._prompts()

        sample_dir = self.output_dir / sample_id
        puzzle_dir = sample_dir / "puzzles"
        solution_dir = sample_dir / "solutions"
        video_dir = sample_dir / "videos"
        steps_dir = sample_dir / "steps"
        for directory in (puzzle_dir, solution_dir, video_dir, steps_dir):
            directory.mkdir(parents=True, exist_ok=True)

        puzzle_image = self._render_state(grid, components, phase="input")
        solution_image = self._render_state(grid, components, phase="final", component_index=best_label - 1)
        puzzle_path = puzzle_dir / "puzzle.png"
        solution_path = solution_dir / "solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        solution_video_path = None
        video_num_frames = None
        if self.video:
            video_path = video_dir / "solution.webm"
            video_num_frames = self._render_video(grid, components, output_path=video_path)
            if video_num_frames > 0:
                solution_video_path = video_path

        size = len(grid)
        left, top, cell_size, grid_size = self._grid_geometry(size)
        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": "Count the cells in the unique largest 4-directionally connected component of 1s.",
            "answer": answer_raw,
            "structured_answer": {
                "input_grid": grid,
                "labels": labels,
                "component_sizes": sizes,
                "components": [[[row, col] for row, col in component] for component in components],
                "largest_component_size": best_size,
                "largest_component_label": best_label,
                "grid_bbox": [left, top, left + grid_size, top + grid_size],
                "cell_size": cell_size,
                "unique_largest": True,
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

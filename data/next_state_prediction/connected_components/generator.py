from __future__ import annotations

import random
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

from data.common.base import AbstractPuzzleGenerator
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.next_state_prediction.core.rendering import (
    discrete_video_frames,
    finish_state_canvas,
    new_state_canvas,
    scaled_font,
)


pattern_instructions = {
    "connected_components": (
        "An 8 by 8 binary grid contains several filled regions. Count the 4-connected components, where only up, "
        "down, left, and right adjacency connects filled cells and diagonal contact does not."
    )
}

_SIZE = 8
_GRID_LEFT = 88
_GRID_TOP = 108
_CELL_SIZE = 42
_GRID_PX = _SIZE * _CELL_SIZE
_EMPTY = -1
_UNVISITED = 0
_ACTIVE = 99

_BG = (246, 248, 247)
_CARD = (255, 255, 253)
_INK = (34, 40, 46)
_MUTED = (88, 96, 103)
_GRID_FILL = (251, 252, 252)
_GRID_LINE = (174, 183, 188)
_ACCENT = (49, 132, 148)
_UNVISITED_COLOR = (72, 83, 103)
_ACTIVE_COLOR = (240, 164, 57)
_COUNTED_COLOR = (46, 151, 166)


def _connected_components_4(grid: Sequence[Sequence[int]]) -> List[List[tuple[int, int]]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    visited: set[tuple[int, int]] = set()
    components: List[List[tuple[int, int]]] = []
    for row in range(rows):
        for col in range(cols):
            if int(grid[row][col]) != 1 or (row, col) in visited:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited.add((row, col))
            component: List[tuple[int, int]] = []
            while queue:
                current = queue.popleft()
                component.append(current)
                rr, cc = current
                for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and int(grid[nr][nc]) == 1
                        and (nr, nc) not in visited
                    ):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            components.append(sorted(component))
    return components


def _component_states(
    grid: Sequence[Sequence[int]],
    components: Sequence[Sequence[tuple[int, int]]],
) -> List[List[List[int]]]:
    input_state = [
        [_UNVISITED if int(value) == 1 else _EMPTY for value in row]
        for row in grid
    ]
    final = [list(row) for row in input_state]
    for component_index, component in enumerate(components, start=1):
        for row, col in component:
            final[row][col] = component_index
    return [input_state, final]


class ConnectedComponentsPattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = "connected_components"
    task_family = "next_state_prediction"
    task_mechanic = "four_connected_component_enumeration"

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        video: bool = True,
        video_fps: int = 16,
    ) -> None:
        if int(video_fps) != 16:
            raise ValueError("connected_components videos must use 16 fps")
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_connected_components"
        super().__init__(resolved)
        self.rng = random.Random(seed)
        self.seed = seed
        self.video = bool(video)
        self.video_fps = int(video_fps)
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _sample_grid(self) -> tuple[List[List[int]], List[List[tuple[int, int]]]]:
        target = self.rng.randint(2, 5)
        directions = ((-1, 0), (0, -1), (0, 1), (1, 0))
        for _ in range(96):
            labels = [[0 for _col in range(_SIZE)] for _row in range(_SIZE)]

            # Seed the first two regions across a diagonal. Their persistent
            # corner contact makes the 4-connectivity rule load-bearing in
            # every generated instance instead of merely mentioning it.
            top = self.rng.randrange(_SIZE - 1)
            left = self.rng.randrange(_SIZE - 1)
            if self.rng.randrange(2) == 0:
                seeds = [(top, left), (top + 1, left + 1)]
            else:
                seeds = [(top, left + 1), (top + 1, left)]

            seed_candidates = [
                (row, col)
                for row in range(_SIZE)
                for col in range(_SIZE)
            ]
            self.rng.shuffle(seed_candidates)
            for cell in seed_candidates:
                if len(seeds) >= target:
                    break
                row, col = cell
                if all(abs(row - seed_row) + abs(col - seed_col) > 1 for seed_row, seed_col in seeds):
                    seeds.append(cell)
            if len(seeds) != target:
                continue

            cells_by_component: List[List[tuple[int, int]]] = []
            for label, (row, col) in enumerate(seeds, start=1):
                labels[row][col] = label
                cells_by_component.append([(row, col)])

            filled_target = self.rng.randint(max(16, 2 * target), 34)
            target_sizes = [2 for _ in range(target)]
            for _extra_cell in range(filled_target - 2 * target):
                target_sizes[self.rng.randrange(target)] += 1
            remaining = [target_sizes[index] - 1 for index in range(target)]

            while any(remaining):
                growth_choices: List[tuple[int, List[tuple[int, int]]]] = []
                for component_index in range(target):
                    if remaining[component_index] <= 0:
                        continue
                    label = component_index + 1
                    frontier: set[tuple[int, int]] = set()
                    for row, col in cells_by_component[component_index]:
                        for delta_row, delta_col in directions:
                            next_row = row + delta_row
                            next_col = col + delta_col
                            if not (0 <= next_row < _SIZE and 0 <= next_col < _SIZE):
                                continue
                            if labels[next_row][next_col] != 0:
                                continue
                            neighbor_labels = {
                                labels[next_row + neighbor_row][next_col + neighbor_col]
                                for neighbor_row, neighbor_col in directions
                                if 0 <= next_row + neighbor_row < _SIZE
                                and 0 <= next_col + neighbor_col < _SIZE
                            }
                            if neighbor_labels - {0, label}:
                                continue
                            frontier.add((next_row, next_col))
                    if frontier:
                        growth_choices.append((component_index, sorted(frontier)))
                if not growth_choices:
                    break
                component_index, frontier = self.rng.choice(growth_choices)
                row, col = self.rng.choice(frontier)
                labels[row][col] = component_index + 1
                cells_by_component[component_index].append((row, col))
                remaining[component_index] -= 1
            if any(remaining):
                continue

            grid = [[int(label > 0) for label in row] for row in labels]
            components = _connected_components_4(grid)
            if (
                len(components) == target
                and sorted(len(component) for component in components) == sorted(target_sizes)
            ):
                return grid, components
        raise RuntimeError("failed to construct a separated connected-components grid")

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The supplied image shows one fixed 8 by 8 grid. Dark filled cells are foreground and white cells are empty. "
                "Small row and column numbers along the edges identify every cell, both starting at 1."
            ),
            symbol_legend=(
                "A dark navy cell is an unvisited filled cell.",
                "A white cell is empty and never belongs to a component.",
                "At the global commit, teal marks every filled cell after all complete components have been identified together.",
            ),
            rules=(
                "Two filled cells are connected only when they share an edge: up, down, left, or right.",
                "Diagonal contact alone does not connect cells.",
                "Every filled cell belongs to exactly one maximal orthogonally connected component.",
                "The final answer is the total number of those complete components.",
            ),
            modality_outputs={
                "ti2v": (
                    "Generate a video that keeps the input grid readable, then commits the complete component "
                    "assignment as one simultaneous global update in which every filled cell turns teal together."
                ),
                "solution_image": "Render the final grid with every complete 4-connected component in the stable counted color.",
            },
            preservation_constraints=(
                "Keep the 8 by 8 grid, every empty and filled cell position, the edge numbers, and the grid geometry unchanged.",
                "Change only filled-cell state colors.",
            ),
            transition_constraints=(
                "Commit the complete assignment as one simultaneous global update; component order is not semantic.",
                "Never show a state in which only a subset of complete components has changed color.",
                "Do not change any foreground/empty occupancy.",
                "Finish with every filled cell showing its stable counted color.",
            ),
            ti2v_continuity={
                "timing": "discrete",
                "moving_nouns": "the filled cells' state colors",
                "static_nouns": "the empty cells, the printed edge numbers, the grid frame, the background, and the camera",
                "hold_noun": "the exact final assignment",
            },
            answer=AnswerContract(
                kind="nonnegative_integer",
                format_template="Answer: <nonnegative integer>.",
                parser_profile="integer_answer",
                verifier_profile="four_connected_component_enumeration_replay",
                format_example="Answer: 99.",
                accepts_multiple_valid_answers=False,
            ),
            render_profile="connected_components_synchronized_global_commit_v6",
        )

    @staticmethod
    def _state_color(value: int) -> tuple[int, int, int]:
        if value == _EMPTY:
            return _GRID_FILL
        if value == _UNVISITED:
            return _UNVISITED_COLOR
        if value == _ACTIVE:
            return _ACTIVE_COLOR
        return _COUNTED_COLOR

    def _render_state(
        self,
        state: Sequence[Sequence[int]],
        *,
        state_index: int,
        component_count: int,
        focus_seed: tuple[int, int] | None = None,
        status_override: str | None = None,
        count_override: str | None = None,
    ) -> Image.Image:
        del state_index, component_count, status_override, count_override
        image, draw = new_state_canvas(_BG, _CARD, (216, 221, 218), card_box=(16, 14, 496, 498))
        small_font = scaled_font(10)

        for index in range(_SIZE):
            draw.text((_GRID_LEFT - 16, _GRID_TOP + (index + 0.5) * _CELL_SIZE), str(index + 1), fill=_MUTED, font=small_font, anchor="mm")
            draw.text((_GRID_LEFT + (index + 0.5) * _CELL_SIZE, _GRID_TOP - 13), str(index + 1), fill=_MUTED, font=small_font, anchor="mm")
        for row in range(_SIZE):
            for col in range(_SIZE):
                x0 = _GRID_LEFT + col * _CELL_SIZE
                y0 = _GRID_TOP + row * _CELL_SIZE
                x1 = x0 + _CELL_SIZE
                y1 = y0 + _CELL_SIZE
                value = int(state[row][col])
                draw.rectangle(
                    (x0, y0, x1, y1),
                    fill=self._state_color(value),
                    outline=_GRID_LINE,
                    width=2,
                )
        if focus_seed is not None:
            row, col = focus_seed
            x0 = _GRID_LEFT + col * _CELL_SIZE
            y0 = _GRID_TOP + row * _CELL_SIZE
            x1 = x0 + _CELL_SIZE
            y1 = y0 + _CELL_SIZE
            draw.rounded_rectangle(
                (x0 + 5, y0 + 5, x1 - 5, y1 - 5),
                radius=7,
                outline=_ACTIVE_COLOR,
                width=4,
            )
        draw.rounded_rectangle(
            (_GRID_LEFT - 6, _GRID_TOP - 6, _GRID_LEFT + _GRID_PX + 6, _GRID_TOP + _GRID_PX + 6),
            radius=13,
            outline=_ACCENT,
            width=3,
        )
        return finish_state_canvas(image)

    def _render_video(
        self,
        states: Sequence[Sequence[Sequence[int]]],
        *,
        output_path: Path,
    ) -> int:
        component_count = max(
            (int(value) for row in states[-1] for value in row if int(value) > 0),
            default=0,
        )
        rendered = [
            self._render_state(state, state_index=index, component_count=component_count)
            for index, state in enumerate(states)
        ]
        frames = [np.asarray(image, dtype=np.uint8) for image in discrete_video_frames(rendered)]
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)

    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1
        grid, components = self._sample_grid()
        states = _component_states(grid, components)
        answer = len(components)
        contract = self._contract()
        prompts = {f"{mode}_prompt": contract.build_prompt(mode) for mode in contract.prompt_modes()}
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid connected-components prompt contract: {'; '.join(issues)}")

        sample_dir = self.output_dir / sample_id
        puzzle_dir = sample_dir / "puzzles"
        solution_dir = sample_dir / "solutions"
        video_dir = sample_dir / "videos"
        for directory in (puzzle_dir, solution_dir, video_dir):
            directory.mkdir(parents=True, exist_ok=True)

        puzzle_image = self._render_state(states[0], state_index=0, component_count=answer)
        solution_image = self._render_state(states[-1], state_index=len(states) - 1, component_count=answer)
        puzzle_path = puzzle_dir / f"{sample_id}__puzzle.png"
        solution_path = solution_dir / f"{sample_id}__solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        solution_video_path: Path | None = None
        video_num_frames: int | None = None
        if self.video:
            video_path = video_dir / f"{sample_id}__solution.webm"
            video_num_frames = self._render_video(states, output_path=video_path)
            if video_num_frames:
                solution_video_path = video_path


        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_mechanic": self.task_mechanic,
            "task_contract": contract.to_dict(),
            "question": "Count the 4-connected foreground components in the indexed grid.",
            "answer": str(answer),
            "structured_answer": {
                "input_grid": grid,
                "components": [[[row, col] for row, col in component] for component in components],
                "component_count": answer,
                "component_states": states,
                "grid_bbox": [_GRID_LEFT, _GRID_TOP, _GRID_LEFT + _GRID_PX, _GRID_TOP + _GRID_PX],
                "cell_size": _CELL_SIZE,
                "connectivity": 4,
                "reference_component_order": None,
                "component_order_is_semantic": False,
                "task_mechanic": self.task_mechanic,
                "filled_cell_count": sum(sum(row) for row in grid),
                "component_sizes": [len(component) for component in components],
            },
            **prompts,
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path.as_posix() if solution_video_path else None,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }

    def make_sample(self) -> tuple[Dict[str, Any], Image.Image, Image.Image]:
        record = self.create_puzzle()
        puzzle = Image.open(record["image"]).convert("RGB").copy()
        solution = Image.open(record["solution_image_path"]).convert("RGB").copy()
        return record, puzzle, solution


__all__ = ["ConnectedComponentsPattern", "_connected_components_4", "_component_states"]

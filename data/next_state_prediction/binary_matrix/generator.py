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
    scaled_font,
)
from data.common.pipeline.structured_grid import matrix_to_json


pattern_instructions = {
    "binary_matrix": (
        "The input image shows a clean square binary matrix of 0s and 1s. For every cell, compute the Manhattan "
        "distance to the nearest 0 and return the full distance matrix."
    )
}

_INPUT_DESC = (
    "The input image shows one square binary matrix of source and non-source cells."
)
_SOLUTION_IMAGE_OUTPUT = "Render the complete distance matrix in the same fixed grid layout with every integer visible."
_TI2V_OUTPUT = (
    "Generate a video that replays the complete ordered distance-wave process. Commit each distance layer as one "
    "complete state: layer 0 shows every zero source as a purple 0 together, and each later layer settles the "
    "previous frontier to teal while all newly reached cells appear as the purple frontier together. Finish by "
    "committing the complete final matrix with every integer visible."
)

_BG = (247, 248, 250)
_CARD = (255, 255, 253)
_INK = (35, 40, 48)
_ZERO = (246, 247, 243)
_ONE = (61, 68, 87)
_UNDISCOVERED = (207, 213, 223)
_FRONTIER = (160, 98, 190)
_SETTLED = (67, 151, 166)


def _multi_source_zero_distance(matrix: List[List[int]]) -> List[List[int]]:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    distances = [[-1 for _ in range(cols)] for _ in range(rows)]
    queue: deque[Tuple[int, int]] = deque()
    for row in range(rows):
        for col in range(cols):
            if matrix[row][col] == 0:
                distances[row][col] = 0
                queue.append((row, col))
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = row + dr
            nc = col + dc
            if 0 <= nr < rows and 0 <= nc < cols and distances[nr][nc] == -1:
                distances[nr][nc] = distances[row][col] + 1
                queue.append((nr, nc))
    return distances


def _influential_zero_sources(matrix: List[List[int]]) -> set[Tuple[int, int]]:
    """Return zero sources that uniquely improve at least one non-source cell."""

    sources = [
        (row, col)
        for row, values in enumerate(matrix)
        for col, value in enumerate(values)
        if value == 0
    ]
    influential: set[Tuple[int, int]] = set()
    for row, values in enumerate(matrix):
        for col, value in enumerate(values):
            if value != 1:
                continue
            source_distances = [
                abs(row - source_row) + abs(col - source_col)
                for source_row, source_col in sources
            ]
            if not source_distances:
                continue
            minimum = min(source_distances)
            if source_distances.count(minimum) == 1:
                influential.add(sources[source_distances.index(minimum)])
    return influential


def _binary_distance_states(
    matrix: List[List[int]],
    distances: List[List[int]],
) -> List[tuple[str, int | None, List[List[str]]]]:
    size = len(matrix)
    states: List[tuple[str, int | None, List[List[str]]]] = [
        (
            "input",
            None,
            [["zero" if matrix[row][col] == 0 else "one" for col in range(size)] for row in range(size)],
        )
    ]
    max_distance = max((value for row in distances for value in row), default=0)
    # Every reached distance must appear as a frontier before the separate
    # all-settled terminal, including the farthest layer.
    for layer in range(max_distance + 1):
        states.append(
            (
                "layer",
                layer,
                [
                    [
                        "frontier"
                        if distances[row][col] == layer
                        else "settled"
                        if distances[row][col] < layer
                        else "undiscovered"
                        for col in range(size)
                    ]
                    for row in range(size)
                ],
            )
        )
    states.append(("final", None, [["settled" for _col in range(size)] for _row in range(size)]))
    return states


class BinaryMatrixPattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = "binary_matrix"
    task_family = "next_state_prediction"
    task_mechanic = "multi_source_manhattan_distance_wave"

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        size_range: Tuple[int, int] = (4, 6),
        video: bool = True,
        video_fps: int = 16,
    ) -> None:
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_binary_matrix"
        super().__init__(resolved)
        if len(size_range) != 2:
            raise ValueError("binary_matrix size_range must contain exactly two bounds")
        minimum_size, maximum_size = (int(size_range[0]), int(size_range[1]))
        if not 4 <= minimum_size <= maximum_size <= 8:
            raise ValueError("binary_matrix size_range must satisfy 4 <= minimum <= maximum <= 8")
        if int(video_fps) != 16:
            raise ValueError("binary_matrix videos must use 16 fps")
        self.rng = random.Random(seed)
        self.seed = seed
        self.size_range = (minimum_size, maximum_size)
        self.video = bool(video)
        self.video_fps = int(video_fps)
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _sample_matrix(self) -> List[List[int]]:
        size = self.rng.randint(*self.size_range)
        for _attempt in range(512):
            matrix = [
                [self.rng.choices([0, 1], weights=(3, 7), k=1)[0] for _ in range(size)]
                for _ in range(size)
            ]
            zero_count = sum(value == 0 for row in matrix for value in row)
            one_count = size * size - zero_count
            if zero_count < 2 or one_count < 4:
                continue
            distances = _multi_source_zero_distance(matrix)
            if (
                max((value for row in distances for value in row), default=0) >= 3
                and len(_influential_zero_sources(matrix)) >= 2
            ):
                return matrix

        matrix = [[1 for _ in range(size)] for _ in range(size)]
        matrix[0][0] = 0
        matrix[size - 1][size - 1] = 0
        fallback_distances = _multi_source_zero_distance(matrix)
        if (
            max((value for row in fallback_distances for value in row), default=0) < 3
            or len(_influential_zero_sources(matrix)) < 2
        ):
            raise RuntimeError("binary_matrix deterministic fallback violated its calibrated multi-source contract")
        return matrix

    @staticmethod
    def _grid_geometry(size: int) -> tuple[int, int, int, int]:
        cell_size = min(72, 312 // max(1, size))
        grid_size = cell_size * size
        left = (512 - grid_size) // 2
        top = 110 + (312 - grid_size) // 2
        return left, top, cell_size, grid_size

    def _render_state(
        self,
        matrix: List[List[int]],
        distances: List[List[int]],
        *,
        phase: str,
        layer: int | None = None,
    ) -> Image.Image:
        image, draw = new_state_canvas(_BG, _CARD, (217, 221, 225))
        value_font = scaled_font(18)

        size = len(matrix)
        left, top, cell_size, grid_size = self._grid_geometry(size)
        for row in range(size):
            for col in range(size):
                distance = int(distances[row][col])
                if phase == "input":
                    # The input encodes each cell's bit in its colour alone
                    #; the distance layers then print values.
                    category = "zero" if matrix[row][col] == 0 else "one"
                    label = ""
                elif phase == "layer":
                    if distance == layer:
                        category, label = "frontier", str(distance)
                    elif distance < int(layer or 0):
                        category, label = "settled", str(distance)
                    else:
                        category, label = "undiscovered", ""
                else:
                    category, label = "settled", str(distance)
                palette = {
                    "zero": _ZERO,
                    "one": _ONE,
                    "undiscovered": _UNDISCOVERED,
                    "frontier": _FRONTIER,
                    "settled": _SETTLED,
                }
                x0 = left + col * cell_size
                y0 = top + row * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                draw.rounded_rectangle(
                    (x0 + 2, y0 + 2, x1 - 2, y1 - 2),
                    radius=8,
                    fill=palette[category],
                    outline=(107, 55, 132) if category == "frontier" else (195, 201, 205),
                    width=3 if category == "frontier" else 2,
                )
                if label:
                    dark_text = category in {"zero", "undiscovered"}
                    draw.text(
                        ((x0 + x1) / 2, (y0 + y1) / 2 + 1),
                        label,
                        font=value_font,
                        fill=_INK if dark_text else (255, 255, 255),
                        anchor="mm",
                    )
        draw.rounded_rectangle(
            (left - 5, top - 5, left + grid_size + 5, top + grid_size + 5),
            radius=12,
            outline=(111, 119, 127),
            width=2,
        )

        return finish_state_canvas(image)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=_INPUT_DESC,
            symbol_legend=(
                "A pale cell = a 0 source cell whose distance is 0.",
                "A dark cell = a 1 cell whose distance to the nearest 0 must be computed.",
            ),
            rules=(
                "Every 0 cell is a simultaneous source at distance 0.",
                "Movement is allowed only one cell up, down, left, or right; diagonal movement is forbidden.",
                "There are no walls, so every cell is reachable from at least one 0.",
                "Each cell receives the minimum Manhattan distance to any 0 source.",
                "Because the grid has no walls, the only factors are the visible zero-source locations and orthogonal distance.",
            ),
            modality_outputs={
                "ti2v": _TI2V_OUTPUT,
                "solution_image": _SOLUTION_IMAGE_OUTPUT,
            },
            preservation_constraints=(
                "Keep the matrix size, the original source locations, and the grid geometry unchanged.",
                "Change only distance-state colors and displayed distance values; a settled value never changes again.",
            ),
            transition_constraints=(
                "Start with the supplied binary matrix, then commit Layer 0 with every 0 source in the purple frontier.",
                "Use purple only for the current frontier, teal for settled distances, and gray for cells not yet reached.",
                "Commit each next layer as one complete state: the previous frontier settles to teal and all newly reached cells appear in purple together.",
            ),
            ti2v_continuity={
                "timing": "discrete",
                "moving_nouns": "the distance layers and their printed values",
                "static_nouns": "the matrix frame, the background, and the camera",
                "hold_noun": "the exact final distance matrix",
            },
            answer=AnswerContract(
                kind="grid",
                format_template="Answer: [[<integer>, ...], ...].",
                parser_profile="json_integer_grid",
                verifier_profile="binary_matrix_zero_source_wave_replay",
                format_example="Answer: [[98, 99, ...], [100, 101, ...], ...].",
                accepts_multiple_valid_answers=False,
            ),
            render_profile="binary_matrix_zero_source_wave_v4",
        )

    def _prompts(self) -> tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {f"{mode}_prompt": contract.build_prompt(mode) for mode in contract.prompt_modes()}
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid binary-matrix prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _render_video(
        self,
        matrix: List[List[int]],
        distances: List[List[int]],
        states: List[tuple[str, int | None, List[List[str]]]],
        *,
        output_path: Path,
    ) -> int:
        renders = [
            self._render_state(matrix, distances, phase=phase, layer=layer)
            for phase, layer, _state in states
        ]
        frames = [np.asarray(image, dtype=np.uint8) for image in discrete_video_frames(renders)]
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)


    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1

        input_matrix = self._sample_matrix()
        answer_matrix = _multi_source_zero_distance(input_matrix)
        states = _binary_distance_states(input_matrix, answer_matrix)
        answer_raw = matrix_to_json(answer_matrix)
        contract, prompts = self._prompts()

        sample_dir = self.output_dir / sample_id
        puzzle_dir = sample_dir / "puzzles"
        solution_dir = sample_dir / "solutions"
        video_dir = sample_dir / "videos"
        steps_dir = sample_dir / "steps"
        for directory in (puzzle_dir, solution_dir, video_dir, steps_dir):
            directory.mkdir(parents=True, exist_ok=True)

        puzzle_image = self._render_state(input_matrix, answer_matrix, phase="input")
        solution_image = self._render_state(input_matrix, answer_matrix, phase="final")
        puzzle_path = puzzle_dir / f"{sample_id}__puzzle.png"
        solution_path = solution_dir / f"{sample_id}__solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        solution_video_path = None
        video_num_frames = None
        if self.video:
            video_path = video_dir / f"{sample_id}__solution.webm"
            video_num_frames = self._render_video(input_matrix, answer_matrix, states, output_path=video_path)
            if video_num_frames > 0:
                solution_video_path = video_path

        size = len(input_matrix)
        left, top, cell_size, grid_size = self._grid_geometry(size)
        max_distance = max((value for row in answer_matrix for value in row), default=0)
        source_count = sum(value == 0 for row in input_matrix for value in row)
        influential_source_count = len(_influential_zero_sources(input_matrix))
        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": "Return the Manhattan distance to the nearest 0 for every cell in the matrix.",
            "answer": answer_raw,
            "structured_answer": {
                "input_matrix": input_matrix,
                "distance_matrix": answer_matrix,
                "max_distance": max_distance,
                "source_count": source_count,
                "influential_source_count": influential_source_count,
                "grid_bbox": [left, top, left + grid_size, top + grid_size],
                "cell_size": cell_size,
                "difficulty_variant": "open_grid",
                "task_mechanic": self.task_mechanic,
                "intentional_overlap_with": "grid_bfs",
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

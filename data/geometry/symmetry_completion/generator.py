"""Complete a half-empty occupancy grid by reflection across a given axis.

The visible source half and the mirror axis determine one exact target-half
occupancy mask.  Four patch candidates are shown as independent hypothesis
plates in the same target-half coordinates; exactly one contains every
required reflected cell and no extra cell.  The axis always lies *between*
two central rows or columns, so no grid cell may sit on the axis.
"""

from __future__ import annotations

from data.common.base import RenderedTaskGenerator
from data.geometry.core.rendered_contract import finalize_geometry_sample

import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.grid_coordinates import GRID_XY_CONVENTION_SENTENCE
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.pipeline.media import LETTER_OPTIONS, load_font
from data.common.choice_cards import attach_visual_choice_card_inputs
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC


pattern_instructions = {
    "symmetry_completion": (
        (
            "A full-canvas even square grid is split by a horizontal or vertical mirror axis. One half contains "
            "a visible occupancy pattern and the other half is empty. Independent A-D plates propose the missing half; "
            "exactly one completes the grid by reflection without adding or omitting a cell."
        )
    ),
}

Cell = Tuple[int, int]
CellSet = Tuple[Cell, ...]

AXES: Tuple[str, ...] = ("vertical", "horizontal")
GRID_SIZES: Tuple[int, ...] = (6, 8)
_LABELS = ("A", "B", "C", "D")

_BG = (255, 255, 255)
_PANEL = (255, 255, 255)
_GRID = (198, 208, 221)
_TEXT = (35, 43, 55)
_AXIS = (210, 72, 72)
_FILLS: Tuple[Tuple[int, int, int], ...] = (
    (61, 111, 181),
    (45, 145, 108),
    (205, 117, 57),
    (137, 85, 180),
    (195, 72, 96),
)

_BOARD_BOX = (66, 66, 446, 446)
_PLATE_CONTENT_BOX = (76, 40, 436, 420)
_PLATE_LABEL_CENTER = (256, 474)


def _normalise_cells(cells: Iterable[Sequence[int]]) -> CellSet:
    return tuple(sorted((int(cell[0]), int(cell[1])) for cell in cells))


def _source_side(cell: Cell, grid_size: int, axis: str) -> bool:
    row, col = cell
    half = grid_size // 2
    if axis == "vertical":
        return col < half
    if axis == "horizontal":
        return row < half
    raise ValueError(f"Unsupported symmetry axis: {axis}")


def _target_side(cell: Cell, grid_size: int, axis: str) -> bool:
    row, col = cell
    half = grid_size // 2
    if axis == "vertical":
        return col >= half
    if axis == "horizontal":
        return row >= half
    raise ValueError(f"Unsupported symmetry axis: {axis}")


def _validate_grid(grid_size: int) -> None:
    if grid_size not in (*GRID_SIZES, 10, 12, 14, 16) or grid_size % 2:
        raise ValueError(
            "symmetry_completion requires an even grid of 6..16 cells with the axis between cells"
        )


def _validate_source_cells(
    cells: Iterable[Sequence[int]], grid_size: int, axis: str
) -> CellSet:
    _validate_grid(grid_size)
    points = [(int(cell[0]), int(cell[1])) for cell in cells]
    if not 5 <= len(points) <= 24:
        raise ValueError(
            "symmetry_completion source must contain 5 to 24 occupied cells"
        )
    if len(set(points)) != len(points):
        raise ValueError("symmetry_completion source cannot contain duplicate cells")
    for row, col in points:
        if not (0 <= row < grid_size and 0 <= col < grid_size):
            raise ValueError("symmetry_completion source cell lies outside the grid")
        if not _source_side((row, col), grid_size, axis):
            raise ValueError("symmetry_completion source cells must stay strictly on the visible side of the axis")
    remaining = set(points)
    frontier = {points[0]}
    visited: set[Cell] = set()
    while frontier:
        cell = frontier.pop()
        if cell in visited:
            continue
        visited.add(cell)
        row, col = cell
        frontier.update(
            neighbor
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
            if neighbor in remaining and neighbor not in visited
        )
    if visited != remaining:
        raise ValueError("symmetry_completion source cells must be edge-connected")
    if len({row for row, _col in points}) < 2 or len({col for _row, col in points}) < 2:
        raise ValueError("symmetry_completion source must span at least two rows and two columns")
    half = grid_size // 2
    depth_values = [col if axis == "vertical" else row for row, col in points]
    if half - 1 not in depth_values or min(depth_values) >= half - 1:
        raise ValueError("symmetry_completion source must touch the axis edge and extend away from it")
    return _normalise_cells(points)


def _validate_candidate_cells(
    cells: Iterable[Sequence[int]], grid_size: int, axis: str
) -> CellSet:
    _validate_grid(grid_size)
    points = [(int(cell[0]), int(cell[1])) for cell in cells]
    if not points:
        raise ValueError("symmetry_completion candidate patch cannot be empty")
    if len(points) > 25:
        raise ValueError("symmetry_completion candidate patch is too dense")
    if len(set(points)) != len(points):
        raise ValueError("symmetry_completion candidate patch cannot contain duplicate cells")
    for row, col in points:
        if not (0 <= row < grid_size and 0 <= col < grid_size):
            raise ValueError("symmetry_completion candidate cell lies outside the grid")
        if not _target_side((row, col), grid_size, axis):
            raise ValueError("symmetry_completion candidates must stay strictly on the empty side of the axis")
    return _normalise_cells(points)


def _reflect_cell(cell: Sequence[int], grid_size: int, axis: str) -> Cell:
    _validate_grid(grid_size)
    row, col = int(cell[0]), int(cell[1])
    if not (0 <= row < grid_size and 0 <= col < grid_size):
        raise ValueError("symmetry_completion reflection input lies outside the grid")
    if axis == "vertical":
        return row, grid_size - 1 - col
    if axis == "horizontal":
        return grid_size - 1 - row, col
    raise ValueError(f"Unsupported symmetry axis: {axis}")


def _expected_completion(cells: Iterable[Sequence[int]], grid_size: int, axis: str) -> CellSet:
    source = _validate_source_cells(cells, grid_size, axis)
    expected = _normalise_cells(_reflect_cell(cell, grid_size, axis) for cell in source)
    return _validate_candidate_cells(expected, grid_size, axis)


def _candidate_flags(
    source_cells: Iterable[Sequence[int]],
    candidates: Sequence[Iterable[Sequence[int]]],
    grid_size: int,
    axis: str,
) -> List[bool]:
    expected = _expected_completion(source_cells, grid_size, axis)
    return [_validate_candidate_cells(candidate, grid_size, axis) == expected for candidate in candidates]


def _validate_task(task: Dict[str, Any]) -> None:
    grid_size = int(task["grid_size"])
    axis = str(task["axis"])
    source = _validate_source_cells(task["source_cells"], grid_size, axis)
    expected = _expected_completion(source, grid_size, axis)
    if _normalise_cells(task["expected_completion_cells"]) != expected:
        raise ValueError("symmetry_completion expected completion does not replay from the visible cells")
    candidates = [task["correct_cells"], *task["distractor_cells"]]
    normalized = [_validate_candidate_cells(candidate, grid_size, axis) for candidate in candidates]
    if len(set(normalized)) != 4:
        raise ValueError("symmetry_completion candidates must be four distinct patches")
    flags = _candidate_flags(source, normalized, grid_size, axis)
    if flags != [True, False, False, False]:
        raise ValueError(f"symmetry_completion requires one unique completion candidate, got {flags}")


def _grow_source(rng: random.Random, grid_size: int, axis: str, count: int) -> List[Cell]:
    half = grid_size // 2
    if axis == "vertical":
        start = (rng.randrange(grid_size), half - 1)
    else:
        start = (half - 1, rng.randrange(grid_size))
    cells = {start}
    for _attempt in range(600):
        if len(cells) >= count:
            break
        row, col = rng.choice(tuple(sorted(cells)))
        delta_row, delta_col = rng.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))
        candidate = (row + delta_row, col + delta_col)
        if 0 <= candidate[0] < grid_size and 0 <= candidate[1] < grid_size and _source_side(candidate, grid_size, axis):
            cells.add(candidate)
    return list(cells)


def _target_cells(grid_size: int, axis: str) -> List[Cell]:
    return [
        (row, col)
        for row in range(grid_size)
        for col in range(grid_size)
        if _target_side((row, col), grid_size, axis)
    ]


def _mutate_completion(rng: random.Random, expected: CellSet, grid_size: int, axis: str) -> CellSet:
    result = set(expected)
    target = _target_cells(grid_size, axis)
    operation = rng.choice(("swap", "drop", "add", "parallel_flip", "depth_shift"))
    if operation == "drop" and len(result) > 1:
        result.remove(rng.choice(tuple(sorted(result))))
    elif operation == "add":
        blanks = [cell for cell in target if cell not in result]
        if blanks:
            result.add(rng.choice(blanks))
    elif operation == "parallel_flip":
        if axis == "vertical":
            result = {(grid_size - 1 - row, col) for row, col in result}
        else:
            result = {(row, grid_size - 1 - col) for row, col in result}
    elif operation == "depth_shift":
        delta = rng.choice((-1, 1))
        if axis == "vertical":
            shifted = {(row, col + delta) for row, col in result}
        else:
            shifted = {(row + delta, col) for row, col in result}
        if shifted and all(cell in target for cell in shifted):
            result = shifted
    else:
        removed = rng.choice(tuple(sorted(result)))
        blanks = [cell for cell in target if cell not in result]
        if blanks:
            result.remove(removed)
            result.add(rng.choice(blanks))
    return _normalise_cells(result)


class SymmetryCompletionPattern(RenderedTaskGenerator):
    task_name = "symmetry_completion"
    task_family = "geometry"
    video_fps = 16
    generate_videos = True

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        image_size: int | Sequence[int] = 512,
        grid_size: int | None = None,
        source_count: int | None = None,
        axis: str | None = None,
        video_fps: int = 16,
        generate_videos: bool | None = None,
        video: bool | None = None,
        difficulty: str = "medium",
        **_kwargs: Any,
    ) -> None:
        size = (int(image_size), int(image_size)) if isinstance(image_size, int) else tuple(int(value) for value in image_size)
        if size != (512, 512):
            raise ValueError("symmetry_completion requires a 512x512 canvas")
        if int(video_fps) != 16:
            raise ValueError("symmetry_completion requires video_fps=16")
        if grid_size is not None:
            _validate_grid(int(grid_size))
        if axis is not None and axis not in AXES:
            raise ValueError(f"symmetry_completion axis must be one of {AXES}")
        resolved_seed = random_seed if random_seed is not None else seed
        self.rng = random.Random(resolved_seed)
        self.image_size = size
        self.grid_size = int(grid_size) if grid_size is not None else None
        self.axis = axis
        self.source_count = source_count
        self.difficulty = str(difficulty)
        self.video_fps = 16
        if generate_videos is None:
            generate_videos = self.generate_videos if video is None else bool(video)
        self.generate_videos = bool(generate_videos)
        self.output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_symmetry_completion"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0

    def _generate_task(self) -> Dict[str, Any]:
        for _attempt in range(800):
            grid_size = self.grid_size or self.rng.choice(GRID_SIZES)
            axis = self.axis or self.rng.choice(AXES)
            max_count = 8 if grid_size == 6 else 10
            cell_count = (
                self.rng.randint(5, max_count)
                if self.source_count is None
                else int(self.source_count)
            )
            source = _grow_source(self.rng, grid_size, axis, cell_count)
            try:
                source_norm = _validate_source_cells(source, grid_size, axis)
            except ValueError:
                continue
            expected = _expected_completion(source_norm, grid_size, axis)
            distractors: List[CellSet] = []
            seen = {expected}
            for _ in range(500):
                candidate = _mutate_completion(self.rng, expected, grid_size, axis)
                try:
                    candidate = _validate_candidate_cells(candidate, grid_size, axis)
                except ValueError:
                    continue
                if candidate in seen:
                    continue
                seen.add(candidate)
                distractors.append(candidate)
                if len(distractors) == 3:
                    break
            if len(distractors) != 3:
                continue
            task = {
                "grid_size": grid_size,
                "axis": axis,
                "source_cells": source_norm,
                "expected_completion_cells": expected,
                "correct_cells": expected,
                "distractor_cells": distractors,
                "fill_color": self.rng.choice(_FILLS),
            }
            _validate_task(task)
            return task
        raise RuntimeError("symmetry_completion: failed to generate a valid unique instance")

    @staticmethod
    def _state_trace(expected_count: int) -> List[Dict[str, Any]]:
        return [
            {"phase": "input", "global_fill_progress": 0.0, "reveal_progress": 0.0},
            {"phase": "global_reflection", "global_fill_progress": 1.0, "reveal_progress": float(expected_count)},
        ]

    @staticmethod
    def _board_layout(grid_size: int) -> Tuple[int, int, int]:
        left, top, right, bottom = _BOARD_BOX
        cell_size = min((right - left) // grid_size, (bottom - top) // grid_size)
        side = cell_size * grid_size
        board_left = left + ((right - left) - side) // 2
        board_top = top + ((bottom - top) - side) // 2
        return board_left, board_top, cell_size

    @classmethod
    def _draw_grid(
        cls,
        draw: Any,
        grid_size: int,
        axis: str,
        source_cells: CellSet,
        completion_cells: CellSet,
        fill_color: Tuple[int, int, int],
        state: Dict[str, Any],
    ) -> None:
        board_left, board_top, cell_size = cls._board_layout(grid_size)
        side = cell_size * grid_size
        label_font = load_font(15 * DOCTRINE_SUPERSAMPLE)
        for index in range(grid_size):
            center = board_left + (index + 0.5) * cell_size
            draw.text((center, board_top - 24), str(index + 1), fill=_TEXT, font=label_font, anchor="mm")
            middle = board_top + (index + 0.5) * cell_size
            draw.text((board_left - 24, middle), str(index + 1), fill=_TEXT, font=label_font, anchor="mm")
        source_lookup = set(source_cells)
        completion_lookup = set(completion_cells)
        global_fill_progress = max(
            0.0,
            min(
                1.0,
                float(
                    state.get(
                        "global_fill_progress",
                        1.0 if float(state.get("reveal_progress", 0.0)) >= len(completion_lookup) else 0.0,
                    )
                ),
            ),
        )

        def blend(start: Sequence[int], end: Sequence[int], fraction: float) -> Tuple[int, int, int]:
            return tuple(int(round(a + (b - a) * fraction)) for a, b in zip(start, end))

        for row in range(grid_size):
            for col in range(grid_size):
                x0 = board_left + col * cell_size
                y0 = board_top + row * cell_size
                x1 = board_left + (col + 1) * cell_size
                y1 = board_top + (row + 1) * cell_size
                cell = (row, col)
                fill = _PANEL
                if cell in source_lookup:
                    fill = fill_color
                elif cell in completion_lookup:
                    fill = blend(_PANEL, fill_color, global_fill_progress)
                draw.rectangle((x0, y0, x1, y1), fill=fill, outline=_GRID, width=2)
        axis_x = board_left + side // 2
        axis_y = board_top + side // 2
        if axis == "vertical":
            draw.line((axis_x, board_top - 8, axis_x, board_top + side + 8), fill=_AXIS, width=5)
        else:
            draw.line((board_left - 8, axis_y, board_left + side + 8, axis_y), fill=_AXIS, width=5)

    def _render_scene(
        self,
        task: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Image.Image:
        grid_size = int(task["grid_size"])
        axis = str(task["axis"])
        image, draw = new_scaled_canvas(512, 512, DOCTRINE_SUPERSAMPLE, background=_BG)
        self._draw_grid(
            draw,
            grid_size,
            axis,
            task["source_cells"],
            task["expected_completion_cells"],
            task["fill_color"],
            state,
        )
        return finish_scaled_canvas(image, (512, 512))

    def _render_option_plate(
        self,
        task: Dict[str, Any],
        cells: CellSet,
        label: str,
    ) -> Image.Image:
        grid_size = int(task["grid_size"])
        axis = str(task["axis"])
        fill_color = tuple(task["fill_color"])
        image, draw = new_scaled_canvas(512, 512, DOCTRINE_SUPERSAMPLE, background=_BG)
        left, top, right, bottom = _PLATE_CONTENT_BOX
        half = grid_size // 2
        rows, cols = (grid_size, half) if axis == "vertical" else (half, grid_size)
        cell_size = min((right - left) // cols, (bottom - top) // rows)
        width, height = cell_size * cols, cell_size * rows
        patch_left = left + ((right - left) - width) // 2
        patch_top = top + ((bottom - top) - height) // 2
        lookup = set(cells)
        for local_row in range(rows):
            for local_col in range(cols):
                row = local_row if axis == "vertical" else local_row + half
                col = local_col + half if axis == "vertical" else local_col
                x0 = patch_left + local_col * cell_size
                y0 = patch_top + local_row * cell_size
                x1 = patch_left + (local_col + 1) * cell_size
                y1 = patch_top + (local_row + 1) * cell_size
                draw.rectangle((x0, y0, x1, y1), fill=fill_color if (row, col) in lookup else _PANEL, outline=_GRID, width=2)
        factor = DOCTRINE_SUPERSAMPLE
        index_font = load_font(15 * factor)
        for local_col in range(cols):
            col = local_col + half if axis == "vertical" else local_col
            center = patch_left + (local_col + 0.5) * cell_size
            draw.text((center, patch_top - 22), str(col + 1), fill=_TEXT, font=index_font, anchor="mm")
        for local_row in range(rows):
            row = local_row if axis == "vertical" else local_row + half
            middle = patch_top + (local_row + 0.5) * cell_size
            draw.text((patch_left - 24, middle), str(row + 1), fill=_TEXT, font=index_font, anchor="mm")
        if axis == "vertical":
            draw.line((patch_left, patch_top - 6, patch_left, patch_top + height + 6), fill=_AXIS, width=5)
        else:
            draw.line((patch_left - 6, patch_top, patch_left + width + 6, patch_top), fill=_AXIS, width=5)
        draw.text(_PLATE_LABEL_CENTER, label, fill=_TEXT, font=load_font(30 * factor), anchor="mm")
        return finish_scaled_canvas(image, (512, 512))

    def _render_video(self, task: Dict[str, Any], output_path: Path) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        count = len(task["expected_completion_cells"])
        trace = self._state_trace(count)
        initial = np.asarray(self._render_scene(task, trace[0]), dtype=np.uint8)
        final = np.asarray(self._render_scene(task, trace[-1]), dtype=np.uint8)
        frames: List[np.ndarray] = [initial.copy() for _ in range(5)]
        for progress in (0.25, 0.50, 0.75, 1.0):
            frames.append(
                np.asarray(
                    self._render_scene(
                        task,
                        {
                            "phase": "global_reflection",
                            "global_fill_progress": progress,
                            "reveal_progress": float(count) * progress,
                        },
                    ),
                    dtype=np.uint8,
                )
            )
        frames.extend(final.copy() for _ in range(8))
        if len(frames) != 17 or (len(frames) - 1) % 4 != 0:
            raise AssertionError("symmetry_completion video must use semantic 4N+1 phases")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    @staticmethod
    def _contract() -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The single image shows an even occupancy grid split by a red mirror axis; colored cells occupy only one "
                f"side of the axis and every cell on the other side is empty. {GRID_XY_CONVENTION_SENTENCE}"
            ),
            modality_input_descriptions={
            },
            symbol_legend=(
                "Colored unit cells are occupied and white cells are empty.",
                "The numbers along the top and left edges instantiate that grid-position convention; candidate patches repeat the indices they cover.",
                "The red line is the given mirror axis and always lies between cells; no cell lies on the axis.",
            ),
            rules=(
                "Reflect every occupied source cell perpendicularly across the red axis: keep its coordinate parallel to the axis unchanged and place it at the equal grid distance on the empty side.",
                "The correct patch contains every reflected cell and no extra cell; missing, shifted, duplicated, source-side, or axis cells are invalid.",
                "Exactly one distinct candidate completes the symmetric grid.",
            ),
            modality_outputs={
                "ti2v": "Hold the input, fade all reflected target cells in together with one shared fill progress, then hold the completed symmetric grid.",
                "solution_image": "Show the completed symmetric grid with every reflected cell filled and nothing else changed.",
            },
            preservation_constraints=(
                "Keep the camera, grid lines, axis, visible source cells, colors, and framing fixed.",
                "Do not move, erase, duplicate, rotate, resize, or recolor any source cell.",
            ),
            transition_constraints=(
                "Begin with the empty target half held unchanged; every reflected cell must use the same smooth fill progress, with no spatial scan.",
                "After the single global fill completes, hold the completed symmetric grid without further change.",
            ),
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="uppercase_option_letter",
                verifier_profile="grid_reflection_replay_exact_patch_and_unique_match",
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_full_canvas_grid_symmetry_completion_v4",
        )

    def _prompts(self) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "question": "Choose the unique target-half patch that exactly completes the occupancy grid by reflection across the given red axis.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid symmetry-completion prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        base_candidates = [task["correct_cells"], *task["distractor_cells"]]
        order = list(range(4))
        self.rng.shuffle(order)
        ordered_candidates = [base_candidates[index] for index in order]
        flags = _candidate_flags(task["source_cells"], ordered_candidates, task["grid_size"], task["axis"])
        if flags.count(True) != 1:
            raise RuntimeError("symmetry_completion ordered candidates lost unique-match semantics")
        correct_index = flags.index(True)
        correct_option = _LABELS[correct_index]

        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"
        trace = self._state_trace(len(task["expected_completion_cells"]))
        puzzle = self._render_scene(task, trace[0])
        solution = self._render_scene(task, trace[-1])
        puzzle.save(puzzle_path)
        solution.save(solution_path)
        option_paths: List[str] = []
        for index, candidate in enumerate(ordered_candidates):
            option_path = sample_dir / f"{sample_id}__option_{_LABELS[index]}.png"
            self._render_option_plate(task, candidate, _LABELS[index]).save(option_path)
            option_paths.append(option_path.as_posix())
        solution_video_path, video_num_frames = self._render_video(task, video_path)

        contract, prompts = self._prompts()
        candidates_payload = [
            {
                "label": LETTER_OPTIONS[index],
                "cells": [list(cell) for cell in candidate],
                "matches_completion": flags[index],
            }
            for index, candidate in enumerate(ordered_candidates)
        ]
        record: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": prompts["question"],
            "answer": correct_option,
            "correct_option": correct_option,
            "structured_answer": {
                "correct_option": correct_option,
                "correct_index": correct_index,
                "axis": task["axis"],
                "axis_location": "between_central_columns" if task["axis"] == "vertical" else "between_central_rows",
                "grid_size": task["grid_size"],
                "source_cells": [list(cell) for cell in task["source_cells"]],
                "expected_completion_cells": [list(cell) for cell in task["expected_completion_cells"]],
                "candidates": candidates_payload,
                "match_flags": flags,
                "state_trace": trace,
                "boundary_rules": {
                    "axis_between_cells": True,
                    "axis_cells_allowed": False,
                    "source_cells_must_remain_fixed": True,
                    "candidate_must_equal_exact_reflection": True,
                },
                "render_geometry": {
                    "canvas_size": [512, 512],
                    "board_box": list(_BOARD_BOX),
                    "plate_content_box": list(_PLATE_CONTENT_BOX),
                    "fill_rgb": list(task["fill_color"]),
                    "axis_rgb": list(_AXIS),
                },
            },
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "reasoning_images": [],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }
        attach_visual_choice_card_inputs(
            record,
            problem_image=puzzle_path.as_posix(),
            option_images=option_paths,
            option_labels=_LABELS,
            ti2v_problem_image=puzzle_path.as_posix(),
        )
        return finalize_geometry_sample(record), puzzle, solution

    def create_puzzle(self) -> Dict[str, Any]:
        return self.make_sample()[0]


__all__ = [
    "AXES",
    "GRID_SIZES",
    "SymmetryCompletionPattern",
    "np",
    "_candidate_flags",
    "_expected_completion",
    "_normalise_cells",
    "_reflect_cell",
    "_validate_candidate_cells",
    "_validate_source_cells",
    "_validate_task",
]

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.base import AbstractPuzzleGenerator
from data.common.animation_policy import (
    DISCRETE_STATE_HOLD_FRAMES,
    end_hold_frames,
    start_hold_frames,
)
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.video_encoding import encode_rgb_frames_to_webm, pad_frames_to_wan_compatible_length
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.scalar_lists import (
    format_nonnegative_integer_list_answer,
)
from data.common.pipeline.path_tasks import draw_path_line
from data.common.pipeline.structured_grid import (
    CANVAS_PX,
    GRID_ACCENT,
    GRID_ACCENT_FILL,
    GRID_CELL_FILL,
    GRID_INK,
    GRID_MINOR_LINE,
    GRID_OUTER_LINE,
    GRID_PAGE_BG,
    INTEGER_GRID_CELL_PX,
    IntegerGridGeometry,
    draw_grid_lines,
    draw_text_centered,
    fill_cell,
    integer_grid_font_size,
    integer_grid_geometry,
    integer_grid_token_window,
    load_font,
)


pattern_instructions = {
    "spiral_matrix": (
        "The input image shows a square matrix. Read the entries in clockwise spiral order starting from the top-left "
        "cell and return the full sequence."
    )
}

_INPUT_DESC = "A square matrix of distinct two-digit values is shown."
_TI2T_OUTPUT = "Return the typed result as `Answer: [<nonnegative integer>, ...].`"
_TERMINAL_DELIVERY = "every cell read and the complete traced reading path"
_SOLUTION_IMAGE_OUTPUT = (
    f"Return one terminal image showing {_TERMINAL_DELIVERY}. Keep every printed value and the grid unchanged."
)
_TI2V_PROCESS = (
    "Step the reading position from cell to cell in clockwise outer-to-inner order as exact discrete states of equal "
    "duration, extending the traced path by one whole segment on each step, and hold the completed path."
)
_TI2V_OUTPUT = (
    f"Return one continuous fixed-camera video. {_TI2V_PROCESS} Preserve unchanged objects throughout; reject "
    "teleportation, hard cuts, whole-frame crossfades, flicker, disappearance, identity swaps, camera drift, and "
    "undeclared mutations."
)
_VERIFIER_PROFILE = "spiral_matrix_printed_number_path_source_replay_v5"
_VERIFIER_COMPONENTS = (
    "parse_score",
    "source_replay_score",
    "coordinate_coverage_score",
    "adjacency_score",
    "path_segment_score",
    "value_preservation_score",
    "visual_parseability_score",
    "state_order_score",
    "state_cadence_score",
    "preservation_score",
    "final_state_score",
    "media_contract_score",
)
_CANVAS = (CANVAS_PX, CANVAS_PX)

# The board carries three cell states.  Unvisited and the current-cell accent
# come straight from the shared integer-grid palette; the "already read" tint is
# the one colour this task adds, and it deliberately matches the weight of the
# unvisited fill so the accent stays the single loudest mark on the board.
_VISITED_FILL = (226, 240, 228)
# The traced reading path.  Its luminance (134) stays above the printed-glyph
# ink threshold, so a segment crossing a cell centre can never be mistaken for
# part of that cell's digits by a template match.
_PATH_COLOR = (96, 140, 190)
_PATH_WIDTH = 6


def _spiral_coordinates(size: int) -> List[Tuple[int, int]]:
    if size <= 0:
        raise ValueError("spiral size must be positive")
    coordinates: List[Tuple[int, int]] = []
    top = 0
    bottom = size - 1
    left = 0
    right = size - 1
    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            coordinates.append((top, col))
        top += 1
        for row in range(top, bottom + 1):
            coordinates.append((row, right))
        right -= 1
        if top <= bottom:
            for col in range(right, left - 1, -1):
                coordinates.append((bottom, col))
            bottom -= 1
        if left <= right:
            for row in range(bottom, top - 1, -1):
                coordinates.append((row, left))
            left += 1
    return coordinates


def _spiral_order(matrix: List[List[int]]) -> List[int]:
    return [matrix[row][col] for row, col in _spiral_coordinates(len(matrix))]


def _spiral_visual_states(size: int) -> List[tuple[str, int | None]]:
    """Exact discrete semantic states: the input board, each read, the settled board."""

    states: List[tuple[str, int | None]] = [("input", None)]
    states.extend(("step", step_index) for step_index in range(len(_spiral_coordinates(size))))
    states.append(("final", None))
    return states


def _spiral_checkpoint_steps(size: int) -> List[int]:
    """Return natural cumulative checkpoints: completed rings, then final delivery."""
    checkpoints: List[int] = []
    consumed = 0
    remaining = int(size)
    while remaining > 2:
        perimeter = 1 if remaining == 1 else 4 * remaining - 4
        consumed += perimeter
        checkpoints.append(consumed - 1)
        remaining -= 2
    return checkpoints


def spiral_grid_geometry(size: int) -> IntegerGridGeometry:
    """Centre the shared integer board on both axes; this task has no header."""

    board_px = INTEGER_GRID_CELL_PX[int(size)] * int(size)
    return integer_grid_geometry(size, top=(CANVAS_PX - board_px) // 2)


def _edge_runs(matrix: Sequence[Sequence[int]], coordinates: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Split one boundary's values into its straight edge runs."""

    if not coordinates:
        return []
    first_row, first_col = coordinates[0]
    runs: List[List[int]] = [[matrix[first_row][first_col]]]
    previous_step: Tuple[int, int] | None = None
    for index, (row, col) in enumerate(coordinates[1:], start=1):
        previous_row, previous_col = coordinates[index - 1]
        step = (row - previous_row, col - previous_col)
        # The first move sets the opening run's direction rather than ending it.
        if previous_step is not None and step != previous_step:
            runs.append([])
        runs[-1].append(matrix[row][col])
        previous_step = step
    return runs


def _join_values(values: Sequence[int]) -> str:
    text = [str(int(value)) for value in values]
    if len(text) == 1:
        return text[0]
    return f"{', '.join(text[:-1])} and {text[-1]}"


def spiral_ring_sentence(matrix: Sequence[Sequence[int]], ring_index: int) -> str:
    """Sample-specific reasoning for one completed boundary, in visible values only."""

    size = len(matrix)
    coordinates = _spiral_coordinates(size)
    checkpoints = _spiral_checkpoint_steps(size)
    end_step = checkpoints[ring_index]
    start_step = 0 if ring_index == 0 else checkpoints[ring_index - 1] + 1
    runs = _edge_runs(matrix, coordinates[start_step : end_step + 1])
    edges = ("top edge", "right edge", "bottom edge", "left edge")
    described = ", then ".join(
        f"the {edge} gives {_join_values(run)}" for edge, run in zip(edges, runs)
    )
    opening = (
        "Reading the outer boundary clockwise"
        if ring_index == 0
        else "Continuing clockwise around the next boundary"
    )
    remaining = len(coordinates) - (end_step + 1)
    inner = "one cell" if remaining == 1 else f"an inner block of {remaining} cells"
    return (
        f"{opening}, {described}. That closes the boundary after {end_step + 1} values and leaves {inner} unread."
    )


def _spiral_tail_values(matrix: Sequence[Sequence[int]]) -> List[int]:
    """Values still unread after the last completed-boundary checkpoint."""

    coordinates = _spiral_coordinates(len(matrix))
    checkpoints = _spiral_checkpoint_steps(len(matrix))
    tail_start = checkpoints[-1] + 1 if checkpoints else 0
    return [matrix[row][col] for row, col in coordinates[tail_start:]]


def spiral_closing_sentence(matrix: Sequence[Sequence[int]]) -> str:
    """Finish the spiral trace from the last committed boundary state.

    Every earlier boundary is already narrated caption by caption above its
    cumulative image, so this block only reads what that last committed state
    leaves unread instead of re-listing the narrated values.
    """

    tail_values = _spiral_tail_values(matrix)
    if len(tail_values) == 1:
        return (
            f"After that committed boundary, only the middle cell is left, so the reading ends on {tail_values[0]}."
        )
    return (
        f"After that committed boundary, only the inner block remains, and the same rule reads it as "
        f"{_join_values(tail_values)} to end the traversal."
    )


class SpiralMatrixPattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = "spiral_matrix"
    task_family = "in_context_learning"

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        size_range: tuple[int, int] = (3, 5),
        video: bool = True,
        video_fps: int = 16,
        generate_videos: bool | None = None,
        random_seed: int | None = None,
    ) -> None:
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_spiral_matrix"
        super().__init__(resolved)
        resolved_seed = seed if seed is not None else random_seed
        self.rng = random.Random(resolved_seed)
        self.size_range = (int(size_range[0]), int(size_range[1]))
        if self.size_range[0] < 3 or self.size_range[1] > 5 or self.size_range[0] > self.size_range[1]:
            raise ValueError("spiral_matrix size_range must stay within 3..5")
        self.video = bool(video if generate_videos is None else generate_videos)
        self.video_fps = int(video_fps)
        if self.video_fps != 16:
            raise ValueError("spiral_matrix videos must use 16fps")
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _sample_matrix(self) -> List[List[int]]:
        size = self.rng.randint(*self.size_range)
        values = self.rng.sample(range(10, 99), k=size * size)
        return [values[index * size:(index + 1) * size] for index in range(size)]

    def _render_state(
        self,
        matrix: List[List[int]],
        *,
        step_index: int | None = None,
        final: bool = False,
    ) -> Image.Image:
        size = len(matrix)
        geometry = spiral_grid_geometry(size)
        coordinates = _spiral_coordinates(size)
        if final:
            read_count = len(coordinates)
            current: Tuple[int, int] | None = None
        elif step_index is None:
            read_count = 0
            current = None
        else:
            if not 0 <= step_index < len(coordinates):
                raise ValueError("spiral step index is outside the traversal")
            read_count = step_index
            current = coordinates[step_index]

        factor = DOCTRINE_SUPERSAMPLE
        image, sdraw = new_scaled_canvas(_CANVAS[0], _CANVAS[1], factor, GRID_PAGE_BG)
        already_read = set(coordinates[:read_count])
        for row in range(size):
            for col in range(size):
                if (row, col) == current:
                    colour = GRID_ACCENT_FILL
                elif final or (row, col) in already_read:
                    colour = _VISITED_FILL
                else:
                    colour = GRID_CELL_FILL
                x0, y0, _x1, _y1 = geometry.cell_box(row, col)
                fill_cell(sdraw, left=x0, top=y0, cell_px=geometry.cell_px, color=colour)
        draw_grid_lines(
            sdraw,
            left=geometry.left,
            top=geometry.top,
            rows=size,
            cols=size,
            cell_px=geometry.cell_px,
            minor_color=GRID_MINOR_LINE,
            major_color=GRID_OUTER_LINE,
        )
        if current is not None:
            x0, y0, x1, y1 = geometry.cell_box(*current)
            sdraw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], outline=GRID_ACCENT, width=4)

        traced = coordinates[: read_count + (1 if current is not None else 0)]
        points = [geometry.cell_center(row, col) for row, col in traced]
        if len(points) >= 2:
            # The path helper draws directly on the canvas, so its axis-aligned
            # segments are laid down in supersampled coordinates and smoothed
            # by the shared LANCZOS finish.
            draw_path_line(
                image,
                [(x * factor, y * factor) for x, y in points],
                _PATH_COLOR,
                _PATH_WIDTH * factor,
                joint_style="extended",
                tip_marker=False,
            )
            sdraw = ScaledDraw(ImageDraw.Draw(image), factor)

        value_font = load_font(integer_grid_font_size(geometry.cell_px) * factor)
        for row in range(size):
            for col in range(size):
                draw_text_centered(
                    sdraw,
                    text=f"{int(matrix[row][col]):02d}",
                    box=geometry.cell_box(row, col),
                    fnt=value_font,
                    fill=GRID_INK,
                )
        return finish_scaled_canvas(image, _CANVAS)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=_INPUT_DESC,
            symbol_legend=("The centered two-digit text is the cell's value.",),
            rules=(
                "Start at the top-left cell and read its value.",
                "Read the current outer boundary clockwise: the top edge left to right, the right edge downward, the bottom edge right to left, then the left edge upward.",
                "After a boundary closes, shrink all four sides inward by one cell and read the next boundary the same way.",
                "Read every cell exactly once; never skip, repeat, or enter an inner boundary early.",
                "The answer is the ordered list of cell values along that reading path.",
            ),
            modality_outputs={
                "ti2v": _TI2V_OUTPUT,
                "solution_image": _SOLUTION_IMAGE_OUTPUT,
            },
            preservation_constraints=(
                "Keep the matrix size, cell positions, every printed value, the canvas, and the camera unchanged.",
                "Only the cell read-state colours and the traced path may change.",
            ),
            transition_constraints=(
                "Start from the supplied matrix with no cell read and no path drawn.",
                "Each state marks exactly the next cell of the reading path, tints the previously read cells, and extends the path by one whole segment.",
                "Hold every state for the same duration and switch directly to the next; do not interpolate, fade, or blend between states.",
                "Move only to the next orthogonally adjacent cell in reading order, with no skip, repeat, backtrack, or early inner-boundary entry.",
                "Hold the fully read board and its complete path at the end.",
            ),
            answer=AnswerContract(
                kind="nonnegative_integer_list",
                format_template="Answer: [<nonnegative integer>, ...].",
                format_example="Answer: [96, 97, 98, ...].",
                parser_profile="strict_canonical_json_two_digit_integer_list",
                verifier_profile=_VERIFIER_PROFILE,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="spiral_matrix_integer_grid_traced_path_v5",
        )

    def _prompts(self) -> tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid spiral-matrix prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _render_video(
        self,
        matrix: List[List[int]],
        states: List[tuple[str, int | None]],
        *,
        output_path: Path,
    ) -> int:
        coordinates = _spiral_coordinates(len(matrix))
        if len(states) != len(coordinates) + 2:
            raise ValueError("spiral video states must cover input, every read, and final")

        frames: List[np.ndarray] = []
        for phase, step_index in states:
            if phase == "input":
                hold = start_hold_frames(self.video_fps)
                frame = self._render_state(matrix)
            elif phase == "final":
                hold = end_hold_frames(self.video_fps)
                frame = self._render_state(matrix, final=True)
            else:
                # Every traversal state is an exact semantic snapshot held for the
                # same number of frames; nothing is interpolated in between.
                hold = DISCRETE_STATE_HOLD_FRAMES
                frame = self._render_state(matrix, step_index=step_index)
            array = np.asarray(frame.convert("RGB"), dtype=np.uint8)
            frames.extend(array.copy() for _ in range(hold))
        frames = pad_frames_to_wan_compatible_length(frames)
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)

    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1

        matrix = self._sample_matrix()
        coordinates = _spiral_coordinates(len(matrix))
        order = [matrix[row][col] for row, col in coordinates]
        expected_cells = {(row, col) for row in range(len(matrix)) for col in range(len(matrix))}
        if (
            len(coordinates) != len(matrix) ** 2
            or len(set(coordinates)) != len(coordinates)
            or set(coordinates) != expected_cells
            or order != _spiral_order(matrix)
        ):
            raise RuntimeError("spiral_matrix generation produced an invalid traversal")
        states = _spiral_visual_states(len(matrix))
        contract, prompts = self._prompts()

        sample_dir = self.output_dir / sample_id
        puzzle_dir = sample_dir / "puzzles"
        solution_dir = sample_dir / "solutions"
        video_dir = sample_dir / "videos"
        steps_dir = sample_dir / "steps"
        for directory in (puzzle_dir, solution_dir, video_dir, steps_dir):
            directory.mkdir(parents=True, exist_ok=True)

        puzzle_image = self._render_state(matrix)
        solution_image = self._render_state(matrix, final=True)
        puzzle_path = puzzle_dir / f"{sample_id}__puzzle.png"
        solution_path = solution_dir / f"{sample_id}__solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        solution_video_path = None
        video_num_frames = None
        if self.video:
            video_path = video_dir / f"{sample_id}__solution.webm"
            video_num_frames = self._render_video(matrix, states, output_path=video_path)
            if video_num_frames > 0:
                solution_video_path = video_path

        answer_text = format_nonnegative_integer_list_answer(order)
        geometry = spiral_grid_geometry(len(matrix))
        token_width, token_height = integer_grid_token_window(geometry.cell_px)
        return {
            "id": sample_id,
            "answer": answer_text,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": "Read every matrix value in clockwise spiral order starting at the top-left cell.",
            "structured_answer": {
                "input_matrix": matrix,
                "spiral_order": order,
                "spiral_coordinates": [[row, col] for row, col in coordinates],
                "grid_bbox": list(geometry.bbox),
                "cell_size": geometry.cell_px,
                "number_token_geometry": {"width": token_width, "height": token_height},
                "path_stroke": {"color": list(_PATH_COLOR), "width": _PATH_WIDTH},
                "value_count": len(order),
                "answer_schema": {
                    "schema_version": "integer_list_v1",
                    "kind": "nonnegative_integer_list",
                    "format_template": "Answer: [<nonnegative integer>, ...].",
                    "format_example": "Answer: [96, 97, 98, ...].",
                    "parser_profile": "strict_canonical_json_two_digit_integer_list",
                    "final_deliverable": {
                        "ti2v": ["full_video"],
                    },
                    "accepts_multiple_valid_answers": False,
                    "extra_text_allowed": False,
                },
                "intermediate_supervision_schema": {
                    "ti2v": ["input", "one_read_cell_per_discrete_state", "final"],
                },
                "verifier_contract": {
                    "profile": _VERIFIER_PROFILE,
                    "semantic_source": "visible_square_grid_values_and_clockwise_boundary_shrink_rule",
                    "component_scores": list(_VERIFIER_COMPONENTS),
                    "hard_gates": [
                        "strict_parse",
                        "independent_clockwise_spiral_replay",
                        "complete_unique_coordinate_coverage",
                        "orthogonal_adjacency",
                        "terminal_path_segment_coverage",
                        "semantic_final_image",
                        "whole_video_exact_state_trace",
                        "ordered_discrete_state_cadence",
                        "fixed_content_preservation",
                        "media_contract",
                    ],
                },
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


from data.in_context_learning.core.polish import polished_icl_generator

SpiralMatrixPattern = polished_icl_generator(SpiralMatrixPattern)

__all__ = [
    "SpiralMatrixPattern",
    "_spiral_coordinates",
    "_spiral_order",
    "_spiral_visual_states",
    "_PATH_COLOR",
    "_PATH_WIDTH",
    "_VISITED_FILL",
    "spiral_grid_geometry",
    "_VERIFIER_COMPONENTS",
    "_VERIFIER_PROFILE",
    "spiral_ring_sentence",
    "spiral_closing_sentence",
]

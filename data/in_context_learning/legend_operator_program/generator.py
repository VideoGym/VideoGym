from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from PIL import Image

from data.common.base import AbstractPuzzleGenerator
from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
)
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.video_encoding import encode_rgb_frames_to_webm, pad_frames_to_wan_compatible_length
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import format_matrix_answer
from data.common.pipeline.structured_grid import load_font, matrix_to_json

_VERIFIER_PROFILE = "legend_operator_independent_example_and_program_replay_v1"
_VERIFIER_COMPONENTS = {
    "ti2v": (
        "legend_inference_score",
        "program_replay_score",
        "stable_state_score",
        "transition_cell_score",
        "state_order_score",
        "preservation_score",
        "media_contract_score",
        "artifact_readability_score",
        "final_state_score",
    ),
}
pattern_instructions = {
    "legend_operator_program": (
        "Infer three binary-matrix operator tokens from their before-and-after legend examples, then execute the two-to-four-token "
        "program from left to right while preserving the legend and target."
    ),
}
_INPUT_DESC = (
    "The left side shows three rows: an A, B, or C token followed by one binary matrix, an arrow, and its transformed "
    "matrix. Each visible pair defines that token's effect. On the right, the upper matrix is the target, the boxed "
    "tokens below it are the program in left-to-right order, and the lower matrix is the evolving result. Dark cells "
    "are bit 1 and white cells are bit 0."
)
_RULE_SUMMARY = (
    "Infer each token's operation only from the legend examples. Then execute the target program from left to right "
    "on the target matrix. Each operation updates the complete matrix simultaneously, and each output becomes "
    "the following operation's input."
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render the solved still image by updating only the current-result matrix to the final program output and marking "
    "the last program token as active."
)
_TI2V_OUTPUT = (
    "Generate one continuous fixed-camera execution video. "
    "Start with the target copied into the current-result matrix and no active token. "
    "For each token from left to right, highlight that token, briefly show every changing result cell in amber, then show "
    "the complete stable updated matrix. Amber means that a cell is currently changing; it is never a stable bit value. "
    "Keep every legend example, the target matrix, the program order, the layout, and the camera fixed. End with a "
    "readable hold of the exact final matrix and the last token active."
)

_CANVAS_SIZE = 512
_RESULT_GRID_LEFT = 354
_RESULT_GRID_TOP = 326


def program_token_bboxes(length: int) -> tuple[tuple[int, int, int, int], ...]:
    if length not in (2, 3, 4, 5):
        raise ValueError("program length must be between 2 and 5")
    width, gap = (48, 22) if length == 2 else (34, 6)
    left = 396 - (length * width + (length - 1) * gap) // 2
    return tuple((left + i * (width + gap), 224, left + i * (width + gap) + width, 264) for i in range(length))


_STEP_LABEL_BBOX = (350, 434, 466, 460)
_TOKEN_LABELS = ("A", "B", "C")
# Amber marks a cell that is currently changing; green marks the active
# program token.  The task evaluator imports both from here.
_TRANSITION_FILL = (238, 168, 51)
_ACTIVE_TOKEN_OUTLINE = (52, 154, 82)


def work_cell_px(size: int) -> int:
    """Result-matrix cell edge in shipped pixels for a given matrix size."""

    return 96 // int(size)


def _rotate_clockwise(matrix: List[List[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*matrix[::-1])]


def _flip_horizontal(matrix: List[List[int]]) -> List[List[int]]:
    return [list(reversed(row)) for row in matrix]


def _transpose(matrix: List[List[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*matrix)]


_OP_LIBRARY: Dict[str, Callable[[List[List[int]]], List[List[int]]]] = {
    "rotate_cw": _rotate_clockwise,
    "flip_horizontal": _flip_horizontal,
    "transpose": _transpose,
}


def _sample_binary_matrix(rng: random.Random, size: int = 4) -> List[List[int]]:
    while True:
        matrix = [[rng.randint(0, 1) for _ in range(size)] for _ in range(size)]
        ones = sum(sum(row) for row in matrix)
        if 3 <= ones <= size * size - 3:
            return matrix


def _draw_binary_matrix(
    sdraw: ScaledDraw,
    matrix: Sequence[Sequence[int]],
    *,
    left: int,
    top: int,
    cell_size: int,
    transition_cells: set[tuple[int, int]] | None = None,
) -> None:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    for row in range(rows):
        for col in range(cols):
            x0 = left + col * cell_size
            y0 = top + row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            if transition_cells is not None and (row, col) in transition_cells:
                fill = _TRANSITION_FILL
            else:
                fill = (38, 42, 50) if int(matrix[row][col]) else (255, 255, 255)
            sdraw.rectangle((x0, y0, x1, y1), fill=fill, outline=(176, 176, 170), width=1)
    sdraw.rectangle(
        (left, top, left + cols * cell_size, top + rows * cell_size),
        outline=(70, 70, 70),
        width=2,
    )


class LegendOperatorProgramPattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = "legend_operator_program"
    task_family = "in_context_learning"

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        video: bool = True,
        video_fps: int = 16,
        size: int = 4,
        program_length: int = 2,
        operator_sequences: Sequence[Sequence[str]] | None = None,
        token_mapping: Sequence[str] | None = None,
    ) -> None:
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_legend_operator_program"
        super().__init__(resolved)
        self.rng = random.Random(seed)
        self.seed = seed
        self.video = bool(video)
        self.video_fps = int(video_fps)
        self.size = int(size)
        if self.size not in {3, 4, 5}:
            raise ValueError("legend_operator_program size must be 3, 4, or 5")
        self.program_length = int(program_length)
        self.program_token_bboxes = program_token_bboxes(self.program_length)
        self.operator_sequences = tuple(tuple(x) for x in operator_sequences) if operator_sequences is not None else None
        if self.operator_sequences is not None and (not self.operator_sequences or any(
            len(seq) != self.program_length or any(op not in _OP_LIBRARY for op in seq)
            for seq in self.operator_sequences
        )):
            raise ValueError("operator_sequences must contain programs of the requested length")
        self.token_mapping = tuple(token_mapping) if token_mapping is not None else None
        if self.token_mapping is not None and (len(self.token_mapping) != 3 or set(self.token_mapping) != set(_OP_LIBRARY)):
            raise ValueError("token_mapping must be a permutation of the three operations")
        self._counter = 0

    @property
    def mini_cell_size(self) -> int:
        return 60 // self.size

    @property
    def work_cell_size(self) -> int:
        return work_cell_px(self.size)

    @property
    def result_grid_bbox(self) -> tuple[int, int, int, int]:
        grid_size = self.size * self.work_cell_size
        return (
            _RESULT_GRID_LEFT,
            _RESULT_GRID_TOP,
            _RESULT_GRID_LEFT + grid_size,
            _RESULT_GRID_TOP + grid_size,
        )

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=f"{_INPUT_DESC} {GRID_XY_CONVENTION_SENTENCE}",
            symbol_legend=(
                "A dark matrix cell is bit 1 and a white matrix cell is bit 0.",
                "Each A, B, or C token is defined only by its displayed before-and-after legend pair.",
                "The program tokens are ordered from left to right.",
                "Amber is a temporary video transition color for a cell currently changing, not a matrix value.",
            ),
            rules=(
                _RULE_SUMMARY,
                "Apply each inferred operation simultaneously to every cell of the complete current matrix.",
                "The output of each token is the input to the next token.",
            ),
            modality_outputs={
                "ti2v": _TI2V_OUTPUT,
                "solution_image": _SOLUTION_IMAGE_OUTPUT,
            },
            preservation_constraints=(
                "Keep all three legend examples, the target matrix, program tokens, layout, colors, and camera unchanged.",
                "Only the active-token highlight and current-result matrix may change.",
            ),
            transition_constraints=(
                "Begin with a readable hold showing the target copied into the current-result matrix and no active token.",
                "Activate token 1, then show every and only changed result cell in amber while all unchanged cells retain their previous stable appearance; replace all amber cells simultaneously with their new bit states and hold.",
                "Repeat the same amber-to-stable update for every following token, using the complete previous result as input.",
                "Show exactly one stable state after each token without skipping, reversing, or reordering tokens.",
                "End with a readable hold of the exact final matrix and the last token active.",
                "Do not use a hard cut, crossfade, cell teleportation, partial row update, stale input, token-order reversal, brief disappearance, legend mutation, pan, zoom, crop, camera drift, or flicker.",
            ),
            answer=AnswerContract(
                kind="grid",
                format_template="Answer: [[<row values>], ...].",
                format_example="Answer: [[0, 1, ...], [1, 0, ...], ...].",
                parser_profile="strict_canonical_binary_square_json_matrix",
                verifier_profile=_VERIFIER_PROFILE,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_binary_operator_program_v4_low_text",
        )

    def _prompts(self) -> tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid legend-operator prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _sample_program(
        self,
    ) -> tuple[
        Dict[str, str],
        Dict[str, List[List[int]]],
        Dict[str, List[List[int]]],
        List[List[int]],
        List[str],
        List[List[List[int]]],
        List[List[int]],
    ]:
        operator_names = list(_OP_LIBRARY)
        self.rng.shuffle(operator_names)
        if self.token_mapping is not None:
            operator_names = list(self.token_mapping)
        operator_by_token = dict(zip(_TOKEN_LABELS, operator_names))
        legend_examples: Dict[str, List[List[int]]] = {}
        legend_outputs: Dict[str, List[List[int]]] = {}
        for token in _TOKEN_LABELS:
            operator_name = operator_by_token[token]
            while True:
                base = _sample_binary_matrix(self.rng, self.size)
                candidate_outputs = {
                    name: candidate([list(row) for row in base])
                    for name, candidate in _OP_LIBRARY.items()
                }
                intended = candidate_outputs[operator_name]
                matching_names = [name for name, output in candidate_outputs.items() if output == intended]
                if intended != base and matching_names == [operator_name]:
                    legend_examples[token] = base
                    legend_outputs[token] = intended
                    break

        for _ in range(2048):
            target = _sample_binary_matrix(self.rng, self.size)
            if self.operator_sequences is not None:
                tokens_by_op = {op: token for token, op in operator_by_token.items()}
                program = [tokens_by_op[op] for op in self.rng.choice(self.operator_sequences)]
            else:
                program = self.rng.choices(list(_TOKEN_LABELS), k=self.program_length)
            states = [target]
            current = target
            for token in program:
                current = _OP_LIBRARY[operator_by_token[token]]([list(row) for row in current])
                states.append(current)
            reversed_current = [list(row) for row in target]
            for token in reversed(program):
                reversed_current = _OP_LIBRARY[operator_by_token[token]](reversed_current)
            if (
                len({matrix_to_json(state) for state in states}) == len(states)
                and reversed_current != states[-1]
            ):
                return (
                    operator_by_token,
                    legend_examples,
                    legend_outputs,
                    target,
                    program,
                    states,
                    reversed_current,
                )

        raise RuntimeError("could not sample an order-sensitive program with distinct states")

    def _draw_polygon_arrow(
        self,
        sdraw: ScaledDraw,
        *,
        cx: int,
        cy: int,
        length: int,
        height: int,
        color: tuple = (70, 70, 70),
    ) -> None:
        """Draw a thick filled arrow centred at (cx, cy) with the given length and tip height."""
        half_l = length // 2
        shaft_h = max(6, height // 3)
        tip_w = max(12, height // 2)
        # Polygon: rectangle shaft + triangular tip on the right.
        x_left = cx - half_l
        x_right = cx + half_l
        # Shaft top/bottom y.
        sy_top = cy - shaft_h // 2
        sy_bot = cy + shaft_h // 2
        # Tip base x is `tip_w` left of the rightmost point.
        tip_base_x = x_right - tip_w
        points = [
            (x_left, sy_top),
            (tip_base_x, sy_top),
            (tip_base_x, cy - height // 2),
            (x_right, cy),
            (tip_base_x, cy + height // 2),
            (tip_base_x, sy_bot),
            (x_left, sy_bot),
        ]
        sdraw.polygon(points, fill=color)

    def _render_scene(
        self,
        legend_examples: Dict[str, List[List[int]]],
        legend_outputs: Dict[str, List[List[int]]],
        target: List[List[int]],
        program: Sequence[str],
        states: Sequence[List[List[int]]],
        *,
        state_index: int,
        transition: bool = False,
    ) -> Image.Image:
        factor = DOCTRINE_SUPERSAMPLE
        canvas, sdraw = new_scaled_canvas(_CANVAS_SIZE, _CANVAS_SIZE, factor, (255, 255, 255))
        token_font = load_font(22 * factor)
        legend_tops = {"A": 62, "B": 202, "C": 342}
        for token in _TOKEN_LABELS:
            top = legend_tops[token]
            sdraw.rounded_rectangle((28, top + 10, 66, top + 48), radius=10, fill=(255, 255, 255), outline=(166, 166, 158), width=2)
            sdraw.text((40, top + 15), token, font=token_font, fill=(36, 36, 36))
            _draw_binary_matrix(sdraw, legend_examples[token], left=86, top=top, cell_size=self.mini_cell_size)
            self._draw_polygon_arrow(sdraw, cx=171, cy=top + 30, length=30, height=20)
            _draw_binary_matrix(
                sdraw,
                legend_outputs[token],
                left=202,
                top=top,
                cell_size=self.mini_cell_size,
            )

        sdraw.line((309, 24, 309, 488), fill=(214, 214, 208), width=2)
        _draw_binary_matrix(sdraw, target, left=354, top=58, cell_size=self.work_cell_size)
        active_token_index = state_index - 1
        for index, token in enumerate(program):
            x0, y0, x1, y1 = self.program_token_bboxes[index]
            active = index == active_token_index
            sdraw.rounded_rectangle(
                (x0, y0, x1, y1),
                radius=10,
                fill=(235, 250, 239) if active else (255, 255, 255),
                outline=_ACTIVE_TOKEN_OUTLINE if active else (166, 166, 158),
                width=3 if active else 2,
            )
            sdraw.text((x0 + (x1 - x0 - 16) // 2, y0 + 5), token, font=token_font, fill=(32, 32, 32))
        if len(program) == 2:
            self._draw_polygon_arrow(sdraw, cx=396, cy=244, length=18, height=16, color=(110, 110, 110))

        arrow_cx = 402
        arrow_top_y = 273
        arrow_bot_y = 306
        shaft_w = 6
        tip_w = 15
        tip_h = 12
        sdraw.polygon(
            [
                (arrow_cx - shaft_w, arrow_top_y),
                (arrow_cx + shaft_w, arrow_top_y),
                (arrow_cx + shaft_w, arrow_bot_y - tip_h),
                (arrow_cx + tip_w, arrow_bot_y - tip_h),
                (arrow_cx, arrow_bot_y),
                (arrow_cx - tip_w, arrow_bot_y - tip_h),
                (arrow_cx - shaft_w, arrow_bot_y - tip_h),
            ],
            fill=(80, 80, 80),
        )
        transition_cells: set[tuple[int, int]] | None = None
        if transition and state_index > 0:
            transition_cells = {
                (row, col)
                for row in range(self.size)
                for col in range(self.size)
                if states[state_index - 1][row][col] != states[state_index][row][col]
            }
        _draw_binary_matrix(
            sdraw,
            states[state_index],
            left=_RESULT_GRID_LEFT,
            top=_RESULT_GRID_TOP,
            cell_size=self.work_cell_size,
            transition_cells=transition_cells,
        )
        return finish_scaled_canvas(canvas, (_CANVAS_SIZE, _CANVAS_SIZE))

    def _render_video(
        self,
        legend_examples: Dict[str, List[List[int]]],
        legend_outputs: Dict[str, List[List[int]]],
        target: List[List[int]],
        program: Sequence[str],
        states: Sequence[List[List[int]]],
        *,
        output_path: Path,
    ) -> int:
        frames: List[np.ndarray] = []
        initial = np.asarray(
            self._render_scene(
                legend_examples,
                legend_outputs,
                target,
                program,
                states,
                state_index=0,
            ),
            dtype=np.uint8,
        )
        frames.extend(initial.copy() for _ in range(4))
        for state_index in range(1, len(states)):
            transition_frame = np.asarray(
                self._render_scene(
                    legend_examples,
                    legend_outputs,
                    target,
                    program,
                    states,
                    state_index=state_index,
                    transition=True,
                ),
                dtype=np.uint8,
            )
            frames.extend(transition_frame.copy() for _ in range(2))
            stable_frame = np.asarray(
                self._render_scene(
                    legend_examples,
                    legend_outputs,
                    target,
                    program,
                    states,
                    state_index=state_index,
                ),
                dtype=np.uint8,
            )
            hold = 4 if state_index < len(states) - 1 else 9
            frames.extend(stable_frame.copy() for _ in range(hold))
        frames = [np.asarray(frame, dtype=np.uint8) for frame in pad_frames_to_wan_compatible_length(frames)]
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)

    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1

        (
            operator_by_token,
            legend_examples,
            legend_outputs,
            target,
            program,
            states,
            reversed_program_output,
        ) = self._sample_program()
        contract, prompts = self._prompts()

        sample_dir = self.output_dir / sample_id
        puzzle_dir = sample_dir / "puzzles"
        solution_dir = sample_dir / "solutions"
        video_dir = sample_dir / "videos"
        for directory in (puzzle_dir, solution_dir, video_dir):
            directory.mkdir(parents=True, exist_ok=True)

        puzzle_image = self._render_scene(legend_examples, legend_outputs, target, program, states, state_index=0)
        solution_image = self._render_scene(
            legend_examples,
            legend_outputs,
            target,
            program,
            states,
            state_index=len(states) - 1,
        )
        puzzle_path = puzzle_dir / f"{sample_id}-puzzle.png"
        solution_path = solution_dir / f"{sample_id}-solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        solution_video_path = None
        video_num_frames = None
        if self.video:
            video_path = video_dir / f"{sample_id}-solution.webm"
            video_num_frames = self._render_video(
                legend_examples,
                legend_outputs,
                target,
                program,
                states,
                output_path=video_path,
            )
            if video_num_frames > 0:
                solution_video_path = video_path

        answer_text = format_matrix_answer(
            states[-1],
            allowed_values={0, 1},
        )
        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": "Infer the legend operators and apply the program tokens from left to right to the target matrix.",
            "answer": answer_text,
            "structured_answer": {
                "answer_schema": {
                    "schema_version": "binary_square_matrix_v1",
                    "kind": "binary_matrix",
                    "format_template": "Answer: [[<row values>], ...].",
                    "format_example": "Answer: [[0, 1, ...], [1, 0, ...], ...].",
                    "parser_profile": "strict_canonical_binary_square_json_matrix",
                    "final_deliverable": {
                        "ti2v": ["full_video"],
                    },
                    "accepts_multiple_valid_answers": False,
                    "extra_text_allowed": False,
                },
                "intermediate_supervision_schema": {
                    "ti2v": [
                        "input_hold",
                        "exact_changed_cells_amber_for_token_1",
                        "stable_matrix_after_token_1",
                        "exact_changed_cells_amber_for_token_2",
                        "stable_final_matrix_after_token_2",
                        "final_hold",
                    ],
                },
                "verifier_contract": {
                    "profile": _VERIFIER_PROFILE,
                    "semantic_source": "three_before_after_legend_pairs_target_matrix_and_ordered_program_tokens",
                    "component_scores": {
                        mode: list(names) for mode, names in _VERIFIER_COMPONENTS.items()
                    },
                    "binary_predicate": (
                        "strict canonical matrix parse, unique independent legend inference, left-to-right program replay, "
                        "and exact cumulative visual-state agreement"
                    ),
                },
                "operator_by_token": operator_by_token,
                "legend_examples": legend_examples,
                "legend_outputs": legend_outputs,
                "target_matrix": target,
                "program_tokens": list(program),
                "state_trace": states,
                "output_matrix": states[-1],
                "reversed_program_output": reversed_program_output,
                "grid_size": self.size,
                "result_grid_bbox": list(self.result_grid_bbox),
                "result_cell_size": self.work_cell_size,
                "program_token_bboxes": [list(box) for box in self.program_token_bboxes],
                "allowed_change_bboxes": [
                    list(self.result_grid_bbox),
                    *[list(box) for box in self.program_token_bboxes],
                    list(_STEP_LABEL_BBOX),
                ],
            },
            "verifier_contract": {
                "profile": _VERIFIER_PROFILE,
                "semantic_source": "three_before_after_legend_pairs_target_matrix_and_ordered_program_tokens",
                "component_scores": {
                    mode: list(names) for mode, names in _VERIFIER_COMPONENTS.items()
                },
                "binary_predicate": (
                    "strict canonical matrix parse, unique independent legend inference, left-to-right program replay, "
                    "and exact cumulative visual-state agreement"
                ),
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

LegendOperatorProgramPattern = polished_icl_generator(LegendOperatorProgramPattern)

__all__ = [
    "LegendOperatorProgramPattern",
    "_OP_LIBRARY",
    "_VERIFIER_COMPONENTS",
    "_VERIFIER_PROFILE",
]

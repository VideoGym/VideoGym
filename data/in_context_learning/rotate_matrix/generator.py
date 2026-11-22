from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from data.common.animation_policy import discrete_state_timeline
from data.common.base import AbstractPuzzleGenerator
from data.common.grid_coordinates import GRID_XY_CONVENTION_SENTENCE
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.video_encoding import encode_rgb_frames_to_webm, pad_frames_to_wan_compatible_length
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.pipeline.structured_grid import (
    GRID_ACCENT,
    GRID_PAGE_BG,
    draw_integer_grid,
    draw_text_centered,
    integer_grid_font_size,
    integer_grid_geometry,
    integer_grid_token_window,
    load_font,
    matrix_to_json,
)


pattern_instructions = {
    "rotate_matrix": (
        "The input image shows a square matrix of distinct two-digit integers. Apply the requested number of "
        "clockwise 90-degree quarter-turns and return the complete final matrix."
    )
}

_INPUT_DESC = (
    "The input image shows one square matrix of distinct two-digit integers, with the requested number of "
    f"clockwise quarter-turns printed in amber above it. Every cell prints its integer once. {GRID_XY_CONVENTION_SENTENCE}"
)
_TI2T_OUTPUT = (
    "Return the complete final matrix as a JSON array of integer rows in the exact format "
    '"Answer: [[<integer>, ...], ...]." with one space after every comma and no extra words.'
)
_TI2TI_OUTPUT = (
    "Produce one generated image for each completed quarter-turn before the last, then the final matrix. Before those "
    "images, write one sample-specific reasoning block that states the matrix size, the requested turn count, where the "
    "source rows and columns end up, and the complete final row-major matrix; end that same block with the exact "
    'final-answer line "Answer: [[<integer>, ...], ...]." Then output the final image and no later text or image block.'
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render the same matrix after all requested clockwise quarter-turns, with every printed integer preserved and "
    "moved to its correct final cell."
)
_TI2V_OUTPUT = (
    "Generate a fixed-camera video that starts by holding the supplied input matrix. Show the matrix after each "
    "clockwise quarter-turn as an exact still state, in order, and switch directly from one state to the next. "
    "End by holding the requested final matrix. Keep each printed integer unchanged, and keep the turn count, grid "
    "dimensions, background, and camera fixed. Do not rotate counterclockwise, skip or repeat a quarter-turn, "
    "change/add/remove/duplicate a value, transpose without rotating, tilt or resize the grid, fade or cross-blend "
    "between states, flicker, crop, zoom, pan, or move the camera."
)

_CANVAS = (512, 512)
_TURN_LABEL_CENTER = (256, 50)
_TURN_LABEL_FONT_PX = 28
_TURN_LABEL_BOX = (
    _TURN_LABEL_CENTER[0] - 40,
    _TURN_LABEL_CENTER[1] - 20,
    _TURN_LABEL_CENTER[0] + 40,
    _TURN_LABEL_CENTER[1] + 20,
)


def _rotate_clockwise(matrix: List[List[int]], turns: int) -> List[List[int]]:
    result = [list(row) for row in matrix]
    for _ in range(turns % 4):
        result = [list(row) for row in zip(*result[::-1])]
    return result


def _rotation_states(matrix: List[List[int]], turns: int) -> List[List[List[int]]]:
    return [_rotate_clockwise(matrix, turn) for turn in range(turns + 1)]


class RotateMatrixPattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = "rotate_matrix"
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
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_rotate_matrix"
        super().__init__(resolved)
        resolved_seed = seed if seed is not None else random_seed
        self.rng = random.Random(resolved_seed)
        self.size_range = (int(size_range[0]), int(size_range[1]))
        if self.size_range[0] < 3 or self.size_range[1] > 5 or self.size_range[0] > self.size_range[1]:
            raise ValueError("rotate_matrix size_range must stay within 3..5")
        self.video = bool(video if generate_videos is None else generate_videos)
        self.video_fps = int(video_fps)
        if self.video_fps != 16:
            raise ValueError("rotate_matrix videos must use 16fps")
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _sample_matrix(self) -> List[List[int]]:
        size = self.rng.randint(*self.size_range)
        values = self.rng.sample(range(10, 100), k=size * size)
        return [values[index * size:(index + 1) * size] for index in range(size)]

    def _render_state(
        self,
        matrix: List[List[int]],
        *,
        turn_index: int = 0,
        target_turns: int = 1,
    ) -> Image.Image:
        """Render one exact matrix state.

        ``turn_index`` tells the caller which state it asked for; the drawn
        frame deliberately does not encode it, because a progress counter would
        be a changing subtitle rather than task state.
        """

        del turn_index
        size = len(matrix)
        geometry = integer_grid_geometry(size)
        factor = DOCTRINE_SUPERSAMPLE
        image, sdraw = new_scaled_canvas(_CANVAS[0], _CANVAS[1], factor, GRID_PAGE_BG)
        draw_text_centered(
            sdraw,
            text=f"×{int(target_turns)}",
            box=_TURN_LABEL_BOX,
            fnt=load_font(_TURN_LABEL_FONT_PX * factor),
            fill=GRID_ACCENT,
        )
        draw_integer_grid(
            sdraw,
            matrix=[[f"{int(value):02d}" for value in row] for row in matrix],
            geometry=geometry,
            fnt=load_font(integer_grid_font_size(geometry.cell_px) * factor),
        )
        return finish_scaled_canvas(image, _CANVAS)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=_INPUT_DESC,
            symbol_legend=(
                "The centered two-digit text is the cell's integer value.",
                "The short amber label above the grid states the requested number of clockwise quarter-turns.",
            ),
            rules=(
                "One clockwise quarter-turn moves the value at (x, y) in an n-by-n matrix to (y, n + 1 - x).",
                "Apply that same movement to every cell simultaneously for each requested quarter-turn.",
                "Use the output of one quarter-turn as the input to the next until the requested count is complete.",
                "Move each printed integer with its cell; no value may be changed, added, removed, duplicated, or replaced.",
                "The answer is the complete matrix after exactly the requested number of clockwise quarter-turns.",
            ),
            modality_outputs={
                "ti2v": _TI2V_OUTPUT,
                "solution_image": _SOLUTION_IMAGE_OUTPUT,
            },
            preservation_constraints=(
                "Keep the matrix size, printed integer identities, cell styling, turn count, canvas, and camera unchanged.",
                "Only which cell holds which value may change between completed states.",
            ),
            transition_constraints=(
                "Begin with a visible hold of the supplied input matrix.",
                "Show the matrix after each completed clockwise quarter-turn as an exact still state, holding every state for the same length of time.",
                "Switch directly from one completed state to the next; no tilted, part-turned, resized, faded, or blended frame belongs in between.",
                "Show every completed 90-degree clockwise state in exact order; do not jump directly to a later turn or insert a counterclockwise, repeated, transposed, or mutated state.",
                "End with a visible hold of the exact requested final matrix.",
            ),
            answer=AnswerContract(
                kind="grid",
                format_template="Answer: [[<integer>, ...], ...].",
                format_example="Answer: [[98, 99, ...], [100, 101, ...], ...].",
                parser_profile="strict_json_square_two_digit_integer_grid",
                verifier_profile="rotate_matrix_printed_number_source_replay_v4",
                accepts_multiple_valid_answers=False,
            ),
            render_profile="rotate_matrix_shared_integer_grid_v4",
        )

    def _prompts(self) -> tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid rotate-matrix prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _render_video(
        self,
        states: List[List[List[int]]],
        *,
        output_path: Path,
    ) -> int:
        """Serialize the turn sequence as exact discrete states.

        The state space is the four orientations of one matrix, so the
        transition model is discrete too: every state is held for the same
        number of frames and switched directly to the next.  A continuous spin
        would have to rotate the printed digits away from upright and snap them
        back at every commit, mutating the very tokens the task preserves.
        """

        target_turns = len(states) - 1
        holds = discrete_state_timeline(len(states), fps=self.video_fps)
        frames: List[np.ndarray] = []
        for state_index, (state, hold) in enumerate(zip(states, holds)):
            rendered = np.asarray(
                self._render_state(state, turn_index=state_index, target_turns=target_turns),
                dtype=np.uint8,
            )
            frames.extend(rendered.copy() for _ in range(hold))
        frames = [np.asarray(frame, dtype=np.uint8) for frame in pad_frames_to_wan_compatible_length(frames)]
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)

    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1

        matrix = self._sample_matrix()
        turns = self.rng.randint(1, 3)
        states = _rotation_states(matrix, turns)
        rotated = states[-1]
        answer_raw = matrix_to_json(rotated)
        contract, prompts = self._prompts()

        sample_dir = self.output_dir / sample_id
        puzzle_dir = sample_dir / "puzzles"
        solution_dir = sample_dir / "solutions"
        video_dir = sample_dir / "videos"
        for directory in (puzzle_dir, solution_dir, video_dir):
            directory.mkdir(parents=True, exist_ok=True)

        puzzle_image = self._render_state(states[0], turn_index=0, target_turns=turns)
        solution_image = self._render_state(states[-1], turn_index=turns, target_turns=turns)
        puzzle_path = puzzle_dir / f"{sample_id}__puzzle.png"
        solution_path = solution_dir / f"{sample_id}__solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        solution_video_path = None
        video_num_frames = None
        if self.video:
            video_path = video_dir / f"{sample_id}__solution.webm"
            video_num_frames = self._render_video(states, output_path=video_path)
            if video_num_frames > 0:
                solution_video_path = video_path

        size = len(matrix)
        geometry = integer_grid_geometry(size)
        left, top, right, bottom = geometry.bbox
        token_width, token_height = integer_grid_token_window(geometry.cell_px)
        # Only the board changes between states.  Everything outside it --
        # including the amber turn count -- is fixed content the verifier must
        # be able to police, so the allowance is the board plus the two pixels
        # of antialiasing its outer rule bleeds into.
        dynamic_bbox = [
            max(0, left - 2),
            max(0, top - 2),
            min(512, right + 2),
            min(512, bottom + 2),
        ]
        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": f"Rotate the matrix clockwise by exactly {turns} quarter-turn{'s' if turns != 1 else ''} ({turns * 90} degrees).",
            "answer": answer_raw,
            "structured_answer": {
                "input_matrix": matrix,
                "rotations": turns,
                "output_matrix": rotated,
                "rotation_states": states,
                "grid_bbox": [left, top, right, bottom],
                "cell_size": geometry.cell_px,
                "allowed_change_bboxes": [dynamic_bbox],
                # The turn count decides the answer but covers well under a
                # percent of the canvas, so it carries its own preservation
                # obligation instead of being averaged into the whole frame.
                "turn_label_bbox": list(_TURN_LABEL_BOX),
                "number_token_geometry": {
                    "width": token_width,
                    "height": token_height,
                },
                "state_hold_frames": discrete_state_timeline(len(states), fps=self.video_fps),
                "answer_schema": {
                    "schema_version": "integer_grid_v2",
                    "kind": "square_two_digit_integer_grid",
                    "format_template": "Answer: [[<integer>, ...], ...].",
                    "format_example": "Answer: [[98, 99, ...], [100, 101, ...], ...].",
                    "parser_profile": "strict_json_square_two_digit_integer_grid",
                    "final_deliverable": {
                        "ti2v": ["full_video"],
                    },
                    "accepts_multiple_valid_answers": False,
                    "extra_text_allowed": False,
                },
                "intermediate_supervision_schema": {
                    "ti2v": ["input_hold", "ordered_stable_turn_states", "equal_state_holds", "final_hold"],
                },
                "verifier_contract": {
                    "profile": "rotate_matrix_printed_number_source_replay_v4",
                    "semantic_source": "input_matrix_plus_requested_clockwise_quarter_turn_count",
                    "component_scores": [
                        "parse_score",
                        "source_replay_score",
                        "shape_validity_score",
                        "value_preservation_score",
                        "state_order_score",
                        "discrete_transition_score",
                        "preservation_score",
                        "final_state_score",
                        "media_contract_score",
                    ],
                    "hard_gates": [
                        "strict_parse",
                        "independent_quarter_turn_replay",
                        "square_shape_and_two_digit_values",
                        "cell_value_identity_preservation",
                        "ordered_clockwise_states",
                        "direct_equal_duration_state_switches",
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

RotateMatrixPattern = polished_icl_generator(RotateMatrixPattern)

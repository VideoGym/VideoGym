from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw

from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.pipeline.structured_grid import matrix_to_json
from data.common.pipeline.rendered_task import RenderedTaskGenerator
from data.next_state_prediction.core.rendering import (
    discrete_video_frames,
    finish_state_canvas,
    new_state_canvas,
    scaled_font,
)


pattern_instructions = {
    "grid_shift": (
        "A fixed indexed 4 by 4 grid contains six colored ID tokens and states one cardinal one-cell translation. "
        "Translate every token simultaneously without wraparound, remove tokens shifted beyond the boundary, and return "
        "the complete final row-major integer grid."
    )
}

_CANVAS = (512, 512)
_BG = (246, 248, 247)
_CARD = (255, 255, 253)
_INK = (34, 40, 46)
_MUTED = (88, 96, 103)
_GRID_FILL = (250, 251, 251)
_GRID_LINE = (163, 173, 179)
_ACCENT = (49, 132, 148)
_ARROW = (225, 139, 40)
_GRID_LEFT = 140
_GRID_TOP = 132
_CELL_SIZE = 58
_GRID_SIZE = 4

# Index zero is unused. The evaluator duplicates this fixed, task-level palette
# so candidate videos can be parsed without trusting generator keyframes.
_PALETTE = (
    (239, 83, 80),
    (66, 165, 245),
    (102, 187, 106),
    (255, 202, 40),
    (171, 71, 188),
    (255, 112, 67),
    (38, 198, 218),
    (141, 110, 99),
)
_SHIFT_LABELS = {
    (-1, 0): "up by 1",
    (1, 0): "down by 1",
    (0, -1): "left by 1",
    (0, 1): "right by 1",
}
_SHIFT_ARROWS = {
    (-1, 0): "↑",
    (1, 0): "↓",
    (0, -1): "←",
    (0, 1): "→",
}


def _shift_matrix(matrix: Sequence[Sequence[int | None]], dr: int, dc: int) -> List[List[int | None]]:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    shifted: List[List[int | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            value = matrix[row][col]
            if value is None:
                continue
            next_row = row + dr
            next_col = col + dc
            if 0 <= next_row < rows and 0 <= next_col < cols:
                shifted[next_row][next_col] = int(value)
    return shifted


def _dense_matrix(matrix: Sequence[Sequence[int | None]]) -> List[List[int]]:
    return [[0 if value is None else int(value) for value in row] for row in matrix]


class GridShiftPattern(RenderedTaskGenerator):
    TASK_NAME = "grid_shift"
    TASK_FAMILY = "next_state_prediction"

    def _generate_task(self) -> Dict[str, Any]:
        shift = self.rng.choice(list(_SHIFT_LABELS))
        dr, dc = shift
        if dr == -1:
            exit_cells = [(0, col) for col in range(_GRID_SIZE)]
        elif dr == 1:
            exit_cells = [(_GRID_SIZE - 1, col) for col in range(_GRID_SIZE)]
        elif dc == -1:
            exit_cells = [(row, 0) for row in range(_GRID_SIZE)]
        else:
            exit_cells = [(row, _GRID_SIZE - 1) for row in range(_GRID_SIZE)]
        non_exit_cells = [
            (row, col)
            for row in range(_GRID_SIZE)
            for col in range(_GRID_SIZE)
            if (row, col) not in exit_cells
        ]
        removed_count = self.rng.choice((1, 2))
        occupied = set(self.rng.sample(exit_cells, removed_count))
        occupied.update(self.rng.sample(non_exit_cells, 6 - removed_count))

        base_matrix: List[List[int | None]] = [[None for _ in range(_GRID_SIZE)] for _ in range(_GRID_SIZE)]
        token_values = list(range(1, 7))
        self.rng.shuffle(token_values)
        for value, (row, col) in zip(token_values, sorted(occupied)):
            base_matrix[row][col] = value

        target_matrix = _shift_matrix(base_matrix, *shift)
        surviving_values = sorted(value for row in target_matrix for value in row if value is not None)
        removed_values = sorted(set(token_values) - set(surviving_values))
        return {
            "base_matrix": [[value if value is None else int(value) for value in row] for row in base_matrix],
            "shift": list(shift),
            "shift_label": _SHIFT_LABELS[shift],
            "target_matrix": [[value if value is None else int(value) for value in row] for row in target_matrix],
            "answer_matrix": _dense_matrix(target_matrix),
            "token_values": sorted(token_values),
            "surviving_values": surviving_values,
            "removed_values": removed_values,
            "rule": "simultaneous_one_cell_translation_no_wrap",
        }

    @staticmethod
    def _draw_grid_background(draw: ImageDraw.ImageDraw) -> None:
        left = _GRID_LEFT
        top = _GRID_TOP
        right = left + _GRID_SIZE * _CELL_SIZE
        bottom = top + _GRID_SIZE * _CELL_SIZE
        index_font = scaled_font(10)
        draw.rounded_rectangle(
            (left - 7, top - 7, right + 7, bottom + 7),
            radius=15,
            fill=(245, 248, 248),
            outline=_ACCENT,
            width=3,
        )
        for row in range(_GRID_SIZE):
            for col in range(_GRID_SIZE):
                x0 = left + col * _CELL_SIZE
                y0 = top + row * _CELL_SIZE
                draw.rectangle(
                    (x0, y0, x0 + _CELL_SIZE, y0 + _CELL_SIZE),
                    fill=_GRID_FILL,
                    outline=_GRID_LINE,
                    width=2,
                )
        for index in range(_GRID_SIZE):
            draw.text(
                (left + (index + 0.5) * _CELL_SIZE, top - 12),
                str(index + 1),
                font=index_font,
                fill=_MUTED,
                anchor="mm",
            )
            draw.text(
                (left - 16, top + (index + 0.5) * _CELL_SIZE),
                str(index + 1),
                font=index_font,
                fill=_MUTED,
                anchor="mm",
            )

    @staticmethod
    def _draw_cells(draw: ImageDraw.ImageDraw, matrix: Sequence[Sequence[int | None]]) -> None:
        inset = 8
        token_font = scaled_font(19)
        for row, values in enumerate(matrix):
            for col, value in enumerate(values):
                if value is None:
                    continue
                x0 = _GRID_LEFT + col * _CELL_SIZE + inset
                y0 = _GRID_TOP + row * _CELL_SIZE + inset
                x1 = _GRID_LEFT + (col + 1) * _CELL_SIZE - inset
                y1 = _GRID_TOP + (row + 1) * _CELL_SIZE - inset
                draw.rounded_rectangle(
                    (x0, y0, x1, y1),
                    radius=9,
                    fill=_PALETTE[int(value)],
                    outline=(255, 255, 255),
                    width=3,
                )
                color = _PALETTE[int(value)]
                luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
                draw.text(
                    ((x0 + x1) / 2, (y0 + y1) / 2 + 1),
                    str(int(value)),
                    font=token_font,
                    fill=_INK if luminance >= 165 else (255, 255, 255),
                    anchor="mm",
                )

    def _render_state(self, task: Dict[str, Any], matrix: Sequence[Sequence[int | None]], *, status: str) -> Image.Image:
        del status
        image, draw = new_state_canvas(_BG, _CARD, (216, 221, 218))
        rule_font = scaled_font(12)
        arrow_font = scaled_font(30)
        shift = tuple(int(value) for value in task["shift"])
        draw.rounded_rectangle((167, 91, 345, 118), radius=13, fill=(249, 244, 232), outline=(232, 196, 128), width=2)
        draw.text((193, 104), _SHIFT_ARROWS[shift], font=arrow_font, fill=_ARROW, anchor="mm")
        draw.text((270, 104), "1", font=rule_font, fill=_INK, anchor="mm")
        self._draw_grid_background(draw)
        self._draw_cells(draw, matrix)
        return finish_state_canvas(image)

    def _render_puzzle_image(self, task: Dict[str, Any]) -> Image.Image:
        return self._render_state(task, task["base_matrix"], status="Input state")

    def _render_solution_image(self, task: Dict[str, Any]) -> Image.Image:
        return self._render_state(task, task["target_matrix"], status="Final state after one legal translation")

    def _question_text(self, task: Dict[str, Any]) -> str:
        return (
            f"Translate every numbered token {task['shift_label']} without wraparound and return the complete final "
            "4 by 4 row-major integer grid, using 0 for an empty cell."
        )

    def _answer_raw(self, task: Dict[str, Any]) -> str:
        return matrix_to_json(task["answer_matrix"])

    def _structured_answer(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "base_matrix": task["base_matrix"],
            "shift": task["shift"],
            "shift_label": task["shift_label"],
            "target_matrix": task["target_matrix"],
            "answer_matrix": task["answer_matrix"],
            "shift_states": [task["base_matrix"], task["target_matrix"]],
            "token_values": task["token_values"],
            "surviving_values": task["surviving_values"],
            "removed_values": task["removed_values"],
            "rule": task["rule"],
            "grid_bbox": [
                _GRID_LEFT,
                _GRID_TOP,
                _GRID_LEFT + _GRID_SIZE * _CELL_SIZE,
                _GRID_TOP + _GRID_SIZE * _CELL_SIZE,
            ],
            "cell_size": _CELL_SIZE,
        }

    def _visual_input_description(self, task: Dict[str, Any]) -> str:
        return (
            "The supplied image shows one fixed 4 by 4 grid with 1-based row numbers on the left edge and column "
            "numbers on the top edge. A white cell is empty. Each colored rounded token is labeled with its "
            "unchanged integer ID from 1 to 6. The compact direction badge shows an arrow and the numeral 1 for a "
            "one-cell displacement."
        )

    def _visual_output_description(self, task: Dict[str, Any]) -> str:
        return f"Move every numbered token exactly one cell {task['shift_label']} at the same time."

    def _contract(self, task: Dict[str, Any]) -> VisualReasoningContract:
        visual_rule = self._visual_output_description(task)
        return VisualReasoningContract(
            input_description=self._visual_input_description(task),
            symbol_legend=(
                "Each colored rounded token is labeled with its fixed integer identity from 1 to 6.",
                "A white grid cell is empty and is represented by integer 0 in the text answer.",
                "The printed edge numbers count rows 1-4 from the top and columns 1-4 from the left.",
                "The arrow badge gives the direction, and its numeral 1 gives the displacement in cells.",
            ),
            rules=(
                visual_rule,
                "Read every source position before moving any token; never continue from a partially translated grid.",
                "A token whose destination lies beyond the 4 by 4 boundary disappears rather than wrapping to the opposite edge.",
            ),
            modality_outputs={
                "ti2v": (
                    "Show the supplied input, then commit the complete translation as one simultaneous update: every "
                    "surviving numbered token appears at its destination cell together, and any token shifted beyond "
                    "the boundary is gone at that same commit."
                ),
                "solution_image": "Render only the completed final grid after the legal translation.",
            },
            preservation_constraints=(
                "Keep the 4 by 4 grid, the direction badge, and every surviving ID-color pair unchanged.",
                "Change only token positions.",
            ),
            transition_constraints=(
                "Commit the translation as one simultaneous state; never show an illegal partial grid in which only some tokens have moved.",
            ),
            ti2v_continuity={
                "timing": "discrete",
                "moving_nouns": "the numbered tokens",
                "static_nouns": "the grid frame, the printed edge numbers, the direction badge, the background, and the camera",
                "hold_noun": "the exact final grid",
            },
            answer=AnswerContract(
                kind="grid",
                format_template="Answer: [[<integer>, ...], ...].",
                parser_profile="json_integer_grid",
                verifier_profile="one_cell_translation_grid_replay",
                format_example="Answer: [[98, 99, ...], [100, 101, ...], ...].",
                accepts_multiple_valid_answers=False,
            ),
            render_profile="grid_shift_palette_cells_v2",
        )

    def _build_prompts(self, task: Dict[str, Any], answer_raw: str) -> Dict[str, str]:
        del answer_raw
        contract = self._contract(task)
        prompts = {
            "question": self._question_text(task),
            **{f"{mode}_prompt": contract.build_prompt(mode) for mode in contract.prompt_modes()},
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid grid-shift prompt contract: {'; '.join(issues)}")
        return prompts

    def _render_video(self, task: Dict[str, Any], output_path: Path) -> int:
        renders = [self._render_puzzle_image(task), self._render_solution_image(task)]
        frames = [np.asarray(image, dtype=np.uint8) for image in discrete_video_frames(renders)]
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)

    def make_sample(self) -> tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        answer_raw = self._answer_raw(task)
        contract = self._contract(task)
        prompts = self._build_prompts(task, answer_raw)
        puzzle_image = self._render_puzzle_image(task)
        solution_image = self._render_solution_image(task)

        sample_id = f"{self.TASK_NAME}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / "puzzle.png"
        solution_path = sample_dir / "solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        solution_video_path: Path | None = None
        video_num_frames: int | None = None
        if self.generate_videos:
            candidate_path = sample_dir / "solution.webm"
            video_num_frames = self._render_video(task, candidate_path)
            if video_num_frames:
                solution_video_path = candidate_path

        structured = self._structured_answer(task)
        input_images = [puzzle_path.as_posix()]
        return {
            "id": sample_id,
            "task_name": self.TASK_NAME,
            "task_family": self.TASK_FAMILY,
            "task_contract": contract.to_dict(),
            "question": prompts["question"],
            "answer": answer_raw,
            "structured_answer": structured,
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "reasoning_images": input_images,
            "option_images": [],
            "option_labels": [],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path.as_posix() if solution_video_path else None,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }, puzzle_image, solution_image


__all__ = ["GridShiftPattern", "_shift_matrix"]

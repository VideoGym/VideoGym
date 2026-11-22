from __future__ import annotations

import random
import tempfile
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
from data.common.scalar_lists import (
    format_nonnegative_integer_list_answer,
)


pattern_instructions = {
    "arc1d_compact_nonzero": (
        "The input shows one row of empty slots and colored nonzero tokens. Repeatedly apply synchronous one-cell "
        "left-compaction rounds until every token is packed against the left edge, preserving token identity and order."
    )
}

VIDEOGEN_INSTRUCTION_COMMON = (
    "Hold the input row, then show every synchronous one-cell left-compaction round in order and hold the final row."
)

_INPUT_DESC = (
    "The input image shows one indexed row of 10 to 14 slots. A pale slot is empty and counts as value 0. Every "
    "nonzero slot contains one colored token labeled with its digit from 1 to 9; that color-digit pair is the "
    "token's unchanged identity."
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render the stable final row after all nonzero tokens have compacted to the left, preserving every token digit, "
    "color, and left-to-right order."
)
_TI2V_OUTPUT = (
    "Generate a video that starts from the supplied input row and commits each synchronous round as one complete "
    "row state: every eligible token appears one slot to the left together. Continue round by round until the row "
    "is stable."
)

_CANVAS = (512, 512)
_BG = (246, 248, 247)
_CARD = (255, 255, 253)
_INK = (34, 40, 46)
_MUTED = (88, 96, 103)
_SLOT_FILL = (232, 236, 239)
_SLOT_LINE = (174, 184, 190)
_ACCENT = (45, 132, 151)

# The evaluator duplicates this fixed palette instead of trusting sample metadata.
_TOKEN_PALETTE = (
    _SLOT_FILL,
    (220, 68, 55),
    (237, 126, 41),
    (205, 170, 35),
    (62, 164, 85),
    (42, 160, 155),
    (55, 119, 210),
    (102, 83, 196),
    (164, 74, 184),
    (218, 73, 137),
)


def _compact_nonzero_left(row: Sequence[int]) -> List[int]:
    nonzero = [int(value) for value in row if int(value) != 0]
    return nonzero + [0] * (len(row) - len(nonzero))


def _eligible_movers(row: Sequence[int]) -> List[int]:
    return [index for index in range(1, len(row)) if int(row[index]) != 0 and int(row[index - 1]) == 0]


def _synchronous_compaction_round(row: Sequence[int]) -> List[int]:
    result = [int(value) for value in row]
    for index in _eligible_movers(row):
        result[index - 1] = int(row[index])
        result[index] = 0
    return result


def _compaction_states(row: Sequence[int]) -> List[List[int]]:
    states = [[int(value) for value in row]]
    while states[-1] != _compact_nonzero_left(states[-1]):
        next_state = _synchronous_compaction_round(states[-1])
        if next_state == states[-1]:
            raise RuntimeError("Stable compaction stalled before reaching the compact row.")
        states.append(next_state)
    return states


class Arc1DCompactNonzeroPattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = "arc1d_compact_nonzero"
    task_family = "next_state_prediction"
    task_mechanic = "synchronous_stable_compaction"

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        sequence_length: int | None = None,
        target_size: tuple[int, int] = _CANVAS,
        unique: bool = True,
        video: bool = True,
        video_fps: int = 16,
    ) -> None:
        if tuple(target_size) != _CANVAS:
            raise ValueError("arc1d_compact_nonzero requires the native 512x512 render contract")
        if int(video_fps) != 16:
            raise ValueError("arc1d_compact_nonzero videos must use 16 fps")
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_arc1d_compact"
        super().__init__(resolved)
        self.rng = random.Random(seed)
        self.sequence_length = int(sequence_length) if sequence_length is not None else None
        self.unique = bool(unique)
        self.video = bool(video)
        self.video_fps = int(video_fps)
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _sample_row(self) -> List[int]:
        target_rounds = self.rng.randint(3, 6)
        for _attempt in range(5000):
            length = self.sequence_length or self.rng.randint(10, 14)
            length = max(10, min(14, int(length)))
            token_count = self.rng.randint(3, min(7, length - 4, 9))
            positions = sorted(self.rng.sample(range(length), token_count))
            if positions[0] == 0:
                continue
            digits = self.rng.sample(range(1, 10), token_count)
            row = [0] * length
            for position, digit in zip(positions, digits):
                row[position] = digit
            states = _compaction_states(row)
            rounds = len(states) - 1
            if rounds != target_rounds:
                continue
            if not any(len(_eligible_movers(state)) >= 2 for state in states[:-1]):
                continue
            if sum(1 for left, right in zip(positions, positions[1:]) if right - left > 1) < 1:
                continue
            return row
        raise RuntimeError("Failed to sample a nontrivial stable left-compaction row.")

    @staticmethod
    def _row_geometry(length: int) -> tuple[int, int, int, int]:
        cell_size = min(38, 400 // max(1, length))
        row_width = cell_size * length
        left = (512 - row_width) // 2
        top = 214
        return left, top, cell_size, row_width

    @staticmethod
    def _token_text_color(value: int) -> tuple[int, int, int]:
        if value == 0:
            return _MUTED
        red, green, blue = _TOKEN_PALETTE[value]
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        return _INK if luminance > 155 else (255, 255, 255)

    def _draw_slots(self, draw, length: int) -> tuple[int, int, int, int]:
        left, top, cell_size, row_width = self._row_geometry(length)
        index_font = scaled_font(max(9, min(12, cell_size // 3)))
        for index in range(length):
            x0 = left + index * cell_size
            x1 = x0 + cell_size
            draw.rounded_rectangle(
                (x0 + 2, top + 2, x1 - 2, top + cell_size - 2),
                radius=max(5, cell_size // 5),
                fill=_SLOT_FILL,
                outline=_SLOT_LINE,
                width=2,
            )
            draw.text((x0 + cell_size / 2, top + cell_size + 15), str(index + 1), font=index_font, fill=_MUTED, anchor="mm")
        draw.rounded_rectangle(
            (left - 7, top - 7, left + row_width + 7, top + cell_size + 7),
            radius=13,
            outline=_ACCENT,
            width=3,
        )
        return left, top, cell_size, row_width

    def _draw_token(self, draw, *, value: int, center_x: float, top: int, cell_size: int) -> None:
        token_font = scaled_font(max(14, min(20, int(round(cell_size * 0.52)))))
        half = cell_size / 2 - 4
        draw.rounded_rectangle(
            (center_x - half, top + 4, center_x + half, top + cell_size - 4),
            radius=max(5, cell_size // 5),
            fill=_TOKEN_PALETTE[value],
            outline=(255, 255, 255),
            width=2,
        )
        draw.text(
            (center_x, top + cell_size / 2 + 1),
            str(value),
            font=token_font,
            fill=self._token_text_color(value),
            anchor="mm",
        )

    def _render_state(self, row: Sequence[int], *, round_index: int, target_rounds: int) -> Image.Image:
        del round_index, target_rounds
        image, draw = new_state_canvas(_BG, _CARD, (216, 221, 218))
        left, top, cell_size, _row_width = self._draw_slots(draw, len(row))
        for index, raw_value in enumerate(row):
            # An empty slot's state lives in its pale colour alone.
            value = int(raw_value)
            if value != 0:
                center_x = left + (index + 0.5) * cell_size
                self._draw_token(draw, value=value, center_x=center_x, top=top, cell_size=cell_size)
        return finish_state_canvas(image)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=_INPUT_DESC,
            symbol_legend=(
                "A pale slot is empty, counts as value 0, and stays empty unless a token moves into it.",
                "Digits 1 through 9 are colored tokens; each digit-color pair is one unchanged token identity.",
                "The small numbers below the row are fixed slot positions counted from 1 at the left end.",
            ),
            rules=(
                "In one synchronous round, inspect the current row before any token moves.",
                "Every nonzero token whose immediately adjacent left slot is 0 moves exactly one slot left.",
                "All eligible moves in that round happen simultaneously; use the completed row as the input to the next round.",
                "Repeat until no 0 occurs anywhere to the left of a nonzero token.",
                "Preserve row length, every token digit and color, and the original left-to-right token order.",
            ),
            modality_outputs={
                "ti2v": _TI2V_OUTPUT,
                "solution_image": _SOLUTION_IMAGE_OUTPUT,
            },
            preservation_constraints=(
                "Keep the row geometry, slot numbers, token styling, and row length unchanged.",
                "Never jump over a slot, move right, swap token order, change a digit or color, add a token, or remove a token.",
            ),
            transition_constraints=(
                "Commit every completed synchronous one-cell round in exact order as one complete row state; do not skip a round or insert a sequential partial state.",
                "After the first stable row appears, do not continue moving tokens.",
            ),
            ti2v_continuity={
                "timing": "discrete",
                "moving_nouns": "the colored tokens",
                "static_nouns": "the slot row, the printed slot numbers, the background, and the camera",
                "hold_noun": "the exact stable final row",
            },
            answer=AnswerContract(
                kind="nonnegative_integer_list",
                format_template="Answer: [<nonnegative integer>, ...].",
                parser_profile="json_integer_list",
                verifier_profile="stable_left_compaction_replay",
                format_example="Answer: [98, 99, 100, ...].",
                accepts_multiple_valid_answers=False,
            ),
            render_profile="arc1d_stable_compaction_tokens_v2",
        )

    def _prompts(self) -> tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {f"{mode}_prompt": contract.build_prompt(mode) for mode in contract.prompt_modes()}
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid arc1d compact prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _render_video(self, states: Sequence[Sequence[int]], *, output_path: Path) -> int:
        target_rounds = len(states) - 1
        renders = [
            self._render_state(state, round_index=index, target_rounds=target_rounds)
            for index, state in enumerate(states)
        ]
        frames = [np.asarray(image, dtype=np.uint8) for image in discrete_video_frames(renders)]
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)


    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1

        input_row = self._sample_row()
        states = _compaction_states(input_row)
        output_row = states[-1]
        rounds = len(states) - 1
        answer_text = format_nonnegative_integer_list_answer(output_row)
        contract, prompts = self._prompts()

        sample_dir = self.output_dir / sample_id
        puzzle_dir = sample_dir / "puzzles"
        solution_dir = sample_dir / "solutions"
        video_dir = sample_dir / "videos"
        steps_dir = sample_dir / "steps"
        for directory in (puzzle_dir, solution_dir, video_dir, steps_dir):
            directory.mkdir(parents=True, exist_ok=True)

        puzzle_image = self._render_state(states[0], round_index=0, target_rounds=rounds)
        solution_image = self._render_state(states[-1], round_index=rounds, target_rounds=rounds)
        puzzle_path = puzzle_dir / f"{sample_id}__puzzle.png"
        solution_path = solution_dir / f"{sample_id}__solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        solution_video_path: Path | None = None
        video_num_frames: int | None = None
        if self.video:
            video_path = video_dir / f"{sample_id}__solution.webm"
            rendered_frames = self._render_video(states, output_path=video_path)
            if rendered_frames > 0:
                solution_video_path = video_path
                video_num_frames = rendered_frames

        left, top, cell_size, row_width = self._row_geometry(len(input_row))
        return {
            "id": sample_id,
            "answer": answer_text,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": "Apply synchronous one-cell left-compaction rounds until the row is stable.",
            "structured_answer": {
                "input_row": list(input_row),
                "output_row": list(output_row),
                "compaction_states": [list(state) for state in states],
                "rounds": rounds,
                "row_length": len(input_row),
                "row_bbox": [left, top, left + row_width, top + cell_size],
                "cell_size": cell_size,
                "token_values": [int(value) for value in input_row if int(value) != 0],
                "rule": "synchronous_stable_compact_nonzero_left",
                "task_mechanic": self.task_mechanic,
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

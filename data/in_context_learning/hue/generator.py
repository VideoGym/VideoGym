from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw

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
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.pipeline.media import load_font
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts


pattern_instructions = {
    "hue": (
        "A fixed indexed 5 by 5 color grid repeats a cyclic five-color progression. One cell is blank; infer its "
        "unique color from the complete rows and columns and return the printed palette label."
    )
}

VIDEOGEN_INSTRUCTION_COMMON = ""

_CANVAS = (512, 512)
_BG = (246, 248, 247)
_CARD = (255, 255, 253)
_INK = (34, 40, 46)
_MUTED = (88, 96, 103)
_GRID_FILL = (249, 250, 250)
_GRID_LINE = (172, 181, 186)
_ACCENT = (49, 132, 148)
_QUESTION = (111, 119, 125)
_GRID_SIZE = 5
_CELL_SIZE = 55
_GRID_LEFT = 118
_GRID_TOP = 103
_ALLOWED_STEP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 3),
    (2, 1),
    (3, 1),
    (2, 3),
    (3, 2),
)

# Fixed task-level palette. The evaluator duplicates these values so it can
# parse candidate videos without trusting generator-emitted color metadata.
HUE_PALETTE: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("RED", (224, 79, 73)),
    ("ORANGE", (232, 139, 56)),
    ("YELLOW", (222, 187, 55)),
    ("GREEN", (77, 166, 104)),
    ("TEAL", (48, 164, 159)),
    ("BLUE", (67, 121, 196)),
    ("PURPLE", (143, 91, 181)),
    ("PINK", (207, 91, 141)),
)
_COLOR_BY_NAME = dict(HUE_PALETTE)


def _build_hue_board(
    palette_names: Sequence[str],
    *,
    row_step: int,
    col_step: int,
) -> List[List[str]]:
    size = len(palette_names)
    return [
        [palette_names[(row * row_step + col * col_step) % size] for col in range(_GRID_SIZE)]
        for row in range(_GRID_SIZE)
    ]


def _replay_hue_visible_state(
    palette_names: Sequence[str],
    *,
    row_step: int,
    col_step: int,
    visible_board_names: Sequence[Sequence[str | None]],
    missing_cell: Sequence[int],
) -> Dict[str, Any] | None:
    """Replay the indexed cycle from only the rule parameters and visible clues."""
    names = [str(name).upper() for name in palette_names]
    if len(names) != _GRID_SIZE or len(set(names)) != _GRID_SIZE:
        return None
    if (row_step, col_step) not in _ALLOWED_STEP_PAIRS:
        return None
    if len(missing_cell) != 2:
        return None
    try:
        missing = (int(missing_cell[0]), int(missing_cell[1]))
    except (TypeError, ValueError):
        return None
    if not all(0 <= coordinate < _GRID_SIZE for coordinate in missing):
        return None
    if len(visible_board_names) != _GRID_SIZE or any(
        len(row) != _GRID_SIZE for row in visible_board_names
    ):
        return None

    absent_cells = [
        (row_index, col_index)
        for row_index, row in enumerate(visible_board_names)
        for col_index, value in enumerate(row)
        if value is None
    ]
    if absent_cells != [missing]:
        return None

    solved_board = _build_hue_board(names, row_step=row_step, col_step=col_step)
    for row_index, row in enumerate(visible_board_names):
        for col_index, value in enumerate(row):
            if value is None:
                continue
            if str(value).upper() != solved_board[row_index][col_index]:
                return None
    return {
        "missing_color_name": solved_board[missing[0]][missing[1]],
        "solved_board_names": solved_board,
    }


class HuePattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = "hue"
    task_family = "in_context_learning"

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        video: bool = True,
        video_fps: int = 16,
        target_size: tuple[int, int] = _CANVAS,
        unique: bool = True,
    ) -> None:
        del target_size, unique
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_hue"
        super().__init__(resolved)
        self.rng = random.Random(seed)
        self.video = bool(video)
        self.video_fps = int(video_fps)
        if self.video_fps != 16:
            raise ValueError("hue videos must be rendered at exactly 16fps")
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _generate_task(self) -> Dict[str, Any]:
        palette_names = self.rng.sample([name for name, _color in HUE_PALETTE], k=5)
        row_step, col_step = self.rng.choice(_ALLOWED_STEP_PAIRS)
        board_names = _build_hue_board(palette_names, row_step=row_step, col_step=col_step)
        missing = (self.rng.randrange(_GRID_SIZE), self.rng.randrange(_GRID_SIZE))
        visible_board_names: List[List[str | None]] = [list(row) for row in board_names]
        visible_board_names[missing[0]][missing[1]] = None
        replay = _replay_hue_visible_state(
            palette_names,
            row_step=row_step,
            col_step=col_step,
            visible_board_names=visible_board_names,
            missing_cell=missing,
        )
        if replay is None or replay["missing_color_name"] != board_names[missing[0]][missing[1]]:
            raise AssertionError("generated hue clues do not replay to a unique missing color")
        return {
            "palette_names": palette_names,
            "row_step": row_step,
            "col_step": col_step,
            "board_names": board_names,
            "visible_board_names": visible_board_names,
            "missing_cell": list(missing),
            "correct_color_name": board_names[missing[0]][missing[1]],
            "rule": "cyclic_five_color_row_column_progression",
        }

    @staticmethod
    def _grid_bbox() -> list[int]:
        return [
            _GRID_LEFT,
            _GRID_TOP,
            _GRID_LEFT + _GRID_SIZE * _CELL_SIZE,
            _GRID_TOP + _GRID_SIZE * _CELL_SIZE,
        ]

    @staticmethod
    def _render_card_hi(task: Dict[str, Any]) -> Image.Image:
        """Draw the card scaffold on a shared supersampled canvas."""

        factor = DOCTRINE_SUPERSAMPLE
        image, sdraw = new_scaled_canvas(512, 512, factor, _BG)
        sdraw.rounded_rectangle((18, 18, 494, 494), radius=22, fill=_CARD, outline=(216, 221, 218), width=2)

        right = _GRID_LEFT + _GRID_SIZE * _CELL_SIZE
        bottom = _GRID_TOP + _GRID_SIZE * _CELL_SIZE
        sdraw.rounded_rectangle(
            (_GRID_LEFT - 8, _GRID_TOP - 8, right + 8, bottom + 8),
            radius=15,
            fill=(245, 248, 248),
            outline=_ACCENT,
            width=3,
        )
        index_font = load_font(10 * factor)
        for index in range(_GRID_SIZE):
            sdraw.text((_GRID_LEFT + (index + 0.5) * _CELL_SIZE, _GRID_TOP - 17), str(index + 1), font=index_font, fill=_MUTED, anchor="mm")
            sdraw.text((_GRID_LEFT - 18, _GRID_TOP + (index + 0.5) * _CELL_SIZE), str(index + 1), font=index_font, fill=_MUTED, anchor="mm")
        for row in range(_GRID_SIZE):
            for col in range(_GRID_SIZE):
                x0 = _GRID_LEFT + col * _CELL_SIZE
                y0 = _GRID_TOP + row * _CELL_SIZE
                sdraw.rectangle((x0, y0, x0 + _CELL_SIZE, y0 + _CELL_SIZE), fill=_GRID_FILL, outline=_GRID_LINE, width=2)

        names = list(task["palette_names"])
        legend_width = 82
        legend_left = (512 - legend_width * len(names)) // 2
        for index, name in enumerate(names):
            center_x = legend_left + index * legend_width + legend_width // 2
            color = _COLOR_BY_NAME[name]
            sdraw.rounded_rectangle((center_x - 24, 421, center_x + 24, 443), radius=8, fill=color, outline=(255, 255, 255), width=2)
            sdraw.text((center_x, 459), name, font=load_font(9 * factor), fill=_INK, anchor="mm")
        return image

    @staticmethod
    def _draw_fixed_cells_hi(image: Image.Image, task: Dict[str, Any], *, show_missing: bool) -> None:
        factor = DOCTRINE_SUPERSAMPLE
        sdraw = ScaledDraw(ImageDraw.Draw(image), factor)
        missing = tuple(int(value) for value in task["missing_cell"])
        for row, names in enumerate(task["board_names"]):
            for col, name in enumerate(names):
                if (row, col) == missing and not show_missing:
                    sdraw.text(
                        (_GRID_LEFT + (col + 0.5) * _CELL_SIZE, _GRID_TOP + (row + 0.5) * _CELL_SIZE + 1),
                        "?",
                        font=load_font(25 * factor),
                        fill=_QUESTION,
                        anchor="mm",
                    )
                    continue
                x0 = _GRID_LEFT + col * _CELL_SIZE + 6
                y0 = _GRID_TOP + row * _CELL_SIZE + 6
                x1 = _GRID_LEFT + (col + 1) * _CELL_SIZE - 6
                y1 = _GRID_TOP + (row + 1) * _CELL_SIZE - 6
                sdraw.rounded_rectangle((x0, y0, x1, y1), radius=9, fill=_COLOR_BY_NAME[name], outline=(255, 255, 255), width=2)

    def _render_state_hi(self, task: Dict[str, Any], *, solved: bool) -> Image.Image:
        image = self._render_card_hi(task)
        self._draw_fixed_cells_hi(image, task, show_missing=solved)
        return image

    def _render_state(self, task: Dict[str, Any], *, solved: bool, status: str) -> Image.Image:
        del status
        return finish_scaled_canvas(self._render_state_hi(task, solved=solved), _CANVAS)

    def _render_transition(self, task: Dict[str, Any], fraction: float) -> Image.Image:
        image = self._render_state_hi(task, solved=False)
        row, col = (int(value) for value in task["missing_cell"])
        center_x = _GRID_LEFT + (col + 0.5) * _CELL_SIZE
        center_y = _GRID_TOP + (row + 0.5) * _CELL_SIZE
        progress = max(0.0, min(1.0, float(fraction)))
        half = _CELL_SIZE / 2 - 6
        if progress <= 0.0:
            return finish_scaled_canvas(image, _CANVAS)
        factor = DOCTRINE_SUPERSAMPLE
        opacity = int(round(255 * progress))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ScaledDraw(ImageDraw.Draw(overlay), factor)
        overlay_draw.rounded_rectangle(
            (center_x - half, center_y - half, center_x + half, center_y + half),
            radius=9,
            # Geometry is complete from the first transition frame.  Only
            # this missing-cell element changes opacity; the rest of the
            # source scene is never part of the blend.
            fill=(*_COLOR_BY_NAME[task["correct_color_name"]], opacity),
            outline=(255, 255, 255, opacity),
            width=2,
        )
        composited = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        return finish_scaled_canvas(composited, _CANVAS)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The supplied image shows one indexed 5 by 5 grid and a five-entry color legend. Exactly one cell "
                "contains a gray question mark; every other cell contains one legend color."
            ),
            symbol_legend=(
                GRID_XY_CONVENTION_SENTENCE,
                "Read the five legend swatches from left to right as cycle positions 1, 2, 3, 4, and 5.",
                "The printed uppercase legend names are the only allowed text answers.",
                "The gray question mark is the single missing color; all colored cells are fixed clues.",
            ),
            rules=(
                "The five displayed colors use the legend's left-to-right cyclic order throughout the grid.",
                "Moving down one row advances every cycle position by the row step; moving right one column advances every cycle position by the column step.",
                "Cycle positions wrap around the five-entry legend, so advancing past position 5 continues from position 1.",
                "The row step and column step are distinct values chosen from 1, 2, and 3; infer both steps from the 24 fixed clues.",
                "Fill the question-mark cell with the unique legend color consistent with both directions.",
                "Do not move, recolor, add, or remove any other cell.",
            ),
            modality_outputs={
                "ti2v": (
                    "Generate a fixed-camera video. First hold the "
                    "exact supplied grid. Then show one complete normal-sized region of the inferred color in only the question-mark "
                    "cell and raise that whole region's local opacity continuously from invisible to fully opaque until it replaces the "
                    "gray question mark. Keep every other cell, label, "
                    "legend swatch, border, caption, and camera fixed throughout. End with a readable hold of the exact completed grid."
                ),
                "solution_image": (
                    "Render one completed grid. Replace only the gray question-mark cell with the inferred legend color, "
                    "using the same rounded cell geometry, white inset border, alignment, and saturation as the 24 fixed cells. "
                    "Keep every fixed cell, row/column label, legend swatch and printed label, caption, border, background, spacing, "
                    "and camera exactly unchanged; do not add steps, arrows, highlights, formulas, or answer banners."
                ),
            },
            preservation_constraints=(
                "Keep every non-missing cell, row/column label, legend swatch, caption, border, and camera unchanged.",
                "Only the question-mark cell may change.",
            ),
            transition_constraints=(
                "Begin with a readable hold of the unchanged input grid.",
                "Fade the complete normal-sized missing-cell fill uniformly at its fixed final footprint; do not grow it from a point, wipe it directionally, hard-cut to the answer, or show a stable wrong color.",
                "Do not crossfade the whole frame, teleport the final fill into place, flicker between colors, zoom, pan, crop, or alter any fixed cell while the missing-cell element is fading.",
                "End with a readable hold of the exact completed grid.",
            ),
            answer=AnswerContract(
                kind="typed_text",
                format_template="Answer: <printed-label>.",
                parser_profile="strict_uppercase_palette_label_with_period",
                verifier_profile="hue_visible_clue_cycle_replay_v3",
                format_example="Answer: ULTRAVIOLET.",
                accepts_multiple_valid_answers=False,
            ),
            render_profile="indexed_hue_cycle_grid_v3",
        )

    def _prompts(self) -> tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid hue prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _render_video(self, task: Dict[str, Any], output_path: Path) -> int:
        start = np.asarray(self._render_state(task, solved=False, status="Input state · one cell is missing"), dtype=np.uint8)
        final = np.asarray(self._render_state(task, solved=True, status="Completed cyclic color grid"), dtype=np.uint8)
        frames: List[np.ndarray] = [start.copy() for _ in range(8)]
        for fraction in (0.16, 0.32, 0.50, 0.70, 0.86):
            frames.append(np.asarray(self._render_transition(task, fraction), dtype=np.uint8))
        frames.extend(final.copy() for _ in range(8))
        if len(frames) % 4 != 1:
            raise AssertionError("hue video frame count must satisfy 4N+1")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)

    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1
        task = self._generate_task()
        contract, prompts = self._prompts()
        puzzle = self._render_state(task, solved=False, status="Input state · one cell is missing")
        solution = self._render_state(task, solved=True, status="Completed cyclic color grid")

        sample_dir = self.output_dir / sample_id
        puzzle_dir = sample_dir / "puzzles"
        solution_dir = sample_dir / "solutions"
        video_dir = sample_dir / "videos"
        for directory in (puzzle_dir, solution_dir, video_dir):
            directory.mkdir(parents=True, exist_ok=True)
        puzzle_path = puzzle_dir / f"{sample_id}__puzzle.png"
        solution_path = solution_dir / f"{sample_id}__solution.png"
        puzzle.save(puzzle_path)
        solution.save(solution_path)

        video_path: Path | None = None
        video_num_frames: int | None = None
        if self.video:
            candidate = video_dir / f"{sample_id}__solution.webm"
            video_num_frames = self._render_video(task, candidate)
            if video_num_frames:
                video_path = candidate

        answer_raw = str(task["correct_color_name"])
        board_colors = [[list(_COLOR_BY_NAME[name]) for name in row] for row in task["board_names"]]
        structured = {
            **task,
            "board_size": _GRID_SIZE,
            "board_colors": board_colors,
            "correct_color": list(_COLOR_BY_NAME[answer_raw]),
            "palette": {name: list(_COLOR_BY_NAME[name]) for name in task["palette_names"]},
            "allowed_step_pairs": [list(pair) for pair in _ALLOWED_STEP_PAIRS],
            "palette_cycle_direction": "legend_left_to_right_modulo_5",
            "grid_bbox": self._grid_bbox(),
            "cell_size": _CELL_SIZE,
            "answer_schema": {
                "schema_version": "color_label_v1",
                "kind": "color_label",
                "format_template": "Answer: <printed-label>.",
                "parser_profile": "strict_uppercase_palette_label_with_period",
                "final_deliverable": {
                    "ti2v": ["full_video"],
                },
                "accepts_multiple_valid_answers": False,
                "extra_text_allowed": False,
            },
            "intermediate_supervision_schema": {
                "ti2v": ["input_hold", "continuous_missing_cell_fill", "final_hold"],
            },
            "verifier_contract": {
                "profile": "hue_visible_clue_cycle_replay_v3",
                "semantic_source": "palette_cycle_plus_twenty_four_visible_grid_clues",
                "component_scores": [
                    "parse_score",
                    "source_replay_score",
                    "step_uniqueness_score",
                    "final_state_score",
                    "state_transition_score",
                    "preservation_score",
                    "media_contract_score",
                ],
                "hard_gates": [
                    "strict_parse",
                    "independent_visible_clue_replay",
                    "unique_step_pair",
                    "correct_missing_color",
                    "fixed_cell_preservation",
                    "complete_footprint_local_opacity",
                    "full_scene_preservation",
                    "media_contract",
                ],
            },
        }
        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": "Which printed palette label uniquely completes the cyclic color pattern?",
            "answer": answer_raw,
            "structured_answer": structured,
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": self.relativize_path(puzzle_path),
            "reasoning_image": self.relativize_path(puzzle_path),
            # The primary puzzle image already contains the complete grid and
            # legend. Empty auxiliary lists keep the canonical layer from
            # materializing a duplicate copy of that same image as an input.
            "reasoning_images": [],
            "option_images": [],
            "option_labels": [],
            "solution_image_path": self.relativize_path(solution_path),
            "solution_video_path": self.relativize_path(video_path) if video_path is not None else None,
            "video_fps": self.video_fps if video_path is not None else None,
            "video_num_frames": video_num_frames,
        }

    def make_sample(self) -> tuple[Dict[str, Any], Image.Image, Image.Image]:
        record = self.create_puzzle()
        puzzle = Image.open(self.output_dir / record["image"]).convert("RGB").copy()
        solution = Image.open(self.output_dir / record["solution_image_path"]).convert("RGB").copy()
        return record, puzzle, solution


from data.in_context_learning.core.polish import polished_icl_generator

HuePattern = polished_icl_generator(HuePattern)

__all__ = [
    "HUE_PALETTE",
    "HuePattern",
    "_ALLOWED_STEP_PAIRS",
    "_build_hue_board",
    "_replay_hue_visible_state",
]

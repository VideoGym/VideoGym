"""Visual A:B::C:? completion with a uniquely identifiable rigid transform.

The input is a single 512x512 card containing one demonstrated transform,
one query shape, and four candidates.  The demonstrated A->B pair is
generated only when exactly one transform in the declared pool explains it.
That transform is then applied to C, whose five transform outcomes are also
pairwise distinct, so exactly one candidate is correct.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.render_primitives import DOCTRINE_SUPERSAMPLE, ScaledDraw
from data.common.video_encoding import encode_rgb_frames_to_webm, pad_frames_to_wan_compatible_length
from data.common.pipeline.media import LETTER_OPTIONS, load_font
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.in_context_learning.core.option_delivery import (
    OPTION_MODALITY_OUTPUTS,
    OPTION_PRESERVATION_CONSTRAINTS,
    OPTION_TRANSITION_CONSTRAINTS,
)
from data.common.pipeline.choice_generator import ChoiceTaskGenerator
from data.common.pipeline.choice_layout import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    ACCENT_RED,
    NEUTRAL_DARK,
    assemble_choice_order,
    compose_tile_panel,
)


_TRANSFORMS = ("rot90cw", "rot90ccw", "rot180", "reflect_v", "reflect_h")
_TRANSFORM_GLYPHS = {
    "rot90cw": "90° ↻",
    "rot90ccw": "90° ↺",
    "rot180": "180°",
    "reflect_v": "↔",
    "reflect_h": "↕",
}
_SOURCE_MARK = (18, 184, 204)
_TARGET_MARK = (236, 92, 179)
_PREVIEW = (151, 86, 190)
_RULE_BADGE = (61, 110, 183)


def _rot(cells: Iterable[Sequence[int]], quarter_turns_cw: int) -> List[Tuple[int, int]]:
    """Rotate grid coordinates clockwise; rows increase downward on screen."""

    out = [(int(col), int(row)) for col, row in cells]
    for _ in range(quarter_turns_cw % 4):
        out = [(-row, col) for col, row in out]
    return out


def _reflect_v(cells: Iterable[Sequence[int]]) -> List[Tuple[int, int]]:
    """Reflect across the vertical axis: x -> -x."""

    return [(-int(col), int(row)) for col, row in cells]


def _reflect_h(cells: Iterable[Sequence[int]]) -> List[Tuple[int, int]]:
    """Reflect across the horizontal axis: y -> -y."""

    return [(int(col), -int(row)) for col, row in cells]


def _apply(cells: Iterable[Sequence[int]], transform: str) -> List[Tuple[int, int]]:
    if transform == "rot90cw":
        return _rot(cells, 1)
    if transform == "rot180":
        return _rot(cells, 2)
    if transform == "rot90ccw":
        return _rot(cells, 3)
    if transform == "reflect_v":
        return _reflect_v(cells)
    if transform == "reflect_h":
        return _reflect_h(cells)
    raise ValueError(f"Unsupported analogy transform: {transform}")


def _normalise(cells: Iterable[Sequence[int]]) -> Tuple[Tuple[int, int], ...]:
    normalized_input = [(int(col), int(row)) for col, row in cells]
    if not normalized_input:
        return tuple()
    min_col = min(col for col, _row in normalized_input)
    min_row = min(row for _col, row in normalized_input)
    return tuple(sorted((col - min_col, row - min_row) for col, row in normalized_input))


def _transform_signatures(cells: Iterable[Sequence[int]]) -> Dict[str, Tuple[Tuple[int, int], ...]]:
    return {transform: _normalise(_apply(cells, transform)) for transform in _TRANSFORMS}


def _has_unique_transform_outcomes(cells: Iterable[Sequence[int]]) -> bool:
    signatures = tuple(_transform_signatures(cells).values())
    return len(set(signatures)) == len(_TRANSFORMS)


def _grid_geometry(
    box: Tuple[int, int, int, int],
    cells: Iterable[Sequence[int]],
    *,
    grid_n: int,
) -> Tuple[Tuple[Tuple[int, int], ...], int, int, int, int, int, int]:
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    side = min(box_width, box_height) - 6
    cell_size = max(8, side // grid_n)
    inner = cell_size * grid_n
    center_x = (box[0] + box[2]) // 2
    center_y = (box[1] + box[3]) // 2
    x0 = center_x - inner // 2
    y0 = center_y - inner // 2
    normalized = _normalise(cells)
    max_col = max((col for col, _row in normalized), default=0)
    max_row = max((row for _col, row in normalized), default=0)
    offset_col = (grid_n - 1 - max_col) // 2
    offset_row = (grid_n - 1 - max_row) // 2
    return normalized, cell_size, inner, x0, y0, offset_col, offset_row


def _mini_grid_cell_centers(
    box: Tuple[int, int, int, int],
    cells: Iterable[Sequence[int]],
    *,
    grid_n: int = 4,
) -> List[List[float]]:
    normalized, cell_size, _inner, x0, y0, offset_col, offset_row = _grid_geometry(
        box, cells, grid_n=grid_n
    )
    return [
        [
            x0 + (col + offset_col + 0.5) * cell_size,
            y0 + (row + offset_row + 0.5) * cell_size,
        ]
        for col, row in normalized
    ]


def _source_sub_boxes(
    box: Tuple[int, int, int, int],
) -> Dict[str, Tuple[int, int, int, int]]:
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    row_height = box_height // 2 - 6
    sub_width = (box_width - 60) // 2
    return {
        "a": (box[0] + 4, box[1] + 4, box[0] + 4 + sub_width, box[1] + 4 + row_height),
        "b": (box[2] - 4 - sub_width, box[1] + 4, box[2] - 4, box[1] + 4 + row_height),
        "c": (
            box[0] + 4,
            box[1] + 4 + row_height + 12,
            box[0] + 4 + sub_width,
            box[1] + 4 + 2 * row_height + 12,
        ),
        "q": (
            box[2] - 4 - sub_width,
            box[1] + 4 + row_height + 12,
            box[2] - 4,
            box[1] + 4 + 2 * row_height + 12,
        ),
    }


def _fixed_render_boxes() -> Dict[str, Any]:
    # ``compose_tile_panel`` places the source frame at y=16 and gives the
    # drawer a 12-pixel inset.  Export the geometry actually drawn rather than
    # the former caption-era y=48 offset.
    source_inner = (30, 28, 482, 236)
    source_boxes = _source_sub_boxes(source_inner)
    candidate_panels = (
        (16, 262, 249, 372),
        (263, 262, 496, 372),
        (16, 386, 249, 496),
        (263, 386, 496, 496),
    )
    candidate_inner = tuple(
        (panel[0] + 10, panel[1] + 32, panel[2] - 10, panel[3] - 10)
        for panel in candidate_panels
    )
    a_box, b_box = source_boxes["a"], source_boxes["b"]
    first_arrow_y = (a_box[1] + a_box[3]) // 2
    center_x = (a_box[2] + b_box[0]) // 2
    return {
        "source_inner_bbox": list(source_inner),
        "source_grid_bboxes": {key: list(value) for key, value in source_boxes.items()},
        "candidate_panel_bboxes": [list(panel) for panel in candidate_panels],
        "candidate_inner_bboxes": [list(panel) for panel in candidate_inner],
        "rule_badge_bbox": [center_x - 27, first_arrow_y - 14, center_x + 27, first_arrow_y + 14],
    }


def _canvas_scaled_draw(image: Image.Image) -> ScaledDraw:
    """Wrap a compose_tile_panel working canvas in unscaled coordinates.

    The compositor may hand drawers a supersampled canvas; the working factor
    is its width over the 512-pixel target, and one for legacy 1x callers.
    """

    return ScaledDraw(ImageDraw.Draw(image), max(1, image.width // 512))


def _draw_mini_grid(
    image: Image.Image,
    box: Tuple[int, int, int, int],
    cells: Iterable[Sequence[int]],
    *,
    color: Sequence[int],
    grid_n: int = 4,
    source_marks: bool = False,
    target_marks: bool = False,
    preview_count: int | None = None,
) -> None:
    sdraw = _canvas_scaled_draw(image)
    normalized, cell_size, inner, x0, y0, offset_col, offset_row = _grid_geometry(
        box, cells, grid_n=grid_n
    )
    for index in range(grid_n + 1):
        sdraw.line(
            ((x0 + index * cell_size, y0), (x0 + index * cell_size, y0 + inner)),
            fill=(228, 232, 238),
            width=1,
        )
        sdraw.line(
            ((x0, y0 + index * cell_size), (x0 + inner, y0 + index * cell_size)),
            fill=(228, 232, 238),
            width=1,
        )
    sdraw.rectangle((x0, y0, x0 + inner, y0 + inner), outline=NEUTRAL_DARK, width=2)
    visible_count = len(normalized) if preview_count is None else max(0, min(preview_count, len(normalized)))
    for cell_index, (col, row) in enumerate(normalized):
        grid_x = x0 + (col + offset_col) * cell_size
        grid_y = y0 + (row + offset_row) * cell_size
        cell_box = (grid_x + 2, grid_y + 2, grid_x + cell_size - 2, grid_y + cell_size - 2)
        if preview_count is None:
            sdraw.rectangle(cell_box, fill=tuple(int(value) for value in color))
        elif cell_index < visible_count:
            sdraw.rounded_rectangle(cell_box, radius=2, fill=None, outline=_PREVIEW, width=3)
        center_x = grid_x + cell_size / 2.0
        center_y = grid_y + cell_size / 2.0
        if source_marks:
            sdraw.ellipse(
                (center_x - 3.5, center_y - 3.5, center_x + 3.5, center_y + 3.5),
                fill=_SOURCE_MARK,
                outline=(35, 85, 92),
                width=1,
            )
        if target_marks:
            sdraw.ellipse(
                (center_x - 5.0, center_y - 5.0, center_x + 5.0, center_y + 5.0),
                fill=None,
                outline=_TARGET_MARK,
                width=3,
            )


def _arrow(sdraw: ScaledDraw, x1: int, y: int, x2: int) -> None:
    sdraw.line(((x1, y), (x2 - 6, y)), fill=NEUTRAL_DARK, width=2)
    sdraw.polygon(((x2, y), (x2 - 8, y - 5), (x2 - 8, y + 5)), fill=NEUTRAL_DARK)


# Every source shape below is asymmetric under the complete transform pool.
# This is deliberately stricter than merely requiring four distinct choices:
# the demonstration pair itself must identify one and only one transform.
_F_SHAPE = ((0, -1), (1, -1), (0, 0), (1, 0), (0, 1))
_L_SHAPE = ((-1, -1), (-1, 0), (-1, 1), (0, 1))
_J_SHAPE = ((1, -1), (1, 0), (0, 1), (1, 1))
_P_SHAPE = ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2))
_N_SHAPE = ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3))
_Y_SHAPE = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1))
_SHAPE_BANK = (_F_SHAPE, _L_SHAPE, _J_SHAPE, _P_SHAPE, _N_SHAPE, _Y_SHAPE)


class AnalogyCompletionPattern(ChoiceTaskGenerator):
    task_name = "analogy_completion"
    task_family = "in_context_learning"
    video_fps = 16

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        video_fps: int = 16,
        generate_videos: bool = True,
        **_kwargs: Any,
    ) -> None:
        resolved_seed = random_seed if random_seed is not None else seed
        self.rng = random.Random(resolved_seed)
        self.video_fps = int(video_fps)
        self.generate_videos = bool(generate_videos)
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.gettempdir()) / f"videogym_{self.task_name}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = (512, 512)
        self._sample_index = 0

    def _generate_task(self) -> Dict[str, Any]:
        for _attempt in range(200):
            transform = self.rng.choice(_TRANSFORMS)
            shape_a, shape_c = self.rng.sample(_SHAPE_BANK, 2)
            if not _has_unique_transform_outcomes(shape_a) or not _has_unique_transform_outcomes(shape_c):
                continue
            output_b = _apply(shape_a, transform)
            output_d = _apply(shape_c, transform)
            matching_transforms = [
                candidate_transform
                for candidate_transform in _TRANSFORMS
                if _normalise(_apply(shape_a, candidate_transform)) == _normalise(output_b)
            ]
            if matching_transforms != [transform]:
                continue
            distractors = [
                _apply(shape_c, candidate_transform)
                for candidate_transform in _TRANSFORMS
                if candidate_transform != transform
            ]
            self.rng.shuffle(distractors)
            distractors = distractors[:3]
            candidate_signatures = {_normalise(output_d), *(_normalise(cells) for cells in distractors)}
            if len(candidate_signatures) != 4:
                continue
            return {
                "transform": transform,
                "shape_a": list(shape_a),
                "shape_b": output_b,
                "shape_c": list(shape_c),
                "answer": output_d,
                "distractors": distractors,
                "color_a": self.rng.choice((ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE)),
                "color_c": self.rng.choice((ACCENT_ORANGE, ACCENT_RED)),
            }
        raise RuntimeError("analogy_completion: failed to find a uniquely identifiable instance")

    @staticmethod
    def _state_trace() -> List[Dict[str, Any]]:
        return [
            {"pair_marks": False, "show_rule": False, "query_marks": False, "preview_count": 0, "fill_query": False, "select": False},
            {"pair_marks": True, "show_rule": False, "query_marks": False, "preview_count": 0, "fill_query": False, "select": False},
            {"pair_marks": True, "show_rule": True, "query_marks": False, "preview_count": 0, "fill_query": False, "select": False},
            {"pair_marks": True, "show_rule": True, "query_marks": True, "preview_count": 0, "fill_query": False, "select": False},
            {"pair_marks": True, "show_rule": True, "query_marks": True, "preview_count": 5, "fill_query": False, "select": False},
            {"pair_marks": True, "show_rule": True, "query_marks": True, "preview_count": 5, "fill_query": True, "select": True},
        ]

    @staticmethod
    def _clean_solution_state() -> Dict[str, Any]:
        return {
            "pair_marks": False,
            "show_rule": False,
            "query_marks": False,
            "preview_count": 5,
            "fill_query": True,
            "select": True,
        }

    def _draw_source(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int],
        task: Dict[str, Any],
    ) -> None:
        sdraw = _canvas_scaled_draw(image)
        state = task.get("_render_state") or self._state_trace()[0]
        boxes = _source_sub_boxes(box)
        a_box, b_box, c_box, q_box = boxes["a"], boxes["b"], boxes["c"], boxes["q"]
        _draw_mini_grid(
            image,
            a_box,
            task["shape_a"],
            color=task["color_a"],
            source_marks=bool(state["pair_marks"]),
        )
        _draw_mini_grid(
            image,
            b_box,
            task["shape_b"],
            color=task["color_a"],
            target_marks=bool(state["pair_marks"]),
        )
        _draw_mini_grid(
            image,
            c_box,
            task["shape_c"],
            color=task["color_c"],
            source_marks=bool(state["query_marks"]),
        )
        if state["fill_query"]:
            _draw_mini_grid(
                image,
                q_box,
                task["answer"],
                color=task["color_c"],
                target_marks=bool(state["query_marks"]),
            )
        elif int(state["preview_count"]) > 0:
            _draw_mini_grid(
                image,
                q_box,
                task["answer"],
                color=task["color_c"],
                preview_count=int(state["preview_count"]),
            )
        else:
            # The placeholder must occupy exactly the same centered square the
            # answer grid will occupy, so the question-mark card and the
            # completed card are the same size.
            _normalized, _cell_size, inner, x0, y0, _oc, _or = _grid_geometry(
                q_box, [], grid_n=4
            )
            sdraw.rounded_rectangle(
                (x0, y0, x0 + inner, y0 + inner),
                radius=4,
                outline=NEUTRAL_DARK,
                width=2,
                fill=(248, 250, 252),
            )
            sdraw.text(
                (x0 + inner // 2, y0 + inner // 2),
                "?",
                fill=NEUTRAL_DARK,
                font=load_font(max(24, int(inner * 0.5)) * sdraw.factor),
                anchor="mm",
            )
        first_arrow_y = (a_box[1] + a_box[3]) // 2
        second_arrow_y = (c_box[1] + c_box[3]) // 2
        _arrow(sdraw, a_box[2] + 4, first_arrow_y, b_box[0] - 4)
        _arrow(sdraw, c_box[2] + 4, second_arrow_y, q_box[0] - 4)
        if state["show_rule"]:
            center_x = (a_box[2] + b_box[0]) // 2
            badge = (center_x - 27, first_arrow_y - 14, center_x + 27, first_arrow_y + 14)
            sdraw.rounded_rectangle(badge, radius=7, fill=_RULE_BADGE, outline=(38, 72, 126), width=1)
            sdraw.text(
                (center_x, first_arrow_y),
                _TRANSFORM_GLYPHS[task["transform"]],
                fill=(255, 255, 255),
                font=load_font(13 * sdraw.factor),
                anchor="mm",
            )

    @staticmethod
    def _candidate_drawer(cells: Sequence[Sequence[int]], color: Sequence[int]) -> Callable:
        captured = [tuple(int(value) for value in cell) for cell in cells]

        def drawer(image: Image.Image, box: Tuple[int, int, int, int]) -> None:
            _draw_mini_grid(image, box, captured, color=color)

        return drawer

    def _make_candidate_drawers(self, task: Dict[str, Any]):
        correct = self._candidate_drawer(task["answer"], task["color_c"])
        distractors = [
            self._candidate_drawer(cells, task["color_c"]) for cells in task["distractors"]
        ]
        return correct, distractors, {
            "transform": task["transform"],
            "shape_a": [list(cell) for cell in task["shape_a"]],
            "shape_b": [list(cell) for cell in task["shape_b"]],
            "shape_c": [list(cell) for cell in task["shape_c"]],
            "correct_d": [list(cell) for cell in task["answer"]],
        }

    def _render_scene(
        self,
        task: Dict[str, Any],
        ordered_candidates: Sequence[Sequence[Sequence[int]]],
        correct_letter: str,
        state: Dict[str, Any],
    ) -> Image.Image:
        render_task = dict(task)
        render_task["_render_state"] = dict(state)

        def draw_source(image: Image.Image, box: Tuple[int, int, int, int]) -> None:
            self._draw_source(image, box, render_task)

        drawers = [
            self._candidate_drawer(candidate, task["color_c"]) for candidate in ordered_candidates
        ]
        return compose_tile_panel(
            draw_source=draw_source,
            candidate_drawers=drawers,
            source_caption="Analogy",
            highlight_letter=correct_letter if state["select"] else None,
            supersample=DOCTRINE_SUPERSAMPLE,
        )

    def _render_video(
        self,
        task: Dict[str, Any],
        ordered_candidates: Sequence[Sequence[Sequence[int]]],
        correct_letter: str,
        output_path: Path,
    ) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        frames: List[np.ndarray] = []
        states = self._state_trace()

        def append_state(state: Dict[str, Any], hold: int) -> None:
            frame = np.asarray(
                self._render_scene(task, ordered_candidates, correct_letter, state), dtype=np.uint8
            )
            frames.extend(frame.copy() for _ in range(hold))

        append_state(states[0], 8)
        append_state(states[1], 6)
        append_state(states[2], 6)
        append_state(states[3], 6)
        for preview_count in range(1, len(_normalise(task["answer"])) + 1):
            preview_state = dict(states[4], preview_count=preview_count)
            append_state(preview_state, 3)
        filled_unselected = dict(states[5], select=False)
        append_state(filled_unselected, 6)
        append_state(states[5], 12)
        frames = [np.asarray(frame, dtype=np.uint8) for frame in pad_frames_to_wan_compatible_length(frames)]
        if len(frames) > 157 or not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    @staticmethod
    def _contract() -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The supplied image is one analogy card. Its top source panel shows A->B and C->?, and its lower half contains four candidate grids A-D."
            ),
            symbol_legend=(
                "Filled cells form the shapes; all mini-grids share the same orientation and rows increase downward.",
                "In intermediate evidence, cyan centres and magenta rings mark corresponding cells, while a purple preview and a transform badge show the inferred change without altering the candidates.",
                "The compact arrow badge denotes clockwise 90 degrees, counterclockwise 90 degrees, 180 degrees, vertical-axis reflection, or horizontal-axis reflection.",
            ),
            rules=(
                "Exactly one declared rigid transform maps A to B after translation within the mini-grid is ignored.",
                "Apply that same transform to C without changing its cell count or color.",
                "Choose the unique candidate whose occupied-cell shape equals the transformed C shape.",
            ),
            modality_outputs=OPTION_MODALITY_OUTPUTS,
            preservation_constraints=OPTION_PRESERVATION_CONSTRAINTS,
            transition_constraints=OPTION_TRANSITION_CONSTRAINTS,
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="uppercase_option_letter",
                verifier_profile="analogy_transform_replay_and_candidate_uniqueness",
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_analogy_transform_trace_v2",
        )

    def _prompts(self) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "question": "Infer the unique transform shown by A->B, apply it to C, and choose the matching candidate.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid analogy-completion prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def create_puzzle(self) -> Dict[str, Any]:
        task = self._generate_task()
        contract, prompts = self._prompts()
        correct_drawer, distractor_drawers, _meta = self._make_candidate_drawers(task)
        _drawers, correct_letter, choice_indices = assemble_choice_order(
            self.rng, correct_drawer, distractor_drawers
        )
        source_candidates = [task["answer"], *task["distractors"]]
        ordered_candidates = [source_candidates[index] for index in choice_indices]
        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        reasoning_final_path = sample_dir / f"{sample_id}__reasoning_final.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"
        puzzle = self._render_scene(task, ordered_candidates, correct_letter, self._state_trace()[0])
        solution = self._render_scene(task, ordered_candidates, correct_letter, self._clean_solution_state())
        reasoning_final = self._render_scene(
            task, ordered_candidates, correct_letter, self._state_trace()[-1]
        )
        puzzle.save(puzzle_path)
        solution.save(solution_path)
        reasoning_final.save(reasoning_final_path)
        solution_video_path, video_num_frames = self._render_video(
            task, ordered_candidates, correct_letter, video_path
        )
        candidate_payload = [
            {"label": LETTER_OPTIONS[index], "cells": [list(cell) for cell in candidate]}
            for index, candidate in enumerate(ordered_candidates)
        ]
        matching_transforms = [
            transform
            for transform in _TRANSFORMS
            if _normalise(_apply(task["shape_a"], transform)) == _normalise(task["shape_b"])
        ]
        render_geometry = _fixed_render_boxes()
        source_boxes = {
            key: tuple(value)
            for key, value in render_geometry["source_grid_bboxes"].items()
        }
        candidate_inner = [tuple(value) for value in render_geometry["candidate_inner_bboxes"]]
        render_geometry.update(
            {
                "a_cell_centers": _mini_grid_cell_centers(source_boxes["a"], task["shape_a"]),
                "b_cell_centers": _mini_grid_cell_centers(source_boxes["b"], task["shape_b"]),
                "c_cell_centers": _mini_grid_cell_centers(source_boxes["c"], task["shape_c"]),
                "query_cell_centers": _mini_grid_cell_centers(source_boxes["q"], task["answer"]),
                "candidate_cell_centers": [
                    _mini_grid_cell_centers(candidate_inner[index], candidate)
                    for index, candidate in enumerate(ordered_candidates)
                ],
            }
        )
        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": prompts["question"],
            "answer": correct_letter,
            "correct_option": correct_letter,
            "structured_answer": {
                "answer_schema": {
                    "schema_version": "option_letter_v1",
                    "kind": "option_letter",
                    "format_template": FOUR_CHOICE_CODEC.schema,
                    "format_example": FOUR_CHOICE_CODEC.example,
                    "parser_profile": "strict_uppercase_option_letter_a_to_d",
                    "final_deliverable": { "ti2v": ["full_video"]},
                    "accepts_multiple_valid_answers": False,
                    "extra_text_allowed": False,
                },
                "intermediate_supervision_schema": {
                    "ti2v": ["input_hold", "uniform_complete_green_border_fade", "final_hold"],
                },
                "verifier_contract": {
                    "profile": "analogy_transform_replay_and_candidate_uniqueness",
                },
                "correct_option": correct_letter,
                "transform": task["transform"],
                "matching_transforms": matching_transforms,
                "shape_a": [list(cell) for cell in task["shape_a"]],
                "shape_b": [list(cell) for cell in task["shape_b"]],
                "shape_c": [list(cell) for cell in task["shape_c"]],
                "correct_d": [list(cell) for cell in task["answer"]],
                "shape_a_rgb": list(task["color_a"]),
                "shape_c_rgb": list(task["color_c"]),
                "candidates": candidate_payload,
                "state_trace": self._state_trace(),
                "render_geometry": {
                    "canvas_size": [512, 512],
                    "grid_n": 4,
                    "source_mark_rgb": list(_SOURCE_MARK),
                    "target_mark_rgb": list(_TARGET_MARK),
                    "preview_rgb": list(_PREVIEW),
                    "rule_badge_rgb": list(_RULE_BADGE),
                    **render_geometry,
                },
            },
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "reasoning_images": [puzzle_path.as_posix()],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }


from data.in_context_learning.core.polish import polished_icl_generator

AnalogyCompletionPattern = polished_icl_generator(AnalogyCompletionPattern)

__all__ = [
    "AnalogyCompletionPattern",
    "_TRANSFORMS",
    "_apply",
    "_has_unique_transform_outcomes",
    "_normalise",
]

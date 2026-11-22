"""Triangle-inequality classification with cumulative side-sum comparisons."""

from __future__ import annotations

from data.common.base import RenderedTaskGenerator
from data.geometry.core.rendered_contract import draw_answer_border, finalize_geometry_sample

import random
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from PIL import Image, ImageDraw

from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.pipeline.media import LETTER_OPTIONS, downsample, load_font
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC


pattern_instructions = {
    "triangle_inequality_check": (
        "A single clean square canvas shows four labeled side-length triples as same-scale colored segments. "
        "The task is to identify the unique triple whose two shorter sides do not sum to more than its longest side."
    ),
}

_LABELS = tuple(LETTER_OPTIONS[:4])
_VERIFIER_PROFILE = "triangle_inequality_visible_length_multimodal_hard_gate_v1"
_PERFORMANCE_DIAGNOSTICS = {
    "required_measurements": (
        "cold_start_ms",
        "warm_p50_ms",
        "warm_p95_ms",
        "samples_per_second",
        "batch_size",
        "peak_memory_mb",
        "decoded_frames_per_video",
        "early_stop_rate",
        "cache_hit_rate",
    ),
    "automatic_failure_thresholds": False,
}
_SCALE = 2
_BG = (255, 255, 255)
_PANEL_FILL = (248, 250, 252)
_PANEL_EDGE = (157, 164, 174)
_GUIDE = (216, 221, 228)
_TEXT = (42, 46, 52)
_SELECT = (196, 36, 46)
_SIDE_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (55, 104, 190),
    (46, 151, 111),
    (224, 139, 49),
)
_PANEL_BBOXES: Tuple[Tuple[int, int, int, int], ...] = (
    (16, 12, 496, 130),
    (16, 136, 496, 254),
    (16, 260, 496, 378),
    (16, 384, 496, 502),
)
_INPUT_BAR_START_X = 70
_INPUT_BAR_SCALE = 6.25
_DIVIDER_X = 254
_COMPARE_START_X = 308
_COMPARE_BAR_SCALE = 4.1
_COMPARE_VALUE_X = 449
# Legibility floor for the eight drawn comparison bars, in side-length units.
# Each bar is stroked 7px wide with a radius-4 endpoint dot at both ends, and
# the serial-stroke observer reads a bar only outside a radius-(width+3)=10px
# disk at each end -- so 20px of every bar is unobservable by construction and
# its eight progress bins collapse below that length.  Measured against this
# renderer: 6 units (24.6px) leaves the two end bins empty, 7 units (28.7px)
# is the first length whose worst bin clears the observer's 4px minimum, and 9
# units (36.9px) is the first whose worst bin doubles that (14px) while making
# each quarter-progress step (9.2px) longer than the 8px endpoint dot, so the
# growth is readable frame to frame rather than swallowed by its own marker.
_MIN_COMPARISON_BAR = 9
# The longest side still has to end well short of the printed value column.
_MAX_COMPARISON_BAR = 22


@lru_cache(maxsize=16)
def _font(size: int):
    return load_font(size)


def _normalize_canvas_size(value: object) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return 512, 512


def _triangle_analysis(triple: Sequence[int]) -> Dict[str, Any]:
    if len(triple) != 3:
        raise ValueError("Each candidate must contain exactly three side lengths.")
    values = [int(value) for value in triple]
    if any(value <= 0 for value in values):
        raise ValueError("Side lengths must be positive integers.")
    ordered_indices = sorted(range(3), key=lambda index: (values[index], index))
    shorter_indices = ordered_indices[:2]
    longest_index = ordered_indices[2]
    shorter_sum = values[shorter_indices[0]] + values[shorter_indices[1]]
    longest = values[longest_index]
    margin = shorter_sum - longest
    return {
        "display_sides": values,
        "sorted_sides": sorted(values),
        "shorter_indices": shorter_indices,
        "longest_index": longest_index,
        "shorter_sum": shorter_sum,
        "longest": longest,
        "margin": margin,
        "can_form_triangle": margin > 0,
    }


def _make_invalid_triple(rng: random.Random) -> Tuple[int, int, int]:
    # The failing row's shorter-side sum is its shorter comparison bar, so the
    # legibility floor lands on `first + second`; the longest side only has to
    # stay inside the comparison zone.
    first = rng.randint(3, 9)
    second = rng.randint(max(first, _MIN_COMPARISON_BAR - first), 12)
    gap = rng.randint(0, min(3, _MAX_COMPARISON_BAR - first - second))
    sides = [first, second, first + second + gap]
    rng.shuffle(sides)
    return tuple(sides)


def _make_valid_triple(rng: random.Random) -> Tuple[int, int, int]:
    # A valid row's shorter bar is its longest side, so the floor lands there.
    # `first + second >= _MIN_COMPARISON_BAR + 2` keeps the margin >= 2 window
    # non-empty above the floor, which makes the whole draw constructive.
    first = rng.randint(4, 11)
    second = rng.randint(max(first, _MIN_COMPARISON_BAR + 2 - first), 15)
    longest = rng.randint(
        max(second, first + second - 8, _MIN_COMPARISON_BAR),
        min(20, first + second - 2),
    )
    sides = [first, second, longest]
    rng.shuffle(sides)
    return tuple(sides)


def _comparison_payload(triples: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for index, triple in enumerate(triples):
        analysis = _triangle_analysis(triple)
        payload.append(
            {
                "label": _LABELS[index],
                "display_sides": analysis["display_sides"],
                "sorted_sides": analysis["sorted_sides"],
                "shorter_indices": analysis["shorter_indices"],
                "longest_index": analysis["longest_index"],
                "shorter_sum": analysis["shorter_sum"],
                "longest": analysis["longest"],
                "margin": analysis["margin"],
                "can_form_triangle": analysis["can_form_triangle"],
            }
        )
    return payload


def _answer_schema() -> Dict[str, Any]:
    return {
        "schema_version": "uppercase_option_letter_v1",
        "kind": "option_letter",
        "parser_profile": "strict_uppercase_option_letter_a_to_d",
        "format_template": FOUR_CHOICE_CODEC.schema,
        "format_example": FOUR_CHOICE_CODEC.example,
        "judged_artifacts": {
            "ti2v": ["full_video"],
        },
        "accepts_multiple_valid_answers": False,
        "extra_text_allowed": False,
    }


def _verifier_contract() -> Dict[str, Any]:
    return {
        "profile": _VERIFIER_PROFILE,
        "semantic_source": (
            "the four visible positive integer side triples, their same-scale colored segments, the strict "
            "two-shorter-sides sum rule, and the fixed comparison-panel geometry"
        ),
        "diagnostic_metrics": {
            "text": [
                "source_replay_valid",
                "unique_failure_valid",
                "predicted_option",
                "expected_option",
            ],
            "image": [
                "source_geometry_valid",
                "row_progress",
                "selection_progress",
                "static_mutation_fraction",
            ],
            "video": [
                "decoded_frame_count",
                "recognized_frame_count",
                "phase_sequence",
                "global_row_synchrony_valid",
                "growth_monotonic",
                "selection_after_all_rows",
                "static_mutation_fraction",
            ],
        },
        "binary_predicate": (
            "strict option parse, independent replay of all four strict inequalities, exactly one failing row, "
            "semantic final-image or whole-video comparison parsing, unique final selection, fixed-content "
            "preservation, and agreement of the visible result with the replayed answer"
        ),
        "performance_diagnostics": {
            "required_measurements": list(_PERFORMANCE_DIAGNOSTICS["required_measurements"]),
            "automatic_failure_thresholds": False,
        },
    }


def _render_geometry() -> Dict[str, Any]:
    return {
        "canvas_size": [512, 512],
        "panel_bboxes": [list(box) for box in _PANEL_BBOXES],
        "input_bar_start_x": _INPUT_BAR_START_X,
        "input_bar_scale": _INPUT_BAR_SCALE,
        "divider_x": _DIVIDER_X,
        "comparison_start_x": _COMPARE_START_X,
        "comparison_bar_scale": _COMPARE_BAR_SCALE,
        "comparison_value_x": _COMPARE_VALUE_X,
        "side_rgbs": [list(color) for color in _SIDE_COLORS],
        "selection_rgb": list(_SELECT),
    }


def _expected_state_trace(correct_option: str) -> List[Dict[str, Any]]:
    return [
        {
            "phase": "input",
            "completed_rows": 0,
            "active_row": None,
            "selected_option": None,
        },
        {
            "phase": "global_comparison",
            "completed_rows": 4,
            "active_row": None,
            "selected_option": None,
        },
        {
            "phase": "final",
            "completed_rows": 4,
            "active_row": None,
            "selected_option": correct_option,
        },
    ]


def _replay_triangle_inequality(structured: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    triples = structured.get("triples")
    if not isinstance(triples, list) or len(triples) != 4:
        return {}, False
    try:
        normalized = [[int(value) for value in triple] for triple in triples]
        comparisons = _comparison_payload(normalized)
    except (TypeError, ValueError):
        return {}, False
    invalid_indices = [
        index for index, comparison in enumerate(comparisons) if not comparison["can_form_triangle"]
    ]
    if len(invalid_indices) != 1:
        return {"comparisons": comparisons, "invalid_indices": invalid_indices}, False
    invalid_index = invalid_indices[0]
    correct_option = _LABELS[invalid_index]
    expected_trace = _expected_state_trace(correct_option)
    expected_candidates = [
        {"label": _LABELS[index], "sides": list(triple)}
        for index, triple in enumerate(normalized)
    ]
    replay = {
        "triples": normalized,
        "comparisons": comparisons,
        "invalid_indices": invalid_indices,
        "invalid_index": invalid_index,
        "invalid_triple": normalized[invalid_index],
        "correct_option": correct_option,
        "state_trace": expected_trace,
        "render_geometry": _render_geometry(),
        "answer_schema": _answer_schema(),
        "verifier_contract": _verifier_contract(),
    }
    passed = (
        structured.get("candidates") == expected_candidates
        and structured.get("comparisons") == comparisons
        and int(structured.get("invalid_index", -1)) == invalid_index
        and structured.get("invalid_triple") == normalized[invalid_index]
        and structured.get("correct_option") == correct_option
        and structured.get("state_trace") == expected_trace
        and structured.get("render_geometry") == _render_geometry()
        and structured.get("answer_schema") == _answer_schema()
        and structured.get("verifier_contract") == _verifier_contract()
        and structured.get("multiple_solution_policy")
        == "exactly_one_of_A_through_D_fails_the_strict_triangle_inequality"
    )
    return replay, bool(passed)


def _panel_center(index: int) -> Tuple[float, float]:
    left, top, right, bottom = _PANEL_BBOXES[index]
    return (left + right) / 2.0, (top + bottom) / 2.0


def _draw_progress_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    lengths: Sequence[float],
    colors: Sequence[Tuple[int, int, int]],
    progress: float,
    *,
    scale: int,
) -> None:
    progress = max(0.0, min(1.0, float(progress)))
    total = sum(lengths)
    remaining = total * progress
    cursor = float(start[0]) * scale
    y = float(start[1]) * scale
    for length, color in zip(lengths, colors):
        visible = min(float(length), remaining)
        if visible > 0.0:
            end = cursor + visible * _COMPARE_BAR_SCALE * scale
            draw.line((cursor, y, end, y), fill=color, width=7 * scale)
            draw.ellipse((cursor - 4 * scale, y - 4 * scale, cursor + 4 * scale, y + 4 * scale), fill=color)
            draw.ellipse((end - 4 * scale, y - 4 * scale, end + 4 * scale, y + 4 * scale), fill=color)
            cursor = end
        remaining -= visible
        if remaining <= 0:
            break


def comparison_stroke_geometry(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Screen-space endpoints of the eight answer-bearing comparison bars.

    Row order follows the panel order A-D; within one row the shorter-side-sum
    bar precedes the longest-side bar.  Verifiers import this single source so
    their stroke grammar can never drift from the renderer.
    """

    strokes: List[Dict[str, Any]] = []
    for index, comparison in enumerate(task["comparisons"]):
        _center_x, center_y = _panel_center(index)
        sum_y = center_y - 18
        long_y = center_y + 18
        sum_width = float(comparison["shorter_sum"]) * _COMPARE_BAR_SCALE
        long_width = float(comparison["longest"]) * _COMPARE_BAR_SCALE
        strokes.append(
            {
                "label": f"{_LABELS[index]}_shorter_sum_bar",
                "start": (float(_COMPARE_START_X), float(sum_y)),
                "end": (float(_COMPARE_START_X) + sum_width, float(sum_y)),
                "width": 7.0,
            }
        )
        strokes.append(
            {
                "label": f"{_LABELS[index]}_longest_bar",
                "start": (float(_COMPARE_START_X), float(long_y)),
                "end": (float(_COMPARE_START_X) + long_width, float(long_y)),
                "width": 7.0,
            }
        )
    return strokes


def render_scene(
    task: Dict[str, Any],
    *,
    reveal_count: int = 0,
    active_index: int | None = None,
    active_progress: float = 1.0,
    completed_strokes: int | None = None,
    active_stroke_progress: float = 0.0,
    selection_progress: float = 0.0,
) -> Image.Image:
    """Render the four-row scene at any input/comparison/selection state."""

    scale = _SCALE
    image = Image.new("RGB", (512 * scale, 512 * scale), _BG)
    draw = ImageDraw.Draw(image)
    for index, panel in enumerate(_PANEL_BBOXES):
        box = tuple(value * scale for value in panel)
        draw.rounded_rectangle(box, radius=12 * scale, fill=_PANEL_FILL, outline=_PANEL_EDGE, width=2 * scale)
        left, top, right, bottom = panel
        center_y = (top + bottom) / 2.0
        draw.line((_DIVIDER_X * scale, (top + 8) * scale, _DIVIDER_X * scale, (bottom - 8) * scale), fill=_GUIDE, width=2 * scale)

        label_x = 39
        draw.ellipse(
            ((label_x - 16) * scale, (center_y - 16) * scale, (label_x + 16) * scale, (center_y + 16) * scale),
            fill=_BG,
            outline=_TEXT,
            width=2 * scale,
        )
        draw.text((label_x * scale, (center_y - 1) * scale), _LABELS[index], fill=_TEXT, font=_font(16 * scale), anchor="mm")

        for side_index, length in enumerate(task["triples"][index]):
            y = center_y + (side_index - 1) * 30
            x1 = _INPUT_BAR_START_X
            x2 = x1 + float(length) * _INPUT_BAR_SCALE
            color = _SIDE_COLORS[side_index]
            draw.line((x1 * scale, y * scale, x2 * scale, y * scale), fill=color, width=7 * scale)
            for x in (x1, x2):
                draw.ellipse(
                    ((x - 4) * scale, (y - 4) * scale, (x + 4) * scale, (y + 4) * scale),
                    fill=color,
                    outline=_TEXT,
                    width=1 * scale,
                )
            draw.text(((x2 + 12) * scale, (y - 1) * scale), str(length), fill=_TEXT, font=_font(13 * scale), anchor="mm")

        sum_y = center_y - 18
        long_y = center_y + 18

        if completed_strokes is not None:
            stroke_count = max(0, min(8, int(completed_strokes)))
            sum_stroke_index = index * 2
            long_stroke_index = sum_stroke_index + 1
            sum_progress = (
                1.0
                if sum_stroke_index < stroke_count
                else max(0.0, min(1.0, float(active_stroke_progress)))
                if sum_stroke_index == stroke_count and stroke_count < 8
                else 0.0
            )
            long_progress = (
                1.0
                if long_stroke_index < stroke_count
                else max(0.0, min(1.0, float(active_stroke_progress)))
                if long_stroke_index == stroke_count and stroke_count < 8
                else 0.0
            )
        else:
            row_progress = (
                1.0
                if index < reveal_count
                else max(0.0, min(1.0, float(active_progress)))
                if active_index == index
                else 0.0
            )
            sum_progress = long_progress = row_progress
        comparison = task["comparisons"][index]
        shorter_indices = comparison["shorter_indices"]
        longest_index = comparison["longest_index"]
        shorter_lengths = [task["triples"][index][side_index] for side_index in shorter_indices]
        shorter_colors = [_SIDE_COLORS[side_index] for side_index in shorter_indices]
        if sum_progress > 0.0:
            _draw_progress_line(
                draw,
                (_COMPARE_START_X, sum_y),
                shorter_lengths,
                shorter_colors,
                sum_progress,
                scale=scale,
            )
        if long_progress > 0.0:
            _draw_progress_line(
                draw,
                (_COMPARE_START_X, long_y),
                [comparison["longest"]],
                [_SIDE_COLORS[longest_index]],
                long_progress,
                scale=scale,
            )
        if sum_progress >= 1.0 and long_progress >= 1.0:
            draw.text((_COMPARE_VALUE_X * scale, sum_y * scale), str(comparison["shorter_sum"]), fill=_TEXT, font=_font(12 * scale), anchor="mm")
            draw.text((_COMPARE_VALUE_X * scale, long_y * scale), str(comparison["longest"]), fill=_TEXT, font=_font(12 * scale), anchor="mm")

        if selection_progress > 0.0 and index == task["invalid_index"]:
            draw_answer_border(
                draw,
                panel,
                scale=scale,
                radius=12,
                width=5,
                progress=selection_progress,
            )

    return downsample(image, scale)


class TriangleInequalityCheckPattern(RenderedTaskGenerator):
    task_name = "triangle_inequality_check"
    task_family = "geometry"
    video_fps = 16

    def __init__(
        self,
        output_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.image_size = _normalize_canvas_size(kwargs.get("image_size", 512))
        if self.image_size != (512, 512):
            raise ValueError("triangle_inequality_check requires a 512x512 canvas")
        self.video_fps = int(kwargs.get("video_fps", self.video_fps))
        self.generate_videos = bool(kwargs.get("generate_videos", kwargs.get("video", True)))
        seed = kwargs.get("random_seed", kwargs.get("seed"))
        self.rng = random.Random(seed)
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.gettempdir()) / "videogym_triangle_inequality_check"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0

    def _generate_task(self) -> Dict[str, Any]:
        for _ in range(80):
            invalid = _make_invalid_triple(self.rng)
            triples: List[Tuple[int, int, int]] = [invalid]
            signatures = {tuple(sorted(invalid))}
            for _candidate in range(300):
                if len(triples) == 4:
                    break
                candidate = _make_valid_triple(self.rng)
                signature = tuple(sorted(candidate))
                if signature in signatures:
                    continue
                signatures.add(signature)
                triples.append(candidate)
            if len(triples) != 4:
                continue
            self.rng.shuffle(triples)
            comparisons = _comparison_payload(triples)
            invalid_indices = [
                index for index, comparison in enumerate(comparisons) if not comparison["can_form_triangle"]
            ]
            if len(invalid_indices) != 1:
                continue
            invalid_index = invalid_indices[0]
            return {
                "triples": [list(triple) for triple in triples],
                "comparisons": comparisons,
                "invalid_index": invalid_index,
                "invalid_triple": list(triples[invalid_index]),
            }
        raise RuntimeError("Failed to construct a unique triangle-inequality task.")

    @staticmethod
    def _contract() -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The image contains rows A-D, each showing three exact positive-integer lengths as same-scale colored segments."
            ),
            symbol_legend=(
                "Within one candidate, the three colored segments are an unordered proposed side-length triple; segment color preserves the identity of the printed input length but does not itself indicate short, long, valid, or invalid.",
            ),
            rules=(
                "For each row, identify the two shorter printed integers and the longest integer.",
                "Three lengths form a non-degenerate triangle if and only if the sum of the two shorter values is strictly greater than the longest value.",
                "Equality is a degenerate boundary failure, and a smaller sum also fails; use the exact integers without visual tolerance.",
                "Select the unique row whose shorter-side sum is less than or equal to its longest value.",
            ),
            modality_outputs={
                "ti2v": "Hold the input, draw each comparison bar from its left endpoint to its right endpoint, completing one bar before starting the next, then smoothly change only the complete failing-row border to green and hold.",
                "solution_image": "Complete the two comparison bars and values for every row, then add exactly one green outer border around the unique failing row. Do not add pass/fail badges, checks, crosses, or a second selection.",
            },
            preservation_constraints=(
                "Keep all input segments, values, row labels, panels, dividers, layout, and camera unchanged.",
                "Only the four comparison-bar pairs, their final values, and one green outer border may be added.",
            ),
            transition_constraints=(
                "Begin with an undecided hold, then process rows A-D in order; within each row draw the shorter-side-sum bar first and the longest-side bar second.",
                "Every bar grows continuously from its left endpoint to its right endpoint, only one bar may be active at a time, and the derived values appear after both bars in that row are complete.",
                "Only then change the whole failing-row border gray-to-green synchronously, without perimeter tracing, and hold the solved state.",
            ),
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="strict_uppercase_option_letter_a_to_d",
                verifier_profile=_VERIFIER_PROFILE,
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_triangle_inequality_single_final_v3",
        )

    def _build_prompts(self) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "question": "Choose the unique side-length triple that cannot form a non-degenerate triangle.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid triangle-inequality prompt contract: {'; '.join(issues)}")
        return contract, prompts

    @staticmethod
    def _panel_center(index: int) -> Tuple[float, float]:
        return _panel_center(index)

    @staticmethod
    def _draw_progress_line(
        draw: ImageDraw.ImageDraw,
        start: Tuple[float, float],
        lengths: Sequence[float],
        colors: Sequence[Tuple[int, int, int]],
        progress: float,
        *,
        scale: int,
    ) -> None:
        _draw_progress_line(draw, start, lengths, colors, progress, scale=scale)

    def _draw_scene(
        self,
        task: Dict[str, Any],
        *,
        reveal_count: int = 0,
        active_index: int | None = None,
        active_progress: float = 1.0,
        completed_strokes: int | None = None,
        active_stroke_progress: float = 0.0,
        selection_progress: float = 0.0,
    ) -> Image.Image:
        return render_scene(
            task,
            reveal_count=reveal_count,
            active_index=active_index,
            active_progress=active_progress,
            completed_strokes=completed_strokes,
            active_stroke_progress=active_stroke_progress,
            selection_progress=selection_progress,
        )

    def _render_video(self, task: Dict[str, Any], *, output_path: Path) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        frames: List[np.ndarray] = []
        puzzle = np.asarray(self._draw_scene(task).convert("RGB"), dtype=np.uint8)
        frames.extend(puzzle.copy() for _ in range(5))
        for stroke_index in range(8):
            for progress in (0.25, 0.50, 0.75, 1.0):
                frame = self._draw_scene(
                    task,
                    completed_strokes=stroke_index,
                    active_stroke_progress=progress,
                )
                frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
        for step in range(1, 5):
            frame = self._draw_scene(task, reveal_count=4, selection_progress=step / 4.0)
            frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
        solution = np.asarray(
            self._draw_scene(task, reveal_count=4, selection_progress=1.0).convert("RGB"),
            dtype=np.uint8,
        )
        frames.extend(solution.copy() for _ in range(8))
        if len(frames) != 49 or (len(frames) - 1) % 4 != 0:
            raise AssertionError("triangle_inequality_check video violates its 49-frame 4N+1 contract")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        contract, prompts = self._build_prompts()
        correct_option = _LABELS[task["invalid_index"]]
        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"

        puzzle_image = self._draw_scene(task)
        solution_image = self._draw_scene(task, reveal_count=4, selection_progress=1.0)
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        solution_video_path, video_num_frames = self._render_video(task, output_path=video_path)

        structured_answer = {
            "correct_option": correct_option,
            "candidates": [
                {"label": _LABELS[index], "sides": list(triple)}
                for index, triple in enumerate(task["triples"])
            ],
            "triples": [list(triple) for triple in task["triples"]],
            "comparisons": task["comparisons"],
            "invalid_index": task["invalid_index"],
            "invalid_triple": list(task["invalid_triple"]),
            "true_coordinate": {
                "invalid_index": task["invalid_index"],
                "panel_bbox": list(_PANEL_BBOXES[task["invalid_index"]]),
            },
            "state_trace": _expected_state_trace(correct_option),
            "render_geometry": _render_geometry(),
            "answer_schema": _answer_schema(),
            "verifier_contract": _verifier_contract(),
            "multiple_solution_policy": "exactly_one_of_A_through_D_fails_the_strict_triangle_inequality",
        }
        _, replay_valid = _replay_triangle_inequality(structured_answer)
        if not replay_valid:
            raise RuntimeError("Generated triangle-inequality sample failed semantic replay.")

        sample: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": prompts["question"],
            "answer": correct_option,
            "correct_option": correct_option,
            "structured_answer": structured_answer,
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }
        return finalize_geometry_sample(sample), puzzle_image, solution_image


__all__ = [
    'TriangleInequalityCheckPattern',
    '_DIVIDER_X',
    '_PANEL_BBOXES',
    '_comparison_payload',
    '_replay_triangle_inequality',
    '_triangle_analysis',
    'np',
]

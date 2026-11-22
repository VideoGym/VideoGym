"""Select the exact left-right reflection of one asymmetric ordered polyline."""

from __future__ import annotations

from data.common.base import RenderedTaskGenerator
from data.geometry.core.rendered_contract import finalize_geometry_sample

import math
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.pipeline.media import load_font
from data.common.choice_cards import attach_visual_choice_card_inputs
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    new_scaled_canvas,
    finish_scaled_canvas,
)
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC


pattern_instructions = {
    "mirror_image_select": (
        (
            "A full-canvas scene shows one asymmetric ordered polyline to the left of a vertical mirror axis. "
            "Independent candidate plates propose poses; select the exact left-right reflection without rotating or resizing it."
        )
    )
}

_LABELS = ("A", "B", "C", "D")
_TRANSFORMS = ("mirror_lr", "identity", "flip_ud", "rotate_180", "rotate_90_cw", "rotate_90_ccw")
_BG = (255, 255, 255)
_INK = (31, 39, 54)
_EXPECTED = (222, 82, 153)
_GUIDE = (24, 169, 190)
_START = (224, 69, 62)
_END = (42, 151, 88)
_AXIS_RGB = (69, 114, 177)
_AXIS_X = 256
_AXIS_Y_RANGE = (36, 476)
_SOURCE_CENTER = (128, 256)
_MIRROR_CENTER = (384, 256)
_SOURCE_SCALE = 1.75
_PLATE_CENTER = (256, 240)
_PLATE_SCALE = 2.6
_PATH_COLORS = ((48, 101, 190), (28, 145, 116), (135, 89, 183), (220, 125, 42))
_VERIFIER_PROFILE = "mirror_image_visible_ordered_polyline_multimodal_hard_gate_v1"
_VERIFIER_COMPONENTS = {
    "text": ("parse_score", "source_replay_score", "candidate_uniqueness_score", "final_state_score"),
    "image": ("parse_score", "source_replay_score", "preservation_score", "final_state_score"),
    "video": (
        "parse_score",
        "source_replay_score",
        "state_order_score",
        "temporal_continuity_score",
        "preservation_score",
        "final_state_score",
    ),
}
_VERIFIER_PERFORMANCE_BUDGET = {
    "text_warm_p95_ms": 5.0,
    "image_warm_p95_ms": 80.0,
    "video_warm_p95_ms": 900.0,
    "video_min_samples_per_second": 1.0,
    "peak_python_memory_mb": 64.0,
    "regression_tolerance_fraction": 0.15,
}
_STATE_TRACE = [
    {"phase": "input", "guide_progress": 0.0, "trace_progress": 0.0},
    {"phase": "endpoint_guides", "guide_progress": 1.0, "trace_progress": 0.0},
    {"phase": "mirror_trace", "guide_progress": 1.0, "trace_progress": 1.0},
]


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
            "the visible ordered six-vertex source polyline, its red start and green end identities, the four candidate "
            "vertex sequences shown on the independent hypothesis plates, path color, vertical axis, and fixed render geometry"
        ),
        "component_scores": {mode: list(names) for mode, names in _VERIFIER_COMPONENTS.items()},
        "binary_predicate": (
            "strict option parse, independent six-transform replay with exactly one left-right reflection, semantic "
            "endpoint-guide/mirror-trace parsing, fixed-content preservation, and monotonic whole-video construction"
        ),
        "performance_budget": dict(_VERIFIER_PERFORMANCE_BUDGET),
    }


def _map_points_expected(
    points: Sequence[Sequence[int]], center: Sequence[int], scale: float
) -> List[List[int]]:
    return [
        [round(int(center[0]) + int(x) * scale), round(int(center[1]) + int(y) * scale)]
        for x, y in points
    ]


def _expected_render_geometry(
    source: Sequence[Sequence[int]],
    candidates: Sequence[Sequence[Sequence[int]]],
    path_color: Sequence[int],
) -> Dict[str, Any]:
    source_screen = _map_points_expected(source, _SOURCE_CENTER, _SOURCE_SCALE)
    expected_screen = _map_points_expected(_apply_transform(source, "mirror_lr"), _MIRROR_CENTER, _SOURCE_SCALE)
    candidate_plate_vertices = [
        _map_points_expected(candidate, _PLATE_CENTER, _PLATE_SCALE)
        for candidate in candidates
    ]
    return {
        "canvas_size": [512, 512],
        "axis_x": _AXIS_X,
        "axis_y_range": list(_AXIS_Y_RANGE),
        "source_center": list(_SOURCE_CENTER),
        "mirror_center": list(_MIRROR_CENTER),
        "source_scale": _SOURCE_SCALE,
        "plate_center": list(_PLATE_CENTER),
        "plate_scale": _PLATE_SCALE,
        "source_screen_vertices": source_screen,
        "expected_screen_vertices": expected_screen,
        "candidate_plate_vertices": candidate_plate_vertices,
        "endpoint_guide_segments": [
            [source_screen[0], expected_screen[0]],
            [source_screen[-1], expected_screen[-1]],
        ],
        "path_color_rgb": [int(value) for value in path_color],
        "guide_rgb": list(_GUIDE),
        "expected_rgb": list(_EXPECTED),
        "start_rgb": list(_START),
        "end_rgb": list(_END),
    }


def _point_segment_distance(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 0.0:
        return math.hypot(px - ax, py - ay)
    ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))


def _segments_intersect(
    left_start: Sequence[int],
    left_end: Sequence[int],
    right_start: Sequence[int],
    right_end: Sequence[int],
) -> bool:
    def orientation(a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> int:
        cross = (int(b[0]) - int(a[0])) * (int(c[1]) - int(a[1])) - (
            int(b[1]) - int(a[1])
        ) * (int(c[0]) - int(a[0]))
        return 0 if cross == 0 else (1 if cross > 0 else -1)

    def on_segment(a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> bool:
        return (
            min(int(a[0]), int(c[0])) <= int(b[0]) <= max(int(a[0]), int(c[0]))
            and min(int(a[1]), int(c[1])) <= int(b[1]) <= max(int(a[1]), int(c[1]))
        )

    o1 = orientation(left_start, left_end, right_start)
    o2 = orientation(left_start, left_end, right_end)
    o3 = orientation(right_start, right_end, left_start)
    o4 = orientation(right_start, right_end, left_end)
    if o1 != o2 and o3 != o4:
        return True
    return bool(
        (o1 == 0 and on_segment(left_start, right_start, left_end))
        or (o2 == 0 and on_segment(left_start, right_end, left_end))
        or (o3 == 0 and on_segment(right_start, left_start, right_end))
        or (o4 == 0 and on_segment(right_start, left_end, right_end))
    )


def _segment_distance(
    left_start: Sequence[int],
    left_end: Sequence[int],
    right_start: Sequence[int],
    right_end: Sequence[int],
) -> float:
    if _segments_intersect(left_start, left_end, right_start, right_end):
        return 0.0
    return min(
        _point_segment_distance(left_start, right_start, right_end),
        _point_segment_distance(left_end, right_start, right_end),
        _point_segment_distance(right_start, left_start, left_end),
        _point_segment_distance(right_end, left_start, left_end),
    )


def _glyph_is_visually_separable(points: Sequence[Sequence[int]]) -> bool:
    segments = list(zip(points, points[1:]))
    for index, (left, pivot, right) in enumerate(zip(points, points[1:], points[2:])):
        incoming = (int(pivot[0]) - int(left[0]), int(pivot[1]) - int(left[1]))
        outgoing = (int(right[0]) - int(pivot[0]), int(right[1]) - int(pivot[1]))
        cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / max(
            1.0,
            math.hypot(*incoming) * math.hypot(*outgoing),
        )
        if cosine < -0.45:
            return False
    for left_index, left_segment in enumerate(segments):
        for right_index, right_segment in enumerate(segments):
            if right_index <= left_index + 1:
                continue
            if _segment_distance(*left_segment, *right_segment) < 18.0:
                return False
    return True


def _generate_glyph(rng: random.Random) -> List[Tuple[int, int]]:
    """Generate six ordered vertices with stable asymmetry and readable segments."""

    for _attempt in range(300):
        points = [(rng.randint(-54, -20), rng.randint(-52, 52))]
        while len(points) < 6:
            x, y = points[-1]
            angle = rng.uniform(0.0, math.tau)
            length = rng.randint(30, 52)
            candidate = (int(round(x + math.cos(angle) * length)), int(round(y + math.sin(angle) * length)))
            if max(abs(candidate[0]), abs(candidate[1])) > 62:
                continue
            if candidate in points or math.dist(points[-1], candidate) < 26:
                continue
            points.append(candidate)
        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        centered = [(int(round(x - cx)), int(round(y - cy))) for x, y in points]
        if any(max(abs(x), abs(y)) > 60 for x, y in centered):
            continue
        width = max(x for x, _ in centered) - min(x for x, _ in centered)
        height = max(y for _, y in centered) - min(y for _, y in centered)
        if max(width, height) < 82 or min(width, height) < 38:
            continue
        if not _glyph_is_visually_separable(centered):
            continue
        signatures = {
            tuple(_apply_transform(centered, transform))
            for transform in _TRANSFORMS
        }
        if len(signatures) == len(_TRANSFORMS):
            return centered
    raise RuntimeError("mirror_image_select failed to generate a non-degenerate glyph")


def _apply_transform(points: Sequence[Sequence[int]], transform: str) -> List[Tuple[int, int]]:
    if transform == "mirror_lr":
        return [(-int(x), int(y)) for x, y in points]
    if transform == "identity":
        return [(int(x), int(y)) for x, y in points]
    if transform == "flip_ud":
        return [(int(x), -int(y)) for x, y in points]
    if transform == "rotate_180":
        return [(-int(x), -int(y)) for x, y in points]
    if transform == "rotate_90_cw":
        return [(int(y), -int(x)) for x, y in points]
    if transform == "rotate_90_ccw":
        return [(-int(y), int(x)) for x, y in points]
    raise ValueError(f"unknown transform {transform!r}")


def _parse_points(raw: Any) -> List[Tuple[int, int]] | None:
    try:
        points = [(int(point[0]), int(point[1])) for point in raw]
    except (TypeError, ValueError, IndexError):
        return None
    if len(points) != 6 or len(set(points)) != 6:
        return None
    if any(max(abs(x), abs(y)) > 62 for x, y in points):
        return None
    if any(math.dist(left, right) < 24 for left, right in zip(points, points[1:])):
        return None
    return points


def replay_mirror_image_select(structured: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Derive the unique mirror candidate without trusting saved answer metadata."""

    try:
        source_raw = structured["source_vertices"]
        candidates_raw = structured["candidate_vertices"]
        path_color = [int(value) for value in structured["path_color"]]
    except (KeyError, TypeError, ValueError):
        return {}, False
    source = _parse_points(source_raw)
    if source is None or not isinstance(candidates_raw, list) or len(candidates_raw) != 4:
        return {}, False
    candidates = [_parse_points(raw) for raw in candidates_raw]
    if any(candidate is None for candidate in candidates):
        return {}, False
    parsed_candidates = [candidate for candidate in candidates if candidate is not None]
    transform_results = {
        transform: _apply_transform(source, transform)
        for transform in _TRANSFORMS
    }
    if len({tuple(value) for value in transform_results.values()}) != len(_TRANSFORMS):
        return {}, False
    inferred_transforms: List[str] = []
    for candidate in parsed_candidates:
        matches = [name for name, expected in transform_results.items() if candidate == expected]
        if len(matches) != 1:
            return {}, False
        inferred_transforms.append(matches[0])
    if len(set(inferred_transforms)) != 4 or inferred_transforms.count("mirror_lr") != 1:
        return {}, False
    correct_index = inferred_transforms.index("mirror_lr")
    expected_geometry = _expected_render_geometry(source, parsed_candidates, path_color)
    verifier = structured.get("verifier_contract")
    if not (
        tuple(path_color) in _PATH_COLORS
        and structured.get("mirror_axis") == "vertical"
        and structured.get("start_marker") == "red"
        and structured.get("end_marker") == "green"
        and structured.get("multiple_solution_policy") == "exactly_one_candidate_is_the_left_right_reflection"
        and structured.get("state_trace") == _STATE_TRACE
        and structured.get("render_geometry") == expected_geometry
        and structured.get("answer_schema") == _answer_schema()
        and isinstance(verifier, dict)
        and verifier.get("profile") == _VERIFIER_PROFILE
        and verifier.get("component_scores")
        == {mode: list(names) for mode, names in _VERIFIER_COMPONENTS.items()}
        and verifier.get("performance_budget") == _VERIFIER_PERFORMANCE_BUDGET
        and isinstance(verifier.get("semantic_source"), str)
        and isinstance(verifier.get("binary_predicate"), str)
    ):
        return {}, False
    return {
        "source_vertices": [list(point) for point in source],
        "candidate_vertices": [[list(point) for point in candidate] for candidate in parsed_candidates],
        "inferred_transforms": inferred_transforms,
        "correct_index": correct_index,
        "correct_option": _LABELS[correct_index],
        "path_color": path_color,
        "state_trace": list(_STATE_TRACE),
        "render_geometry": expected_geometry,
    }, True


def _replay_mirror_image_select(structured: Dict[str, Any]) -> Tuple[Dict[str, float], bool]:
    metrics = {
        "parse_valid": 0.0,
        "source_valid": 0.0,
        "candidate_geometry_valid": 0.0,
        "answer_valid": 0.0,
    }
    _replay, valid = replay_mirror_image_select(structured)
    if valid:
        metrics = {name: 1.0 for name in metrics}
    return metrics, valid


class MirrorImageSelectPattern(RenderedTaskGenerator):
    task_name = "mirror_image_select"
    task_family = "geometry"
    video_fps = 16

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        image_size: Tuple[int, int] | int | None = None,
        video_fps: int | None = None,
        generate_videos: bool = True,
        video: bool | None = None,
        **_: Any,
    ) -> None:
        size = (int(image_size), int(image_size)) if isinstance(image_size, int) else tuple(image_size or (512, 512))
        if size != (512, 512):
            raise ValueError("mirror_image_select requires a 512x512 canvas")
        if video_fps not in (None, 16):
            raise ValueError("mirror_image_select videos require 16 fps")
        self.rng = random.Random(seed if seed is not None else random_seed)
        self.generate_videos = bool(generate_videos if video is None else video)
        self.output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_mirror_image_select"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0

    def _generate_task(self) -> Dict[str, Any]:
        source = _generate_glyph(self.rng)
        distractors = self.rng.sample([name for name in _TRANSFORMS if name != "mirror_lr"], 3)
        transforms = ["mirror_lr", *distractors]
        self.rng.shuffle(transforms)
        return {
            "source": source,
            "candidate_transforms": transforms,
            "candidates": [_apply_transform(source, name) for name in transforms],
            "correct_index": transforms.index("mirror_lr"),
            "color": self.rng.choice(_PATH_COLORS),
        }

    @staticmethod
    def _map_points(points: Sequence[Sequence[int]], center: Sequence[int], scale: float) -> List[Tuple[int, int]]:
        return [(round(center[0] + int(x) * scale), round(center[1] + int(y) * scale)) for x, y in points]

    @staticmethod
    def _draw_polyline(draw: Any, points: Sequence[Tuple[int, int]], color: Tuple[int, int, int], *, width: int = 6) -> None:
        # The path stays open — its start and end markers carry the ordering the
        # task asks about — so it is stroked as a mitered open path rather than
        # with PIL's round `joint="curve"`, whose disc blunts every sharp turn.
        draw.open_path(points, fill=color, width=width)
        sx, sy = points[0]
        ex, ey = points[-1]
        draw.ellipse((sx - 8, sy - 8, sx + 8, sy + 8), fill=_START, outline=_INK, width=1)
        draw.ellipse((ex - 8, ey - 8, ex + 8, ey + 8), fill=_END, outline=_INK, width=1)

    @staticmethod
    def _draw_expected_trace(
        draw: Any,
        points: Sequence[Tuple[int, int]],
        segment_progress: float,
    ) -> None:
        progress = max(0.0, min(float(len(points) - 1), float(segment_progress)))
        if progress <= 0.0:
            return
        complete_segments = min(len(points) - 1, int(math.floor(progress)))
        fraction = progress - complete_segments
        visible = list(points[: complete_segments + 1])
        if complete_segments < len(points) - 1 and fraction > 0.0:
            left = points[complete_segments]
            right = points[complete_segments + 1]
            visible.append(
                (
                    int(round(left[0] + (right[0] - left[0]) * fraction)),
                    int(round(left[1] + (right[1] - left[1]) * fraction)),
                )
            )
        if len(visible) >= 2:
            draw.open_path(visible, fill=_EXPECTED, width=6)
        sx, sy = points[0]
        draw.ellipse((sx - 8, sy - 8, sx + 8, sy + 8), fill=_START, outline=_INK, width=1)
        if progress >= len(points) - 1:
            ex, ey = points[-1]
            draw.ellipse((ex - 8, ey - 8, ex + 8, ey + 8), fill=_END, outline=_INK, width=1)

    def _render_scene(
        self,
        task: Dict[str, Any],
        *,
        show_endpoint_guides: bool = False,
        completed_guides: int | None = None,
        active_guide_progress: float = 0.0,
        mirror_progress: float = 0.0,
    ) -> Image.Image:
        image, draw = new_scaled_canvas(512, 512, DOCTRINE_SUPERSAMPLE, background=_BG)
        draw.line((_AXIS_X, _AXIS_Y_RANGE[0], _AXIS_X, _AXIS_Y_RANGE[1]), fill=_AXIS_RGB, width=3)

        source_points = self._map_points(task["source"], _SOURCE_CENTER, _SOURCE_SCALE)
        expected_points = self._map_points(_apply_transform(task["source"], "mirror_lr"), _MIRROR_CENTER, _SOURCE_SCALE)
        self._draw_polyline(draw, source_points, task["color"])
        guide_pairs = (
            (source_points[0], expected_points[0]),
            (source_points[-1], expected_points[-1]),
        )
        guide_count = 2 if show_endpoint_guides and completed_guides is None else int(completed_guides or 0)
        guide_count = max(0, min(len(guide_pairs), guide_count))
        for guide_index, (source_point, expected_point) in enumerate(guide_pairs):
            progress = (
                1.0
                if guide_index < guide_count
                else max(0.0, min(1.0, float(active_guide_progress)))
                if guide_index == guide_count and guide_count < len(guide_pairs)
                else 0.0
            )
            if progress > 0.0:
                endpoint = (
                    int(round(source_point[0] + (expected_point[0] - source_point[0]) * progress)),
                    int(round(source_point[1] + (expected_point[1] - source_point[1]) * progress)),
                )
                draw.line((source_point[0], source_point[1], endpoint[0], endpoint[1]), fill=_GUIDE, width=2)
            if progress >= 1.0:
                draw.ellipse(
                    (expected_point[0] - 6, expected_point[1] - 6, expected_point[0] + 6, expected_point[1] + 6),
                    fill=_BG,
                    outline=_GUIDE,
                    width=2,
                )
        self._draw_expected_trace(draw, expected_points, mirror_progress)
        return finish_scaled_canvas(image, (512, 512))

    def _render_option_plate(self, candidate: Sequence[Sequence[int]], label: str, color: Tuple[int, int, int]) -> Image.Image:
        image, draw = new_scaled_canvas(512, 512, DOCTRINE_SUPERSAMPLE, background=_BG)
        points = self._map_points(candidate, _PLATE_CENTER, _PLATE_SCALE)
        self._draw_polyline(draw, points, color)
        factor = DOCTRINE_SUPERSAMPLE
        draw.text((256, 474), label, fill=(42, 46, 52), font=load_font(30 * factor), anchor="mm")
        return finish_scaled_canvas(image, (512, 512))

    def _structured_answer(self, task: Dict[str, Any]) -> Dict[str, Any]:
        index = int(task["correct_index"])
        structured = {
            "source_vertices": [list(point) for point in task["source"]],
            "candidate_vertices": [[list(point) for point in candidate] for candidate in task["candidates"]],
            "path_color": list(task["color"]),
            "candidate_transforms": list(task["candidate_transforms"]),
            "correct_index": index,
            "correct_option": _LABELS[index],
            "mirror_axis": "vertical",
            "start_marker": "red",
            "end_marker": "green",
            "multiple_solution_policy": "exactly_one_candidate_is_the_left_right_reflection",
            "state_trace": list(_STATE_TRACE),
            "render_geometry": _expected_render_geometry(
                task["source"], task["candidates"], task["color"]
            ),
            "answer_schema": _answer_schema(),
            "verifier_contract": _verifier_contract(),
        }
        _replay, valid = replay_mirror_image_select(structured)
        if not valid:
            raise RuntimeError("Generated mirror-image sample failed independent semantic replay.")
        return structured

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The single image shows one asymmetric ordered polyline to the left of a vertical blue mirror axis "
                "and an empty mirror workspace to the right."
            ),
            modality_input_descriptions={
            },
            symbol_legend=(
                "The red dot is the ordered start vertex and the green dot is the ordered end vertex; both identities must be reflected with the complete path.",
                "A vertical reflection reverses left and right while preserving height, segment lengths, vertex order, color, and scale.",
            ),
            rules=(
                "Reflect every source vertex across the vertical axis into the empty right workspace.",
                "Do not rotate, resize, translate individual vertices, reverse the ordered endpoints, or use an up-down reflection.",
                "Exactly one candidate pose equals that complete left-right reflection.",
            ),
            modality_outputs={
                "ti2v": (
                    "Deliver the complete video: hold the exact input, draw the start-endpoint cyan correspondence guide "
                    "from the source endpoint to its reflection, finish it before drawing the other endpoint guide, then grow "
                    "the pink expected mirror polyline continuously in vertex order and hold the completed scene."
                ),
                "solution_image": "Show the two cyan endpoint guides and the complete pink mirrored polyline with its endpoint dots in the right workspace.",
            },
            preservation_constraints=(
                "Keep the source path, axis, background, colors, scale, framing, and camera unchanged.",
                "Only the cyan correspondence guides and the pink expected mirror trace may be added.",
            ),
            transition_constraints=(
                "Begin with a readable hold of the exact unmarked input.",
                "Draw each cyan endpoint guide from its source endpoint to its reflected endpoint, one complete guide at a time; only then extend the pink expected path in source vertex order without erasing earlier segments.",
                "After the complete expected path is visible, hold the completed scene without further change.",
            ),
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="strict_uppercase_option_letter_a_to_d",
                verifier_profile=_VERIFIER_PROFILE,
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_full_canvas_mirror_image_select_v3",
        )

    def _prompts(self) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "question": "Reflect the ordered source polyline left-to-right and choose the exact matching candidate.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid mirror-image prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _render_video(self, task: Dict[str, Any], output_path: Path) -> int:
        if not self.generate_videos:
            return 0
        frames: List[np.ndarray] = []

        def append(image: Image.Image, count: int) -> None:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            frames.extend(array.copy() for _ in range(count))

        append(self._render_scene(task), 5)
        for guide_index in range(2):
            for progress in (0.25, 0.50, 0.75, 1.0):
                append(
                    self._render_scene(
                        task,
                        completed_guides=guide_index,
                        active_guide_progress=progress,
                    ),
                    1,
                )
        for step in range(1, 16):
            append(
                self._render_scene(
                    task,
                    completed_guides=2,
                    mirror_progress=step / 3.0,
                ),
                1,
            )
        append(self._render_scene(task, completed_guides=2, mirror_progress=5.0), 5)
        if len(frames) != 33 or (len(frames) - 1) % 4 != 0:
            raise AssertionError("mirror_image_select video must contain 33 frames")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=16):
            return 0
        return len(frames)

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        structured = self._structured_answer(task)
        replay_metrics, replay_valid = _replay_mirror_image_select(structured)
        if not replay_valid:
            raise RuntimeError(f"mirror_image_select semantic replay failed: {replay_metrics}")
        contract, prompts = self._prompts()
        puzzle = self._render_scene(task)
        step_one = self._render_scene(task, show_endpoint_guides=True)
        solution = self._render_scene(task, show_endpoint_guides=True, mirror_progress=5.0)

        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        step_one_path = sample_dir / f"{sample_id}__step_01.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        puzzle.save(puzzle_path)
        step_one.save(step_one_path)
        solution.save(solution_path)
        option_paths: List[str] = []
        for index, candidate in enumerate(task["candidates"]):
            option_path = sample_dir / f"{sample_id}__option_{_LABELS[index]}.png"
            self._render_option_plate(candidate, _LABELS[index], task["color"]).save(option_path)
            option_paths.append(option_path.as_posix())

        video_path: Path | None = None
        video_frames: int | None = None
        if self.generate_videos:
            candidate = sample_dir / f"{sample_id}__solution.webm"
            video_frames = self._render_video(task, candidate)
            if video_frames:
                video_path = candidate

        sample: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": prompts["question"],
            "answer": structured["correct_option"],
            "correct_option": structured["correct_option"],
            "structured_answer": structured,
            "verifier_contract": structured["verifier_contract"],
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "reasoning_images": [],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": video_path.as_posix() if video_path else None,
            "video_fps": 16 if video_path else None,
            "video_num_frames": video_frames,
        }
        attach_visual_choice_card_inputs(
            sample,
            problem_image=puzzle_path.as_posix(),
            option_images=option_paths,
            option_labels=_LABELS,
            ti2v_problem_image=puzzle_path.as_posix(),
        )
        return finalize_geometry_sample(sample), puzzle, solution


__all__ = [
    "MirrorImageSelectPattern",
    "_MIRROR_CENTER",
    "_SOURCE_CENTER",
    "_SOURCE_SCALE",
    "_TRANSFORMS",
    "_apply_transform",
    "_expected_render_geometry",
    "_replay_mirror_image_select",
    "replay_mirror_image_select",
]

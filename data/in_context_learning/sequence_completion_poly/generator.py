"""Marked-polygon D4 sequence with direct final-card completion.

Three cards show one asymmetric marked polygon under successive applications
of a single repeated D4 operation; the final card is completed with the fourth
state.  The typed payload names the repeated operation from a fixed
seven-word vocabulary, which for an asymmetric marked polygon is a faithful
encoding of the fourth state given the visible third state.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data.common.render_primitives import ScaledDraw
from data.in_context_learning.core.sequence_cards import (
    SequenceCompletionGenerator,
    final_content_bbox,
    render_sequence_scene,
)
from data.in_context_learning.core.sequence_completion_verifier import (
    SequenceReplay,
    make_template_decoder,
)

_ARRANGEMENT = "row4"
_TOTAL_CARDS = 4

_D4_MATRICES = {
    "identity": (1, 0, 0, 1),
    "rot90_cw": (0, -1, 1, 0),
    "rot180": (-1, 0, 0, -1),
    "rot90_ccw": (0, 1, -1, 0),
    "flip_h": (-1, 0, 0, 1),
    "flip_v": (1, 0, 0, -1),
    "flip_diag": (0, 1, 1, 0),
    "flip_antidiag": (0, -1, -1, 0),
}
_MATRIX_TO_TRANSFORM = {matrix: name for name, matrix in _D4_MATRICES.items()}
_ALL_TRANSFORMS = tuple(_D4_MATRICES)
_STEP_TRANSFORMS = tuple(name for name in _ALL_TRANSFORMS if name != "identity")

OPERATION_WORDS = {
    "rot90_cw": "90 clockwise",
    "rot90_ccw": "90 counterclockwise",
    "rot180": "180",
    "flip_h": "left-right flip",
    "flip_v": "top-bottom flip",
    "flip_diag": "main-diagonal flip",
    "flip_antidiag": "anti-diagonal flip",
}
_WORD_TO_OPERATION = {word: name for name, word in OPERATION_WORDS.items()}
_OPERATION_PROSE = {
    "rot90_cw": "a quarter-turn clockwise",
    "rot90_ccw": "a quarter-turn counterclockwise",
    "rot180": "a half-turn",
    "flip_h": "a left-right flip across the vertical center line",
    "flip_v": "a top-bottom flip across the horizontal center line",
    "flip_diag": "a flip across the top-left to bottom-right diagonal",
    "flip_antidiag": "a flip across the top-right to bottom-left diagonal",
}

_COLORS = (
    (63, 111, 212),
    (36, 150, 100),
    (224, 120, 60),
    (149, 88, 192),
)
_MARKER = (238, 177, 52)
_BACKDROP_LINE = (222, 227, 233)
_AXIS_LINE = (198, 204, 212)
_OUTLINE = (44, 48, 54)


def transform_point(point: Sequence[float], transform: str) -> Tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    a, b, c, d = _D4_MATRICES[transform]
    return a * x + b * y, c * x + d * y


def compose_transform(outer: str, inner: str) -> str:
    """Return the D4 transform produced by applying ``inner``, then ``outer``."""

    a, b, c, d = _D4_MATRICES[outer]
    e, f, g, h = _D4_MATRICES[inner]
    matrix = (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)
    return _MATRIX_TO_TRANSFORM[matrix]


def inverse_transform(transform: str) -> str:
    for candidate in _ALL_TRANSFORMS:
        if compose_transform(candidate, transform) == "identity":
            return candidate
    raise ValueError(f"transform has no D4 inverse: {transform}")


def _paint_backdrop(sdraw: ScaledDraw, box) -> None:
    left, top, right, bottom = (float(value) for value in box)
    size = min(right - left, bottom - top) - 8.0
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    x0, y0 = cx - size / 2.0, cy - size / 2.0
    quarter = size / 4.0
    for index in range(5):
        offset = index * quarter
        weight_color = _AXIS_LINE if index == 2 else _BACKDROP_LINE
        sdraw.line((x0 + offset, y0, x0 + offset, y0 + size), fill=weight_color, width=1)
        sdraw.line((x0, y0 + offset, x0 + size, y0 + offset), fill=weight_color, width=1)


def _paint_polygon_state(sdraw: ScaledDraw, box, item: Mapping[str, Any], position: int) -> None:
    del position
    _paint_backdrop(sdraw, box)
    left, top, right, bottom = (float(value) for value in box)
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    scale = (min(right - left, bottom - top) - 8.0) * 0.46
    transform = str(item["transform"])
    base_points = item["base_points"]
    color = tuple(int(value) for value in item["color"])
    points = [
        (cx + x * scale, cy + y * scale)
        for x, y in (transform_point(point, transform) for point in base_points)
    ]
    sdraw.polygon(points, fill=color, outline=_OUTLINE, width=2)
    marker_index = int(item["marker_index"])
    mx, my = points[marker_index]
    sdraw.ellipse((mx - 4.2, my - 4.2, mx + 4.2, my + 4.2), fill=_MARKER, outline=_OUTLINE, width=1)


def _items_from_semantics(semantics: Mapping[str, Any], transform: str) -> Dict[str, Any]:
    return {
        "transform": transform,
        "base_points": [list(point) for point in semantics["base_points"]],
        "marker_index": int(semantics["marker_index"]),
        "color": list(semantics["color"]),
    }


def _visible_items(semantics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [_items_from_semantics(semantics, str(value)) for value in semantics["shown_transforms"]]


def _answer_item(semantics: Mapping[str, Any]) -> Dict[str, Any]:
    return _items_from_semantics(semantics, str(semantics["answer_transform"]))


def render_pair_from_semantics(semantics: Mapping[str, Any]):
    items = _visible_items(semantics)
    answer = _answer_item(semantics)
    problem = render_sequence_scene(
        arrangement=_ARRANGEMENT,
        items=items,
        answer_item=None,
        painter=_paint_polygon_state,
        solved=False,
        placeholder_painter=_paint_backdrop,
    )
    terminal = render_sequence_scene(
        arrangement=_ARRANGEMENT,
        items=items,
        answer_item=answer,
        painter=_paint_polygon_state,
        solved=True,
    )
    return problem, terminal


def _infer_step(shown: Sequence[str]) -> Optional[str]:
    values = [str(value) for value in shown]
    if len(values) != 3 or any(value not in _D4_MATRICES for value in values):
        return None
    matching = [
        step
        for step in _STEP_TRANSFORMS
        if compose_transform(step, values[0]) == values[1]
        and compose_transform(step, values[1]) == values[2]
    ]
    return matching[0] if len(matching) == 1 else None


def _orbit_distinct(base_points: Sequence[Sequence[float]]) -> bool:
    """Require all eight D4 images of the vertex list to be pairwise distinct."""

    signatures = []
    for transform in _ALL_TRANSFORMS:
        signature = tuple(
            (round(x, 2), round(y, 2))
            for x, y in (transform_point(point, transform) for point in base_points)
        )
        signatures.append(signature)
    return len(set(signatures)) == len(_ALL_TRANSFORMS)


def _marker_direction(semantics: Mapping[str, Any], transform: str) -> str:
    point = transform_point(semantics["base_points"][int(semantics["marker_index"])], transform)
    x, y = point
    horizontal = "right" if x > 0.14 else "left" if x < -0.14 else ""
    vertical = "lower" if y > 0.14 else "upper" if y < -0.14 else ""
    if not horizontal and not vertical:
        return "near the center"
    if not horizontal:
        return f"at the {vertical} side"
    if not vertical:
        return f"at the {horizontal} side"
    return f"at the {vertical} {horizontal}"


class SequenceCompletionPolyPattern(SequenceCompletionGenerator):
    task_name = "sequence_completion_poly"
    arrangement = _ARRANGEMENT

    def _sample_polygon(self) -> Tuple[List[List[float]], int]:
        count = self.rng.randint(6, 8)
        for _attempt in range(300):
            angles = sorted(self.rng.uniform(0.0, math.tau) for _ in range(count))
            gaps = [(angles[(index + 1) % count] - angles[index]) % math.tau for index in range(count)]
            if min(gaps) < 0.30:
                continue
            marker_index = self.rng.randrange(count)
            points: List[List[float]] = []
            for index, angle in enumerate(angles):
                radius = 0.82 if index == marker_index else self.rng.uniform(0.42, 0.74)
                points.append([round(math.cos(angle) * radius, 4), round(math.sin(angle) * radius, 4)])
            marker = points[marker_index]
            if abs(marker[0]) < 0.18 and abs(marker[1]) < 0.18:
                continue
            if not _orbit_distinct(points):
                continue
            return points, marker_index
        raise RuntimeError("sequence_completion_poly failed to sample an asymmetric polygon")

    def sample_semantics(self) -> Dict[str, Any]:
        base_points, marker_index = self._sample_polygon()
        step = self.rng.choice(_STEP_TRANSFORMS)
        initial = self.rng.choice(_ALL_TRANSFORMS)
        shown = [initial]
        for _ in range(2):
            shown.append(compose_transform(step, shown[-1]))
        answer = compose_transform(step, shown[-1])
        if _infer_step(shown) != step:
            raise RuntimeError("sequence_completion_poly replay disagrees with the sampled step")
        return {
            "base_points": base_points,
            "marker_index": marker_index,
            "step_transform": step,
            "shown_transforms": shown,
            "answer_transform": answer,
            "color": list(self.rng.choice(_COLORS)),
        }

    def visible_items(self, semantics):
        return _visible_items(semantics)

    def answer_item(self, semantics):
        return _answer_item(semantics)

    def paint_content(self, sdraw, box, item, position):
        _paint_polygon_state(sdraw, box, item, position)

    def paint_placeholder(self, sdraw, box):
        _paint_backdrop(sdraw, box)

    def answer_payload(self, semantics) -> str:
        return OPERATION_WORDS[str(semantics["step_transform"])]

    def answer_template(self) -> str:
        return "Answer: <operation>."

    def author_reasoning(self, semantics) -> str:
        shown = [str(value) for value in semantics["shown_transforms"]]
        step = str(semantics["step_transform"])
        directions = [_marker_direction(semantics, transform) for transform in shown]
        answer_direction = _marker_direction(semantics, str(semantics["answer_transform"]))
        return (
            f"On card 1 the gold vertex sits {directions[0]}. On card 2 it sits {directions[1]}, "
            f"and the outline matches card 1 taken through {_OPERATION_PROSE[step]}. Card 3 shows the "
            f"same operation applied again, with the gold vertex {directions[2]}. So the repeated "
            f"operation is the {OPERATION_WORDS[step]}, and applying it once more puts card 4's "
            f"polygon with its gold vertex {answer_direction}."
        )

    def prompt_copy(self, semantics) -> Dict[str, str]:
        return {
            "input_copy": (
                "Four numbered cards show one orientation sequence: cards 1-3 each hold the same "
                "irregular polygon with one gold corner dot, drawn over a fixed square backdrop grid "
                "whose center lines mark the flip axes, and card 4 holds only a question mark over "
                "the same backdrop."
            ),
            "rule_copy": (
                "Exactly one operation repeats at every step, taken from this set: 90 clockwise, "
                "90 counterclockwise, 180, left-right flip, top-bottom flip, main-diagonal flip "
                "along the top-left to bottom-right center diagonal, or anti-diagonal flip along the "
                "top-right to bottom-left center diagonal. The polygon and its gold corner move "
                "together, and card 4 applies the repeated operation once more to card 3."
            ),
            "answer_meaning": "the repeated operation's name from that set.",
            "terminal_copy": (
                "the polygon and its gold corner in the orientation the repeated operation produces "
                "from card 3, at the same drawing size and colors."
            ),
        }

    def structured_semantics(self, semantics) -> Dict[str, Any]:
        return {
            "base_points": [list(point) for point in semantics["base_points"]],
            "marker_index": int(semantics["marker_index"]),
            "shown_transforms": [str(value) for value in semantics["shown_transforms"]],
            "step_transform": str(semantics["step_transform"]),
            "answer_transform": str(semantics["answer_transform"]),
            "operation_vocabulary": dict(OPERATION_WORDS),
            "render_geometry": {
                "arrangement": _ARRANGEMENT,
                "polygon_rgb": [int(value) for value in semantics["color"]],
            },
        }

    def glyph_inventory(self, semantics) -> Dict[str, Any]:
        del semantics
        return {
            "schema_version": "icl_rendered_glyph_inventory_v1",
            "static_whitelist": ["1", "2", "3", "4", "?"],
            "bounded_dynamic_schema": [],
            "observed_dynamic_values": [],
            "answer_fields_excluded": True,
        }


def build_sequence_replay(state: Mapping[str, Any]) -> SequenceReplay:
    base_points = [list(point) for point in state.get("base_points") or []]
    marker_index = int(state.get("marker_index", -1))
    shown = [str(value) for value in state.get("shown_transforms") or []]
    if len(base_points) < 6 or not 0 <= marker_index < len(base_points):
        raise ValueError("polygon-sequence source geometry is malformed")
    if not _orbit_distinct(base_points):
        raise ValueError("polygon-sequence source polygon is not D4-asymmetric")
    step = _infer_step(shown)
    if step is None:
        raise ValueError("polygon-sequence source does not identify one repeated D4 operation")
    answer_transform = compose_transform(step, shown[2])
    geometry = state.get("render_geometry") or {}
    color = [int(value) for value in (geometry.get("polygon_rgb") or [])]
    if len(color) != 3 or tuple(color) not in _COLORS:
        raise ValueError("polygon-sequence source is missing its sampled polygon color")
    semantics = {
        "base_points": base_points,
        "marker_index": marker_index,
        "step_transform": step,
        "shown_transforms": shown,
        "answer_transform": answer_transform,
        "color": color,
    }
    problem, terminal = render_pair_from_semantics(semantics)
    third_inverse = inverse_transform(shown[2])
    templates = {}
    for orientation in _ALL_TRANSFORMS:
        implied_step = compose_transform(orientation, third_inverse)
        if implied_step == "identity":
            continue
        variant = dict(semantics, answer_transform=orientation)
        templates[OPERATION_WORDS[implied_step]] = np.asarray(
            render_pair_from_semantics(variant)[1], dtype=np.uint8
        )
    decoder = make_template_decoder(templates, final_content_bbox(_ARRANGEMENT, _TOTAL_CARDS))
    expected_word = OPERATION_WORDS[step]
    pattern = "|".join(sorted(_WORD_TO_OPERATION, key=len, reverse=True))
    return SequenceReplay(
        expected_payload=expected_word,
        payload_pattern=pattern,
        problem=np.asarray(problem, dtype=np.uint8),
        terminal=np.asarray(terminal, dtype=np.uint8),
        decode_final_card=decoder,
        diagnostics={
            "source_replay_step": step,
            "source_replay_answer_transform": answer_transform,
        },
    )


__all__ = [
    "OPERATION_WORDS",
    "SequenceCompletionPolyPattern",
    "build_sequence_replay",
    "compose_transform",
    "inverse_transform",
    "render_pair_from_semantics",
    "transform_point",
]

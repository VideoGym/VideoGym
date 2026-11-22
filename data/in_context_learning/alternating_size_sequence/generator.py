"""Alternating-height shape sequence with direct final-card completion."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from data.in_context_learning.core.sequence_cards import (
    SequenceCompletionGenerator,
    final_content_bbox,
    render_sequence_scene,
)
from data.in_context_learning.core.sequence_completion_verifier import (
    SequenceReplay,
    make_template_decoder,
)
from data.in_context_learning.core.size_cards import (
    RULER_UNITS,
    SIZE_CARD_GLYPHS,
    SIZE_COLORS,
    SIZE_INPUT_COPY,
    SIZE_RULER_SENTENCE,
    SIZE_TERMINAL_COPY,
    SIZE_SHAPES,
    paint_ruler,
    paint_sized_shape,
)

_ARRANGEMENT = "grid4"
_TOTAL_CARDS = 4
_MIN_GAP = 2


def _visible_items(semantics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    shape = str(semantics["shape"])
    color = list(semantics["color"])
    return [
        {"shape": shape, "color": color, "height": int(value)}
        for value in semantics["heights"][:3]
    ]


def _answer_item(semantics: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "shape": str(semantics["shape"]),
        "color": list(semantics["color"]),
        "height": int(semantics["heights"][3]),
    }


def render_pair_from_semantics(semantics: Mapping[str, Any]):
    items = _visible_items(semantics)
    answer = _answer_item(semantics)
    problem = render_sequence_scene(
        arrangement=_ARRANGEMENT,
        items=items,
        answer_item=None,
        painter=paint_sized_shape,
        solved=False,
        placeholder_painter=paint_ruler,
    )
    terminal = render_sequence_scene(
        arrangement=_ARRANGEMENT,
        items=items,
        answer_item=answer,
        painter=paint_sized_shape,
        solved=True,
    )
    return problem, terminal


def _infer_fourth(observed: Sequence[int]) -> int | None:
    values = [int(value) for value in observed]
    if len(values) != 3:
        return None
    if not all(1 <= value <= RULER_UNITS for value in values):
        return None
    if values[0] != values[2] or abs(values[0] - values[1]) < _MIN_GAP:
        return None
    return values[1]


class AlternatingSizeSequencePattern(SequenceCompletionGenerator):
    task_name = "alternating_size_sequence"
    arrangement = _ARRANGEMENT

    def sample_semantics(self) -> Dict[str, Any]:
        first = self.rng.randint(1, RULER_UNITS)
        other_pool = [
            value
            for value in range(1, RULER_UNITS + 1)
            if abs(value - first) >= _MIN_GAP
        ]
        second = self.rng.choice(other_pool)
        heights = [first, second, first, second]
        if _infer_fourth(heights[:3]) != heights[3]:
            raise RuntimeError("alternating_size_sequence replay disagrees with the sampled heights")
        return {
            "shape": self.rng.choice(SIZE_SHAPES),
            "color": list(self.rng.choice(SIZE_COLORS)),
            "heights": heights,
        }

    def visible_items(self, semantics):
        return _visible_items(semantics)

    def answer_item(self, semantics):
        return _answer_item(semantics)

    def paint_content(self, sdraw, box, item, position):
        paint_sized_shape(sdraw, box, item, position)

    def paint_placeholder(self, sdraw, box):
        paint_ruler(sdraw, box)

    def answer_payload(self, semantics) -> str:
        return str(int(semantics["heights"][3]))

    def answer_template(self) -> str:
        return "Answer: <height>."

    def author_reasoning(self, semantics) -> str:
        heights = [int(value) for value in semantics["heights"]]
        shape = str(semantics["shape"])
        return (
            f"Reading each {shape}'s top edge against the numbered lines: card 1 reaches line "
            f"{heights[0]}, card 2 reaches line {heights[1]}, and card 3 returns to line {heights[2]}. "
            f"The heights alternate between {heights[0]} and {heights[1]}, so card 4 takes the other "
            f"value again: height {heights[3]}."
        )

    def prompt_copy(self, semantics) -> Dict[str, str]:
        return {
            "input_copy": SIZE_INPUT_COPY,
            "rule_copy": (
                f"{SIZE_RULER_SENTENCE} The three visible heights alternate between two fixed values, "
                "and card 4 continues that alternation once; the shape and its color never change."
            ),
            "answer_meaning": "the numbered line card 4's shape must rise to.",
            "terminal_copy": SIZE_TERMINAL_COPY,
        }

    def structured_semantics(self, semantics) -> Dict[str, Any]:
        return {
            "shape": str(semantics["shape"]),
            "heights": [int(value) for value in semantics["heights"]],
            "observed_heights": [int(value) for value in semantics["heights"][:3]],
            "correct_height": int(semantics["heights"][3]),
            "render_geometry": {
                "arrangement": _ARRANGEMENT,
                "shape_rgb": [int(value) for value in semantics["color"]],
                "ruler_units": RULER_UNITS,
            },
        }

    def glyph_inventory(self, semantics) -> Dict[str, Any]:
        del semantics
        return dict(SIZE_CARD_GLYPHS)


def build_sequence_replay(state: Mapping[str, Any]) -> SequenceReplay:
    observed = [int(value) for value in state.get("observed_heights") or []]
    fourth = _infer_fourth(observed)
    if fourth is None:
        raise ValueError("alternating-size source does not define the stated alternation")
    shape = str(state.get("shape") or "")
    geometry = state.get("render_geometry") or {}
    color = [int(value) for value in (geometry.get("shape_rgb") or [])]
    if shape not in SIZE_SHAPES or len(color) != 3 or tuple(color) not in SIZE_COLORS:
        raise ValueError("alternating-size source is missing its sampled shape or color")
    semantics = {"shape": shape, "color": color, "heights": [*observed, fourth]}
    problem, terminal = render_pair_from_semantics(semantics)
    templates = {}
    for height in range(1, RULER_UNITS + 1):
        variant = dict(semantics, heights=[*observed, height])
        templates[str(height)] = np.asarray(render_pair_from_semantics(variant)[1], dtype=np.uint8)
    decoder = make_template_decoder(templates, final_content_bbox(_ARRANGEMENT, _TOTAL_CARDS))
    from data.common.prompt_paraphrase import active_profile
    if active_profile('height_solids'):
        from data.in_context_learning.core.size_cards import make_ruled_height_decoder
        decoder = make_ruled_height_decoder(templates, final_content_bbox(_ARRANGEMENT, _TOTAL_CARDS), color)
    return SequenceReplay(
        expected_payload=str(fourth),
        payload_pattern=r"\d",
        problem=np.asarray(problem, dtype=np.uint8),
        terminal=np.asarray(terminal, dtype=np.uint8),
        decode_final_card=decoder,
        diagnostics={"source_replay_fourth_height": int(fourth)},
    )


__all__ = ["AlternatingSizeSequencePattern", "build_sequence_replay", "render_pair_from_semantics"]

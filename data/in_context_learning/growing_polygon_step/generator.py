"""Regular-polygon vertex-count sequence with direct final-card completion."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

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

_COLORS = (
    (63, 111, 212),
    (36, 150, 100),
    (224, 120, 60),
    (149, 88, 192),
)
_VERTEX_DOT = (44, 48, 54)
_ARRANGEMENT = "row4"
_TOTAL_CARDS = 4
_MIN_COUNT, _MAX_COUNT = 3, 9
_START_RANGE = (3, 4, 5)
_STEPS = (1, 2)


def _paint_polygon(sdraw: ScaledDraw, box, item: Mapping[str, Any], position: int) -> None:
    del position
    count = int(item["vertex_count"])
    color = tuple(int(value) for value in item["color"])
    left, top, right, bottom = (float(value) for value in box)
    center = ((left + right) / 2.0, (top + bottom) / 2.0)
    radius = min(right - left, bottom - top) * 0.42
    points = [
        (
            center[0] + radius * math.cos(-math.pi / 2.0 + 2.0 * math.pi * index / count),
            center[1] + radius * math.sin(-math.pi / 2.0 + 2.0 * math.pi * index / count),
        )
        for index in range(count)
    ]
    sdraw.polygon(points, fill=color, outline=_VERTEX_DOT, width=2)
    for x, y in points:
        sdraw.ellipse((x - 3.2, y - 3.2, x + 3.2, y + 3.2), fill=_VERTEX_DOT)


def _visible_items(semantics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    color = list(semantics["color"])
    return [{"vertex_count": int(value), "color": color} for value in semantics["sequence"][:3]]


def _answer_item(semantics: Mapping[str, Any]) -> Dict[str, Any]:
    return {"vertex_count": int(semantics["sequence"][3]), "color": list(semantics["color"])}


def render_pair_from_semantics(semantics: Mapping[str, Any]):
    items = _visible_items(semantics)
    answer = _answer_item(semantics)
    problem = render_sequence_scene(
        arrangement=_ARRANGEMENT, items=items, answer_item=None, painter=_paint_polygon, solved=False
    )
    terminal = render_sequence_scene(
        arrangement=_ARRANGEMENT, items=items, answer_item=answer, painter=_paint_polygon, solved=True
    )
    return problem, terminal


def _infer_fourth(observed: Sequence[int]) -> Optional[int]:
    values = [int(value) for value in observed]
    if len(values) != 3 or values[0] < _MIN_COUNT:
        return None
    step = values[1] - values[0]
    if abs(step) not in _STEPS or values[2] - values[1] != step:
        return None
    fourth = values[2] + step
    return fourth if _MIN_COUNT <= fourth <= _MAX_COUNT else None


class GrowingPolygonStepPattern(SequenceCompletionGenerator):
    task_name = "growing_polygon_step"
    arrangement = _ARRANGEMENT

    def __init__(self, *args, descending=False, **kwargs):
        self.descending = bool(descending)
        super().__init__(*args, **kwargs)

    def sample_semantics(self) -> Dict[str, Any]:
        for _attempt in range(40):
            start = self.rng.choice(_START_RANGE)
            step = self.rng.choice(_STEPS)
            sequence = [start + step * index for index in range(4)]
            if sequence[3] > _MAX_COUNT:
                continue
            if self.descending:
                sequence.reverse()
                step = -step
            if _infer_fourth(sequence[:3]) != sequence[3]:
                raise RuntimeError("growing_polygon_step replay disagrees with the sampled sequence")
            return {
                "sequence": sequence,
                "step": step,
                "color": list(self.rng.choice(_COLORS)),
            }
        raise RuntimeError("growing_polygon_step failed to sample a bounded sequence")

    def visible_items(self, semantics):
        return _visible_items(semantics)

    def answer_item(self, semantics):
        return _answer_item(semantics)

    def paint_content(self, sdraw, box, item, position):
        _paint_polygon(sdraw, box, item, position)

    def answer_payload(self, semantics) -> str:
        return str(int(semantics["sequence"][3]))

    def answer_template(self) -> str:
        return "Answer: <count>."

    def author_reasoning(self, semantics) -> str:
        sequence = [int(value) for value in semantics["sequence"]]
        step = int(semantics["step"])
        names = {
            3: "triangle",
            4: "square",
            5: "pentagon",
            6: "hexagon",
            7: "heptagon",
            8: "octagon",
            9: "nine-sided polygon",
        }
        return (
            f"Counting the marked corners: card 1 shows a {names[sequence[0]]} with {sequence[0]} vertices, "
            f"card 2 has {sequence[1]}, and card 3 has {sequence[2]}. Each card gains "
            f"{step} vertices algebraically, so card 4 needs a regular polygon with "
            f"{sequence[3]} vertices."
        )

    def prompt_copy(self, semantics) -> Dict[str, str]:
        return {
            "input_copy": (
                "Four numbered cards show one polygon sequence: cards 1-3 each hold a regular polygon "
                "with a small dot on every vertex, and card 4 holds only a question mark."
            ),
            "rule_copy": (
                f"The vertex counts of cards 1-3 {'decrease' if self.descending else 'increase'} by one constant whole-number step of 1 or 2. "
                "Card 4 continues that same signed step once; orientation and drawing size never change a "
                "vertex count."
            ),
            "answer_meaning": "the number of vertices card 4's polygon needs.",
            "terminal_copy": (
                "a regular polygon with exactly that many vertices, drawn at the same color, size, and "
                "vertex dots as the visible polygons."
            ),
        }

    def structured_semantics(self, semantics) -> Dict[str, Any]:
        return {
            "sequence": [int(value) for value in semantics["sequence"]],
            "observed_counts": [int(value) for value in semantics["sequence"][:3]],
            "correct_count": int(semantics["sequence"][3]),
            "step": int(semantics["step"]),
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
    observed = [int(value) for value in state.get("observed_counts") or []]
    fourth = _infer_fourth(observed)
    if fourth is None:
        raise ValueError("growing-polygon source does not define one allowed constant step")
    geometry = state.get("render_geometry") or {}
    color = [int(value) for value in (geometry.get("polygon_rgb") or [])]
    if len(color) != 3 or tuple(color) not in _COLORS:
        raise ValueError("growing-polygon source is missing its sampled polygon color")
    semantics = {"sequence": [*observed, fourth], "step": observed[1] - observed[0], "color": color}
    problem, terminal = render_pair_from_semantics(semantics)
    templates = {}
    for count in range(_MIN_COUNT, _MAX_COUNT + 1):
        variant = dict(semantics, sequence=[*observed, count])
        templates[str(count)] = np.asarray(render_pair_from_semantics(variant)[1], dtype=np.uint8)
    decoder = make_template_decoder(templates, final_content_bbox(_ARRANGEMENT, _TOTAL_CARDS))
    return SequenceReplay(
        expected_payload=str(fourth),
        payload_pattern=r"\d",
        problem=np.asarray(problem, dtype=np.uint8),
        terminal=np.asarray(terminal, dtype=np.uint8),
        decode_final_card=decoder,
        diagnostics={"source_replay_fourth_count": int(fourth)},
    )


__all__ = ["GrowingPolygonStepPattern", "build_sequence_replay", "render_pair_from_semantics"]

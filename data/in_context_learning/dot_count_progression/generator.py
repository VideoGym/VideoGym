"""Dot-count sequence extrapolation with direct final-card completion."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data.common.render_primitives import ScaledDraw
from data.in_context_learning.core.sequence_cards import (
    SequenceCompletionGenerator,
    final_content_bbox,
    render_sequence_scene,
)
from data.in_context_learning.core.sequence_completion_verifier import SequenceReplay

_RULES = ("arithmetic", "geometric", "squares", "triangulars", "powers_of_2")
_COLORS = (
    (63, 111, 212),
    (36, 150, 100),
    (224, 120, 60),
    (149, 88, 192),
)
_DOT_EDGE = (52, 56, 62)
_GRID_CAPACITY = 36
_ARRANGEMENT = "row4"
_TOTAL_CARDS = 4


def _seq_arith(a: int, d: int, k: int) -> List[int]:
    return [a + index * d for index in range(k)]


def _seq_geo(a: int, ratio: int, k: int) -> List[int]:
    return [a * (ratio**index) for index in range(k)]


def _matching_rule_families(observed: Sequence[int]) -> List[str]:
    """Return every allowed family matching the three observed counts.

    The canonical powers-of-two counts are reserved for the ``powers_of_2``
    family rather than also being admitted as a generic integer-ratio
    sequence, keeping the five named families disjoint for every sample.
    """

    values = [int(value) for value in observed]
    if len(values) != 3:
        return []
    matches: List[str] = []
    first, second, third = values
    step = second - first
    if step > 0 and third - second == step:
        matches.append("arithmetic")
    if values != [1, 2, 4] and first > 0 and second % first == 0:
        ratio = second // first
        if ratio > 1 and second * ratio == third:
            matches.append("geometric")
    if values == [1, 4, 9]:
        matches.append("squares")
    if values == [1, 3, 6]:
        matches.append("triangulars")
    if values == [1, 2, 4]:
        matches.append("powers_of_2")
    return matches


def _infer_rule_and_fourth(observed: Sequence[int]) -> Tuple[List[str], Optional[int]]:
    values = [int(value) for value in observed]
    if len(values) != 3 or not all(1 <= value <= _GRID_CAPACITY for value in values):
        return [], None
    matches: List[Tuple[str, int]] = []
    if values == [1, 4, 9]:
        matches.append(("squares", 16))
    if values == [1, 3, 6]:
        matches.append(("triangulars", 10))
    if values == [1, 2, 4]:
        matches.append(("powers_of_2", 8))
    if values == [3, 6, 12]:
        matches.append(("geometric", 24))
    if values == [1, 3, 9]:
        matches.append(("geometric", 27))
    step = values[1] - values[0]
    if step > 0 and values[2] - values[1] == step:
        matches.append(("arithmetic", values[2] + step))
    valid = [(rule, fourth) for rule, fourth in matches if 1 <= fourth <= _GRID_CAPACITY]
    return [rule for rule, _fourth in valid], valid[0][1] if len(valid) == 1 else None


def _grid_geometry(box: Sequence[float]) -> Tuple[float, float, float]:
    left, top, right, bottom = (float(value) for value in box)
    cell = min((right - left) / 6.0, (bottom - top) / 6.0)
    grid_left = left + (right - left - cell * 6.0) / 2.0
    grid_top = top + (bottom - top - cell * 6.0) / 2.0
    return grid_left, grid_top, cell


def grid_centers(box: Sequence[float]) -> List[Tuple[float, float]]:
    """Reading-order dot centers of one card's six-by-six grid."""

    grid_left, grid_top, cell = _grid_geometry(box)
    return [
        (grid_left + (index % 6 + 0.5) * cell, grid_top + (index // 6 + 0.5) * cell)
        for index in range(_GRID_CAPACITY)
    ]


def _paint_dots(sdraw: ScaledDraw, box, item: Mapping[str, Any], position: int) -> None:
    del position
    count = int(item["count"])
    color = tuple(int(value) for value in item["color"])
    centers = grid_centers(box)
    _left, _top, cell = _grid_geometry(box)
    radius = max(3.0, cell * 0.30)
    for center_x, center_y in centers[:count]:
        sdraw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill=color,
            outline=_DOT_EDGE,
            width=1,
        )


def _visible_items(semantics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    color = list(semantics["color"])
    return [{"count": int(value), "color": color} for value in semantics["sequence"][:3]]


def _answer_item(semantics: Mapping[str, Any]) -> Dict[str, Any]:
    return {"count": int(semantics["sequence"][3]), "color": list(semantics["color"])}


def render_pair_from_semantics(semantics: Mapping[str, Any]):
    items = _visible_items(semantics)
    answer = _answer_item(semantics)
    problem = render_sequence_scene(
        arrangement=_ARRANGEMENT, items=items, answer_item=None, painter=_paint_dots, solved=False
    )
    terminal = render_sequence_scene(
        arrangement=_ARRANGEMENT, items=items, answer_item=answer, painter=_paint_dots, solved=True
    )
    return problem, terminal


def _rule_explanation(observed: Sequence[int], rule: str, target: int) -> str:
    first, second, third = (int(value) for value in observed)
    if rule == "arithmetic":
        step = second - first
        return f"The increases are both {step}, so adding {step} once more gives {target}."
    if rule == "geometric":
        ratio = second // first if first else 0
        return f"Each count is multiplied by {ratio}, so multiplying {third} by {ratio} gives {target}."
    if rule == "squares":
        return f"The counts are 1, 4, and 9 - the first three perfect squares - so the next square is {target}."
    if rule == "triangulars":
        return f"The increases are 2 and 3; the next increase is 4, giving {third} + 4 = {target}."
    return f"Each count doubles, so doubling {third} gives {target}."


class DotCountProgressionPattern(SequenceCompletionGenerator):
    task_name = "dot_count_progression"
    arrangement = _ARRANGEMENT

    def sample_semantics(self) -> Dict[str, Any]:
        rule = self.rng.choice(_RULES)
        if rule == "arithmetic":
            sequence = _seq_arith(self.rng.randint(1, 4), self.rng.randint(2, 4), 4)
        elif rule == "geometric":
            start, ratio = self.rng.choice(((3, 2), (1, 3)))
            sequence = _seq_geo(start, ratio, 4)
        elif rule == "squares":
            sequence = [1, 4, 9, 16]
        elif rule == "triangulars":
            sequence = [1, 3, 6, 10]
        else:
            sequence = [1, 2, 4, 8]
        if not all(1 <= value <= _GRID_CAPACITY for value in sequence):
            raise RuntimeError("dot_count_progression sampled a count outside the six-by-six grid")
        matches = _matching_rule_families(sequence[:3])
        if matches != [rule]:
            raise RuntimeError(
                f"dot_count_progression ambiguous rule family: {sequence[:3]} matches {matches}"
            )
        inferred_rules, fourth = _infer_rule_and_fourth(sequence[:3])
        if inferred_rules != [rule] or fourth != sequence[3]:
            raise RuntimeError("dot_count_progression replay disagrees with the sampled sequence")
        return {
            "rule": rule,
            "sequence": [int(value) for value in sequence],
            "color": list(self.rng.choice(_COLORS)),
        }

    def visible_items(self, semantics):
        return _visible_items(semantics)

    def answer_item(self, semantics):
        return _answer_item(semantics)

    def paint_content(self, sdraw, box, item, position):
        _paint_dots(sdraw, box, item, position)

    def answer_payload(self, semantics) -> str:
        return str(int(semantics["sequence"][3]))

    def answer_template(self) -> str:
        return "Answer: <count>."

    def author_reasoning(self, semantics) -> str:
        observed = [int(value) for value in semantics["sequence"][:3]]
        target = int(semantics["sequence"][3])
        rule = str(semantics["rule"])
        return (
            f"Card 1 holds {observed[0]} {'dot' if observed[0] == 1 else 'dots'}, card 2 holds "
            f"{observed[1]}, and card 3 holds {observed[2]}. "
            f"{_rule_explanation(observed, rule, target)} "
            f"So card 4 needs {target} dots."
        )

    def prompt_copy(self, semantics) -> Dict[str, str]:
        return {
            "input_copy": (
                "Four numbered cards show one dot-count sequence: cards 1-3 each hold colored dots on a "
                "six-by-six grid, and card 4 holds only a question mark."
            ),
            "rule_copy": (
                "The three visible counts match exactly one of these families: a positive constant "
                "difference; a constant integer ratio, as in 3, 6, 12 or 1, 3, 9; the perfect squares "
                "1, 4, 9; the triangular numbers 1, 3, 6; or the powers of two 1, 2, 4, which always "
                "count as powers of two rather than as a generic ratio. Card 4 continues the matching "
                "family by one more term."
            ),
            "answer_meaning": "the number of dots card 4 needs.",
            "terminal_copy": (
                "exactly that many dots at the same dot color and size, filling card 4's grid cells "
                "in reading order."
            ),
        }

    def structured_semantics(self, semantics) -> Dict[str, Any]:
        return {
            "rule": str(semantics["rule"]),
            "matching_rule_families": _matching_rule_families(semantics["sequence"][:3]),
            "sequence": [int(value) for value in semantics["sequence"]],
            "observed_counts": [int(value) for value in semantics["sequence"][:3]],
            "correct_count": int(semantics["sequence"][3]),
            "render_geometry": {
                "arrangement": _ARRANGEMENT,
                "dot_rgb": [int(value) for value in semantics["color"]],
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


def _decode_dot_card(candidate: np.ndarray, color: Sequence[int]) -> Tuple[Optional[str], Dict[str, Any]]:
    box = final_content_bbox(_ARRANGEMENT, _TOTAL_CARDS)
    centers = grid_centers(box)
    target = np.asarray(color, dtype=np.int16)
    flags: List[bool] = []
    for center_x, center_y in centers:
        x, y = int(round(center_x)), int(round(center_y))
        patch = candidate[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4].astype(np.int16)
        if patch.size == 0:
            flags.append(False)
            continue
        distance = np.linalg.norm(patch - target, axis=2)
        flags.append(int(np.count_nonzero(distance <= 80.0)) >= 4)
    count = 0
    while count < len(flags) and flags[count]:
        count += 1
    prefix_exact = not any(flags[count:])
    if not prefix_exact or count == 0:
        return None, {"occupied_flags": flags, "prefix_exact": prefix_exact}
    return str(count), {"decoded_count": count, "prefix_exact": True}


def build_sequence_replay(state: Mapping[str, Any]) -> SequenceReplay:
    observed = [int(value) for value in state.get("observed_counts") or []]
    rules, fourth = _infer_rule_and_fourth(observed)
    if len(rules) != 1 or fourth is None:
        raise ValueError("dot-count source does not identify one allowed progression")
    geometry = state.get("render_geometry") or {}
    color = [int(value) for value in (geometry.get("dot_rgb") or [])]
    if len(color) != 3 or tuple(color) not in _COLORS:
        raise ValueError("dot-count source is missing its sampled dot color")
    semantics = {"rule": rules[0], "sequence": [*observed, int(fourth)], "color": color}
    problem, terminal = render_pair_from_semantics(semantics)
    return SequenceReplay(
        expected_payload=str(int(fourth)),
        payload_pattern=r"\d{1,2}",
        problem=np.asarray(problem, dtype=np.uint8),
        terminal=np.asarray(terminal, dtype=np.uint8),
        decode_final_card=lambda candidate: _decode_dot_card(candidate, color),
        diagnostics={"source_replay_rule": rules[0], "source_replay_fourth_count": int(fourth)},
    )


__all__ = [
    "DotCountProgressionPattern",
    "build_sequence_replay",
    "grid_centers",
    "render_pair_from_semantics",
]

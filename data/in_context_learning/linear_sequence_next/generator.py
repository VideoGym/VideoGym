"""Repeated grid-transformation sequence with direct final-card completion."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from data.common.grid_coordinates import (
    GRID_POSITION_LIST_ANSWER_SCHEMA,
    GRID_XY_CONVENTION_SENTENCE,
    cell_to_xy,
    format_xy_position_list_payload,
    format_xy_position,
    parse_xy_positions_answer,
)
from data.common.pipeline.media import load_font
from data.common.render_primitives import DOCTRINE_SUPERSAMPLE, ScaledDraw
from data.in_context_learning.core.sequence_cards import (
    SequenceCompletionGenerator,
    final_content_bbox,
    render_sequence_scene,
)
from data.in_context_learning.core.sequence_completion_verifier import SequenceReplay

_ARRANGEMENT = "grid4"
_TOTAL_CARDS = 4
_GRID_N = 6
_TRAINING_RULES = ("translate_right", "translate_down", "translate_diag", "rotate_cw")
_RULES = (*_TRAINING_RULES, "translate_left")
_RULE_COPY = {
    "translate_left": "every filled cell moves one column to the left",
    "translate_right": "every filled cell moves one column to the right",
    "translate_down": "every filled cell moves one row down",
    "translate_diag": "every filled cell moves one row down and one column to the right",
    "rotate_cw": "the whole pattern turns a quarter-turn clockwise about the grid center",
}
_COLORS = (
    (63, 111, 212),
    (36, 150, 100),
    (224, 120, 60),
    (149, 88, 192),
)
_GRID_LINE = (206, 211, 219)
_LABEL_COLOR = (122, 128, 138)
_CELL_EDGE = (52, 56, 62)
_BASE_SHAPES: Tuple[Tuple[Tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
)

Cell = Tuple[int, int]  # (row, col), 0-based internally


def _normalise(cells) -> Tuple[Cell, ...]:
    return tuple(sorted((int(cell[0]), int(cell[1])) for cell in cells))


def _step(cells: Sequence[Cell], rule: str) -> Tuple[Cell, ...]:
    if rule == "translate_left":
        return _normalise((row, col - 1) for row, col in cells)
    if rule == "translate_right":
        return _normalise((row, col + 1) for row, col in cells)
    if rule == "translate_down":
        return _normalise((row + 1, col) for row, col in cells)
    if rule == "translate_diag":
        return _normalise((row + 1, col + 1) for row, col in cells)
    if rule == "rotate_cw":
        return _normalise((col, _GRID_N - 1 - row) for row, col in cells)
    raise ValueError(f"unsupported linear-sequence rule: {rule}")


def _in_bounds(cells: Sequence[Cell]) -> bool:
    return all(0 <= row < _GRID_N and 0 <= col < _GRID_N for row, col in cells)


def _infer_repeated_rule(frames: Sequence[Sequence[Cell]]) -> Optional[str]:
    normalized = [_normalise(frame) for frame in frames[:3]]
    if len(normalized) != 3:
        return None
    matching = [
        rule
        for rule in _RULES
        if _step(normalized[0], rule) == normalized[1] and _step(normalized[1], rule) == normalized[2]
    ]
    return matching[0] if len(matching) == 1 else None


def _grid_geometry(box) -> Tuple[float, float, float]:
    """Return (grid_left, grid_top, cell) for one content box's labeled grid."""

    left, top, right, bottom = (float(value) for value in box)
    label_band = 15.0
    cell = min(
        (right - left - label_band - 6.0) / _GRID_N,
        (bottom - top - label_band - 6.0) / _GRID_N,
    )
    grid_left = left + label_band + (right - left - label_band - cell * _GRID_N) / 2.0
    grid_top = top + label_band + (bottom - top - label_band - cell * _GRID_N) / 2.0
    return grid_left, grid_top, cell


def cell_center(box, row: int, col: int) -> Tuple[float, float]:
    grid_left, grid_top, cell = _grid_geometry(box)
    return grid_left + (col + 0.5) * cell, grid_top + (row + 0.5) * cell


def _paint_grid_frame(sdraw: ScaledDraw, box) -> None:
    grid_left, grid_top, cell = _grid_geometry(box)
    size = cell * _GRID_N
    label_font = load_font(9 * DOCTRINE_SUPERSAMPLE)
    for index in range(_GRID_N + 1):
        offset = index * cell
        sdraw.line((grid_left + offset, grid_top, grid_left + offset, grid_top + size), fill=_GRID_LINE, width=1)
        sdraw.line((grid_left, grid_top + offset, grid_left + size, grid_top + offset), fill=_GRID_LINE, width=1)
    for index in range(_GRID_N):
        center = (index + 0.5) * cell
        sdraw.text((grid_left + center, grid_top - 7.0), str(index + 1), fill=_LABEL_COLOR, font=label_font, anchor="mm")
        sdraw.text((grid_left - 8.0, grid_top + center), str(index + 1), fill=_LABEL_COLOR, font=label_font, anchor="mm")


def _paint_cells(sdraw: ScaledDraw, box, item: Mapping[str, Any], position: int) -> None:
    del position
    _paint_grid_frame(sdraw, box)
    color = tuple(int(value) for value in item["color"])
    grid_left, grid_top, cell = _grid_geometry(box)
    inset = max(2.0, cell * 0.12)
    for row, col in item["cells"]:
        sdraw.rounded_rectangle(
            (
                grid_left + col * cell + inset,
                grid_top + row * cell + inset,
                grid_left + (col + 1) * cell - inset,
                grid_top + (row + 1) * cell - inset,
            ),
            radius=3,
            fill=color,
            outline=_CELL_EDGE,
            width=1,
        )


def _visible_items(semantics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    color = list(semantics["color"])
    return [{"cells": [tuple(cell) for cell in frame], "color": color} for frame in semantics["frames"][:3]]


def _answer_item(semantics: Mapping[str, Any]) -> Dict[str, Any]:
    return {"cells": [tuple(cell) for cell in semantics["frames"][3]], "color": list(semantics["color"])}


def render_pair_from_semantics(semantics: Mapping[str, Any]):
    items = _visible_items(semantics)
    answer = _answer_item(semantics)
    problem = render_sequence_scene(
        arrangement=_ARRANGEMENT,
        items=items,
        answer_item=None,
        painter=_paint_cells,
        solved=False,
        placeholder_painter=_paint_grid_frame,
    )
    terminal = render_sequence_scene(
        arrangement=_ARRANGEMENT,
        items=items,
        answer_item=answer,
        painter=_paint_cells,
        solved=True,
    )
    return problem, terminal


def _payload_from_cells(cells: Sequence[Cell]) -> str:
    return format_xy_position_list_payload(_normalise(cells))


def _parse_payload_cells(payload: str) -> Optional[Set[Cell]]:
    parsed = parse_xy_positions_answer(f"Answer: {str(payload or '').strip()}.")
    if parsed is None:
        return None
    ordered = _normalise(parsed)
    if tuple(parsed) != ordered or len(set(parsed)) != len(parsed):
        return None
    if not _in_bounds(parsed):
        return None
    return set(parsed)


def _cells_sentence(cells: Sequence[Cell]) -> str:
    ordered = _normalise(cells)
    return ", ".join(format_xy_position(cell_to_xy(cell)) for cell in ordered)


class LinearSequenceNextPattern(SequenceCompletionGenerator):
    task_name = "linear_sequence_next"
    arrangement = _ARRANGEMENT

    def __init__(self, *args, reverse_translation=False, **kwargs):
        self.rule_pool = ("translate_left",) if reverse_translation else _TRAINING_RULES
        super().__init__(*args, **kwargs)

    def sample_semantics(self) -> Dict[str, Any]:
        for _attempt in range(200):
            rule = self.rng.choice(self.rule_pool)
            base = _normalise(self.rng.choice(_BASE_SHAPES))
            max_row = max(row for row, _col in base)
            max_col = max(col for _row, col in base)
            if rule == "rotate_cw":
                row_offset = self.rng.randint(0, _GRID_N - 1 - max_row)
                col_offset = self.rng.randint(0, _GRID_N - 1 - max_col)
            else:
                row_budget = 3 if rule in {"translate_down", "translate_diag"} else 0
                col_budget = 3 if rule in {"translate_right", "translate_diag"} else 0
                if (
                    _GRID_N - 1 - max_row - row_budget < 0
                    or _GRID_N - 1 - max_col - col_budget < 0
                ):
                    continue
                row_offset = self.rng.randint(0, _GRID_N - 1 - max_row - row_budget)
                col_offset = self.rng.randint(0, _GRID_N - 1 - max_col - col_budget)
            start = _normalise(
                (row + row_offset, col + col_offset) for row, col in base
            )
            frames = [start]
            for _ in range(3):
                frames.append(_step(frames[-1], rule))
            if not all(_in_bounds(frame) for frame in frames):
                continue
            if frames[3] == frames[2]:
                continue
            if _infer_repeated_rule(frames) != rule:
                continue
            return {
                "rule": rule,
                "frames": [list(frame) for frame in frames],
                "color": list(self.rng.choice(_COLORS)),
            }
        raise RuntimeError("linear_sequence_next failed to sample an unambiguous in-bounds sequence")

    def visible_items(self, semantics):
        return _visible_items(semantics)

    def answer_item(self, semantics):
        return _answer_item(semantics)

    def paint_content(self, sdraw, box, item, position):
        _paint_cells(sdraw, box, item, position)

    def paint_placeholder(self, sdraw, box):
        _paint_grid_frame(sdraw, box)

    def answer_payload(self, semantics) -> str:
        return _payload_from_cells(semantics["frames"][3])

    def answer_template(self) -> str:
        return GRID_POSITION_LIST_ANSWER_SCHEMA

    def author_reasoning(self, semantics) -> str:
        frames = [_normalise(frame) for frame in semantics["frames"]]
        rule = str(semantics["rule"])
        return (
            f"Card 1 fills {_cells_sentence(frames[0])}, and card 2 fills {_cells_sentence(frames[1])}: "
            f"{_RULE_COPY[rule]}. Card 3 fills {_cells_sentence(frames[2])}, which is the same change "
            f"applied again. Applying it once more to card 3 fills {_cells_sentence(frames[3])}, so "
            f"card 4's cells are {_payload_from_cells(frames[3])}."
        )

    def prompt_copy(self, semantics) -> Dict[str, str]:
        return {
            "input_copy": (
                "Four numbered cards show one grid sequence: cards 1-3 each hold filled cells on the "
                "same six-by-six grid with row and column numbers 1-6 along its edges, and card 4 "
                f"holds only a question mark over an empty copy of that grid. {GRID_XY_CONVENTION_SENTENCE}"
            ),
            "rule_copy": (
                "One fixed change maps card 1 to card 2 and card 2 to card 3: either every filled "
                f"cell shifts one column right{', one column left' if self.rule_pool == ('translate_left',) else ''}, one row down, or one row down and one column right, or "
                "the whole pattern turns a quarter-turn clockwise about the grid center. Exactly one "
                "of these changes fits both visible transitions, and card 4 applies it once more."
            ),
            "answer_meaning": (
                "listing card 4's filled grid positions in increasing x order and then increasing y order, "
                "with one space after every comma."
            ),
            "terminal_copy": (
                "exactly those cells filled at the same cell color on card 4's grid, with every other "
                "cell left empty."
            ),
        }

    def structured_semantics(self, semantics) -> Dict[str, Any]:
        return {
            "rule": str(semantics["rule"]),
            "frames": [[list(cell) for cell in _normalise(frame)] for frame in semantics["frames"]],
            "observed_frames": [[list(cell) for cell in _normalise(frame)] for frame in semantics["frames"][:3]],
            "correct_cells": [list(cell) for cell in _normalise(semantics["frames"][3])],
            "grid_size": _GRID_N,
            "render_geometry": {
                "arrangement": _ARRANGEMENT,
                "shape_rgb": [int(value) for value in semantics["color"]],
            },
        }

    def glyph_inventory(self, semantics) -> Dict[str, Any]:
        del semantics
        return {
            "schema_version": "icl_rendered_glyph_inventory_v1",
            "static_whitelist": ["1", "2", "3", "4", "5", "6", "?"],
            "bounded_dynamic_schema": [],
            "observed_dynamic_values": [],
            "answer_fields_excluded": True,
        }


def _decode_cells_card(candidate: np.ndarray, color: Sequence[int]) -> Tuple[Optional[str], Dict[str, Any]]:
    box = final_content_bbox(_ARRANGEMENT, _TOTAL_CARDS)
    target = np.asarray(color, dtype=np.int16)
    occupied: List[Cell] = []
    for row in range(_GRID_N):
        for col in range(_GRID_N):
            x, y = cell_center(box, row, col)
            xi, yi = int(round(x)), int(round(y))
            patch = candidate[max(0, yi - 3) : yi + 4, max(0, xi - 3) : xi + 4].astype(np.int16)
            if patch.size == 0:
                continue
            distance = np.linalg.norm(patch - target, axis=2)
            if int(np.count_nonzero(distance <= 80.0)) >= 6:
                occupied.append((row, col))
    if not occupied:
        return None, {"decoded_xy_positions": []}
    return _payload_from_cells(occupied), {
        "decoded_xy_positions": [
            format_xy_position(cell_to_xy(cell))
            for cell in _normalise(occupied)
        ]
    }


def build_sequence_replay(state: Mapping[str, Any]) -> SequenceReplay:
    observed = [
        _normalise(frame) for frame in (state.get("observed_frames") or []) if isinstance(frame, list)
    ]
    if len(observed) != 3 or not all(_in_bounds(frame) for frame in observed):
        raise ValueError("linear-sequence source lacks three visible in-bounds states")
    rule = _infer_repeated_rule(observed)
    if rule is None:
        raise ValueError("linear-sequence source does not identify one repeated change")
    fourth = _step(observed[2], rule)
    if not _in_bounds(fourth):
        raise ValueError("linear-sequence replayed fourth state leaves the grid")
    geometry = state.get("render_geometry") or {}
    color = [int(value) for value in (geometry.get("shape_rgb") or [])]
    if len(color) != 3 or tuple(color) not in _COLORS:
        raise ValueError("linear-sequence source is missing its sampled cell color")
    semantics = {"rule": rule, "frames": [list(frame) for frame in [*observed, fourth]], "color": color}
    problem, terminal = render_pair_from_semantics(semantics)
    expected_payload = _payload_from_cells(fourth)

    def _payload_equal(candidate: str, expected: str) -> bool:
        parsed = _parse_payload_cells(candidate)
        return parsed is not None and parsed == _parse_payload_cells(expected)

    return SequenceReplay(
        expected_payload=expected_payload,
        payload_pattern=r"\[\([1-6], [1-6]\)(?:, \([1-6], [1-6]\))*\]",
        problem=np.asarray(problem, dtype=np.uint8),
        terminal=np.asarray(terminal, dtype=np.uint8),
        decode_final_card=lambda candidate: _decode_cells_card(candidate, color),
        payload_equal=_payload_equal,
        diagnostics={"source_replay_rule": rule, "source_replay_fourth_cells": [list(cell) for cell in fourth]},
    )


__all__ = ["LinearSequenceNextPattern", "build_sequence_replay", "render_pair_from_semantics"]

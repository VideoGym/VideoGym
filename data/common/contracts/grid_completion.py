"""Shared rendering and serialization utilities for constraint-grid tasks."""

from __future__ import annotations

from typing import List, Mapping, Sequence

from PIL import ImageFont

from data.common.assets import render_font
from data.common.grid_coordinates import (
    format_visible_named_xy_paths_answer,
    format_xy_position,
    format_xy_positions_payload,
)
from data.common.typed_answers import (
    format_answer_line,
    format_canonical_json_answer,
)



def load_font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)


def answer_json(payload: object) -> str:
    return format_canonical_json_answer(payload)


def answer_xy_positions(positions: Sequence[Sequence[int]]) -> str:
    """Serialize visible 1-based cells as canonical ``(x, y)`` tuples."""

    return format_answer_line(format_xy_positions_payload(positions))


def answer_named_xy_paths(
    paths: Mapping[str, Sequence[Sequence[int]]],
) -> str:
    """Serialize visible named paths while preserving their mapping order."""

    return format_visible_named_xy_paths_answer(paths)


def answer_xy_composites(
    composites: Sequence[Sequence[Sequence[int]]],
) -> str:
    """Serialize rectangles or edges as pairs of visible ``(x, y)`` positions."""

    if not composites:
        raise ValueError("at least one coordinate-pair composite is required")
    items: List[str] = []
    for composite in composites:
        if len(composite) != 2:
            raise ValueError("each coordinate composite must contain exactly two positions")
        items.append(
            f"({format_xy_position(composite[0])}, {format_xy_position(composite[1])})"
        )
    return format_answer_line("[" + ", ".join(items) + "]")


__all__ = [
    "answer_json",
    "answer_named_xy_paths",
    "answer_xy_composites",
    "answer_xy_positions",
    "load_font",
]

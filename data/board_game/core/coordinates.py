"""Shared numeric ``(x, y)`` coordinate contract for Board Game tasks.

Semantic engines may index arrays with zero-based ``(row, column)`` tuples,
but every model-facing coordinate is numeric and one-based: ``x`` is the row
counted from the top and ``y`` is the column counted from the left.  Keeping
the conversion here prevents task-local Prompt, CoT, Answer, and verifier code
from drifting in terminology or exposing Python indices.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    format_xy_position,
)

Cell = Tuple[int, int]


def require_internal_cell(cell: Sequence[int], *, size: int, name: str = "cell") -> Cell:
    """Validate one internal zero-based cell on a square board."""

    if (
        isinstance(cell, (str, bytes))
        or not isinstance(cell, Sequence)
        or len(cell) != 2
        or isinstance(cell[0], bool)
        or isinstance(cell[1], bool)
        or not isinstance(cell[0], int)
        or not isinstance(cell[1], int)
        or not (0 <= cell[0] < size)
        or not (0 <= cell[1] < size)
    ):
        raise ValueError(f"{name} must be an internal 0-based cell on the {size}x{size} board")
    return int(cell[0]), int(cell[1])


def to_visible_cell(cell: Sequence[int], *, size: int, name: str = "cell") -> Cell:
    """Convert an internal cell to the visible one-based coordinate."""

    row, col = require_internal_cell(cell, size=size, name=name)
    return row + 1, col + 1


def from_visible_cell(cell: Sequence[int], *, size: int, name: str = "cell") -> Cell:
    """Convert a visible one-based coordinate to an internal cell."""

    if (
        isinstance(cell, (str, bytes))
        or not isinstance(cell, Sequence)
        or len(cell) != 2
        or isinstance(cell[0], bool)
        or isinstance(cell[1], bool)
        or not isinstance(cell[0], int)
        or not isinstance(cell[1], int)
        or not (1 <= cell[0] <= size)
        or not (1 <= cell[1] <= size)
    ):
        raise ValueError(f"{name} must be a visible (x, y) pair from 1 through {size}")
    return int(cell[0]) - 1, int(cell[1]) - 1


def visible_cell_text(cell: Sequence[int], *, size: int) -> str:
    """Serialize one internal cell as canonical visible ``(x, y)`` text."""

    return format_xy_position(to_visible_cell(cell, size=size))


__all__ = [
    "Cell",
    "GRID_XY_CONVENTION_SENTENCE",
    "from_visible_cell",
    "require_internal_cell",
    "to_visible_cell",
    "visible_cell_text",
]

"""Model-facing coordinate prose for Constraint Grid tasks."""

from __future__ import annotations

from typing import Sequence

from data.common.grid_coordinates import (
    cell_to_xy,
    format_xy_position,
)


def display_cell_text(cell: Sequence[int]) -> str:
    """Format one internal zero-based cell as the public ``(x, y)`` position."""

    return format_xy_position(cell_to_xy(cell))


def display_edge_text(first: Sequence[int], second: Sequence[int]) -> str:
    """Format one internal edge using two public positions."""

    return f"{display_cell_text(first)}–{display_cell_text(second)}"


__all__ = ["display_cell_text", "display_edge_text"]

"""One-based coordinate formatting for model-facing Dynamics artifacts."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    cell_to_xy,
    format_xy_path_payload,
    format_xy_position,
)


def display_index(index: int) -> int:
    """Convert an internal zero-based index to the visible one-based index."""
    return int(index) + 1


def with_grid_xy_convention(description: str) -> str:
    """Append the shared public coordinate convention exactly once."""

    text = str(description).strip()
    if GRID_XY_CONVENTION_SENTENCE in text:
        return text
    return f"{text} {GRID_XY_CONVENTION_SENTENCE}"


def display_cell(cell: Sequence[int]) -> Tuple[int, int]:
    """Convert an internal ``(row, column)`` cell to visible coordinates."""
    return cell_to_xy(cell)


def display_cell_text(cell: Sequence[int]) -> str:
    return format_xy_position(display_cell(cell))


def display_cells_text(cells: Iterable[Sequence[int]]) -> str:
    return ", ".join(display_cell_text(cell) for cell in cells)


def display_path_text(cells: Iterable[Sequence[int]]) -> str:
    """Format an ordered internal cell path with the public ``(x, y)`` syntax."""

    return format_xy_path_payload(list(cells))


def display_indices_text(indices: Iterable[int]) -> str:
    return ", ".join(str(display_index(index)) for index in indices)


__all__ = [
    "display_cell",
    "display_cell_text",
    "display_cells_text",
    "display_path_text",
    "display_index",
    "display_indices_text",
    "with_grid_xy_convention",
]

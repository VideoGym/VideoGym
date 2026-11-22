"""Shared supersampled board scenery for Constraint Grid renderers.

Every task draws a square-cell board on a 512x512 canvas. This module owns the
canvas allocation (4x supersampling with LANCZOS finish), standard centered
board geometry, and 1-based axis-label style so task renderers and pixel
decoders derive the same integer geometry from one source.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from PIL import Image

from data.common.contracts.grid_completion import load_font
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)

CANVAS = 512
BG = (250, 251, 254)
GRID_LIGHT = (210, 215, 225)
GRID_BORDER = (40, 44, 52)
TEXT = (24, 28, 36)
# Family-standard axis ordinals: quiet gray, visually subordinate to clue
# numbers so counts and givens keep the dark/navy foreground.
AXIS_LABEL = (110, 118, 130)




def standard_board_geometry(rows: int, cols: int, *, margin: int = 32) -> Tuple[int, int, int]:
    """Return ``(cell, left, top)`` for the family's centered square board.

    This is the same integer formula the family's pixel decoders replay; the
    supersampling factor never changes it because scaling happens inside the
    draw proxy only.
    """

    usable = CANVAS - 2 * margin
    cell = (usable // max(int(rows), int(cols))) & ~1
    left = (CANVAS - int(cols) * cell) // 2
    top = (CANVAS - int(rows) * cell) // 2
    return cell, left, top


def standard_cell_centers(rows: int, cols: int, *, margin: int = 32) -> Dict[Tuple[int, int], Tuple[int, int]]:
    cell, left, top = standard_board_geometry(rows, cols, margin=margin)
    return {
        (r, c): (left + c * cell + cell // 2, top + r * cell + cell // 2)
        for r in range(int(rows))
        for c in range(int(cols))
    }


def new_board_scene(*, factor: int = DOCTRINE_SUPERSAMPLE) -> Tuple[Image.Image, ScaledDraw]:
    return new_scaled_canvas(CANVAS, CANVAS, factor, background=BG)


def finish_board_scene(image: Image.Image) -> Image.Image:
    return finish_scaled_canvas(image, (CANVAS, CANVAS))


def board_font(size: int, *, factor: int = DOCTRINE_SUPERSAMPLE):
    return load_font(max(8, int(size)) * factor)


def draw_board_grid(
    draw: ScaledDraw,
    rows: int,
    cols: int,
    *,
    cell: int,
    left: int,
    top: int,
    light: Tuple[int, int, int] = GRID_LIGHT,
    border: Tuple[int, int, int] = GRID_BORDER,
    border_width: int = 3,
) -> None:
    for r in range(rows + 1):
        y = top + r * cell
        draw.line([(left, y), (left + cols * cell, y)], fill=light, width=1)
    for c in range(cols + 1):
        x = left + c * cell
        draw.line([(x, top), (x, top + rows * cell)], fill=light, width=1)
    draw.rectangle(
        [left, top, left + cols * cell, top + rows * cell],
        outline=border,
        width=border_width,
    )


def draw_axis_labels(
    draw: ScaledDraw,
    rows: int,
    cols: int,
    *,
    cell: int,
    left: int,
    top: int,
    font,
    color: Tuple[int, int, int] = AXIS_LABEL,
    row_side: str = "left",
    col_side: str = "top",
    offset: int = 13,
) -> List[str]:
    """Draw 1-based row/column ordinals along the board edges.

    Returns the exact label strings drawn so glyph inventories can be derived
    from real draw calls instead of being asserted.
    """

    drawn: List[str] = []
    for index in range(cols):
        x = left + index * cell + cell // 2
        y = top - offset if col_side == "top" else top + rows * cell + offset
        draw.text((x, y), str(index + 1), fill=color, font=font, anchor="mm")
        drawn.append(str(index + 1))
    for index in range(rows):
        x = left - offset if row_side == "left" else left + cols * cell + offset
        y = top + index * cell + cell // 2
        draw.text((x, y), str(index + 1), fill=color, font=font, anchor="mm")
        drawn.append(str(index + 1))
    return drawn


__all__ = [
    "AXIS_LABEL",
    "BG",
    "CANVAS",
    "GRID_BORDER",
    "GRID_LIGHT",
    "TEXT",
    "board_font",
    "draw_axis_labels",
    "draw_board_grid",
    "finish_board_scene",
    "new_board_scene",
    "standard_board_geometry",
    "standard_cell_centers",
]

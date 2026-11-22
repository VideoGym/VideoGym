"""Shared structured-grid rendering primitives.

One integer-grid visual language is used wherever a task shows a square board
of printed numbers that the solver reads cell by cell.  Keeping the palette,
stroke weights, cell geometry and glyph metrics here means the constraint-grid
completion boards and the in-context-learning matrix tasks cannot drift into
per-task dialects.

The palette constants and the three low-level drawing helpers below are the
canonical definitions, so there is exactly one implementation and one set of
colour values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from PIL import ImageDraw, ImageFont

from data.common.assets import font_path



RGB = Tuple[int, int, int]

# ---------------------------------------------------------------------------
# Canonical integer-grid palette.
# ---------------------------------------------------------------------------
GRID_PAGE_BG: RGB = (247, 249, 252)
GRID_CELL_FILL: RGB = (234, 242, 252)
GRID_MINOR_LINE: RGB = (153, 160, 171)
GRID_OUTER_LINE: RGB = (52, 60, 78)
GRID_INK: RGB = (29, 33, 41)
# The single accent.  It is reserved for the one bearing, sample-varying mark a
# board may carry; it is never spent on decorative frames or panel outlines.
GRID_ACCENT: RGB = (210, 130, 30)
GRID_ACCENT_FILL: RGB = (255, 240, 220)

GRID_MINOR_WIDTH = 2
GRID_OUTER_WIDTH = 4

CANVAS_PX = 512


def load_font(size: int) -> ImageFont.ImageFont:
    return ImageFont.truetype(str(font_path()), size=size)


def matrix_to_json(matrix: Sequence[Sequence[object]]) -> str:
    return json.dumps([[value for value in row] for row in matrix], ensure_ascii=False)


# ---------------------------------------------------------------------------
# Low-level drawing helpers (canonical definitions).
# ---------------------------------------------------------------------------
def draw_text_centered(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    box: Tuple[int, int, int, int],
    fnt: ImageFont.ImageFont,
    fill: RGB = GRID_INK,
    dy: int = -2,
) -> None:
    if not text:
        return
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = left + (right - left - tw) / 2 - bbox[0]
    y = top + (bottom - top - th) / 2 - bbox[1] + dy
    draw.text((x, y), text, fill=fill, font=fnt)


def draw_grid_lines(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    rows: int,
    cols: int,
    cell_px: int,
    minor_color: RGB = GRID_MINOR_LINE,
    minor_width: int = GRID_MINOR_WIDTH,
    major_every_row: Optional[int] = None,
    major_every_col: Optional[int] = None,
    major_color: RGB = GRID_OUTER_LINE,
    major_width: int = GRID_OUTER_WIDTH,
) -> None:
    """Draw a rows x cols rectangular grid with optional thick separators."""
    width_px = cols * cell_px
    height_px = rows * cell_px

    for c in range(cols + 1):
        x = left + c * cell_px
        is_major = (
            major_every_col is not None
            and (c % major_every_col == 0)
        )
        is_edge = c == 0 or c == cols
        color = major_color if (is_major or is_edge) else minor_color
        w = major_width if (is_major or is_edge) else minor_width
        draw.line([(x, top), (x, top + height_px)], fill=color, width=w)

    for r in range(rows + 1):
        y = top + r * cell_px
        is_major = (
            major_every_row is not None
            and (r % major_every_row == 0)
        )
        is_edge = r == 0 or r == rows
        color = major_color if (is_major or is_edge) else minor_color
        w = major_width if (is_major or is_edge) else minor_width
        draw.line([(left, y), (left + width_px, y)], fill=color, width=w)


def fill_cell(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    cell_px: int,
    color: RGB,
) -> None:
    draw.rectangle(
        [left + 1, top + 1, left + cell_px - 1, top + cell_px - 1],
        fill=color,
    )


# ---------------------------------------------------------------------------
# Square integer board geometry.
# ---------------------------------------------------------------------------
# Cell sizes are chosen so the drawn board keeps a near-constant ~392px
# footprint for every supported size.  A constant footprint is what makes the
# board land in the same place on the canvas whatever the matrix size, so a
# reader (and a frame-differencing verifier) sees stable alignment.

# Every cell size is even and every board is centred on an even offset, so each
# cell centre falls on a whole pixel.  A verifier that crops a fixed window
# around a cell centre then gets the same patch size for every cell; with a
# half-pixel centre the rounding alternates and two cells of the same board
# yield differently shaped patches, which silently breaks template matching.
INTEGER_GRID_CELL_PX: Mapping[int, int] = {3: 132, 4: 98, 5: 78, 6: 66}
INTEGER_GRID_TOP_PX = 84


@dataclass(frozen=True)
class IntegerGridGeometry:
    """Placement of one square integer board on the 512x512 canvas."""

    size: int
    left: int
    top: int
    cell_px: int

    @property
    def board_px(self) -> int:
        return self.cell_px * self.size

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.left + self.board_px, self.top + self.board_px)

    def cell_box(self, row: int, col: int) -> Tuple[int, int, int, int]:
        x0 = self.left + col * self.cell_px
        y0 = self.top + row * self.cell_px
        return (x0, y0, x0 + self.cell_px, y0 + self.cell_px)

    def cell_center(self, row: int, col: int) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.cell_box(row, col)
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def integer_grid_geometry(
    size: int,
    *,
    top: int = INTEGER_GRID_TOP_PX,
    canvas: int = CANVAS_PX,
    cell_px: Optional[int] = None,
) -> IntegerGridGeometry:
    """Centre a square integer board horizontally on the canvas."""

    board_size = int(size)
    if board_size < 2:
        raise ValueError("integer grid geometry requires at least a 2x2 board")
    resolved_cell = int(cell_px) if cell_px is not None else INTEGER_GRID_CELL_PX.get(board_size)
    if resolved_cell is None:
        raise ValueError(f"no default integer-grid cell size for a {board_size}x{board_size} board")
    if resolved_cell % 2:
        raise ValueError("integer grid cell size must be even so cell centres land on whole pixels")
    board_px = resolved_cell * board_size
    if board_px > canvas:
        raise ValueError("integer grid board does not fit the canvas")
    left = (canvas - board_px) // 2
    return IntegerGridGeometry(
        size=board_size,
        left=left - (left % 2),
        top=int(top) - (int(top) % 2),
        cell_px=resolved_cell,
    )


def integer_grid_font_size(cell_px: int) -> int:
    """Digit size that stays proportional to the cell without crowding it."""

    return min(52, max(12, int(round(int(cell_px) * 0.42))))


def integer_grid_token_window(cell_px: int) -> Tuple[int, int]:
    """Sampling window a verifier uses to read one cell's printed digits.

    The window is centred on the cell and stays well inside the cell walls, so
    a neighbouring cell's glyphs can never intrude on a template match.  Both
    dimensions are even: together with whole-pixel cell centres that makes the
    cropped patch exactly the same size for every cell of every board.
    """

    cell = int(cell_px)
    width = 2 * int(round(cell * 0.62 / 2))
    height = 2 * int(round(cell * 0.42 / 2))
    return (width, height)


def draw_integer_grid(
    draw: ImageDraw.ImageDraw,
    *,
    matrix: Sequence[Sequence[object]],
    geometry: IntegerGridGeometry,
    fnt: ImageFont.ImageFont,
    cell_fill: RGB = GRID_CELL_FILL,
    ink: RGB = GRID_INK,
    minor_color: RGB = GRID_MINOR_LINE,
    outer_color: RGB = GRID_OUTER_LINE,
    cell_fills: Optional[Mapping[Tuple[int, int], RGB]] = None,
    cell_inks: Optional[Mapping[Tuple[int, int], RGB]] = None,
) -> None:
    """Render a square board of printed values in the shared grid language.

    Every cell is a flat filled square separated by the shared grid rule; there
    is no per-cell frame, badge or pill behind a value.  ``None`` entries render
    as an empty cell.  ``cell_fills``/``cell_inks`` exist for tasks that give a
    cell bearing emphasis, and are keyed by ``(row, col)``.
    """

    size = geometry.size
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("integer grid matrix does not match its geometry size")
    for row in range(size):
        for col in range(size):
            x0, y0, _x1, _y1 = geometry.cell_box(row, col)
            fill_cell(
                draw,
                left=x0,
                top=y0,
                cell_px=geometry.cell_px,
                color=(cell_fills or {}).get((row, col), cell_fill),
            )
    draw_grid_lines(
        draw,
        left=geometry.left,
        top=geometry.top,
        rows=size,
        cols=size,
        cell_px=geometry.cell_px,
        minor_color=minor_color,
        major_color=outer_color,
    )
    for row in range(size):
        for col in range(size):
            value = matrix[row][col]
            if value is None:
                continue
            draw_text_centered(
                draw,
                text=str(value),
                box=geometry.cell_box(row, col),
                fnt=fnt,
                fill=(cell_inks or {}).get((row, col), ink),
            )


__all__ = [
    "CANVAS_PX",
    "GRID_ACCENT",
    "GRID_ACCENT_FILL",
    "GRID_CELL_FILL",
    "GRID_INK",
    "GRID_MINOR_LINE",
    "GRID_MINOR_WIDTH",
    "GRID_OUTER_LINE",
    "GRID_OUTER_WIDTH",
    "GRID_PAGE_BG",
    "INTEGER_GRID_CELL_PX",
    "INTEGER_GRID_TOP_PX",
    "IntegerGridGeometry",
    "draw_grid_lines",
    "draw_integer_grid",
    "draw_text_centered",
    "fill_cell",
    "integer_grid_font_size",
    "integer_grid_geometry",
    "integer_grid_token_window",
    "load_font",
    "matrix_to_json",
]

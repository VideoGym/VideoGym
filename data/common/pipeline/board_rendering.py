"""Shared rendering primitives for abstract board-game tasks."""

from __future__ import annotations

import math
from typing import Tuple

from PIL import ImageDraw, ImageFont

from data.common.assets import render_font

PIECE_WHITE = (245, 240, 220)
PIECE_WHITE_OUT = (160, 150, 110)
PIECE_BLACK = (30, 32, 40)
PIECE_BLACK_OUT = (5, 5, 10)
PIECE_RED = (210, 60, 50)
PIECE_RED_OUT = (140, 30, 20)
PIECE_BLUE = (60, 100, 200)
PIECE_BLUE_OUT = (30, 60, 140)


def load_font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)


def draw_go_stone(
    draw: ImageDraw.ImageDraw,
    *,
    centre: Tuple[float, float],
    radius: int,
    black: bool,
) -> None:
    cx, cy = centre
    body = PIECE_BLACK if black else PIECE_WHITE
    edge = PIECE_BLACK_OUT if black else PIECE_WHITE_OUT
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=body,
        outline=edge,
        width=2,
    )
    hi_r = int(radius * 0.35)
    hi_off = int(radius * 0.4)
    hi_colour = (90, 90, 100) if black else (255, 255, 250)
    draw.ellipse(
        [cx - hi_off, cy - hi_off, cx - hi_off + hi_r, cy - hi_off + hi_r],
        fill=hi_colour,
    )


def draw_hex_cell(
    draw: ImageDraw.ImageDraw,
    *,
    centre: Tuple[float, float],
    radius: int,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    width: int = 2,
) -> None:
    """Draw a pointy-top hex cell."""
    cx, cy = centre
    pts = []
    for i in range(6):
        ang = math.pi / 3 * i + math.pi / 6
        pts.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
    draw.polygon(pts, fill=fill, outline=outline)
    pts2 = pts + [pts[0]]
    for i in range(6):
        draw.line([pts2[i], pts2[i + 1]], fill=outline, width=width)


__all__ = [
    "PIECE_WHITE",
    "PIECE_WHITE_OUT",
    "PIECE_BLACK",
    "PIECE_BLACK_OUT",
    "PIECE_RED",
    "PIECE_RED_OUT",
    "PIECE_BLUE",
    "PIECE_BLUE_OUT",
    "load_font",
    "draw_go_stone",
    "draw_hex_cell",
]

"""Shared Spatial Navigation path rendering primitives.

Paths are semantic polylines through exact cell centres.  Square-grid routes
in this family are always drawn as square-cornered polylines: every segment is
a filled axis-aligned rectangle and every interior turn is capped with a full
square, so corners carry no notch and no rounded bulge.
"""

from __future__ import annotations

import math
from typing import (
    Sequence,
    Tuple,
)

from PIL import ImageDraw

Point = Tuple[float, float]


def _draw_segment(
    draw: ImageDraw.ImageDraw,
    p1: Point,
    p2: Point,
    color: Tuple[int, int, int],
    width: int,
) -> None:
    half = width / 2.0
    x1, y1 = p1
    x2, y2 = p2
    if math.isclose(x1, x2, abs_tol=1e-6) and math.isclose(y1, y2, abs_tol=1e-6):
        return
    if math.isclose(x1, x2, abs_tol=1e-6):
        draw.rectangle((x1 - half, min(y1, y2), x1 + half, max(y1, y2)), fill=color)
        return
    if math.isclose(y1, y2, abs_tol=1e-6):
        draw.rectangle((min(x1, x2), y1 - half, max(x1, x2), y1 + half), fill=color)
        return
    # Rare non-axis-aligned hop (e.g. across a diagonal shortcut): draw an
    # oriented filled quad so the stroke still has straight edges.
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    px, py = -dy / length * half, dx / length * half
    draw.polygon(
        [(x1 + px, y1 + py), (x1 - px, y1 - py), (x2 - px, y2 - py), (x2 + px, y2 + py)],
        fill=color,
    )


def _square_cap(
    draw: ImageDraw.ImageDraw,
    point: Point,
    color: Tuple[int, int, int],
    width: int,
) -> None:
    half = width / 2.0
    x, y = point
    draw.rectangle((x - half, y - half, x + half, y + half), fill=color)


def draw_spatial_navigation_path(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Point],
    *,
    color: Tuple[int, int, int],
    width: int,
    round_caps: bool = True,
) -> None:
    """Draw a square-cornered polyline through the given points.

    ``round_caps`` keeps its historical name but now paints square end caps,
    so a stroke never ends in a half-circle bulge; interior turns are always
    capped with full squares regardless of the flag.
    """

    deduped: list[Point] = []
    for point in points:
        normalized = (float(point[0]), float(point[1]))
        if not deduped or normalized != deduped[-1]:
            deduped.append(normalized)
    if len(deduped) < 2:
        return

    line_width = max(1, int(width))
    for start, end in zip(deduped, deduped[1:]):
        _draw_segment(draw, start, end, color, line_width)
    for joint in deduped[1:-1]:
        _square_cap(draw, joint, color, line_width)
    if round_caps:
        _square_cap(draw, deduped[0], color, line_width)
        _square_cap(draw, deduped[-1], color, line_width)


__all__ = [ "draw_spatial_navigation_path"]

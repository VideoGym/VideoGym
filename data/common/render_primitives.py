"""Shared supersampled drawing primitives.

This module is the single implementation of the library-wide line-quality
floor: every line, polyline, arc, circle, and polygon outline is drawn onto a
canvas allocated at ``factor`` times the target resolution — through a
:class:`ScaledDraw` proxy that keeps callers in unscaled target coordinates —
and then LANCZOS-downsampled back to the target size.  Families must consume
these primitives instead of maintaining parallel supersample implementations.

It is also the single implementation of the library's joint convention:
:func:`stroke_polyline`, :func:`stroke_open_path` and :func:`stroke_closed_ring`
miter every multi-vertex stroke, and work on either a :class:`ScaledDraw` (still
images, unscaled target coordinates) or a raw PIL ``ImageDraw`` already in
supersampled coordinates (video frames, reveal masks, task-local card
renderers).

The supersample factor is always an explicit argument, defaulting to
:data:`DOCTRINE_SUPERSAMPLE` (4x).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Default supersample factor for new work (the geometry-family quality floor).
DOCTRINE_SUPERSAMPLE: int = 4


def scale_xy(xy: Any, factor: int) -> Any:
    """Scale a coordinate sequence (list of point tuples or a flat bbox)."""
    if not xy:
        return xy
    if isinstance(xy[0], (tuple, list)):
        return [(x * factor, y * factor) for x, y in xy]
    return tuple(value * factor for value in xy)


def scale_stroke_width(width: Any, factor: int) -> int:
    """Scale a stroke width, keeping at least one supersampled pixel."""
    return max(1, round(width * factor))


def as_point_pairs(xy, factor: float = 1.0) -> list:
    """Normalize a PIL coordinate sequence to a list of ``(x, y)`` float pairs.

    Accepts either a sequence of points or PIL's flat ``[x0, y0, x1, y1, …]``
    form.  ``factor`` is for callers that stroke onto a raw ``ImageDraw`` already
    working in supersampled coordinates: they hand over unscaled vertices and
    scale them once here instead of at every call site.
    """

    seq = list(xy)
    if seq and isinstance(seq[0], (int, float)):
        seq = [(seq[index], seq[index + 1]) for index in range(0, len(seq) - 1, 2)]
    return [(float(point[0]) * factor, float(point[1]) * factor) for point in seq]


def drop_repeated_points(points: Sequence[Sequence[float]]) -> list:
    """Collapse consecutive duplicate vertices.

    Callers legitimately produce them — an animation that reveals a stroke by
    walking a moving tip repeats the final vertex once the tip arrives, and a
    closed ring may be handed over with its first point already repeated at the
    end.  A zero-length segment has no direction, so leaving the duplicates in
    would make :func:`miter_outline` degenerate and silently fall back to PIL's
    blunt round joint.
    """

    deduped: list = []
    for point in points:
        pair = (float(point[0]), float(point[1]))
        if not deduped or pair != deduped[-1]:
            deduped.append(pair)
    return deduped


def miter_outline(
    points: Sequence[Sequence[float]],
    width: float,
    *,
    closed: bool = False,
    miter_limit: float = 4.0,
):
    """Offset an open path to its stroke outline with mitered joins.

    Returns a single closed polygon that can be filled to render the stroke, or
    ``None`` when the path degenerates (repeated points, zero width, or an inner
    join beyond a finite segment) and the caller should stroke the segments.
    """

    pts = as_point_pairs(points)
    if width <= 0 or len(pts) < 2 or closed:
        return None
    half = float(width) / 2.0

    def unit(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return None
        return dx / length, dy / length

    directions = []
    for index in range(len(pts) - 1):
        step = unit(pts[index], pts[index + 1])
        if step is None:
            return None
        directions.append(step)

    def offsets(sign: float):
        side = []
        for index, point in enumerate(pts):
            before = directions[index - 1] if index > 0 else None
            after = directions[index] if index < len(directions) else None
            if before is None or after is None:
                step = after or before
                normal = (-step[1] * sign, step[0] * sign)
                side.append((point[0] + normal[0] * half, point[1] + normal[1] * half))
                continue
            n1 = (-before[1] * sign, before[0] * sign)
            n2 = (-after[1] * sign, after[0] * sign)
            mx, my = n1[0] + n2[0], n1[1] + n2[1]
            length = math.hypot(mx, my)
            if length < 1e-9:
                side.append((point[0] + n1[0] * half, point[1] + n1[1] * half))
                continue
            mx, my = mx / length, my / length
            cosine = max(1e-6, mx * n1[0] + my * n1[1])
            reach = half / cosine
            turn = before[0] * after[1] - before[1] * after[0]
            inner_overrun = turn * sign > 0 and math.sqrt(max(0.0, reach * reach - half * half)) > min(
                math.dist(pts[index - 1], point), math.dist(point, pts[index + 1])
            )
            if inner_overrun:
                # A nearly reversing path can put the inner intersection past
                # an adjoining segment's endpoint. The offset outline then
                # self-intersects: filling it draws spikes or cancels real ink.
                # Stroke the finite segments instead, as for an edge-on face.
                return None
            elif turn * sign < 0 and reach > half * miter_limit:
                # Bevel the outside corner without tapering adjoining edges.
                # Keep inner intersections that lie on both finite segments.
                side.extend((
                    (point[0] + n1[0] * half, point[1] + n1[1] * half),
                    (point[0] + n2[0] * half, point[1] + n2[1] * half),
                ))
            else:
                side.append((point[0] + mx * reach, point[1] + my * reach))
        return side

    left, right = offsets(1.0), offsets(-1.0)
    if left is None or right is None:
        return None
    return left + list(reversed(right))


# ---------------------------------------------------------------------------
# Joint-correct strokes
# ---------------------------------------------------------------------------

# PIL only rounds the *interior* joints of a polyline, and its round joint is a
# disc of the stroke width, which cannot fill an acute miter.  A closed outline
# stroked as ``line(ring + [ring[0]], joint="curve")`` therefore blunts every
# sharp vertex into a flat step and leaves the closing vertex unjoined, and an
# open multi-vertex path gets rounded, blunted corners.  The three functions
# below are the library's only stroke convention for multi-vertex geometry.

# They take the drawing surface as their first argument and work on either of
# the two surfaces families render onto:

# * a :class:`ScaledDraw`, which takes unscaled target coordinates and scales
#   them onto the supersampled canvas, and
# * a raw PIL ``ImageDraw`` already working in supersampled coordinates (video
#   frames, reveal masks, and task-local card renderers).

# Only ``line``, ``polygon`` and ``ellipse`` are used, so the caller's
# coordinates and stroke width are simply interpreted in whichever space its
# surface expects.  Keeping one implementation for both is what stops the
# still-image and video paths from drifting apart.


def stroke_open_path(draw, points: Sequence[Sequence[float]], *, fill, width) -> None:
    """Stroke an open polyline with mitered joins and butt caps."""

    pts = drop_repeated_points(as_point_pairs(points))
    if len(pts) < 2:
        return
    if len(pts) == 2:
        draw.line(pts, fill=fill, width=width)
        return
    outline = miter_outline(pts, width, closed=False)
    if outline is None:
        draw.line(pts, fill=fill, width=width, joint="curve")
        return
    draw.polygon(outline, fill=fill)


def stroke_closed_ring(draw, ring: Sequence[Sequence[float]], *, fill, width) -> None:
    """Stroke a closed ring, centred on the ring, mitering *every* vertex.

    ``ring`` is the vertex ring *without* a repeated first point.

    PIL's ``polygon(outline=…, width=…)`` is deliberately not used: it masks the
    stroke to the polygon interior, so the band sits fully *inside* the ring
    instead of straddling it.  That shifts an outline inward by half the stroke
    width, doubles every shared interior edge of a mesh and halves silhouette
    edges — a layout change, and in tasks whose answer depends on outline
    geometry, a coarse shortcut.

    The ring is instead stroked as an open mitered path from the closure vertex
    all the way round back to it, plus a short two-segment patch centred on that
    closure vertex.  The first pass miters every other vertex but meets itself
    with two butt caps at the closure; the patch miters the closure and ends at
    the two adjacent edge midpoints, where the first pass already lays down the
    full-width band, so the union is exactly the centred, fully mitered closed
    band.  Cutting the ring into a *single* open path whose two ends coincide
    looks equivalent but is not: the offset outline then carries a zero-width
    bridge at the cut, and PIL's scan-line fill can drop one whole scan-line
    across the band there — a hairline crack that both lightens the outline and
    lets a flood fill escape the interior.
    """

    pts = drop_repeated_points(as_point_pairs(ring))
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        stroke_open_path(draw, pts, fill=fill, width=width)
        return
    stroke_open_path(draw, pts + [pts[0]], fill=fill, width=width)
    previous, first, following = pts[-1], pts[0], pts[1]
    stroke_open_path(
        draw,
        [
            ((previous[0] + first[0]) / 2.0, (previous[1] + first[1]) / 2.0),
            first,
            ((first[0] + following[0]) / 2.0, (first[1] + following[1]) / 2.0),
        ],
        fill=fill,
        width=width,
    )


def stroke_polyline(
    draw,
    xy,
    *,
    fill,
    width,
    join_start: bool = False,
    join_end: bool = False,
) -> None:
    """Stroke a polyline with mitered joins, closing it when it is a ring.

    A polyline whose first and last vertex coincide is treated as a closed ring
    and mitered at every vertex; an open polyline keeps butt caps at its two
    endpoints and gets a sealing disc only where the caller declares that the
    endpoint is shared with another stroke via ``join_start`` / ``join_end``.
    That disc covers both strokes' butt caps and the wedge between them, which
    is what a construction figure needs where a bisector starts on a triangle
    edge; isolated endpoints such as ray tips stay cap-free.
    """

    pts = drop_repeated_points(as_point_pairs(xy))
    if len(pts) < 2:
        return
    if len(pts) >= 4 and pts[0] == pts[-1]:
        stroke_closed_ring(draw, pts[:-1], fill=fill, width=width)
        return
    stroke_open_path(draw, pts, fill=fill, width=width)
    if not (join_start or join_end):
        return
    radius = max(1, width // 2)
    for enabled, (cx, cy) in ((join_start, pts[0]), (join_end, pts[-1])):
        if enabled:
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)


def arrow_stroke_points(start, end, *, head_length):
    """The geometry family's vector-arrow head and shaft, in target coordinates.

    Extracted from TranslationCompositionGenerator so other diagram renderers
    reuse its 25-degree open head and sealed shaft joint.
    """
    ax, ay = map(float, start)
    bx, by = map(float, end)
    angle = math.atan2(by - ay, bx - ax)
    wings = [
        (bx + head_length * math.cos(angle + math.pi + sign * math.radians(25)),
         by + head_length * math.sin(angle + math.pi + sign * math.radians(25)))
        for sign in (1, -1)
    ]
    return [wings[0], (bx, by), wings[1]], [(ax, ay), (bx, by)]


def draw_arrow(draw, start, end, *, fill, width=2, head_length=10):
    """Stroke the existing vector-arrow convention through shared miter joins."""
    length = math.dist(start, end)
    if length <= 0:
        return
    head, shaft = arrow_stroke_points(start, end, head_length=min(head_length, length * .45))
    stroke_polyline(draw, head, fill=fill, width=width)
    stroke_polyline(draw, shaft, fill=fill, width=width, join_end=True)


class ScaledDraw:
    """Transparent wrapper that scales all coordinates for supersampled rendering.

    Every proxied PIL ImageDraw method multiplies positional coordinates and
    stroke widths by ``factor``, so callers work in the original target
    coordinate space while drawing onto the higher-resolution canvas.  Fonts
    are passed through unchanged: text callers load their font at
    ``size * factor`` themselves (and may use :attr:`raw` with :attr:`factor`
    for manual glyph-metric work in supersampled space).
    """

    __slots__ = ("raw", "factor")

    def __init__(self, draw: ImageDraw.ImageDraw, factor: int) -> None:
        self.raw = draw
        self.factor = factor

    # -- coordinate helpers --------------------------------------------------

    def _sxy(self, xy):
        return scale_xy(xy, self.factor)

    def _sw(self, w):
        return scale_stroke_width(w, self.factor)

    # -- PIL ImageDraw proxies -----------------------------------------------

    def line(self, xy, fill=None, width=0, joint=None):
        self.raw.line(self._sxy(xy), fill=fill, width=self._sw(width) if width else 0, joint=joint)

    def ellipse(self, xy, fill=None, outline=None, width=1):
        self.raw.ellipse(self._sxy(xy), fill=fill, outline=outline, width=self._sw(width))

    def arc(self, xy, start, end, fill=None, width=1):
        self.raw.arc(self._sxy(xy), start, end, fill=fill, width=self._sw(width))

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.raw.rectangle(self._sxy(xy), fill=fill, outline=outline, width=self._sw(width))

    def rounded_rectangle(self, xy, radius=0, fill=None, outline=None, width=1, corners=None):
        self.raw.rounded_rectangle(
            self._sxy(xy),
            radius=radius * self.factor,
            fill=fill,
            outline=outline,
            width=self._sw(width),
            corners=corners,
        )

    def polygon(self, xy, fill=None, outline=None, width=0):
        self.raw.polygon(self._sxy(xy), fill=fill, outline=outline,
                         width=self._sw(width) if width else 0)

    def text(self, xy, text, fill=None, font=None, **kwargs):
        factor = self.factor
        self.raw.text((xy[0] * factor, xy[1] * factor), text, fill=fill, font=font, **kwargs)

    def textbbox(self, xy, text, font=None, **kwargs):
        """Text bounds in unscaled coordinates for supersampled fonts.

        Callers load fonts at ``size * factor``; dividing the raw bounds back
        by ``factor`` lets centring math stay in target coordinates.
        """

        factor = self.factor
        bbox = self.raw.textbbox((xy[0] * factor, xy[1] * factor), text, font=font, **kwargs)
        return tuple(value / factor for value in bbox)

    def point(self, xy, fill=None):
        self.raw.point(self._sxy(xy), fill=fill)

    # -- joint-correct strokes ------------------------------------------------

    # Thin bindings onto the module-level stroke convention, so a ScaledDraw
    # caller working in unscaled target coordinates and a raw-ImageDraw caller
    # working in supersampled coordinates share one implementation.

    def closed_outline(self, xy, fill=None, width=1):
        """Stroke a closed ring, centred on the ring, with mitered corners.

        See :func:`stroke_closed_ring`; ``xy`` is the vertex ring *without* a
        repeated first point.
        """

        stroke_closed_ring(self, xy, fill=fill, width=width)

    def open_path(self, xy, fill=None, width=1):
        """Stroke an open path with mitered corners and butt ends.

        See :func:`stroke_open_path`.
        """

        stroke_open_path(self, xy, fill=fill, width=width)

    def chord(self, xy, start, end, fill=None, outline=None, width=1):
        self.raw.chord(self._sxy(xy), start, end, fill=fill, outline=outline, width=self._sw(width))

    def pieslice(self, xy, start, end, fill=None, outline=None, width=1):
        self.raw.pieslice(self._sxy(xy), start, end, fill=fill, outline=outline, width=self._sw(width))


def new_scaled_canvas(
    width: int,
    height: int,
    factor: int,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[Image.Image, ScaledDraw]:
    """Allocate a ``factor``-times supersampled RGB canvas with its ScaledDraw."""
    image = Image.new("RGB", (width * factor, height * factor), background)
    return image, ScaledDraw(ImageDraw.Draw(image), factor)


def finish_scaled_canvas(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """LANCZOS-downsample a supersampled canvas back to its target size."""
    if image.size == tuple(target_size):
        return image
    return image.resize(tuple(target_size), Image.Resampling.LANCZOS)


def scaled_polyline_points(points: Sequence[Sequence[float]], factor: int) -> np.ndarray:
    """Round unscaled polyline vertices to integer supersampled pixels."""
    return np.array(
        [
            [round(float(point[0]) * factor), round(float(point[1]) * factor)]
            for point in points
        ],
        dtype=np.int32,
    )


def draw_smooth_polyline(
    canvas: Image.Image,
    points: Sequence[Sequence[float]],
    *,
    color: Optional[Tuple[int, int, int]] = None,
    width: float = 1,
    factor: int = 1,
) -> None:
    """Draw an anti-aliased polyline in place on a supersampled canvas.

    Vertices are given in unscaled target coordinates; the stroke is rendered
    with OpenCV's ``LINE_AA`` at the supersampled resolution so dense
    parametric curves stay smooth after the LANCZOS downsample.
    """
    scaled_points = scaled_polyline_points(points, factor)
    if len(scaled_points) < 2:
        return
    array = np.array(canvas)
    cv2.polylines(
        array,
        [scaled_points],
        isClosed=False,
        color=tuple(color or (32, 32, 32)),
        thickness=scale_stroke_width(width, factor),
        lineType=cv2.LINE_AA,
    )
    canvas.paste(Image.fromarray(array))


__all__ = [
    "DOCTRINE_SUPERSAMPLE",
    "arrow_stroke_points",
    "draw_arrow",
    "ScaledDraw",
    "as_point_pairs",
    "draw_smooth_polyline",
    "drop_repeated_points",
    "finish_scaled_canvas",
    "miter_outline",
    "new_scaled_canvas",
    "scale_stroke_width",
    "scale_xy",
    "scaled_polyline_points",
    "stroke_closed_ring",
    "stroke_open_path",
    "stroke_polyline",
]

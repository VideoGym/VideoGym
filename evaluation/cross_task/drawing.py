"""Thin scene adapters over the established shared rendering components.

The task families own layout and semantics. Antialiasing, joins, vector arrows,
grids, stones and voxel meshes remain the existing library's responsibility.
"""

import json
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from data.common.assets import render_font
from data.common.pipeline.board_rendering import draw_go_stone
from data.common.pipeline.structured_grid import draw_grid_lines
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    draw_arrow,
    finish_scaled_canvas,
    new_scaled_canvas,
    stroke_closed_ring,
    stroke_polyline,
)

font = lru_cache(64)(render_font)
FACTOR = DOCTRINE_SUPERSAMPLE
INK = "#243247"
BLUE = "#347bb0"
TEAL = "#168679"
GOLD = "#e7ae42"
RED = "#c95764"
ORANGE = "#ed7339"
PALETTE = [TEAL, BLUE, RED, "#9874c0", GOLD, "#687d8f"]


class Canvas:
    def __init__(self):
        self.surface, self.d = new_scaled_canvas(512, 512, FACTOR)

    @property
    def image(self):
        return finish_scaled_canvas(self.surface, (512, 512))

    def text(self, xy, value, size=16, fill=INK, anchor="mm"):
        self.d.text(
            tuple(xy),
            str(value),
            font=font(round(size * FACTOR)),
            fill=fill,
            anchor=anchor,
        )

    def line(self, points, fill=INK, width=2):
        stroke_polyline(self.d, points, fill=fill, width=width)

    def polygon(self, points, fill=None, outline=None, width=1):
        if fill is not None:
            self.d.polygon([tuple(p) for p in points], fill=fill)
        if outline is not None:
            stroke_closed_ring(self.d, points, fill=outline, width=width)

    def dot(self, xy, radius=5, fill=BLUE, label=None):
        x, y = xy
        self.d.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline="white",
            width=1.5,
        )
        if label is not None:
            self.text((x, y - 14), label, 12)

    def arrow(
        self, a, b, fill=INK, width=2, inset=0, *, start_inset=None, end_inset=None
    ):
        a, b = np.asarray(a, float), np.asarray(b, float)
        length = float(np.linalg.norm(b - a))
        left = inset if start_inset is None else start_inset
        right = inset if end_inset is None else end_inset
        if length <= left + right:
            return
        direction = (b - a) / length
        draw_arrow(
            self.d, a + left * direction, b - right * direction, fill=fill, width=width
        )

    def panel(self, box, fill="#f7f9fb"):
        self.d.rounded_rectangle(
            tuple(box), radius=8, fill=fill, outline="#d6e0e7", width=1
        )

    def grid(self, rows, cols, x=56, y=100, size=44):
        self.d.rectangle((x, y, x + cols * size, y + rows * size), fill="#f5f8fb")
        draw_grid_lines(
            self.d,
            left=x,
            top=y,
            rows=rows,
            cols=cols,
            cell_px=size,
            minor_color="#c2cfdb",
            minor_width=1,
            major_color="#aebfcd",
            major_width=1,
        )
        return [
            (x + (col + 0.5) * size, y + (row + 0.5) * size)
            for row in range(rows)
            for col in range(cols)
        ]


@dataclass
class Slot:
    kind: str
    shape: object
    options: tuple = (1,)
    colors: tuple = ()
    size: int = 20
    guide: bool = True

    def _paint(self, draw, value, factor, stroke_width=3.5, mark_style="badge"):
        color = self.colors[self.options.index(value)] if self.colors else ORANGE
        if self.kind == "line":
            stroke_polyline(draw, self.shape, fill=color, width=stroke_width)
        elif self.kind == "arc":
            bounds, start, end = self.shape
            draw.arc(bounds, start, end, fill=color, width=stroke_width)
        elif self.kind == "ring":
            draw.ellipse(tuple(self.shape), outline=color, width=stroke_width)
        elif self.kind in ("cell", "board_cell"):
            x0, y0, x1, y1 = self.shape
            draw.rounded_rectangle(
                (x0 + 3, y0 + 3, x1 - 3, y1 - 3),
                radius=2,
                fill=TEAL if self.kind == "board_cell" else color,
            )
            if self.kind == "board_cell" and value:
                draw_go_stone(
                    draw,
                    centre=((x0 + x1) / 2, (y0 + y1) / 2),
                    radius=round((x1 - x0) * 0.33),
                    black=value == 1,
                )
        else:
            x, y = self.shape
            if mark_style == "badge":
                draw.rounded_rectangle(
                    (x - 16, y - 15, x + 16, y + 15), radius=4, fill=color
                )
            draw.text(
                (x, y),
                str(value),
                font=font(round(self.size * factor)),
                fill="white" if mark_style == "badge" else INK,
                anchor="mm",
            )

    def paint(self, image, value, *, stroke_width=3.5, mark_style="badge"):
        if value == -1:
            return
        factor = image.width // 512
        if factor > 1:
            self._paint(
                ScaledDraw(ImageDraw.Draw(image), factor),
                value,
                factor,
                stroke_width,
                mark_style,
            )
        else:
            color = self.colors[self.options.index(value)] if self.colors else ORANGE
            left, top, layer = _observation_layer(
                self.kind,
                json.dumps(self.shape, default=lambda v: v.tolist()),
                value,
                color,
                self.size,
                stroke_width,
                mark_style,
            )
            image.paste(layer, (left, top), layer)


@lru_cache(256)
def _observation_layer(
    kind, encoded_shape, value, color, size, stroke_width, mark_style
):
    """Small cached AA patches used by the pixel decoder, never by media export."""
    shape = json.loads(encoded_shape)
    if kind == "number":
        x, y = shape
        bounds = (x - 20, y - 18, x + 20, y + 18)
    elif kind in ("ring", "cell", "board_cell"):
        bounds = shape
    elif kind == "arc":
        bounds = shape[0]
    else:
        bounds = (
            min(p[0] for p in shape),
            min(p[1] for p in shape),
            max(p[0] for p in shape),
            max(p[1] for p in shape),
        )
    left, top = math.floor(bounds[0]) - 4, math.floor(bounds[1]) - 4
    width, height = math.ceil(bounds[2]) - left + 5, math.ceil(bounds[3]) - top + 5
    if kind == "number":
        local = [shape[0] - left, shape[1] - top]
    elif kind in ("ring", "cell", "board_cell"):
        local = [shape[0] - left, shape[1] - top, shape[2] - left, shape[3] - top]
    elif kind == "arc":
        b = shape[0]
        local = [[b[0] - left, b[1] - top, b[2] - left, b[3] - top], shape[1], shape[2]]
    else:
        local = [[x - left, y - top] for x, y in shape]
    layer = Image.new("RGBA", (width * FACTOR, height * FACTOR), (0, 0, 0, 0))
    Slot(kind, local, (value,), (color,), size)._paint(
        ScaledDraw(ImageDraw.Draw(layer), FACTOR),
        value,
        FACTOR,
        stroke_width,
        mark_style,
    )
    return left, top, finish_scaled_canvas(layer, (width, height))


def cell_slots(centres, size, options=(1,), colors=()):
    return [
        Slot(
            "cell",
            (x - size / 2, y - size / 2, x + size / 2, y + size / 2),
            tuple(options),
            tuple(colors),
        )
        for x, y in centres
    ]


def answer_vector(count, selected, value=1):
    selected = set(selected)
    return tuple(value if i in selected else -1 for i in range(count))

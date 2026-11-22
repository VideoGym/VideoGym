from __future__ import annotations

import math
from typing import Tuple

from PIL import ImageDraw


RGB = Tuple[int, int, int]


def draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    center: Tuple[float, float],
    size: float,
    *,
    fill: RGB,
    outline: RGB = (30, 30, 30),
    width: int = 3,
) -> None:
    cx, cy = center
    half = size / 2.0
    shape = shape.lower()
    if shape == "circle":
        draw.ellipse((cx - half, cy - half, cx + half, cy + half), fill=fill, outline=outline, width=width)
        return
    if shape == "square":
        draw.rectangle((cx - half, cy - half, cx + half, cy + half), fill=fill, outline=outline, width=width)
        return
    if shape == "triangle":
        points = [(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)]
    elif shape == "diamond":
        points = [(cx, cy - half), (cx - half, cy), (cx, cy + half), (cx + half, cy)]
    elif shape == "hexagon":
        points = [
            (cx + half * math.cos(angle), cy + half * math.sin(angle))
            for angle in (math.pi / 6 + math.pi * index / 3 for index in range(6))
        ]
    else:
        outer = half
        inner = half * 0.45
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            radius = outer if index % 2 == 0 else inner
            points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    draw.polygon(points, fill=fill, outline=outline, width=width)



__all__ = [
    "draw_shape",
]

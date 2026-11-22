"""Family-wide supersampled scene canvas helpers.

Every Dynamics renderer draws through the shared :mod:`render_primitives`
ScaledDraw proxy at the doctrine 4x supersample factor and LANCZOS-downsamples
back to the fixed 512-square canvas.
Task renderers keep working in unscaled 512-space coordinates; only canvas
allocation, font sizing, and the final downsample go through these helpers.
"""

from __future__ import annotations

from typing import Tuple

from PIL import Image, ImageFont

from data.common.pipeline.board_motion import CANVAS_SIZE, load_font
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)

SS = DOCTRINE_SUPERSAMPLE


def new_scene(background: Tuple[int, int, int]) -> Tuple[Image.Image, ScaledDraw]:
    """Allocate one supersampled 512-square scene canvas with its ScaledDraw."""

    return new_scaled_canvas(CANVAS_SIZE, CANVAS_SIZE, SS, background=background)


def new_scene_rgba(background: Tuple[int, int, int]) -> Tuple[Image.Image, ScaledDraw]:
    """Scene canvas whose draw supports RGBA fills for opacity-faded elements."""

    from PIL import ImageDraw

    image = Image.new("RGB", (CANVAS_SIZE * SS, CANVAS_SIZE * SS), background)
    return image, ScaledDraw(ImageDraw.Draw(image, "RGBA"), SS)


def finish_scene(image: Image.Image) -> Image.Image:
    """LANCZOS-downsample a supersampled scene back to the 512-square canvas."""

    return finish_scaled_canvas(image, (CANVAS_SIZE, CANVAS_SIZE))


def scene_font(size: int) -> ImageFont.ImageFont:
    """Load a font sized for the supersampled canvas from a 512-space size."""

    return load_font(size * SS)


__all__ = ["SS", "finish_scene", "new_scene", "new_scene_rgba", "scene_font"]

from __future__ import annotations

from typing import Tuple


from PIL import Image, ImageOps


DEFAULT_CANVAS_SIZE: Tuple[int, int] = (512, 512)
DEFAULT_FILL = (255, 255, 255)


def fit_canvas(
    image: Image.Image,
    target_size: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
    fill: Tuple[int, int, int] = DEFAULT_FILL,
) -> Image.Image:
    target_width, target_height = target_size
    if image.size == target_size:
        return image

    fitted = image.convert("RGB")
    if fitted.width > target_width or fitted.height > target_height:
        fitted = ImageOps.contain(fitted, target_size, Image.LANCZOS)

    canvas = Image.new("RGB", target_size, fill)
    offset_x = max(0, (target_width - fitted.width) // 2)
    offset_y = max(0, (target_height - fitted.height) // 2)
    canvas.paste(fitted, (offset_x, offset_y))
    return canvas

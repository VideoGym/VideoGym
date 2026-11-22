from __future__ import annotations

from data.common.assets import render_font

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from data.common.pipeline.media import LETTER_OPTIONS
from data.common.typed_answers import FOUR_CHOICE_CODEC


OPTION_CANVAS_SIZE = (512, 512)


def _load_font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)


def option_label(index: int) -> str:
    return LETTER_OPTIONS[int(index)]


def option_answer_text(index: int) -> str:
    return FOUR_CHOICE_CODEC.format(option_label(index))


def render_labeled_option_image(
    tile: Image.Image,
    *,
    label: str,
    target_size: tuple[int, int] = OPTION_CANVAS_SIZE,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    canvas = Image.new("RGB", target_size, background)
    label_font = _load_font(max(28, target_size[1] // 14))
    label_band = max(72, target_size[1] // 6)
    padding = max(24, target_size[0] // 16)
    available_width = target_size[0] - 2 * padding
    available_height = target_size[1] - label_band - 2 * padding
    fitted = ImageOps.contain(tile.convert("RGB"), (available_width, available_height), Image.Resampling.LANCZOS)
    paste_x = (target_size[0] - fitted.width) // 2
    paste_y = max(padding, (available_height - fitted.height) // 2 + padding)
    canvas.paste(fitted, (paste_x, paste_y))

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (target_size[0] // 2, target_size[1] - label_band // 2),
        str(label).strip().upper(),
        fill=(20, 20, 20),
        font=label_font,
        anchor="mm",
    )
    return canvas


def save_labeled_option_images(
    option_tiles: Sequence[Image.Image],
    output_dir: Path,
    *,
    stem: str,
    labels: Sequence[str] | None = None,
    target_size: tuple[int, int] = OPTION_CANVAS_SIZE,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_labels = list(labels or LETTER_OPTIONS[: len(option_tiles)])
    saved_paths: list[Path] = []
    for index, tile in enumerate(option_tiles):
        label = resolved_labels[index]
        option_path = output_dir / f"{stem}_{label}.png"
        render_labeled_option_image(tile, label=label, target_size=target_size).save(option_path)
        saved_paths.append(option_path)
    return saved_paths

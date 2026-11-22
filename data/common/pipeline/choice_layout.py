"""Shared tile-composition and prompt helpers for four-choice visual pattern tasks."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.common.pipeline.media import LETTER_OPTIONS, load_font
from data.common.render_primitives import (
    ScaledDraw,
    finish_scaled_canvas,
)


CANVAS_SIZE: Tuple[int, int] = (512, 512)
BACKGROUND_COLOR: Tuple[int, int, int] = (250, 250, 248)
PANEL_BORDER_COLOR: Tuple[int, int, int] = (44, 50, 60)
PANEL_FILL_COLOR: Tuple[int, int, int] = (255, 255, 255)
SOURCE_PANEL_TINT: Tuple[int, int, int] = (235, 244, 255)
LABEL_COLOR: Tuple[int, int, int] = (28, 32, 40)
ACCENT_RED: Tuple[int, int, int] = (220, 38, 38)
ACCENT_GREEN: Tuple[int, int, int] = (34, 139, 88)
ACCENT_BLUE: Tuple[int, int, int] = (47, 92, 168)
ACCENT_PURPLE: Tuple[int, int, int] = (149, 88, 192)
ACCENT_ORANGE: Tuple[int, int, int] = (224, 120, 60)
NEUTRAL_DARK: Tuple[int, int, int] = (54, 60, 72)

# Layout constants for the 1+4 tile composition.
SOURCE_HEIGHT = 232
GUTTER = 14
EDGE_PADDING = 16
def compose_tile_panel(
    *,
    draw_source: Callable[[Image.Image, Tuple[int, int, int, int]], None],
    candidate_drawers: Sequence[Callable[[Image.Image, Tuple[int, int, int, int]], None]],
    source_caption: str = "",
    highlight_letter: str | None = None,
    supersample: int = 1,
) -> Image.Image:
    """Compose a 1+4 panel layout into a 512x512 image.

    The source panel sits at top, occupying the full canvas width minus
    edge padding; the four candidate panels (A-D) sit in a 2x2 grid below.
    Each ``drawer`` receives the inner content box (after frame/label) and
    must paint into ``img`` directly.

    If ``highlight_letter`` is provided, the corresponding candidate frame
    is drawn with one thicker green outer border so the solution image visually
    points at the answer; the source panel is unchanged.

    With ``supersample`` greater than one, the working canvas is allocated at
    ``supersample`` times the target resolution and LANCZOS-downsampled at the
    end (the shared line-quality floor).  Drawers still receive unscaled boxes
    and must derive the working factor as ``img.width // CANVAS_SIZE[0]``.
    """
    if len(candidate_drawers) != 4:
        raise ValueError("Pattern MCQ tasks render exactly four candidates.")
    factor = max(1, int(supersample))
    cw, ch = CANVAS_SIZE
    img = Image.new("RGB", (cw * factor, ch * factor), BACKGROUND_COLOR)
    sdraw = ScaledDraw(ImageDraw.Draw(img), factor)

    # Source panel (top).
    src_left = EDGE_PADDING
    src_right = cw - EDGE_PADDING
    src_top = EDGE_PADDING
    src_bottom = src_top + SOURCE_HEIGHT
    src_box = (src_left, src_top, src_right, src_bottom)
    sdraw.rounded_rectangle(src_box, radius=8, fill=SOURCE_PANEL_TINT, outline=PANEL_BORDER_COLOR, width=2)
    # Inner content box for the source.
    inner_src = (src_left + 14, src_top + 12, src_right - 14, src_bottom - 12)
    draw_source(img, inner_src)

    # Candidate panels (2x2).
    cand_top0 = src_bottom + GUTTER
    cand_bottom1 = ch - EDGE_PADDING
    cand_height = (cand_bottom1 - cand_top0 - GUTTER) // 2
    half_w = (cw - 2 * EDGE_PADDING - GUTTER) // 2
    positions = []
    for row in range(2):
        for col in range(2):
            left = EDGE_PADDING + col * (half_w + GUTTER)
            top = cand_top0 + row * (cand_height + GUTTER)
            right = left + half_w
            bottom = top + cand_height
            positions.append((left, top, right, bottom))

    for index, (box, drawer) in enumerate(zip(positions, candidate_drawers)):
        letter = LETTER_OPTIONS[index]
        is_highlight = (highlight_letter is not None and letter == highlight_letter)
        border = ACCENT_GREEN if is_highlight else PANEL_BORDER_COLOR
        bw = 5 if is_highlight else 2
        sdraw.rounded_rectangle(box, radius=8, fill=PANEL_FILL_COLOR, outline=border, width=bw)
        # Letter label on a small chip in the top-left corner for legibility.
        font = load_font(18 * factor)
        chip_x = box[0] + 8
        chip_y = box[1] + 6
        sdraw.rounded_rectangle(
            (chip_x, chip_y, chip_x + 28, chip_y + 22),
            radius=4,
            fill=(28, 32, 40),
            outline=None,
        )
        sdraw.text(
            (chip_x + 14, chip_y + 11),
            letter,
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
        )
        # Inner box leaves room for the letter chip.
        inner = (box[0] + 10, box[1] + 32, box[2] - 10, box[3] - 10)
        drawer(img, inner)

    return finish_scaled_canvas(img, CANVAS_SIZE)


def fit_pad(image: Image.Image, target: Tuple[int, int] = CANVAS_SIZE) -> Image.Image:
    """Pad/crop an image to ``target`` size, centered, white-background."""
    if image.size == target:
        return image.copy()
    canvas = Image.new("RGB", target, BACKGROUND_COLOR)
    src = image.convert("RGB")
    fx = target[0] / src.width
    fy = target[1] / src.height
    f = min(fx, fy)
    new_w = max(1, int(round(src.width * f)))
    new_h = max(1, int(round(src.height * f)))
    if (new_w, new_h) != src.size:
        src = src.resize((new_w, new_h), Image.Resampling.LANCZOS)
    px = (target[0] - new_w) // 2
    py = (target[1] - new_h) // 2
    canvas.paste(src, (px, py))
    return canvas


def build_standard_prompts(
    task_name: str,
    *,
    intro: str,
    rules: str,
    output_format: str,
) -> Dict[str, str]:
    """Build the choice-card prompt bundle with Input Image / Rule / Required Output sections."""
    answer_format = FOUR_CHOICE_CODEC.instruction
    question = f"Input Image: {intro}\n\nRule: {rules}\n\nRequired Output: {output_format} {answer_format}"
    ti2v_prompt = (
        f"Input Image: {intro}\n\nRule: {rules}\n\nRequired Output: Start from the unchanged input, then "
        "change the correct candidate's complete outer frame smoothly and simultaneously to green. Keep every panel, "
        "label, and drawing fixed, and hold the final state."
    )
    return {
        "question": question,
        "ti2v_prompt": ti2v_prompt,
    }


def assemble_choice_order(
    rng: random.Random,
    correct_drawer: Callable[[Image.Image, Tuple[int, int, int, int]], None],
    distractor_drawers: Sequence[Callable[[Image.Image, Tuple[int, int, int, int]], None]],
) -> tuple[List[Callable[[Image.Image, Tuple[int, int, int, int]], None]], str, List[int]]:
    """Shuffle (correct + distractors) and preserve the semantic source indices."""
    if len(distractor_drawers) != 3:
        raise ValueError("Pattern MCQ tasks supply exactly three distractors.")
    drawers: List[Callable[[Image.Image, Tuple[int, int, int, int]], None]] = [
        correct_drawer,
        *distractor_drawers,
    ]
    indices = list(range(4))
    rng.shuffle(indices)
    correct_index = indices.index(0)
    correct_letter = LETTER_OPTIONS[correct_index]
    return [drawers[i] for i in indices], correct_letter, indices


__all__ = [
    "CANVAS_SIZE",
    "BACKGROUND_COLOR",
    "PANEL_BORDER_COLOR",
    "PANEL_FILL_COLOR",
    "SOURCE_PANEL_TINT",
    "LABEL_COLOR",
    "ACCENT_RED",
    "ACCENT_GREEN",
    "ACCENT_BLUE",
    "ACCENT_PURPLE",
    "ACCENT_ORANGE",
    "NEUTRAL_DARK",
    "compose_tile_panel",
    "fit_pad",
    "build_standard_prompts",
    "assemble_choice_order",
]

from __future__ import annotations

from data.common.assets import render_font

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageFont

from data.common.video_encoding import (
    SIMPLE_VIDEO_FRAMES,
    encode_rgb_frames_to_webm,
    fit_frames_to_video_budget,
)


LETTER_OPTIONS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")
RENDER_SCALE = 2
DEFAULT_SAFE_MARGIN = 24


def resolve_safe_margin(
    target_size: tuple[int, int],
    safe_margin: int | tuple[int, int] | None = None,
) -> tuple[int, int]:
    if isinstance(safe_margin, int):
        margin_x = margin_y = max(0, int(safe_margin))
    elif isinstance(safe_margin, (tuple, list)) and len(safe_margin) == 2:
        margin_x = max(0, int(safe_margin[0]))
        margin_y = max(0, int(safe_margin[1]))
    else:
        margin_x = margin_y = max(DEFAULT_SAFE_MARGIN, min(target_size) // 20)
    max_margin_x = max(0, target_size[0] // 2 - 1)
    max_margin_y = max(0, target_size[1] // 2 - 1)
    return min(margin_x, max_margin_x), min(margin_y, max_margin_y)


def resize_to_fit(
    image: Image.Image,
    target_box: tuple[int, int],
    *,
    allow_scale_up: bool = False,
) -> Image.Image:
    source = image.convert("RGB")
    max_width = max(1, int(target_box[0]))
    max_height = max(1, int(target_box[1]))
    width_scale = max_width / max(1, source.width)
    height_scale = max_height / max(1, source.height)
    scale = min(width_scale, height_scale)
    if not allow_scale_up:
        scale = min(scale, 1.0)
    new_size = (
        max(1, int(round(source.width * scale))),
        max(1, int(round(source.height * scale))),
    )
    if new_size == source.size:
        return source.copy()
    resized = source.resize(new_size, Image.Resampling.LANCZOS)
    if scale < 1.0:
        resized = resized.filter(ImageFilter.UnsharpMask(radius=0.8, percent=140, threshold=2))
    return resized


def fit_image_with_safe_margin(
    image: Image.Image,
    *,
    target_size: tuple[int, int] = (512, 512),
    fill: tuple[int, int, int] = (255, 255, 255),
    safe_margin: int | tuple[int, int] | None = None,
    allow_scale_up: bool = False,
) -> Image.Image:
    margin_x, margin_y = resolve_safe_margin(target_size, safe_margin)
    fitted = resize_to_fit(
        image,
        (max(1, target_size[0] - margin_x * 2), max(1, target_size[1] - margin_y * 2)),
        allow_scale_up=allow_scale_up,
    )
    canvas = Image.new("RGB", target_size, fill)
    paste_x = (target_size[0] - fitted.width) // 2
    paste_y = (target_size[1] - fitted.height) // 2
    canvas.paste(fitted, (paste_x, paste_y))
    return canvas


def pad_image(
    img: Image.Image,
    target_size: tuple[int, int] = (512, 512),
    *,
    safe_margin: int | tuple[int, int] | None = None,
) -> Image.Image:
    return fit_image_with_safe_margin(img, target_size=target_size, safe_margin=safe_margin)


def _smoothstep(alpha: float) -> float:
    clamped = min(1.0, max(0.0, float(alpha)))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def downsample(image: Image.Image, scale: int) -> Image.Image:
    """LANCZOS downsample with sharpening for supersampled renders."""
    if scale <= 1:
        return image
    target = (image.width // scale, image.height // scale)
    result = image.resize(target, Image.Resampling.LANCZOS)
    return result.filter(ImageFilter.UnsharpMask(radius=0.7, percent=130, threshold=2))


def build_complete_footprint_fade_frames(
    puzzle_image: Image.Image,
    solution_image: Image.Image,
    *,
    hold_frames: int,
    fade_frames: int,
    transition_repeats: int = 1,
    final_hold_frames: int | None = None,
) -> list[np.ndarray]:
    """Fade every changed pixel together at the complete terminal footprint.

    Pixels that are identical at both endpoints remain bit-identical in every
    frame. All changed pixels transition together, so an atomic answer is not
    exposed by radial growth, a directional scan, or an input-to-final hard
    cut. Tasks with real motion/contact phases must provide their own renderer
    instead of using this atomic appearance helper.
    """
    puzzle_arr = np.array(puzzle_image.convert("RGB"), dtype=np.uint8)
    solution_arr = np.array(solution_image.convert("RGB"), dtype=np.uint8)
    if puzzle_arr.shape != solution_arr.shape:
        raise ValueError("Atomic semantic update endpoints must have the same shape")
    base_hold = max(1, int(hold_frames))
    transition_steps = max(2, int(fade_frames))
    repeats = max(1, int(transition_repeats))
    final_hold = base_hold if final_hold_frames is None else max(1, int(final_hold_frames))
    changed = np.any(puzzle_arr != solution_arr, axis=2)
    frames: list[np.ndarray] = [puzzle_arr.copy() for _ in range(base_hold)]
    for step in range(1, transition_steps + 1):
        alpha = _smoothstep(step / transition_steps)
        frame = puzzle_arr.copy()
        blended = np.rint(
            puzzle_arr[changed].astype(np.float32) * (1.0 - alpha)
            + solution_arr[changed].astype(np.float32) * alpha
        ).astype(np.uint8)
        frame[changed] = blended
        frames.extend(frame.copy() for _ in range(repeats))
    frames.extend(solution_arr.copy() for _ in range(final_hold))
    return frames


def save_visual_reasoning_video(
    puzzle_img: Image.Image,
    solution_img: Image.Image,
    video_path: str,
    fps: int = 16,
    *,
    cut_mode: str = "semantic",
) -> int:
    hold_frames = max(6, fps // 2)
    fade_frames = max(8, fps)
    if cut_mode != "semantic":
        raise ValueError(
            f"Unsupported cut_mode: {cut_mode!r}; atomic appearance transitions require 'semantic'"
        )
    frames = build_complete_footprint_fade_frames(
        puzzle_img,
        solution_img,
        hold_frames=hold_frames,
        fade_frames=fade_frames,
    )
    frames = fit_frames_to_video_budget(
        frames,
        max_frames=SIMPLE_VIDEO_FRAMES,
        final_hold_frames=max(4, fps // 4),
    )

    if not encode_rgb_frames_to_webm(frames, Path(video_path), fps=fps):
        return 0
    return len(frames)


def load_font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)

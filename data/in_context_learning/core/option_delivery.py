"""Authoring and panel-geometry helpers for ICL multiple-choice tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from data.common.render_primitives import DOCTRINE_SUPERSAMPLE, ScaledDraw
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.common.video_encoding import encode_rgb_frames_to_webm


OPTION_BORDER_RGB = (34, 139, 88)
OPTION_PANEL_KEYS = (
    "candidate_panel_bboxes",
    "candidate_panel_boxes",
    "option_panel_bboxes",
    "option_panels",
    "option_bboxes",
    "option_boxes",
    "panel_bboxes",
)

OPTION_MODALITY_OUTPUTS = {
    "ti2v": (
        "Return one fixed-camera video that holds the unchanged input, fades all four sides of "
        "the correct option border together and smoothly to green, then holds the selected card. "
        "No reasoning evidence, query content, candidate content, label, panel, background, or "
        "camera may change."
    ),
    "solution_image": (
        "Reproduce the supplied card exactly except for one complete uniform green outer border "
        "on the selected option; do not fill a query or retain a reasoning mark."
    ),
}
OPTION_PRESERVATION_CONSTRAINTS = (
    "The terminal image and every video frame preserve all input evidence, query cells, option contents, labels, panels, background, framing, and camera.",
    "Only the complete outer-border band of the independently correct option may change, and it may change only toward the shared selection green.",
)
OPTION_TRANSITION_CONSTRAINTS = (
    "Begin with a readable hold of the exact unselected input.",
    "Increase the opacity of all four sides of the correct option border together and smoothly; perimeter tracing, a hard commit, regression, and partial sides are invalid.",
    "End with a readable hold of the complete uniform green border while every other visual element remains unchanged.",
)


def _valid_panel_box(raw: object) -> tuple[int, int, int, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        left, top, right, bottom = (int(round(float(value))) for value in raw)
    except (TypeError, ValueError):
        return None
    if not (0 <= left < right <= 512 and 0 <= top < bottom <= 512):
        return None
    return left, top, right, bottom


def option_panel_boxes(structured: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    """Recover visible option-card geometry without consulting saved answers."""

    containers: list[dict[str, Any]] = [structured]
    render_geometry = structured.get("render_geometry")
    task_meta = structured.get("task_meta")
    if isinstance(render_geometry, dict):
        containers.append(render_geometry)
    if isinstance(task_meta, dict):
        containers.append(task_meta)
        nested_geometry = task_meta.get("render_geometry")
        if isinstance(nested_geometry, dict):
            containers.append(nested_geometry)
    for container in containers:
        for key in OPTION_PANEL_KEYS:
            raw = container.get(key)
            if isinstance(raw, dict):
                raw = [raw.get(letter) for letter in FOUR_CHOICE_CODEC.labels]
                raw = [value for value in raw if value is not None]
            if (
                not isinstance(raw, (list, tuple))
                or len(raw) != len(FOUR_CHOICE_CODEC.labels)
            ):
                continue
            boxes = [_valid_panel_box(value) for value in raw]
            if all(box is not None for box in boxes):
                return [box for box in boxes if box is not None]
    return []


def _draw_uniform_border(
    image: Image.Image,
    box: Sequence[int],
    *,
    opacity: float = 1.0,
) -> None:
    """Composite one uniform green option border onto a finished 512-square card."""

    opacity = min(1.0, max(0.0, float(opacity)))
    if opacity <= 0.0:
        return
    factor = DOCTRINE_SUPERSAMPLE
    mask = Image.new("L", (image.width * factor, image.height * factor), 0)
    ScaledDraw(ImageDraw.Draw(mask), factor).rounded_rectangle(
        tuple(int(value) for value in box),
        radius=8,
        outline=int(round(255 * opacity)),
        width=5,
    )
    image.paste(
        Image.new("RGB", image.size, OPTION_BORDER_RGB),
        (0, 0),
        mask.resize(image.size, Image.Resampling.LANCZOS),
    )


def render_terminal_image(
    target_path: Path,
    *,
    puzzle_path: Path,
    boxes: Sequence[Sequence[int]],
    selected_index: int,
) -> None:
    """Author the formal terminal from the clean input plus one green border."""

    if not target_path.is_file() or not puzzle_path.is_file():
        return
    if selected_index not in range(len(boxes)):
        raise ValueError("selected option is outside the visible candidate set")
    with Image.open(puzzle_path) as opened:
        target = opened.convert("RGB")
    _draw_uniform_border(target, boxes[selected_index])
    target.save(target_path)


def render_selection_video(
    target_path: Path,
    *,
    puzzle_path: Path,
    boxes: Sequence[Sequence[int]],
    selected_index: int,
) -> None:
    """Author input hold -> simultaneous full-border fade -> final hold."""

    if not puzzle_path.is_file():
        return
    if selected_index not in range(len(boxes)):
        raise ValueError("selected option is outside the visible candidate set")
    with Image.open(puzzle_path) as opened:
        puzzle = opened.convert("RGB")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    opacities = [0.0] * 5 + [step / 11.0 for step in range(1, 12)] + [1.0] * 5
    frames: list[np.ndarray] = []
    for opacity in opacities:
        frame = puzzle.copy()
        _draw_uniform_border(frame, boxes[selected_index], opacity=opacity)
        frames.append(np.asarray(frame, dtype=np.uint8))
    if len(frames) % 4 != 1:
        raise AssertionError("ICL option video must use a 4N+1 frame count")
    if not encode_rgb_frames_to_webm(frames, target_path, fps=16):
        raise RuntimeError(f"Failed to encode standardized option video: {target_path}")


__all__ = [
    "OPTION_BORDER_RGB",
    "OPTION_MODALITY_OUTPUTS",
    "OPTION_PRESERVATION_CONSTRAINTS",
    "OPTION_TRANSITION_CONSTRAINTS",
    "option_panel_boxes",
    "render_selection_video",
    "render_terminal_image",
]

"""Shared board layout and snapshot-dissolve video helpers on a 512x512 canvas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFont

from data.common.assets import render_font
from data.common.video_encoding import (
    SIMPLE_VIDEO_FRAMES,
    encode_rgb_frames_to_webm,
    fit_frames_to_video_budget,
)


CANVAS_SIZE = 512
DEFAULT_FPS = 16




def load_font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)


def even_cell_size(rows: int, cols: int, *, board_max: int = 440, board_min: int = 32) -> int:
    """Return an even cell size that keeps the board centred inside CANVAS_SIZE.

    The result is at least board_min and at most board_max // max(rows, cols), and
    rounded down to the nearest even integer so pixel centres land on integer
    pixels.
    """
    extent = max(rows, cols)
    raw = max(board_min, board_max // extent)
    even = raw - (raw % 2)
    return max(2, even)


@dataclass
class BoardLayout:
    rows: int
    cols: int
    cell: int
    origin_x: int
    origin_y: int

    @classmethod
    def for_grid(cls, rows: int, cols: int, *, board_max: int = 440, reserve_top: int = 0) -> "BoardLayout":
        """Build a layout for a rows x cols board.

        reserve_top reserves space above the board (e.g. for column labels) so
        the board area does not overlap labels. The board still centres inside
        the remaining canvas height.
        """
        cell = even_cell_size(rows, cols, board_max=board_max - reserve_top)
        ox = (CANVAS_SIZE - cols * cell) // 2
        oy = reserve_top + (CANVAS_SIZE - reserve_top - rows * cell) // 2
        return cls(rows=rows, cols=cols, cell=cell,
                   origin_x=ox - (ox % 2),
                   origin_y=oy - (oy % 2))

    def cell_box(self, row: int, col: int) -> Tuple[int, int, int, int]:
        x0 = self.origin_x + col * self.cell
        y0 = self.origin_y + row * self.cell
        return x0, y0, x0 + self.cell, y0 + self.cell

    def cell_center(self, row: int, col: int) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.cell_box(row, col)
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def build_state_dissolve_video(
    *,
    video_path: Path,
    states: Sequence[Image.Image],
    transition_frames: int = 6,
    hold_frames: int = 8,
    fps: int = DEFAULT_FPS,
) -> int:
    """Smooth alpha dissolve between successive board snapshots.

    Used for transitions that are NOT a single piece sliding (e.g. multi-cell
    Tetris lock, Lights Out toggle, Jewel2 cascade): sub-pixel interpolation
    cannot represent these, so we instead alpha-blend between holds. Hold
    frames are bit-identical to the snapshot so the start/end state is always
    a clean reference frame; only mid-dissolve frames are blended.
    """
    if not states:
        return 0
    frames: List[np.ndarray] = []
    arrays = [np.array(image.convert("RGB"), dtype=np.uint8) for image in states]

    def push(arr: np.ndarray, count: int) -> None:
        for _ in range(count):
            frames.append(arr.copy())

    push(arrays[0], hold_frames)
    for i in range(len(arrays) - 1):
        a = arrays[i].astype(np.float32)
        b = arrays[i + 1].astype(np.float32)
        for k in range(1, transition_frames + 1):
            t = k / (transition_frames + 1)
            blended = (a * (1.0 - t) + b * t).clip(0, 255).astype(np.uint8)
            frames.append(blended)
        push(arrays[i + 1], hold_frames)

    frames = fit_frames_to_video_budget(
        frames,
        max_frames=SIMPLE_VIDEO_FRAMES,
        final_hold_frames=hold_frames,
    )

    if not encode_rgb_frames_to_webm(
        frames,
        video_path,
        fps=fps,
        extra_args=["-g", "1", "-bf", "0"],
    ):
        return 0
    return len(frames)


__all__ = [
    "BoardLayout",
    "CANVAS_SIZE",
    "DEFAULT_FPS",
    "build_state_dissolve_video",
    "even_cell_size",
    "load_font",
]

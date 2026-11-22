"""Shared base for mechanic-rich single-agent spatial_navigation tasks.

Provides:
- ``MechanicGridGeneratorBase``: 512x512 auto-fit canvas, exact discrete agent states,
  shared mechanic icon primitives, and internal row-major indexing helpers.
- ``MechanicScenario``: dataclass capturing per-task render state.

Each concrete task supplies a ``sample_scenario`` that builds a ``MechanicScenario``
plus an animation script (a list of frame deltas), and writes an
``answer_payload`` whose canonical answer is an ordered list of visible
1-based ``(x, y)`` positions along the trajectory.
"""

from __future__ import annotations

from data.common.assets import render_font

import math
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.common.base import AbstractPuzzleGenerator
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.video_encoding import encode_rgb_frames_to_webm, fit_frames_to_video_budget
from data.common.animation_policy import (
    DISCRETE_STATE_HOLD_FRAMES,
    frame_budget_for_frame_count,
)
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path
from data.spatial_navigation.core.prompt_contract import build_spatial_navigation_prompt_bundle


DIR_TO_DELTA: Dict[str, Tuple[int, int]] = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "stay": (0, 0),
}
DELTA_TO_DIR: Dict[Tuple[int, int], str] = {v: k for k, v in DIR_TO_DELTA.items()}

# Palette: all tasks share this set so the family looks like one design system.
COLOR_BG = (250, 249, 246)
COLOR_GRID = (179, 181, 186)
COLOR_FLOOR = (255, 255, 255)
COLOR_WALL = (54, 64, 82)
COLOR_WALL_OUTLINE = (32, 38, 52)
COLOR_START = (58, 163, 92)
COLOR_START_DARK = (30, 106, 56)
COLOR_GOAL = (67, 124, 230)
COLOR_GOAL_DARK = (37, 82, 163)
COLOR_AGENT = (239, 133, 59)
COLOR_AGENT_DARK = (149, 78, 34)
COLOR_PATH = (220, 41, 41)
COLOR_TEXT = (30, 30, 30)
# Family-wide royal-blue 1-based axis numbers (same as the maze scenes).
COLOR_LABEL = (30, 90, 220)

# Mechanic accent palette
COLOR_PLATE_OFF = (180, 174, 158)
COLOR_PLATE_ON = (255, 187, 60)
COLOR_LEVER_BASE = (97, 71, 50)
COLOR_LEVER_OFF = (213, 165, 92)
COLOR_LEVER_ON = (236, 121, 70)
COLOR_DOOR_LOCKED = (103, 92, 67)
COLOR_DOOR_OPEN = (170, 214, 154)
COLOR_DOOR_OUTLINE = (67, 61, 44)
COLOR_KEY = (245, 213, 89)
COLOR_KEY_OUTLINE = (168, 133, 18)
COLOR_PORTAL_A = (132, 94, 247)
COLOR_PORTAL_B = (236, 121, 70)
COLOR_PORTAL_C = (66, 184, 131)
COLOR_PORTAL_D = (233, 103, 186)
COLOR_BOMB = (38, 38, 38)
COLOR_BOMB_FUSE = (236, 121, 70)
COLOR_OIL = (138, 92, 56)
COLOR_OIL_FLAME = (255, 187, 60)
COLOR_MAGNET = (210, 60, 60)
COLOR_MAGNET_BLUE = (35, 102, 168)
COLOR_ICE = (199, 230, 244)
COLOR_ICE_OUTLINE = (135, 188, 220)
COLOR_WIND = (190, 220, 240)
COLOR_CONVEYOR = (240, 230, 200)
COLOR_CONVEYOR_DARK = (180, 160, 120)
COLOR_BRIDGE = (172, 130, 78)
COLOR_BRIDGE_DARK = (115, 80, 40)
COLOR_WATER = (90, 152, 210)
COLOR_WATER_DARK = (40, 90, 150)


@dataclass
class MechanicScenario:
    """One mechanic-rich spatial_navigation instance."""

    task_name: str
    task_family: str
    title: str
    input_desc: str
    rules_desc: str
    ti2v_output: str
    solution_image_output: str

    size: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    agent_path: List[Tuple[int, int]]      # all cells the agent visits in order
    walls: List[Tuple[int, int]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    # answer payload helpers
    structured_answer: Dict[str, Any] = field(default_factory=dict)


class MechanicGridGeneratorBase(AbstractPuzzleGenerator[Dict[str, Any]]):
    """Abstract parent for mechanic-rich single-agent spatial_navigation tasks."""

    TASK_NAME = "mechanic_grid_task"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Mechanic Grid Task"
    DEFAULT_SIZE = 7

    CANVAS_SIDE = 512
    # Family line-quality floor: scenes draw supersampled through the shared
    # render primitives and LANCZOS-downsample back to the 512 canvas.
    SCENE_SUPERSAMPLE = DOCTRINE_SUPERSAMPLE

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: Optional[int] = None,
        size: Optional[int] = None,
        video: bool = True,
        video_fps: int = 16,
    ) -> None:
        resolved_output_dir = (
            output_dir if output_dir is not None
            else Path(tempfile.gettempdir()) / f"videogym_{self.TASK_NAME}"
        )
        super().__init__(resolved_output_dir)
        self.rng = random.Random(seed)
        self.seed = seed
        self.size = max(4, int(size if size is not None else self.DEFAULT_SIZE))
        self.video = bool(video)
        self.video_fps = int(video_fps)
        # 512x512 auto-fit, even-pixel cell.
        ideal = self.CANVAS_SIDE / (self.size + 1)
        self.cell_size = max(28, int(ideal) // 2 * 2)
        remaining = self.CANVAS_SIDE - self.size * self.cell_size
        self.margin = remaining // 2 if remaining >= 24 else max(16, self.cell_size // 2)
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def create_random_puzzle(self) -> Dict[str, Any]:
        return self.create_puzzle()

    # -------- subclass hook --------

    def sample_scenario(self) -> MechanicScenario:
        raise NotImplementedError

    # -------- canvas / cells --------

    def _canvas_size(self) -> Tuple[int, int]:
        side = self.margin * 2 + self.size * self.cell_size
        return side, side

    def _cell_box(self, cell: Tuple[int, int]) -> Tuple[int, int, int, int]:
        row, col = cell
        left = self.margin + col * self.cell_size
        top = self.margin + row * self.cell_size
        return left, top, left + self.cell_size, top + self.cell_size

    def _cell_center(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        left, top, right, bottom = self._cell_box(cell)
        return (left + right) / 2.0, (top + bottom) / 2.0

    def _cell_id(self, cell: Tuple[int, int]) -> int:
        # 1-based row-major id over the full grid
        return cell[0] * self.size + cell[1] + 1

    def _path_cell_ids(self, path: Sequence[Tuple[int, int]]) -> List[int]:
        return [self._cell_id(cell) for cell in path]

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        return render_font(int(size) * self.SCENE_SUPERSAMPLE, bold=False)

    def _begin_scene(self) -> Tuple[Image.Image, ScaledDraw]:
        """Allocate the supersampled scene canvas for one board render."""

        width, height = self._canvas_size()
        from data.common.prompt_paraphrase import active_profile
        return new_scaled_canvas(width,height,self.SCENE_SUPERSAMPLE,(217,232,224) if active_profile("portal_gates") else COLOR_BG)

    def _finish_scene(self, image: Image.Image) -> Image.Image:
        """Downsample the supersampled scene back to the 512 target canvas."""

        return finish_scaled_canvas(image, self._canvas_size())

    def _next_id(self) -> str:
        puzzle_id = f"{self.TASK_NAME}-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    # -------- shared icon primitives (product-grade, layered PIL) --------

    def _draw_grid_and_labels(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        wall_set: Iterable[Tuple[int, int]] = (),
    ) -> None:
        from data.common.prompt_paraphrase import active_profile
        wall_set = set(wall_set)
        label_font = self._load_font(max(14, self.cell_size // 4))
        for row in range(self.size):
            for col in range(self.size):
                left, top, right, bottom = self._cell_box((row, col))
                if (row, col) in wall_set:
                    draw.rectangle(
                        [left, top, right, bottom],
                        fill=COLOR_WALL,
                        outline=COLOR_WALL_OUTLINE,
                        width=1,
                    )
                elif active_profile("portal_gates"):
                    cx,cy=(left+right)/2,(top+bottom)/2;rad=self.cell_size*.31
                    for rr,cc in ((row+1,col),(row,col+1)):
                        if rr<self.size and cc<self.size and (rr,cc) not in wall_set:
                            xx,yy=self._cell_center((rr,cc))
                            draw.line((cx,cy,xx,yy),fill=(89,117,117),width=4)
                    draw.polygon([(cx,cy-rad),(cx+rad,cy),(cx,cy+rad),(cx-rad,cy)],fill=(246,233,194),outline=(89,117,117),width=2)
                else:
                    draw.rectangle(
                        [left, top, right, bottom],
                        fill=COLOR_FLOOR,
                        outline=COLOR_GRID,
                        width=1,
                    )
        for row in range(self.size):
            draw.text(
                (self.margin - self.cell_size * 0.32, self.margin + row * self.cell_size + self.cell_size * 0.5),
                str(row + 1),
                fill=COLOR_LABEL,
                font=label_font,
                anchor="mm",
            )
        for col in range(self.size):
            draw.text(
                (self.margin + col * self.cell_size + self.cell_size * 0.5, self.margin - self.cell_size * 0.32),
                str(col + 1),
                fill=COLOR_LABEL,
                font=label_font,
                anchor="mm",
            )

    def _draw_start_goal(
        self,
        draw: ImageDraw.ImageDraw,
        scenario: MechanicScenario,
        *,
        token_font: Optional[ImageFont.ImageFont] = None,
    ) -> None:
        """Draw the S and G tiles with corner-seated letters.

        The agent disc is centred in whichever endpoint cell it occupies, so
        a centred letter would be hidden in every input (S) and terminal (G)
        state.  Seating each letter in the top-left corner of its tile keeps
        both identities readable while the agent shares the cell.
        """

        if token_font is None:
            token_font = self._load_font(max(12, self.cell_size // 4 - 2))
        inset = max(5, self.cell_size // 12)
        offset = inset + max(6, self.cell_size // 9)
        for cell, letter, fill, outline in (
            (scenario.start, "S", COLOR_START, COLOR_START_DARK),
            (scenario.goal, "G", COLOR_GOAL, COLOR_GOAL_DARK),
        ):
            left, top, right, bottom = self._cell_box(cell)
            draw.rounded_rectangle(
                [left + inset, top + inset, right - inset, bottom - inset],
                radius=max(6, inset),
                fill=fill,
                outline=outline,
                width=2,
            )
            draw.text(
                (left + offset, top + offset),
                letter,
                fill=(255, 255, 255),
                font=token_font,
                anchor="mm",
            )

    def _draw_agent(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
    ) -> None:
        cx, cy = center
        radius = max(6, (self.cell_size - 2 * max(10, self.cell_size // 4)) / 2.0)
        # outer halo
        draw.ellipse(
            [cx - radius - 2, cy - radius - 2, cx + radius + 2, cy + radius + 2],
            fill=(255, 255, 255),
            outline=COLOR_AGENT_DARK,
            width=2,
        )
        # body
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=COLOR_AGENT,
            outline=COLOR_AGENT_DARK,
            width=2,
        )
        # highlight glint
        glint_r = radius * 0.32
        draw.ellipse(
            [cx - radius * 0.4, cy - radius * 0.5, cx - radius * 0.4 + glint_r, cy - radius * 0.5 + glint_r],
            fill=(255, 215, 170),
        )

    # Gauge geometry is shared with the family verifier: pips sit in the
    # bottom margin, right-aligned to the grid's right edge, so the whole
    # gauge stays inside the verifier's below-grid right-half variation mask.
    def _gauge_pip_boxes(self, total: int) -> List[Tuple[float, float, float, float]]:
        width, height = self._canvas_size()
        strip_top = self.margin + self.size * self.cell_size
        strip_h = height - strip_top
        side = max(9, min(16, strip_h - 10))
        gap = max(3, side // 4)
        right_edge = width - self.margin
        top = strip_top + (strip_h - side) / 2.0
        boxes: List[Tuple[float, float, float, float]] = []
        for index in range(int(total)):
            right = right_edge - (int(total) - 1 - index) * (side + gap)
            boxes.append((right - side, top, right, top + side))
        return boxes

    def _draw_inventory_gauge(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        filled: int,
        total: int,
        fill_color: Tuple[int, int, int],
        outline_color: Tuple[int, int, int],
    ) -> None:
        """Draw a discrete pip gauge: the leftmost ``filled`` pips are full.

        Each quantity state is a prefix of the same stable pip positions, so
        spending or restoring one unit never rearranges the others and no
        per-step numeric counter is needed.
        """

        filled = max(0, min(int(filled), int(total)))
        for index, box in enumerate(self._gauge_pip_boxes(int(total))):
            if index < filled:
                draw.rounded_rectangle(box, radius=3, fill=fill_color, outline=outline_color, width=1)
            else:
                draw.rounded_rectangle(box, radius=3, fill=(255, 255, 255), outline=outline_color, width=1)

    def _draw_pressure_plate(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
        *,
        active: bool,
        label: Optional[str] = None,
        token_font: Optional[ImageFont.ImageFont] = None,
    ) -> None:
        l, t, r, b = self._cell_box(cell)
        cx, cy = self._cell_center(cell)
        outer_r = self.cell_size * 0.36
        inner_r = self.cell_size * 0.26
        rim = COLOR_PLATE_ON if active else COLOR_PLATE_OFF
        face = (255, 220, 140) if active else (220, 215, 200)
        draw.ellipse(
            [cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r],
            fill=rim,
            outline=tuple(max(0, c - 50) for c in rim),
            width=2,
        )
        draw.ellipse(
            [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
            fill=face,
            outline=tuple(max(0, c - 50) for c in rim),
            width=2,
        )
        if label is not None:
            if token_font is None:
                token_font = self._load_font(max(12, self.cell_size // 5))
            draw.text((cx, cy), label, fill=COLOR_TEXT, font=token_font, anchor="mm")

    def _draw_lever(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
        *,
        on: bool,
        label: Optional[str] = None,
        token_font: Optional[ImageFont.ImageFont] = None,
    ) -> None:
        left, top, right, bottom = self._cell_box(cell)
        cx, cy = self._cell_center(cell)
        # Station tile so the lever reads as a fixed installation, not floor
        # clutter; the tile itself never changes state.
        inset = max(3, self.cell_size // 16)
        draw.rounded_rectangle(
            [left + inset, top + inset, right - inset, bottom - inset],
            radius=max(4, inset),
            fill=(246, 238, 220),
            outline=(190, 170, 130),
            width=2,
        )
        # base block
        bw = self.cell_size * 0.52
        bh = self.cell_size * 0.15
        base_y = cy + self.cell_size * 0.16
        draw.rounded_rectangle(
            [cx - bw / 2, base_y, cx + bw / 2, base_y + bh],
            radius=4,
            fill=COLOR_LEVER_BASE,
            outline=tuple(max(0, c - 40) for c in COLOR_LEVER_BASE),
            width=2,
        )
        # The pulled state has exactly one encoding: the arm leans right when
        # un-pulled and left when pulled.  Colour stays constant.
        arm_len = self.cell_size * 0.38
        direction = -1.0 if on else 1.0
        color = COLOR_LEVER_ON
        ax = cx + direction * arm_len * 0.707
        ay = base_y - arm_len * 0.707
        draw.line([(cx, base_y), (ax, ay)], fill=color, width=max(4, self.cell_size // 12))
        kr = max(4, self.cell_size // 10)
        draw.ellipse(
            [ax - kr, ay - kr, ax + kr, ay + kr],
            fill=color,
            outline=tuple(max(0, c - 40) for c in color),
            width=2,
        )
        if label is not None:
            if token_font is None:
                token_font = self._load_font(max(11, self.cell_size // 6))
            draw.text(
                (left + inset + self.cell_size * 0.14, top + inset + self.cell_size * 0.13),
                label,
                fill=COLOR_TEXT,
                font=token_font,
                anchor="mm",
            )

    def _draw_door(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
        *,
        open_state: bool,
        label: Optional[str] = None,
        token_font: Optional[ImageFont.ImageFont] = None,
    ) -> None:
        l, t, r, b = self._cell_box(cell)
        inset = max(5, self.cell_size // 10)
        fill = COLOR_DOOR_OPEN if open_state else COLOR_DOOR_LOCKED
        draw.rounded_rectangle(
            [l + inset, t + inset, r - inset, b - inset],
            radius=max(5, inset),
            fill=fill,
            outline=COLOR_DOOR_OUTLINE,
            width=2,
        )
        # door grain
        cx, cy = self._cell_center(cell)
        if not open_state:
            for k in (-1, 0, 1):
                offset = (self.cell_size // 6) * k
                draw.line(
                    [(l + inset + 4, cy + offset), (r - inset - 4, cy + offset)],
                    fill=tuple(max(0, c - 25) for c in fill),
                    width=1,
                )
            # knob
            kx = r - inset - 8
            draw.ellipse(
                [kx - 4, cy - 4, kx + 4, cy + 4],
                fill=(220, 200, 130),
                outline=(120, 90, 30),
                width=1,
            )
        if label is not None:
            if token_font is None:
                token_font = self._load_font(max(10, self.cell_size // 7))
            draw.text((cx, t + inset + (self.cell_size // 5)), label, fill=COLOR_TEXT, font=token_font, anchor="mm")

    def _draw_portal(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
        *,
        color: Tuple[int, int, int],
        label: Optional[str] = None,
        token_font: Optional[ImageFont.ImageFont] = None,
    ) -> None:
        cx, cy = self._cell_center(cell)
        outer = self.cell_size * 0.36
        from data.common.prompt_paraphrase import active_profile
        if active_profile("portal_gates"):
            dark=tuple(max(0,c-65) for c in color)
            draw.polygon([(cx-outer,cy-outer*.7),(cx+outer*.3,cy-outer*.7),(cx+outer,cy),(cx+outer*.3,cy+outer*.7),(cx-outer,cy+outer*.7)],fill=color,outline=dark,width=2)
            draw.line((cx-outer*.65,cy,cx+outer*.45,cy),fill=(255,255,240),width=4)
            draw.line((cx+outer*.1,cy-outer*.3,cx+outer*.45,cy,cx+outer*.1,cy+outer*.3),fill=(255,255,240),width=3)
        else:
            # 3 nested rings
            for k, factor in enumerate((1.0, 0.78, 0.55)):
                r = outer * factor
                shade = tuple(min(255, c + 25 * k) for c in color)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=shade, width=3)
            # core
            core_r = outer * 0.25
            draw.ellipse(
                [cx - core_r, cy - core_r, cx + core_r, cy + core_r],
                fill=color,
                outline=tuple(max(0, c - 50) for c in color),
                width=2,
            )
        if label is not None:
            if token_font is None:
                token_font = self._load_font(max(10, self.cell_size // 7))
            draw.text((cx, cy - outer - 6), label, fill=COLOR_TEXT, font=token_font, anchor="mm")

    def _draw_conveyor(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
        *,
        direction: str,
    ) -> None:
        l, t, r, b = self._cell_box(cell)
        # base
        draw.rectangle([l + 1, t + 1, r - 1, b - 1], fill=COLOR_CONVEYOR, outline=COLOR_CONVEYOR_DARK, width=1)
        # treads
        cx, cy = self._cell_center(cell)
        for k in (-2, 0, 2):
            offset = (self.cell_size // 8) * k
            if direction in ("right", "left"):
                draw.line(
                    [(l + 4, cy + offset), (r - 4, cy + offset)],
                    fill=COLOR_CONVEYOR_DARK, width=1,
                )
            else:
                draw.line(
                    [(cx + offset, t + 4), (cx + offset, b - 4)],
                    fill=COLOR_CONVEYOR_DARK, width=1,
                )
        # arrow
        arrow_len = self.cell_size * 0.4
        if direction == "right":
            tip = (cx + arrow_len / 2, cy)
            tail = (cx - arrow_len / 2, cy)
        elif direction == "left":
            tip = (cx - arrow_len / 2, cy)
            tail = (cx + arrow_len / 2, cy)
        elif direction == "up":
            tip = (cx, cy - arrow_len / 2)
            tail = (cx, cy + arrow_len / 2)
        else:
            tip = (cx, cy + arrow_len / 2)
            tail = (cx, cy - arrow_len / 2)
        # shaft
        draw.line([tail, tip], fill=COLOR_LEVER_ON, width=max(3, self.cell_size // 14))
        # head
        dx = tip[0] - tail[0]
        dy = tip[1] - tail[1]
        length = math.hypot(dx, dy)
        if length > 0.001:
            ux, uy = dx / length, dy / length
            head = self.cell_size * 0.18
            wing = self.cell_size * 0.13
            left_p = (tip[0] - ux * head - uy * wing, tip[1] - uy * head + ux * wing)
            right_p = (tip[0] - ux * head + uy * wing, tip[1] - uy * head - ux * wing)
            draw.polygon([tip, left_p, right_p], fill=COLOR_LEVER_ON, outline=tuple(max(0, c - 30) for c in COLOR_LEVER_ON))

    def _draw_oil_station(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
    ) -> None:
        cx, cy = self._cell_center(cell)
        r_outer = self.cell_size * 0.38
        r_inner = self.cell_size * 0.30
        # circular pad
        draw.ellipse(
            [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
            fill=(255, 230, 180),
            outline=(200, 140, 50),
            width=2,
        )
        # droplet
        dx = cx
        dy = cy - r_inner * 0.05
        dr = self.cell_size * 0.13
        draw.polygon(
            [
                (dx, dy - dr * 1.2),
                (dx - dr * 0.85, dy + dr * 0.4),
                (dx, dy + dr * 0.9),
                (dx + dr * 0.85, dy + dr * 0.4),
            ],
            fill=(70, 130, 200),
            outline=(30, 80, 150),
        )


    def _draw_ice(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
    ) -> None:
        l, t, r, b = self._cell_box(cell)
        draw.rectangle([l + 1, t + 1, r - 1, b - 1], fill=COLOR_ICE, outline=COLOR_ICE_OUTLINE, width=1)
        cx, cy = self._cell_center(cell)
        sz = self.cell_size * 0.18
        # snowflake glyph: 3 lines
        for k in range(3):
            angle = math.radians(60 * k)
            dx = math.cos(angle) * sz
            dy = math.sin(angle) * sz
            draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=(120, 170, 210), width=2)

    def _draw_water(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
    ) -> None:
        l, t, r, b = self._cell_box(cell)
        draw.rectangle([l + 1, t + 1, r - 1, b - 1], fill=COLOR_WATER, outline=COLOR_WATER_DARK, width=1)
        # wavy lines
        for k, frac in enumerate((0.3, 0.55, 0.8)):
            wy = t + (b - t) * frac
            n = 5
            pts = []
            for i in range(n + 1):
                fx = l + (r - l) * (i / n)
                fy = wy + (4 if i % 2 == 0 else -4)
                pts.append((fx, fy))
            draw.open_path(pts, fill=(220, 240, 255), width=2)

    def _draw_bridge_tile(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
    ) -> None:
        l, t, r, b = self._cell_box(cell)
        draw.rectangle([l + 1, t + 1, r - 1, b - 1], fill=COLOR_BRIDGE, outline=COLOR_BRIDGE_DARK, width=1)
        # planks
        for k in range(3):
            ly = t + 6 + k * (self.cell_size - 12) // 3
            draw.line([(l + 4, ly), (r - 4, ly)], fill=COLOR_BRIDGE_DARK, width=2)

    def _draw_path_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        path_cells: Sequence[Tuple[int, int]],
        *,
        teleport_breaks: Sequence[int] = (),
    ) -> None:
        """Draw the red answer path. teleport_breaks are indices i where path[i]→path[i+1] is a teleport."""
        if len(path_cells) < 2:
            return
        teleport_set = set(teleport_breaks)
        line_w = max(5, self.cell_size // 7)
        # split into segments — a teleport at index i means path[i] is the source
        # portal and path[i+1] is the destination portal; the line ends at path[i]
        # and resumes at path[i+1] (no straight bridge across the teleport).
        segments: List[List[Tuple[float, float]]] = []
        current: List[Tuple[float, float]] = [self._cell_center(path_cells[0])]
        for i in range(len(path_cells) - 1):
            if i in teleport_set:
                # close current segment at path[i] (already in `current`); start fresh at path[i+1]
                segments.append(current)
                current = [self._cell_center(path_cells[i + 1])]
            else:
                current.append(self._cell_center(path_cells[i + 1]))
        if current:
            segments.append(current)
        for seg in segments:
            if len(seg) >= 2:
                draw_spatial_navigation_path(draw, seg, color=COLOR_PATH, width=line_w)

    # -------- prompt assembly --------

    def _build_prompts(self, scenario: MechanicScenario) -> Dict[str, str]:
        return build_spatial_navigation_prompt_bundle(
            task_name=scenario.task_name,
            input_description=scenario.input_desc,
            rule_description=scenario.rules_desc,
            ti2v_output=scenario.ti2v_output,
        )

    # -------- video animation --------

    def _animate_path(
        self,
        scenario: MechanicScenario,
        render_frame: Callable[[Tuple[int, int], Tuple[float, float], int], Image.Image],
        *,
        teleport_steps: Sequence[int] = (),
    ) -> List[np.ndarray]:
        """Build a fixed-cadence sequence of exact semantic grid states."""
        path = list(scenario.agent_path)
        if not path:
            return []
        teleport_set = {int(step) for step in teleport_steps}

        frames: List[np.ndarray] = []
        protected_indices: List[int] = []

        def append_state(cell: Tuple[int, int], step_index: int) -> None:
            image = render_frame(cell, self._cell_center(cell), step_index)
            start_index = len(frames)
            frame = np.array(image.convert("RGB"), dtype=np.uint8)
            frames.extend(frame.copy() for _ in range(DISCRETE_STATE_HOLD_FRAMES))
            protected_indices.append(start_index)

        append_state(path[0], 0)

        for i, cell in enumerate(path[1:], start=1):
            prev = path[i - 1]
            is_teleport = (i - 1) in teleport_set
            if is_teleport:
                # A teleport is still represented by two exact states: source
                # and destination.  There is no interpolated activation frame.
                append_state(prev, i - 1)

            append_state(cell, i)

        budget = frame_budget_for_frame_count(len(frames))
        return fit_frames_to_video_budget(
            frames,
            max_frames=budget,
            final_hold_frames=DISCRETE_STATE_HOLD_FRAMES,
            protected_indices=protected_indices,
        )

    def _moves_for_path(self, path: Sequence[Tuple[int, int]]) -> List[str]:
        moves: List[str] = []
        for prev, cell in zip(path, path[1:]):
            delta = (cell[0] - prev[0], cell[1] - prev[1])
            moves.append(DELTA_TO_DIR.get(delta, "jump"))
        return moves

    def _cells_from_extra(self, raw: Any) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        if isinstance(raw, dict):
            if "cell" in raw:
                cells.extend(self._cells_from_extra(raw["cell"]))
            return cells
        if isinstance(raw, (list, tuple)):
            if len(raw) == 2 and all(isinstance(part, int) for part in raw):
                return [(int(raw[0]), int(raw[1]))]
            for item in raw:
                cells.extend(self._cells_from_extra(item))
        return cells


    def _authored_reasoning(self, scenario: MechanicScenario) -> Optional[str]:
        """Return sample-specific solving reasoning for this scenario.

        Each task derives this text from its own semantic truth: observe the
        layout, name the load-bearing mechanism, walk the route segment by
        segment with its state changes, and confirm the terminal state.  The
        base returns ``None`` so tasks without an authored trace fail loudly
        at SFT reference build instead of shipping template prose.
        """

        return None

    def _authored_reasoning_segments(
        self,
        scenario: MechanicScenario,
    ) -> Optional[List[str]]:
        """Return the interleaved reasoning segments for an interleaved task.

        Segment ``k`` precedes reasoning image ``k+1`` and the final segment
        closes the trace after the last image, so a task emitting ``n``
        nonterminal states must return ``n + 1`` segments, joining the
        same segments, keeping one authored text source for both modalities.
        """

        return None


    def _write_video(self, frames: List[np.ndarray], video_path: Path) -> int:
        if not frames:
            return 0
        budget = frame_budget_for_frame_count(len(frames))
        frames = fit_frames_to_video_budget(
            frames,
            max_frames=budget,
            final_hold_frames=max(6, self.video_fps // 2),
        )
        if not encode_rgb_frames_to_webm(frames, video_path, fps=self.video_fps):
            return 0
        return len(frames)

    # -------- final record assembly --------

    def _format_answer_text(
        self,
        *,
        path: Sequence[Tuple[int, int]],
    ) -> str:
        """Serialize the formal route as visible 1-based ``(x, y)`` positions."""

        from data.common.grid_coordinates import format_xy_path_answer

        return format_xy_path_answer(path)

    def _assemble_record(
        self,
        scenario: MechanicScenario,
        *,
        puzzle_image: Image.Image,
        solution_image: Image.Image,
        video_frames: Optional[List[np.ndarray]],
    ) -> Dict[str, Any]:
        puzzle_id = self._next_id()
        sample_dir = self.output_dir / puzzle_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / "puzzle.png"
        solution_path = sample_dir / "solution.png"
        video_path = sample_dir / "solution.webm"

        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        video_num_frames = None
        solution_video_path = None
        if self.video and video_frames is not None:
            n = self._write_video(video_frames, video_path)
            if n > 0:
                video_num_frames = n
                solution_video_path = self.relativize_path(video_path)

        cell_ids = self._path_cell_ids(scenario.agent_path)
        answer_line = self._format_answer_text(path=scenario.agent_path)
        prompts = self._build_prompts(scenario)
        answer_text = answer_line

        return {
            "id": puzzle_id,
            "task_name": scenario.task_name,
            "task_family": scenario.task_family,
            "title": scenario.title,
            "image": self.relativize_path(puzzle_path),
            "reasoning_image": self.relativize_path(puzzle_path),
            "solution_image_path": self.relativize_path(solution_path),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
            "answer": answer_text,
            "answer_payload": {
                "grid_size": scenario.size,
                "start": [scenario.start[0], scenario.start[1]],
                "goal": [scenario.goal[0], scenario.goal[1]],
                "walls": [[r, c] for r, c in scenario.walls],
                "agent_path": [[r, c] for r, c in scenario.agent_path],
                "agent_path_cell_ids": cell_ids,
                "extras": dict(scenario.extras),
                "structured_answer": dict(scenario.structured_answer),
            },
            "seed": self.seed,
            **prompts,
        }


__all__ = [
    "DELTA_TO_DIR",
    "DIR_TO_DELTA",
    "MechanicGridGeneratorBase",
    "MechanicScenario",
    "COLOR_AGENT",
    "COLOR_AGENT_DARK",
    "COLOR_BG",
    "COLOR_BOMB",
    "COLOR_BOMB_FUSE",
    "COLOR_BRIDGE",
    "COLOR_BRIDGE_DARK",
    "COLOR_CONVEYOR",
    "COLOR_CONVEYOR_DARK",
    "COLOR_DOOR_LOCKED",
    "COLOR_DOOR_OPEN",
    "COLOR_DOOR_OUTLINE",
    "COLOR_FLOOR",
    "COLOR_GOAL",
    "COLOR_GOAL_DARK",
    "COLOR_GRID",
    "COLOR_ICE",
    "COLOR_ICE_OUTLINE",
    "COLOR_KEY",
    "COLOR_KEY_OUTLINE",
    "COLOR_LABEL",
    "COLOR_LEVER_BASE",
    "COLOR_LEVER_OFF",
    "COLOR_LEVER_ON",
    "COLOR_MAGNET",
    "COLOR_MAGNET_BLUE",
    "COLOR_OIL",
    "COLOR_OIL_FLAME",
    "COLOR_PATH",
    "COLOR_PLATE_OFF",
    "COLOR_PLATE_ON",
    "COLOR_PORTAL_A",
    "COLOR_PORTAL_B",
    "COLOR_PORTAL_C",
    "COLOR_PORTAL_D",
    "COLOR_START",
    "COLOR_START_DARK",
    "COLOR_TEXT",
    "COLOR_WALL",
    "COLOR_WALL_OUTLINE",
    "COLOR_WATER",
    "COLOR_WATER_DARK",
    "COLOR_WIND",
]

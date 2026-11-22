"""Shared scaffolding for object-rearrangement task generators.

This module supplies a reusable :class:`ObjectPlanGenerator` parent class
plus a small toolbox of layered-PIL drawing primitives. Each subclass owns its
own mechanic (Tower of Hanoi, pancake sort, train-coupling reorder, etc.) but
inherits a uniform asset pipeline:

* 512×512 canvases with no captions, no panel labels, no letterboxing.
* The "before" and "after" states are conveyed by SPATIAL POSITIONING (left
  half versus right half, separated by a chevron arrow), never by text.
 * Deterministic videos. Rearrangement tasks retain their existing animation
   hooks, while Spatial Navigation subclasses can opt into exact semantic snapshots
   with a fixed hold cadence and no sub-cell transition frames.
* 4N+1 video frame counts via :func:`pad_frames_to_wan_compatible_length`.
* Lossless gbrp WebM source via :func:`encode_rgb_frames_to_webm`.
* Clean PIL primitives with a single fill, a stable outline, and only
  semantically necessary labels. Decorative shadows and glossy highlights are
  intentionally avoided because they add unstable visual detail.

The base class is deliberately neutral about *what* gets rearranged. A
subclass implements:

* :meth:`_sample_problem` — return ``(initial_state, target_state, moves)``.
* :meth:`_state_signature` — produce a hashable representation for invariant
  checks.
* :meth:`_render_state_panel` — render either the "before" or "after" state
  inside an axis-aligned bounding box. The base class handles canvas
  composition, chevron arrow, and animation.
* :meth:`_animate_move` — provide the per-move animation hook used by
  rearrangement tasks outside spatial_navigation.

The base class also offers small drawing helpers that subclasses can rely on:
``_draw_block_icon``, ``_draw_disc_icon``, ``_draw_arrow``, and ``_load_font``.
Each helper paints a layered icon (base fill, outline, top
highlight, optional label) inside an axis-aligned bbox.

The canonical answer is
the move list rendered as ``Answer: [(from, to), (from, to), ...]`` (a plain
Python-style list of tuples). Subclasses can override ``_format_move`` and
``_format_answer`` if they need non-tuple move semantics (e.g. pancake flips
take a single integer ``k``; railcar swaps name a single car ID). The default
formatter handles ``(from, to)`` tuples directly.
"""

from __future__ import annotations

from data.common.sample_identity import RECORD_NAMESPACE
from data.common.assets import render_font

import random
import tempfile
import uuid

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from data.common.base import AbstractPuzzleGenerator, PathLike
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.video_encoding import (
    MAX_VIDEO_FRAMES as VIDEO_CONTRACT_MAX_FRAMES,
    MOTION_VIDEO_FRAMES,
    SIMPLE_VIDEO_FRAMES,
    encode_rgb_frames_to_webm,
    fit_frames_to_video_budget,
)
from data.common.typed_answers import format_answer_line
from data.common.animation_policy import DISCRETE_STATE_HOLD_FRAMES

# ---------------------------------------------------------------------------
# Public canvas + animation contract
# ---------------------------------------------------------------------------

CANVAS_SIZE: int = 512
DEFAULT_FPS: int = 16
DEFAULT_HOLD_FRAMES: int = 4
DEFAULT_MOVE_FRAMES: int = 8
DEFAULT_PAUSE_FRAMES: int = 2
MAX_VIDEO_FRAMES: int = VIDEO_CONTRACT_MAX_FRAMES

CANVAS_BG: Tuple[int, int, int] = (248, 247, 244)
ARROW_FILL: Tuple[int, int, int] = (82, 112, 172)

# A 14-color saturated palette for items. Layered icons mix in lighter and
# darker variants; helpers handle lightness shifts automatically.
ITEM_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (231, 84, 76),   # red
    (243, 162, 56),  # orange
    (243, 199, 81),  # yellow
    (138, 196, 102), # green
    (84, 168, 207),  # sky
    (88, 116, 196),  # blue
    (149, 110, 196), # purple
    (213, 113, 169), # pink
    (88, 188, 178),  # teal
    (199, 138, 90),  # tan
    (108, 128, 145), # slate
    (208, 174, 132), # sand
    (90, 168, 138),  # mint
    (188, 100, 122), # rose
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shade(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    """Lighten (factor>1) or darken (factor<1) a color while keeping channels in [0,255]."""

    return tuple(max(0, min(255, int(round(c * factor)))) for c in color)  # type: ignore[return-value]


def _luminance(color: Tuple[int, int, int]) -> float:
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]


def _readable_text_color(background: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (32, 30, 26) if _luminance(background) > 145 else (250, 248, 244)


def _font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=True)


# ---------------------------------------------------------------------------
# Record dataclass
# ---------------------------------------------------------------------------

@dataclass
class ObjectPlanRecord:
    """Serializable metadata for an object-rearrangement sample."""

    id: str
    task_name: str
    task_family: str
    canvas_dimensions: Tuple[int, int]
    image: str
    reasoning_image: str
    solution_image_path: str
    ti2v_prompt: str
    seed: Optional[int]
    extra: Dict[str, Any] = field(default_factory=dict)
    solution_video_path: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None

    answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "answer": self.answer,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "canvas_dimensions": [int(self.canvas_dimensions[0]), int(self.canvas_dimensions[1])],
            "image": self.image,
            "reasoning_image": self.reasoning_image,
            "solution_image_path": self.solution_image_path,
            "solution_video_path": self.solution_video_path,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
            "ti2v_prompt": self.ti2v_prompt,
            "seed": self.seed,
        }
        for key, value in self.extra.items():
            if key not in payload:
                payload[key] = value
        return payload


# ---------------------------------------------------------------------------
# Base generator
# ---------------------------------------------------------------------------

class ObjectPlanGenerator(AbstractPuzzleGenerator[ObjectPlanRecord]):
    """Reusable parent for object-rearrangement task generators.

    Subclasses MUST implement:

    * :meth:`_sample_problem` — return a 3-tuple of
      ``(initial_state, target_state, moves)``. ``moves`` is a list whose
      element type is up to the subclass (commonly ``Tuple[int, int]`` for
      ``(from, to)`` pegs, an ``int`` for a pancake flip ``k``, or a
      ``Tuple[str, str]`` for label-based reorderings).
    * :meth:`_render_state` — render an entire state into a ``PIL.Image`` of
      a given target size. The base class composites two such renders side
      by side with a chevron arrow between them.
    * :meth:`_describe_input` — short, anchored prose used in the prompt's
      Input image section.
    * :meth:`_describe_rules` — the core mechanic, given as a few bullet-like
      sentences (the base class wraps them under "Rules:").
    * :meth:`_describe_output` — single sentence describing the output (e.g.
      "Return the move sequence as ``[(from, to), ...]`` …"). The base class
      converts that into mode-specific prompts via
      :func:`build_sectioned_prompt`.
    * :meth:`_apply_move` — apply a move to a state, returning the next state.
      Used to validate the move sequence.

    Subclasses MAY override:

    * :meth:`_animate_move` — yield frames for a single move. This hook is
      retained for rearrangement families outside spatial_navigation;
      spatial_navigation subclasses use exact semantic snapshots instead.
    * :meth:`_format_move` — turn a move into its ``(from, to)`` text form.
    * :meth:`_format_answer` — wrap the move list into the canonical
      ``Answer: [...]`` text. Subclasses that need richer answer payloads
      (e.g. include the final state) can override.
    """

    TASK_NAME: str = "object_plan_base"
    TASK_FAMILY: Optional[str] = None
    DEFAULT_OUTPUT_DIR: str = "outputs/rearrangement_game"
    # Only Spatial Navigation object-plan tasks enable this.  Other rearrangement
    # families continue to use their existing _animate_move implementations.
    SPATIAL_NAVIGATION_DISCRETE_ANIMATION: bool = False
    # Shared line-quality floor: panels allocated through _panel_canvas draw at
    # this supersample factor, stay at it through _panel_at_scene_scale, and are
    # LANCZOS-downsampled once when _compose_scene finishes the whole scene.
    SUPERSAMPLE: int = DOCTRINE_SUPERSAMPLE
    # Rearrangement tasks author their reasoning through the observation /
    # per-move / confirmation hooks (all 21 ObjectPlan tasks set this True);
    # subclasses in other families override the delivery hook instead.
    AUTHORED_REASONING_HOOKS: bool = False

    PUZZLE_INPUT_DESCRIPTION: str = (
        "The input puzzle image shows the starting arrangement on the left half of the canvas and the desired "
        "final arrangement on the right half, separated by a blue chevron arrow that points left to right."
    )

    REASONING_INPUT_DESCRIPTION: str = PUZZLE_INPUT_DESCRIPTION

    def __init__(
        self,
        output_dir: Optional[PathLike] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        canvas_size: int = CANVAS_SIZE,
        fps: int = DEFAULT_FPS,
        move_frames: int = DEFAULT_MOVE_FRAMES,
        pause_frames: int = DEFAULT_PAUSE_FRAMES,
        hold_frames: int = DEFAULT_HOLD_FRAMES,
    ) -> None:
        if not self.TASK_FAMILY:
            raise TypeError(f"{type(self).__name__} must define a non-empty TASK_FAMILY")
        resolved_output = output_dir if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        if resolved_output is None:
            resolved_output = Path(tempfile.gettempdir()) / "videogym_rearrangement_default"
        super().__init__(resolved_output)
        self.seed = seed
        self._rng = random.Random(seed)
        self.video = bool(video)
        self.canvas_size = int(canvas_size)
        self.fps = int(fps)
        self.move_frames = int(move_frames)
        self.pause_frames = int(pause_frames)
        self.hold_frames = int(hold_frames)
        root = Path(self.output_dir)
        self.puzzle_dir = root / "puzzles"
        self.solution_dir = root / "solutions"
        for directory in (self.puzzle_dir, self.solution_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._last_video_fps: Optional[int] = None
        self._last_video_num_frames: Optional[int] = None
        self._record_counter = 0

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @property
    def rng(self) -> random.Random:
        return self._rng

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        raise NotImplementedError

    def _state_signature(self, state: Any) -> Any:
        """Hashable representation of a state. Default: tuple-cast nested lists."""

        if isinstance(state, list):
            return tuple(self._state_signature(item) for item in state)
        if isinstance(state, dict):
            return tuple(sorted((k, self._state_signature(v)) for k, v in state.items()))
        return state

    def _apply_move(self, state: Any, move: Any) -> Any:
        raise NotImplementedError

    def _render_state(
        self,
        state: Any,
        panel_width: int,
        *,
        panel_height: Optional[int] = None,
        animation: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        """Render a state inside a panel of ``panel_width x panel_height``.

        Panels are drawn on a :meth:`_panel_canvas` canvas and returned through
        :meth:`_panel_at_scene_scale`, i.e. still at ``SUPERSAMPLE`` scale — the
        composed scene owns the single LANCZOS downsample.

        ``panel_height`` defaults to ``panel_width`` (square); most
        object-rearrangement layouts pass an explicit non-square height that
        fills the canvas vertically.

        ``animation`` is a free-form dict subclasses use to draw a
        moving-item overlay during a transition (the source slot can show
        an empty pedestal; the destination slot can be drawn early; etc.).
        ``animation`` is ``None`` for static rendering.
        """

        raise NotImplementedError

    def _describe_input(self) -> str:
        return self.PUZZLE_INPUT_DESCRIPTION

    def _describe_rules(self) -> str:
        raise NotImplementedError

    def _describe_output(self) -> str:
        raise NotImplementedError

    def _format_move(self, move: Any) -> str:
        if isinstance(move, tuple) and len(move) == 2:
            return f"({move[0]}, {move[1]})"
        if isinstance(move, int):
            return str(move)
        if isinstance(move, str):
            return move
        return repr(move)

    def _format_answer(self, moves: Sequence[Any]) -> str:
        """Return the payload; the shared Answer envelope is added exactly once."""

        items = ", ".join(self._format_move(m) for m in moves)
        return f"[{items}]"

    def _format_answer_from_trace(
        self,
        moves: Sequence[Any],
        state_trace: Sequence[Any],
    ) -> str:
        """Format an answer when its canonical delivery depends on visited states."""

        del state_trace
        return self._format_answer(moves)

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        return _font(size)

    def _panel_canvas(
        self,
        width: int,
        height: int,
        background: Tuple[int, int, int],
    ) -> Tuple[Image.Image, ScaledDraw]:
        """Allocate a supersampled panel canvas; callers keep 1x coordinates."""

        return new_scaled_canvas(int(width), int(height), self.SUPERSAMPLE, background)

    def _panel_at_scene_scale(self, image: Image.Image, width: int, height: int) -> Image.Image:
        """Hand a supersampled panel to the scene compositor at its own scale.

        Panels are *not* downsampled here: :meth:`_compose_scene` pastes them
        into a scene canvas allocated at the same supersample factor and takes
        the single LANCZOS pass for the whole artifact, so panel interiors,
        paste seams, and the chevron divider are all resampled exactly once.
        """

        expected = (int(width) * self.SUPERSAMPLE, int(height) * self.SUPERSAMPLE)
        if image.size != expected:
            raise ValueError(
                f"{type(self).__name__} panel is {image.size}, expected {expected}; "
                "allocate panels through _panel_canvas so the scene composes at "
                f"{self.SUPERSAMPLE}x"
            )
        return image

    def _draw_font(self, draw: Any, size: int) -> ImageFont.ImageFont:
        """Load a font matched to the draw target's supersample factor.

        ``ScaledDraw`` scales coordinates but passes fonts through, so text
        drawn through it must load its font at ``size * factor``.
        """

        return self._load_font(max(1, int(round(size * getattr(draw, "factor", 1)))))

    def _draw_block_icon(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: Tuple[float, float, float, float],
        color: Tuple[int, int, int],
        *,
        label: Optional[str] = None,
        radius: int = 8,
        outline_color: Optional[Tuple[int, int, int]] = None,
        highlight: bool = True,
    ) -> None:
        """Clean block icon with stable geometry and an optional short label."""

        x0, y0, x1, y1 = bbox
        outline_col = outline_color if outline_color is not None else _shade(color, 0.55)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=color, outline=outline_col, width=2)
        # Label
        if label is not None and str(label):
            text = str(label)
            font_size = max(12, int(round(min(x1 - x0, y1 - y0) * 0.42)))
            font = self._draw_font(draw, font_size)
            text_color = _readable_text_color(color)
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            draw.text((cx, cy - 1), text, font=font, fill=text_color, anchor="mm")

    def _draw_disc_icon(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        radius: float,
        color: Tuple[int, int, int],
        *,
        label: Optional[str] = None,
        outline_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        cx, cy = center
        outline_col = outline_color if outline_color is not None else _shade(color, 0.55)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=outline_col, width=2)
        if label is not None and str(label):
            font_size = max(12, int(round(radius * 0.95)))
            font = self._draw_font(draw, font_size)
            text_color = _readable_text_color(color)
            draw.text((cx, cy - 1), str(label), font=font, fill=text_color, anchor="mm")


    def _draw_chevron(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        size: float,
    ) -> None:
        cx, cy = center
        half = size * 0.42
        shaft_end = cx + half * 0.15
        head_tip = cx + half
        line_width = max(4, int(round(size * 0.10)))
        draw.line((cx - half, cy, shaft_end, cy), fill=ARROW_FILL, width=line_width)
        draw.polygon(
            [
                (shaft_end, cy - half * 0.52),
                (head_tip, cy),
                (shaft_end, cy + half * 0.52),
            ],
            fill=ARROW_FILL,
        )

    # ------------------------------------------------------------------
    # Composition: side-by-side panels with chevron divider
    # ------------------------------------------------------------------

    def _panel_geometry(self) -> Dict[str, int]:
        """Two equal panels sharing the canvas, with a thin vertical divider zone.

        Each panel is a rectangle that fills the full vertical extent of the
        canvas (minus a small outer margin) and roughly half the horizontal
        extent. The chevron arrow lives in a narrow vertical band centred
        between the two panels. ``panel_w`` and ``panel_h`` give the rendered
        panel dimensions; subclasses receive both numbers via :meth:`_render_state`.
        """

        cs = self.canvas_size
        outer_margin = int(getattr(self, "SCENE_OUTER_MARGIN", 12))
        chevron_band = int(getattr(self, "SCENE_CHEVRON_BAND", 44))
        panel_w = (cs - 2 * outer_margin - chevron_band) // 2
        panel_w -= panel_w % 2
        panel_h = cs - 2 * outer_margin
        panel_h -= panel_h % 2
        return {
            "canvas": cs,
            "outer_margin": outer_margin,
            "chevron_band": chevron_band,
            "panel_w": panel_w,
            "panel_h": panel_h,
            "left_panel_x0": outer_margin,
            "left_panel_y0": outer_margin,
            "right_panel_x0": outer_margin + panel_w + chevron_band,
            "right_panel_y0": outer_margin,
            "chevron_cx": outer_margin + panel_w + chevron_band // 2,
            "chevron_cy": cs // 2,
        }

    def _scene_scale(self, panel: Image.Image, geom: Dict[str, int]) -> int:
        """Supersample factor a rendered panel is expressed in.

        ``SUPERSAMPLE`` for panels handed over by :meth:`_panel_at_scene_scale`
        (the whole scene then composes supersampled), ``1`` for consumers that
        LANCZOS-downsample their own panels — those cannot be re-supersampled
        without a second resample, so their scene composes at target scale.
        """

        if panel.size == (geom["panel_w"] * self.SUPERSAMPLE, geom["panel_h"] * self.SUPERSAMPLE):
            return int(self.SUPERSAMPLE)
        if panel.size == (geom["panel_w"], geom["panel_h"]):
            return 1
        raise ValueError(
            f"{type(self).__name__} panel is {panel.size}, expected "
            f"{(geom['panel_w'], geom['panel_h'])} at 1x or {self.SUPERSAMPLE}x"
        )

    def _compose_scene(
        self,
        left_image: Image.Image,
        right_image: Image.Image,
    ) -> Image.Image:
        """Paste two panel renders into the canvas with a chevron divider.

        The chevron is the only diagonal canvas-level geometry, so it is always
        drawn through the shared supersampled primitives.  When the panels
        themselves arrive supersampled the whole scene is composed at that
        factor and LANCZOS-downsampled once; otherwise the chevron is drawn on
        a supersampled band and pasted between the finished panels.
        """

        cs = self.canvas_size
        geom = self._panel_geometry()
        chevron_scale = float(getattr(self, "SCENE_CHEVRON_SCALE", 1.1))
        chevron_size = geom["chevron_band"] * chevron_scale
        factor = self._scene_scale(left_image, geom)

        if factor > 1:
            scene, scene_draw = new_scaled_canvas(cs, cs, factor, CANVAS_BG)
            scene.paste(
                left_image,
                (geom["left_panel_x0"] * factor, geom["left_panel_y0"] * factor),
            )
            scene.paste(
                right_image,
                (geom["right_panel_x0"] * factor, geom["right_panel_y0"] * factor),
            )
            if geom["chevron_band"] > 0:
                self._draw_chevron(
                    scene_draw,
                    (geom["chevron_cx"], geom["chevron_cy"]),
                    chevron_size,
                )
            return finish_scaled_canvas(scene, (cs, cs))

        canvas = Image.new("RGB", (cs, cs), CANVAS_BG)
        canvas.paste(left_image, (geom["left_panel_x0"], geom["left_panel_y0"]))
        canvas.paste(right_image, (geom["right_panel_x0"], geom["right_panel_y0"]))
        band_w = int(geom["chevron_band"])
        if band_w > 0:
            band_image, band_draw = new_scaled_canvas(band_w, cs, self.SUPERSAMPLE, CANVAS_BG)
            self._draw_chevron(band_draw, (band_w / 2, geom["chevron_cy"]), chevron_size)
            band = finish_scaled_canvas(band_image, (band_w, cs))
            canvas.paste(band, (geom["left_panel_x0"] + geom["panel_w"], 0))
        return canvas

    def _render_static_scene(
        self,
        initial_state: Any,
        target_state: Any,
    ) -> Image.Image:
        geom = self._panel_geometry()
        left = self._render_state(initial_state, geom["panel_w"], panel_height=geom["panel_h"])
        right = self._render_state(target_state, geom["panel_w"], panel_height=geom["panel_h"])
        return self._compose_scene(left, right)

    # ------------------------------------------------------------------
    # Video assembly
    # ------------------------------------------------------------------

    def _interpolate_pixel_linear(
        self,
        src: Tuple[float, float],
        dst: Tuple[float, float],
        progress: float,
    ) -> Tuple[float, float]:
        """Return a clamped linear interpolation between two pixel positions."""

        p = max(0.0, min(1.0, float(progress)))
        return (
            src[0] + (dst[0] - src[0]) * p,
            src[1] + (dst[1] - src[1]) * p,
        )

    def _build_video_frames(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> List[np.ndarray]:
        """Default video assembler: hold initial → animate each move → hold final.

        Subclasses with bespoke animation (e.g. pancake flips that animate
        a stack of items rotating, or trains that slide laterally) should
        override :meth:`_build_video_frames` directly.
        """

        if self.SPATIAL_NAVIGATION_DISCRETE_ANIMATION:
            return self._build_discrete_spatial_navigation_video_frames(initial_state, target_state, moves)

        # Static initial scene held for hold_frames.
        initial_image = self._render_static_scene(initial_state, target_state)
        frames: List[np.ndarray] = []
        for _ in range(self.hold_frames):
            frames.append(np.array(initial_image, dtype=np.uint8))
        protected_indices: List[int] = [0, len(frames) - 1]

        current = initial_state
        for move in moves:
            move_frames = self._animate_move(current, move, target_state)
            for frame in move_frames:
                frames.append(np.array(frame, dtype=np.uint8))
            # The final motion frame is the contact/landing boundary.  Keep it
            # even when long videos are retimed to the media budget.
            if frames:
                protected_indices.append(len(frames) - 1)
            current = self._apply_move(current, move)
            after_image = self._render_static_scene(current, target_state)
            state_hold_start = len(frames)
            for _ in range(self.pause_frames):
                frames.append(np.array(after_image, dtype=np.uint8))
            protected_indices.extend((state_hold_start, len(frames) - 1))

        # Trailing hold so the solved state is clearly visible.
        for _ in range(self.hold_frames):
            frames.append(np.array(after_image if moves else initial_image, dtype=np.uint8))

        simple_move_limit = int(getattr(self, "SIMPLE_VIDEO_MOVE_LIMIT", 4))
        motion_move_limit = int(getattr(self, "MOTION_VIDEO_MOVE_LIMIT", 7))
        if len(moves) <= simple_move_limit:
            budget = SIMPLE_VIDEO_FRAMES
        elif len(moves) <= motion_move_limit:
            budget = MOTION_VIDEO_FRAMES
        else:
            # Preserve a per-action observation budget as execution grows.
            budget = max(MAX_VIDEO_FRAMES, 16 * len(moves) + 1)
        return fit_frames_to_video_budget(
            frames,
            max_frames=budget,
            final_hold_frames=max(4, self.hold_frames),
            protected_indices=protected_indices,
        )

    def _build_discrete_spatial_navigation_video_frames(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> List[np.ndarray]:
        """Render one exact scene per logical Spatial Navigation state.

        The left panel is the current semantic state and the right panel is
        the fixed target state.  A move is committed before its destination
        state is rendered, so attached counters, carried objects, and other
        stateful details change atomically with the moving object.
        """

        current = initial_state
        states = [current]
        for move in moves:
            current = self._apply_move(current, move)
            states.append(current)

        frames: List[np.ndarray] = []
        protected_indices: List[int] = []
        for state in states:
            image = self._render_static_scene(state, target_state)
            start_index = len(frames)
            array = np.array(image.convert("RGB"), dtype=np.uint8)
            frames.extend(array.copy() for _ in range(DISCRETE_STATE_HOLD_FRAMES))
            protected_indices.append(start_index)

        simple_move_limit = int(getattr(self, "SIMPLE_VIDEO_MOVE_LIMIT", 4))
        motion_move_limit = int(getattr(self, "MOTION_VIDEO_MOVE_LIMIT", 7))
        if len(moves) <= simple_move_limit:
            budget = SIMPLE_VIDEO_FRAMES
        elif len(moves) <= motion_move_limit:
            budget = MOTION_VIDEO_FRAMES
        else:
            # Preserve a per-action observation budget as execution grows.
            budget = max(MAX_VIDEO_FRAMES, 16 * len(moves) + 1)
        return fit_frames_to_video_budget(
            frames,
            max_frames=budget,
            final_hold_frames=DISCRETE_STATE_HOLD_FRAMES,
            protected_indices=protected_indices,
        )

    def _animate_move(
        self,
        state: Any,
        move: Any,
        target_state: Any,
    ) -> Iterable[Image.Image]:
        """Per-move animation hook for tasks outside spatial_navigation."""

        raise NotImplementedError(
            "Subclasses must override _animate_move when using the default _build_video_frames."
        )

    # ------------------------------------------------------------------
    # Encoding + I/O
    # ------------------------------------------------------------------

    def _encode_video(
        self,
        record_id: str,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> Tuple[Optional[Path], Optional[int]]:
        if not self.video:
            return None, None
        frames = self._build_video_frames(initial_state, target_state, moves)
        if not frames:
            return None, None
        video_path = self.solution_dir / f"{record_id}_solution.webm"
        ok = encode_rgb_frames_to_webm(
            frames,
            video_path,
            fps=self.fps,
            extra_args=["-g", "1", "-bf", "0"],
        )
        if not ok:
            return None, None
        self._last_video_fps = self.fps
        self._last_video_num_frames = len(frames)
        return video_path, len(frames)

    def _save_image(self, path: Path, image: Image.Image) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

    def next_id(self) -> str:
        # Stable across process scheduling: the CLI assigns one deterministic
        # seed per logical record, while the local counter also supports
        # direct multi-record use of one generator instance.  uuid4 made
        # otherwise byte-identical worker outputs differ only in filenames.
        identity = (
            f"{RECORD_NAMESPACE}:{self.TASK_FAMILY}:{self.TASK_NAME}:"
            f"{self.seed!r}:{self._record_counter}"
        )
        self._record_counter += 1
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_prompts(self) -> Dict[str, str]:
        return self._build_task_prompts()

    def _build_task_prompts(self) -> Dict[str, str]:
        """Build prompts through the owning Task Family's contract.

        The general object-plan pipeline owns semantic replay and artifact
        assembly, but it must not import a particular Family's vocabulary or
        Prompt grammar.  Each Family-specific subclass provides that boundary.
        """

        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_task_prompts through its Task Family"
        )

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> ObjectPlanRecord:
        record_id = puzzle_id or self.next_id()
        initial_state, target_state, moves = self._sample_problem()

        # Validate the move sequence by replay (load-bearing safety net).
        replay = initial_state
        state_trace = [initial_state]
        for m in moves:
            replay = self._apply_move(replay, m)
            state_trace.append(replay)
        if self._state_signature(replay) != self._state_signature(target_state):
            raise RuntimeError(
                f"{self.__class__.__name__}: replayed move sequence does not reach the target state."
            )

        # Static images.  The reasoning input is the puzzle image itself; no
        # separate reasoning PNG is written (a byte-identical copy would sit
        # unreferenced in the canonical export tree, which section 13.1 of the
        # doctrine forbids).
        puzzle_image = self._render_static_scene(initial_state, target_state)
        solution_image = self._render_static_scene(target_state, target_state)

        puzzle_path = self._save_image(self.puzzle_dir / f"{record_id}_puzzle.png", puzzle_image)
        solution_path = self._save_image(self.solution_dir / f"{record_id}_solution.png", solution_image)

        video_path, video_frames = self._encode_video(record_id, initial_state, target_state, moves)

        answer_text = format_answer_line(
            self._format_answer_from_trace(moves, state_trace)
        )
        self._prompt_state_trace = state_trace
        self._prompt_moves = list(moves)
        try:
            prompts = self._build_prompts()
        finally:
            del self._prompt_state_trace
            del self._prompt_moves

        extra: Dict[str, Any] = {
            "initial_state": self._serialize_state(initial_state),
            "target_state": self._serialize_state(target_state),
            "state_trace": [self._serialize_state(state) for state in state_trace],
            "move_sequence": [self._format_move(m) for m in moves],
            "moves": list(moves) if isinstance(moves, list) else list(moves),
            "move_count": len(moves),
        }
        extra.update(self._extra_payload(initial_state, target_state, moves) or {})
        record = ObjectPlanRecord(
            answer=answer_text,
            id=record_id,
            task_name=self.TASK_NAME,
            task_family=self.TASK_FAMILY,
            canvas_dimensions=(self.canvas_size, self.canvas_size),
            image=self.relativize_path(puzzle_path),
            reasoning_image=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
            ti2v_prompt=prompts["ti2v_prompt"],

            seed=self.seed,
            extra=extra,
            solution_video_path=self.relativize_path(video_path) if video_path is not None else None,
            video_fps=self._last_video_fps if video_path is not None else None,
            video_num_frames=video_frames if video_path is not None else None,
        )
        return record

    def create_random_puzzle(self) -> ObjectPlanRecord:
        return self.create_puzzle()

    def record_to_dict(self, record: ObjectPlanRecord | Dict[str, Any]) -> Dict[str, Any]:
        return super().record_to_dict(record)


    def _author_observation(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> str:
        """Open the trace: read the visible layout and name the load-bearing facts."""

        raise NotImplementedError(
            f"{type(self).__name__} enables AUTHORED_REASONING_HOOKS but does not "
            "author an observation paragraph"
        )

    def _author_move_clause(
        self,
        move_index: int,
        move: Any,
        previous_state: Any,
        next_state: Any,
        moves: Sequence[Any],
        target_state: Any,
    ) -> str:
        """Narrate one committed action: why this move, and what visibly changes."""

        raise NotImplementedError(
            f"{type(self).__name__} enables AUTHORED_REASONING_HOOKS but does not "
            "author per-move reasoning clauses"
        )

    def _author_confirmation(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> str:
        """Close the trace: confirm the reached state satisfies the goal."""

        raise NotImplementedError(
            f"{type(self).__name__} enables AUTHORED_REASONING_HOOKS but does not "
            "author a confirmation paragraph"
        )

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------

    def _serialize_state(self, state: Any) -> Any:
        """Return a JSON-serializable representation of ``state``.

        Default: convert nested lists / tuples / dicts of primitives into
        JSON-friendly Python objects.
        """

        if isinstance(state, (list, tuple)):
            return [self._serialize_state(item) for item in state]
        if isinstance(state, dict):
            return {str(k): self._serialize_state(v) for k, v in state.items()}
        if isinstance(state, (int, float, str, bool)) or state is None:
            return state
        # Fall back to repr.
        return repr(state)

    def _extra_payload(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> Optional[Dict[str, Any]]:
        return None


# ---------------------------------------------------------------------------
# Slot-stack mixin: many of our 17 tasks share the "items live in vertical
# slots" geometry (Hanoi, crane stack, dish pile, library shelf, ...). This
# mixin gives them a shared layout helper and a default animator.
# ---------------------------------------------------------------------------

class SlotStackMixin:
    """Helpers for tasks that place items inside a fixed number of vertical slots.

    Subclasses must define ``self.slot_count`` (number of slots), and the
    helpers below compute slot geometry inside an arbitrary panel.
    """

    SLOT_PEDESTAL_COLOR: Tuple[int, int, int] = (148, 130, 102)
    SLOT_PEDESTAL_TOP: Tuple[int, int, int] = (190, 168, 132)

    def _slot_layout(
        self,
        panel_width: int,
        slot_count: int,
        item_size: float,
        max_stack: int,
        *,
        panel_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        pw = int(panel_width)
        margin_x = max(8, int(round(pw * 0.05)))
        margin_top = max(8, int(round(ph * 0.06)))
        margin_bottom = max(14, int(round(ph * 0.10)))
        usable_w = pw - 2 * margin_x
        slot_w = usable_w / max(1, slot_count)
        # Pedestal sits at the bottom; items stack upward inside its slot.
        pedestal_h = max(10, int(round(ph * 0.045)))
        pedestal_y = ph - margin_bottom
        slot_floor_y = pedestal_y - pedestal_h - 2
        item_w = min(item_size, slot_w * 0.82)
        item_h = item_w * 0.55
        # Clamp item_h so the deepest stack fits between margin_top and slot_floor_y.
        usable_h = slot_floor_y - margin_top
        ideal_total = item_h * max(1, max_stack) + max(0, max_stack - 1) * max(2, item_h * 0.05)
        if ideal_total > usable_h:
            shrink = usable_h / ideal_total
            item_w *= shrink
            item_h *= shrink
        item_w = max(20.0, item_w)
        item_h = max(14.0, item_h)
        slot_centers_x = [
            margin_x + slot_w * (i + 0.5) for i in range(slot_count)
        ]
        return {
            "margin_x": margin_x,
            "margin_top": margin_top,
            "margin_bottom": margin_bottom,
            "slot_w": slot_w,
            "slot_centers_x": slot_centers_x,
            "pedestal_h": pedestal_h,
            "pedestal_y": pedestal_y,
            "slot_floor_y": slot_floor_y,
            "item_w": item_w,
            "item_h": item_h,
            "panel_width": pw,
            "panel_height": ph,
        }

    def _slot_pedestal(
        self,
        draw: ImageDraw.ImageDraw,
        layout: Dict[str, Any],
    ) -> None:
        margin_x = layout["margin_x"]
        pedestal_h = layout["pedestal_h"]
        pedestal_y = layout["pedestal_y"]
        panel_width = layout.get("panel_width", layout.get("panel_size"))
        # Pedestal as a long horizontal bar.
        x0 = margin_x // 2
        x1 = int(panel_width) - margin_x // 2
        draw.rectangle((x0, pedestal_y, x1, pedestal_y + pedestal_h), fill=self.SLOT_PEDESTAL_COLOR)
        draw.rectangle((x0, pedestal_y, x1, pedestal_y + max(2, pedestal_h // 4)), fill=self.SLOT_PEDESTAL_TOP)


__all__ = [
    "ObjectPlanGenerator",
    "ObjectPlanRecord",
    "SlotStackMixin",
    "CANVAS_SIZE",
    "DEFAULT_FPS",
    "DEFAULT_HOLD_FRAMES",
    "DEFAULT_MOVE_FRAMES",
    "DEFAULT_PAUSE_FRAMES",
    "MAX_VIDEO_FRAMES",
    "ITEM_PALETTE",
    "CANVAS_BG",
    "ARROW_FILL",
    "_shade",
    "_readable_text_color",
    "_luminance",
    "_font",
]

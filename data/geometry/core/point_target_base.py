"""Shared scaffolding for point-target option puzzles.

These puzzles feature a hidden or implicit key point on the canvas and a fixed
set of labeled candidate markers positioned nearby. Generators reuse the
candidate-placement helpers, while offline verification decodes the terminal
answer image from the highlighted marker.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import re
import uuid


import cv2
import numpy as np

from data.geometry.core.prompt_paraphrase import (workbench_active, draw_workbench, draw_survey_markers, draw_survey_rail, given_colour)
from data.common.sample_identity import RECORD_NAMESPACE
from data.common.base import AbstractPuzzleEvaluator, AbstractPuzzleGenerator, PathLike
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    as_point_pairs,
    draw_smooth_polyline,
    finish_scaled_canvas,
    new_scaled_canvas,
    scale_stroke_width,
    scale_xy,
    scaled_polyline_points,
    stroke_polyline,
)
from data.common.prompts import build_point_target_prompt
from data.common.prompts import build_still_image_prompt_from_video_description
from data.common.prompts import split_reasoning_description_and_output
from data.common.prompts import split_visual_description_and_output
from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.common.scalar_lists import format_string_list_answer
from data.geometry.core.semantic_contract import (
    SEMANTIC_PHASE_COMMAND_COUNTS,
    build_point_prompts,
    semantic_command_groups,
    semantic_phase_roles,
)
from data.common.video_encoding import (
    MOTION_VIDEO_FRAMES,
    MULTISTEP_VIDEO_FRAMES,
    SIMPLE_VIDEO_FRAMES,
    encode_rgb_frames_to_webm,
    fit_frames_to_video_budget,
)


from PIL import Image, ImageFont, ImageDraw


def _strip_video_instruction(prompt: Optional[str]) -> Optional[str]:
    if prompt is None:
        return None
    stripped = build_still_image_prompt_from_video_description(prompt)
    return stripped or prompt.strip()


_EXPLICIT_PROMPT_SECTION_RE = re.compile(
    r"(?is)(?:^|\s)(Input Image|Rule|Required Output|Output|Deliverable)\s*:\s*"
)


def _explicit_prompt_section(prompt: Optional[str], *labels: str) -> str:
    """Extract a compact labeled section from legacy inline prompts.

    Some task-local constants use ``Input Image: ... Rule: ... Required
    Output: ...`` on one logical line.  The general sentence splitter cannot
    infer those boundaries reliably, so consume the labels before building the
    family-standard three-section prompt.
    """

    value = str(prompt or "")
    matches = list(_EXPLICIT_PROMPT_SECTION_RE.finditer(value))
    wanted = {label.casefold() for label in labels}
    for index, match in enumerate(matches):
        if match.group(1).casefold() not in wanted:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        return value[match.end() : end].strip()
    return ""


def _is_translation_of(
    before: "Image.Image",
    after: "Image.Image",
    dx: int,
    dy: int,
) -> bool:
    """True when ``after`` is ``before`` slid by (dx, dy) and nothing else.

    The ink keeps clearance on every edge, so the plate's own corner is
    background and pasting the whole plate at the offset reproduces exactly
    what a rigid slide should look like.  A stroke whose position did not
    move with the rest of the figure lands somewhere else entirely and shows
    up as whole strokes' worth of disagreement.
    """

    expected = Image.new(before.mode, before.size, before.getpixel((0, 0)))
    expected.paste(before, (int(dx), int(dy)))
    wanted = (np.asarray(expected.convert("L")) < 200).astype(np.uint8)
    drawn = (np.asarray(after.convert("L")) < 200).astype(np.uint8)
    # Rounding a coordinate before the slide and after it can put a stroke's
    # tip or a glyph's edge a pixel or two elsewhere, so ink is matched within
    # that reach.  A part of the figure that did not travel with the rest is
    # a whole marker or a whole stroke away from where it belongs and has no
    # such match.
    window = np.ones((5, 5), dtype=np.uint8)
    # The strip the slide uncovers has no counterpart on the reference plate:
    # a frame-cut stroke that reaches past it there was cut by the plate's own
    # edge, so ink appearing in that strip is not evidence of anything.  Ink
    # *missing* from where the reference puts it is checked everywhere, and
    # that is the half a figure left behind fails.
    known = np.ones(drawn.shape, dtype=bool)
    if dy > 0:
        known[:dy, :] = False
    elif dy < 0:
        known[dy:, :] = False
    if dx > 0:
        known[:, :dx] = False
    elif dx < 0:
        known[:, dx:] = False
    stray = int(np.count_nonzero(drawn & ~cv2.dilate(wanted, window) & known))
    absent = int(np.count_nonzero(wanted & ~cv2.dilate(drawn, window)))
    return stray + absent <= 64


@dataclass
class Point:
    x: float
    y: float

    def to_list(self) -> List[float]:
        return [self.x, self.y]

@dataclass
class PointCandidate:
    """Serializable representation of a labeled candidate point."""

    x: float
    y: float
    label: str

    def to_dict(self) -> Dict[str, object]:
        return {"x": self.x, "y": self.y, "label": self.label}


@dataclass(frozen=True)
class Incidence:
    """One drawn primitive that the answer point lies on.

    A task whose answer sits on visible ink -- a segment, a full line, a ray, a
    circle rim, an arc, or a drawn polyline/curve -- declares that primitive
    here and hands it to :meth:`place_candidates_on_incidence`.  The placer then
    puts every distractor on the *same* ink, so "which marker is on the figure?"
    carries no signal and the option set differs only in the load-bearing
    semantics the task actually asks about.

    Each primitive exposes a uniform arc-length parameterization, so "a wrong
    position along the same primitive" is a single scalar offset and the
    perceptual margin between the answer and its neighbours (7.5) is a
    directly measurable distance rather than an emergent property of rejection
    sampling.
    """

    kind: str
    points: Tuple[Point, ...] = ()
    center: Optional[Point] = None
    radius: float = 0.0
    start_angle: float = 0.0
    sweep: float = 0.0

    # -- constructors ----------------------------------------------------
    @classmethod
    def segment(cls, start: Point, end: Point) -> "Incidence":
        return cls("segment", (start, end))

    @classmethod
    def line(cls, first: Point, second: Point) -> "Incidence":
        return cls("line", (first, second))

    @classmethod
    def ray(cls, origin: Point, through: Point) -> "Incidence":
        return cls("ray", (origin, through))

    @classmethod
    def polyline(cls, points: Sequence[Point]) -> "Incidence":
        return cls("polyline", tuple(points))

    @classmethod
    def circle(cls, center: Point, radius: float) -> "Incidence":
        return cls("circle", (), center, float(radius))

    @classmethod
    def arc(cls, center: Point, radius: float, start_angle: float, sweep: float) -> "Incidence":
        return cls("arc", (), center, float(radius), float(start_angle), float(sweep))

    # -- geometry --------------------------------------------------------
    @property
    def closed(self) -> bool:
        return self.kind == "circle"

    def _unit(self) -> Tuple[float, float, float]:
        a, b = self.points[0], self.points[1]
        dx, dy = b.x - a.x, b.y - a.y
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            raise ValueError("Incidence direction is degenerate")
        return dx / norm, dy / norm, norm

    def _cumulative(self) -> List[float]:
        totals = [0.0]
        for index in range(len(self.points) - 1):
            first, second = self.points[index], self.points[index + 1]
            totals.append(totals[-1] + math.hypot(second.x - first.x, second.y - first.y))
        return totals

    def extent(self) -> Tuple[float, float]:
        """Inclusive arc-length range that still lies on the drawn primitive."""

        if self.kind == "segment":
            return 0.0, self._unit()[2]
        if self.kind == "polyline":
            return 0.0, self._cumulative()[-1]
        if self.kind == "ray":
            return 0.0, float("inf")
        if self.kind == "line":
            return float("-inf"), float("inf")
        if self.kind == "circle":
            return 0.0, 2.0 * math.pi * self.radius
        if self.kind == "arc":
            return 0.0, abs(self.sweep) * self.radius
        raise ValueError(f"Unknown incidence kind {self.kind!r}")

    def project(self, point: Point) -> float:
        """Arc-length coordinate of ``point``'s foot on this primitive."""

        if self.kind in ("segment", "line", "ray"):
            ux, uy, length = self._unit()
            anchor = self.points[0]
            along = (point.x - anchor.x) * ux + (point.y - anchor.y) * uy
            if self.kind == "segment":
                return max(0.0, min(length, along))
            if self.kind == "ray":
                return max(0.0, along)
            return along
        if self.kind == "polyline":
            totals = self._cumulative()
            best = (float("inf"), 0.0)
            for index in range(len(self.points) - 1):
                first, second = self.points[index], self.points[index + 1]
                dx, dy = second.x - first.x, second.y - first.y
                den = dx * dx + dy * dy
                if den < 1e-12:
                    continue
                t = max(0.0, min(1.0, ((point.x - first.x) * dx + (point.y - first.y) * dy) / den))
                foot_x, foot_y = first.x + t * dx, first.y + t * dy
                gap = math.hypot(point.x - foot_x, point.y - foot_y)
                if gap < best[0]:
                    best = (gap, totals[index] + t * math.sqrt(den))
            return best[1]
        if self.kind in ("circle", "arc"):
            if self.center is None or self.radius <= 0.0:
                raise ValueError("Circular incidence needs a center and a positive radius")
            angle = math.atan2(point.y - self.center.y, point.x - self.center.x)
            if self.kind == "circle":
                return (angle % (2.0 * math.pi)) * self.radius
            direction = 1.0 if self.sweep >= 0.0 else -1.0
            delta = ((angle - self.start_angle) * direction) % (2.0 * math.pi)
            return delta * self.radius
        raise ValueError(f"Unknown incidence kind {self.kind!r}")

    def point_at(self, along: float) -> Optional[Point]:
        """Point at arc-length ``along``, or ``None`` when it leaves the ink."""

        low, high = self.extent()
        if self.kind == "circle":
            along = along % high
        elif along < low - 1e-6 or along > high + 1e-6:
            return None
        if self.kind in ("segment", "line", "ray"):
            ux, uy, _ = self._unit()
            anchor = self.points[0]
            return Point(anchor.x + ux * along, anchor.y + uy * along)
        if self.kind == "polyline":
            totals = self._cumulative()
            for index in range(len(self.points) - 1):
                if along <= totals[index + 1] + 1e-9:
                    span = totals[index + 1] - totals[index]
                    t = 0.0 if span < 1e-12 else (along - totals[index]) / span
                    first, second = self.points[index], self.points[index + 1]
                    return Point(
                        first.x + (second.x - first.x) * t,
                        first.y + (second.y - first.y) * t,
                    )
            return self.points[-1]
        if self.center is None:
            raise ValueError("Circular incidence needs a center")
        if self.kind == "circle":
            angle = along / self.radius
        else:
            direction = 1.0 if self.sweep >= 0.0 else -1.0
            angle = self.start_angle + direction * (along / self.radius)
        return Point(
            self.center.x + math.cos(angle) * self.radius,
            self.center.y + math.sin(angle) * self.radius,
        )

    def distance(self, point: Point) -> float:
        if self.kind in ("circle", "arc"):
            if self.center is None:
                raise ValueError("Circular incidence needs a center")
            radial = abs(math.hypot(point.x - self.center.x, point.y - self.center.y) - self.radius)
            if self.kind == "circle":
                return radial
            low, high = self.extent()
            along = self.project(point)
            if along <= high + 1e-6:
                return radial
            foot = self.point_at(min(max(along, low), high))
            return math.hypot(point.x - foot.x, point.y - foot.y) if foot else radial
        foot = self.point_at(self.project(point))
        if foot is None:
            return float("inf")
        return math.hypot(point.x - foot.x, point.y - foot.y)


@dataclass
class PointTargetPuzzleRecord:
    """Base record fields for point-target puzzles."""

    id: str
    ti2v_prompt: str
    canvas_dimensions: Tuple[int, int]
    margin: int
    candidates: List[PointCandidate]
    correct_option: str
    image: str
    solution_image_path: str
    point_radius: int
    line_width: int
    solution_image_prompt: Optional[str] = None
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    reasoning_image: Optional[str] = None
    solution_video_path: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "id": self.id,
            "ti2v_prompt": self.ti2v_prompt,
            "canvas_dimensions": list(self.canvas_dimensions),
            "margin": self.margin,
            # Handle list of dataclass objects
            "candidates": [c.to_dict() if hasattr(c, "to_dict") else c for c in self.candidates],
            "correct_option": self.correct_option,
            "answer": (FIVE_CHOICE_CODEC.format(self.correct_option) if len(self.correct_option) == 1 else format_string_list_answer(sorted(self.correct_option))),
            "image": self.image,
            "reasoning_image": self.reasoning_image,
            "solution_image_path": self.solution_image_path,
            "solution_video_path": self.solution_video_path,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
            "point_radius": self.point_radius,
            "line_width": self.line_width,
            "seed": self.seed,
        }
        for key, value in self.extra.items():
            if key not in payload:
                payload[key] = value
        return payload

class PointTargetPuzzleGenerator(AbstractPuzzleGenerator):
    """Base generator providing canvas configuration and candidate placement."""

    SUPERSAMPLE: int = DOCTRINE_SUPERSAMPLE
    POINT_RADIUS: int = 11
    LINE_WIDTH: int = 5
    CANDIDATE_OUTLINE_COLOR: Tuple[int, int, int] = (32, 32, 32)
    CANDIDATE_HIGHLIGHT_COLOR: Tuple[int, int, int] = (198, 24, 24)
    CANDIDATE_TEXT_COLOR: Tuple[int, int, int] = (0, 0, 0)
    CANDIDATE_BASE_FILL: Tuple[int, int, int] = (255, 255, 255)
    CANDIDATE_HIGHLIGHT_FILL: Tuple[int, int, int] = (255, 220, 220)
    CONSTRUCTION_COLOR: Tuple[int, int, int] = (20, 151, 188)
    CANDIDATE_OUTLINE_WIDTH: int = 3
    CANDIDATE_HIGHLIGHT_OUTLINE_WIDTH: int = 3
    CANDIDATE_LABEL_FONT_SIZE: int = 17
    CANDIDATE_LABEL_OFFSET_Y: int = 0
    # Extra canvas-edge buffer beyond the label/outline footprint, applied
    # only when placing candidates (not when computing anchor padding for
    # base geometry), so markers stay clear of the frame without changing
    # task-specific fit budgets.
    CANDIDATE_EDGE_BUFFER: int = 4
    # How far the finished figure may be slid off the middle of the frame,
    # as a share of the canvas' short side, and the ink clearance the slide
    # must leave on every edge.  See ``_compose_off_centre``.
    COMPOSITION_REACH_RATIO: float = 0.18
    COMPOSITION_EDGE_FLOOR: float = 6.0
    _clip_offset: Tuple[float, float] = (0.0, 0.0)
    MAX_VIDEO_FRAMES: int = MULTISTEP_VIDEO_FRAMES
    DEFAULT_OUTPUT_DIR: str = None
    DEFAULT_TI2V_PROMPT: str = None
    DEFAULT_SOLUTION_IMAGE_PROMPT: str = None
    DEFAULT_RULE_PROMPT: str = None

    def __init__(
        self,
        output_dir: PathLike,
        *,
        canvas_width: int = 512,
        aspect: Optional[float] = None,
        seed: Optional[int] = None,
        ti2v_prompt: Optional[str] = None,
        option_labels: Sequence[str] = ("A", "B", "C", "D", "E"),
        margin_ratio: float = 0.10,
        record_video: bool = False,
        point_radius: Optional[int] = None,
        line_width: Optional[int] = None,
    ) -> None:
        output_dir = output_dir if output_dir is not None else Path(self.DEFAULT_OUTPUT_DIR)
        reasoning_prompt = self.DEFAULT_RULE_PROMPT or ""
        if ti2v_prompt is not None:
            raw_ti2v_prompt = ti2v_prompt
        else:
            raw_ti2v_prompt = self.DEFAULT_TI2V_PROMPT or build_point_target_prompt(reasoning_prompt, mode="ti2v")
        super().__init__(output_dir)
        width = int(canvas_width)
        if width <= 0:
            raise ValueError("canvas_width must be positive")
        if aspect and aspect > 0:
            height = round(width / float(aspect))
        else:
            height = width
        if height <= 0:
            raise ValueError("Derived canvas height must be positive")
        self.canvas_dimensions = (width, height)
        margin_base = min(width, height)
        computed_margin = round(margin_base * max(0.0, margin_ratio))
        self.margin = max(18, computed_margin)
        self._rng = random.Random(seed)
        self._sample_index = 0
        if not option_labels:
            raise ValueError("option_labels must contain at least one label")
        self.option_labels = tuple(option_labels)
        self.seed = seed
        ti2v_input_desc, ti2v_output_desc = split_visual_description_and_output(raw_ti2v_prompt)
        explicit_ti2v_input = _explicit_prompt_section(raw_ti2v_prompt, "Input Image")
        explicit_ti2v_output = _explicit_prompt_section(
            raw_ti2v_prompt,
            "Required Output",
            "Output",
            "Deliverable",
        )
        if explicit_ti2v_input:
            ti2v_input_desc = explicit_ti2v_input
        if explicit_ti2v_output:
            ti2v_output_desc = explicit_ti2v_output
        raw_solution_image_prompt = self.DEFAULT_SOLUTION_IMAGE_PROMPT or _strip_video_instruction(raw_ti2v_prompt)
        if not raw_solution_image_prompt:
            raw_solution_image_prompt = build_point_target_prompt(reasoning_prompt, mode="solution_image")
        solution_image_input_desc, solution_image_output_desc = split_visual_description_and_output(raw_solution_image_prompt)
        raw_reasoning_prompt = reasoning_prompt
        reasoning_input_desc, reasoning_output_desc = split_reasoning_description_and_output(raw_reasoning_prompt)
        explicit_reasoning_input = _explicit_prompt_section(raw_reasoning_prompt, "Input Image")
        explicit_reasoning_rule = _explicit_prompt_section(raw_reasoning_prompt, "Rule")
        if explicit_reasoning_input:
            reasoning_input_desc = explicit_reasoning_input
        if explicit_reasoning_rule:
            reasoning_output_desc = explicit_reasoning_rule
        shared_input_desc = reasoning_input_desc or solution_image_input_desc or ti2v_input_desc
        self._geometry_input_description = shared_input_desc
        self._geometry_rule_description = reasoning_output_desc
        self._geometry_solution_description = solution_image_output_desc or str(raw_solution_image_prompt or "")
        self._geometry_video_description = ti2v_output_desc or str(raw_ti2v_prompt or "")
        self.ti2v_prompt = ""
        self.solution_image_prompt = ""
        self._refresh_geometry_prompts()
        self._candidate_font: Optional[Any] = None
        self.point_radius = int(point_radius) if point_radius is not None else int(self.POINT_RADIUS)
        if self.point_radius <= 0:
            raise ValueError("point_radius must be positive")
        self.line_width = int(line_width) if line_width is not None else int(self.LINE_WIDTH)
        if self.line_width <= 0:
            raise ValueError("line_width must be positive")
        out_root = Path(self.output_dir)
        self.puzzle_dir = out_root / "puzzles"
        self.solution_dir = out_root / "solutions"
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)
        self.record_video = record_video
        self._recording_active = False
        self._recorder: Optional[DrawingRecorder] = None

    def _point_within_canvas_coords(
        self,
        x: float,
        y: float,
        *,
        padding_x: float = 0.0,
        padding_y: Optional[float] = None,
    ) -> bool:
        if padding_y is None:
            padding_y = padding_x
        left, top, right, bottom = self.canvas_bounds()
        return (
            left + padding_x <= x <= right - padding_x
            and top + padding_y <= y <= bottom - padding_y
        )

    def inside_canvas(
        self,
        point: Point,
        *,
        padding: float = 0.0,
    ) -> bool:
        x, y = point.to_list()
        return self._point_within_canvas_coords(x, y, padding_x=padding, padding_y=padding)

    def distance(
        self,
        p1: Point,
        p2: Point,
    ) -> float:
        return math.hypot(p1.x - p2.x, p1.y - p2.y)


    def angle_between(
        self,
        angle1: float,
        angle2: float,
    ) -> float:
        return abs((angle2 - angle1 + math.pi) % (2 * math.pi) - math.pi)

    def point_on_ray(
        self,
        origin: Point,
        angle: float,
        distance: float,
    ) -> Point:
        return Point(
            x=origin.x + distance * math.cos(angle),
            y=origin.y + distance * math.sin(angle),
        )

    def project_point_onto_line(
        self,
        point: Point,
        line_point1: Point,
        line_point2: Point,
    ) -> Tuple[Point, float]:
        dx = line_point2.x - line_point1.x
        dy = line_point2.y - line_point1.y
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return line_point1, 0.0
        t = (
            (point.x - line_point1.x) * dx
            + (point.y - line_point1.y) * dy
        ) / denom
        projection = Point(
            x=line_point1.x + dx * t,
            y=line_point1.y + dy * t,
        )
        return projection, t

    def distance_point_to_line(
        self,
        point: Point,
        line_point1: Point,
        line_point2: Point,
    ) -> float:
        projection, _ = self.project_point_onto_line(point, line_point1, line_point2)
        return self.distance(point, projection)

    def angle_at_vertex(
        self,
        p1: Point,
        vertex: Point,
        p2: Point,
    ) -> float:
        v1x = p1.x - vertex.x
        v1y = p1.y - vertex.y
        v2x = p2.x - vertex.x
        v2y = p2.y - vertex.y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 <= 1e-6 or mag2 <= 1e-6:
            return 0.0
        cosine = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (mag1 * mag2)))
        return math.degrees(math.acos(cosine))

    def triangle_angles(
        self,
        p1: Point,
        p2: Point,
        p3: Point,
    ) -> Tuple[float, float, float]:
        return (
            self.angle_at_vertex(p2, p1, p3),
            self.angle_at_vertex(p1, p2, p3),
            self.angle_at_vertex(p1, p3, p2),
        )

    @property
    def canvas_short_side(self) -> int:
        width, height = self.canvas_dimensions
        return min(width, height)

    def triangle_area(
        self,
        p1: Point,
        p2: Point,
        p3: Point,
    ) -> float:
        return abs((p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)) / 2.0

    def triangle_side_lengths(
        self,
        p1: Point,
        p2: Point,
        p3: Point,
    ) -> Tuple[float, float, float]:
        return (
            self.distance(p1, p2),
            self.distance(p2, p3),
            self.distance(p3, p1),
        )

    def sample_triangle_vertices(
        self,
        *,
        jitter_ratio: float = 0.56,
        min_side_ratio: float = 0.18,
        min_area_ratio: float = 0.03,
        min_altitude_ratio: float = 0.12,
        min_angle_deg: Optional[float] = None,
        max_angle_deg: Optional[float] = None,
        forbidden_angle_windows: Sequence[Tuple[float, float]] = (),
        max_attempts: int = 999,
        validator: Optional[Callable[[Point, Point, Point], bool]] = None,
    ) -> Tuple[Point, Point, Point]:
        min_side = self.canvas_short_side * min_side_ratio
        min_area = (self.canvas_short_side ** 2) * min_area_ratio
        min_altitude = self.canvas_short_side * min_altitude_ratio
        for _ in range(max_attempts):
            p1 = self.pick_target_point(jitter_ratio)
            p2 = self.pick_target_point(jitter_ratio)
            p3 = self.pick_target_point(jitter_ratio)
            area = self.triangle_area(p1, p2, p3)
            if area < min_area:
                continue
            side_lengths = self.triangle_side_lengths(p1, p2, p3)
            if min(side_lengths) < min_side:
                continue
            longest_side = max(side_lengths)
            if longest_side <= 1e-6:
                continue
            if (2.0 * area / longest_side) < min_altitude:
                continue
            angles = self.triangle_angles(p1, p2, p3)
            if min_angle_deg is not None and min(angles) < min_angle_deg:
                continue
            if max_angle_deg is not None and max(angles) > max_angle_deg:
                continue
            if any(
                lower <= angle <= upper
                for angle in angles
                for lower, upper in forbidden_angle_windows
            ):
                continue
            if validator is not None and not validator(p1, p2, p3):
                continue
            return p1, p2, p3
        raise RuntimeError("Failed to sample a valid triangle configuration")

    def candidate_anchor_padding(
        self,
        *,
        extra: float = 0.0,
    ) -> float:
        pad_x, pad_y = self._candidate_safe_padding()
        return max(pad_x, pad_y) + float(extra)

    def minimum_candidate_spacing(
        self,
        *,
        scale: float = 1.0,
    ) -> float:
        pad_x, pad_y = self._candidate_safe_padding()
        base = max(self.point_radius * 2.8, pad_x * 1.9, pad_y * 1.9)
        return base * max(0.1, scale)

    def point_can_host_candidate(
        self,
        point: Point,
        *,
        extra_padding: float = 0.0,
    ) -> bool:
        padding = self.candidate_anchor_padding(extra=extra_padding)
        return self.inside_canvas(point, padding=padding)

    def points_are_well_spaced(
        self,
        points: Sequence[Point],
        *,
        min_distance: Optional[float] = None,
    ) -> bool:
        threshold = self.minimum_candidate_spacing() if min_distance is None else float(min_distance)
        for idx, point in enumerate(points):
            for other in points[idx + 1 :]:
                if self.distance(point, other) < threshold:
                    return False
        return True

    def validate_candidate_layout(
        self,
        candidates: Sequence[PointCandidate],
        *,
        min_distance: Optional[float] = None,
    ) -> bool:
        if not all(self._candidate_fits(candidate) for candidate in candidates):
            return False
        points = [Point(candidate.x, candidate.y) for candidate in candidates]
        threshold = self.minimum_candidate_spacing(scale=0.82) if min_distance is None else min_distance
        return self.points_are_well_spaced(points, min_distance=threshold)

    def circle_fits(
        self,
        center: Point,
        radius: float,
        *,
        extra_padding: float = 0.0,
    ) -> bool:
        return self._point_within_canvas_coords(
            center.x,
            center.y,
            padding_x=radius + extra_padding,
            padding_y=radius + extra_padding,
        )

    def sample_point_along_direction(
        self,
        origin: Point,
        angle: float,
        *,
        min_distance: float,
        max_distance: Optional[float] = None,
        padding: float = 0.0,
    ) -> Point:
        budget = self._max_travel_distance(origin, angle, padding_x=padding, padding_y=padding)
        if budget < min_distance:
            raise RuntimeError("No feasible distance along direction inside canvas")
        safe_budget = max(0.0, budget * 0.98)
        upper = safe_budget if max_distance is None else min(safe_budget, max_distance)
        if upper < min_distance:
            raise RuntimeError("Requested distance range does not fit inside canvas")
        distance = self._rng.uniform(min_distance, upper)
        return self.point_on_ray(origin, angle, distance)

    def sample_symmetric_segment(
        self,
        midpoint: Point,
        angle: float,
        *,
        min_half_length: float,
        max_half_length: Optional[float] = None,
        padding: float = 0.0,
    ) -> Tuple[Point, Point]:
        forward_budget = self._max_travel_distance(
            midpoint, angle, padding_x=padding, padding_y=padding,
        )
        backward_budget = self._max_travel_distance(
            midpoint, angle + math.pi, padding_x=padding, padding_y=padding,
        )
        symmetric_budget = min(forward_budget, backward_budget)
        if symmetric_budget < min_half_length:
            raise RuntimeError("No feasible symmetric segment fits inside canvas")
        upper = symmetric_budget if max_half_length is None else min(symmetric_budget, max_half_length)
        if upper < min_half_length:
            raise RuntimeError("Requested symmetric segment range does not fit inside canvas")
        half_length = self._rng.uniform(min_half_length, upper * 0.98 if upper > min_half_length else upper)
        return (
            self.point_on_ray(midpoint, angle + math.pi, half_length),
            self.point_on_ray(midpoint, angle, half_length),
        )


    def clip_line_to_canvas(
        self,
        anchor: Point,
        angle: float,
        *,
        padding: float = 0.0,
    ) -> Tuple[Point, Point]:
        forward = self._max_travel_distance(anchor, angle, padding_x=padding, padding_y=padding)
        backward = self._max_travel_distance(anchor, angle + math.pi, padding_x=padding, padding_y=padding)
        return (
            self.point_on_ray(anchor, angle + math.pi, backward),
            self.point_on_ray(anchor, angle, forward),
        )

    def _candidate_safe_padding(
        self,
        label: Optional[str] = None,
        *,
        include_edge_buffer: bool = False,
    ) -> Tuple[float, float]:
        font = self._get_candidate_font()
        labels = [label] if label is not None else list(self.option_labels)
        max_text_width = 0
        max_text_height = 0
        for item in labels:
            text_bbox = font.getbbox(item)
            max_text_width = max(max_text_width, text_bbox[2] - text_bbox[0])
            max_text_height = max(max_text_height, text_bbox[3] - text_bbox[1])
        pad_x = max(
            self.point_radius + self.CANDIDATE_HIGHLIGHT_OUTLINE_WIDTH,
            math.ceil(max_text_width / 2),
        )
        pad_y = max(
            self.point_radius + self.CANDIDATE_HIGHLIGHT_OUTLINE_WIDTH,
            max_text_height + max(0, -self.CANDIDATE_LABEL_OFFSET_Y),
        )
        # The extra edge buffer is opt-in so task-specific generators'
        # anchor-placement budgets stay intact; it is applied only by
        # `_candidate_fits` when validating a specific candidate's position
        # and by the candidate-placement helpers themselves.
        buffer = float(self.CANDIDATE_EDGE_BUFFER) if include_edge_buffer else 0.0
        return float(pad_x + 1) + buffer, float(pad_y + 1) + buffer

    def _candidate_fits(self, candidate: PointCandidate) -> bool:
        pad_x, pad_y = self._candidate_safe_padding(
            candidate.label, include_edge_buffer=True,
        )
        return self._point_within_canvas_coords(
            candidate.x,
            candidate.y,
            padding_x=pad_x,
            padding_y=pad_y,
        )

    def _max_travel_distance(
        self,
        origin: Point,
        angle: float,
        *,
        padding_x: float = 0.0,
        padding_y: Optional[float] = None,
    ) -> float:
        if padding_y is None:
            padding_y = padding_x
        left, top, right, bottom = self.canvas_bounds()
        # A stroke that runs until the frame stops it has no ends of its own,
        # so for a slid figure to keep the same shape it is the frame that has
        # to travel with it.  ``_clip_offset`` moves the frame this reach is
        # measured against; only ``_compose_off_centre`` sets it, to draw the
        # plate its slide is checked against.
        offset_x, offset_y = self._clip_offset
        left += offset_x
        right += offset_x
        top += offset_y
        bottom += offset_y
        dx = math.cos(angle)
        dy = math.sin(angle)
        limits: List[float] = []
        if dx > 1e-6:
            limits.append((right - padding_x - origin.x) / dx)
        elif dx < -1e-6:
            limits.append((left + padding_x - origin.x) / dx)
        if dy > 1e-6:
            limits.append((bottom - padding_y - origin.y) / dy)
        elif dy < -1e-6:
            limits.append((top + padding_y - origin.y) / dy)
        positives = [value for value in limits if value >= 0.0]
        return min(positives) if positives else 0.0

    def pick_target_point(
        self,
        jitter_ratio: float = 0.36,
        padding: float = 0.0,
    ) -> Point:
        jitter_ratio/=2 # jitter_ratio = 1 means full spread across the canvas
        left, top, right, bottom = self.canvas_bounds()
        left += padding
        top += padding
        right -= padding
        bottom -= padding
        if left >= right or top >= bottom:
            raise ValueError("padding leaves no drawable area on canvas")
        width, height = right - left, bottom - top
        center_x = left + width * 0.5
        center_y = top + height * 0.5
        jitter_x = self._rng.uniform(-jitter_ratio * width, jitter_ratio * width)
        jitter_y = self._rng.uniform(-jitter_ratio * height, jitter_ratio * height)
        x = center_x + jitter_x
        y = center_y + jitter_y
        return Point(x, y)

    # ------------------------------------------------------------------
    # Candidate cloud seating

    # Every candidate layout in this family is built shape first.  The
    # arrangement of the markers is drawn without knowing which of them will
    # be the answer; only then is one of its slots picked uniformly, and the
    # whole rigid arrangement translated so that slot lands on the answer.

    # That order is what makes the layout answer-free.  A reading that looks
    # only at where the markers sit relative to one another -- distance to the
    # cloud's centroid, extremeness along an axis, rank in reading order,
    # isolation from the nearest neighbour -- is invariant under translation,
    # so it names a slot the shape alone already fixed.  The answer occupies
    # that slot exactly once in ``len(option_labels)`` layouts.  Seating the
    # answer first and scattering distractors around it, as this family used
    # to, puts the answer at the middle of its own cloud by construction and
    # hands every one of those readings the answer for free.

    # Feasibility is kept independent of the chosen slot: a shape is accepted
    # only when *all* of its slots can host the answer, so a layout is never
    # discarded for a reason that correlates with where the answer sits.
    # ------------------------------------------------------------------

    CLOUD_SHAPES: Tuple[str, ...] = ("ring", "ladder", "arc", "stagger")

    def cloud_shape_offsets(
        self,
        kind: str,
        count: int,
        spacing: float,
        *,
        angle: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Local offsets for one deliberate marker arrangement, centroid at 0.

        ``spacing`` is the distance between neighbouring markers, so every
        shape here holds at least that gap between any two of its members.
        """

        if count <= 0:
            raise ValueError("A candidate cloud needs at least one marker")
        if count == 1:
            return [(0.0, 0.0)]
        points: List[Tuple[float, float]]
        if kind == "ladder":
            points = [(index * spacing, 0.0) for index in range(count)]
        elif kind == "ring":
            radius = spacing / (2.0 * math.sin(math.pi / count)) if count > 2 else spacing / 2.0
            step = math.tau / count
            points = [
                (radius * math.cos(index * step), radius * math.sin(index * step))
                for index in range(count)
            ]
        elif kind == "arc":
            sweep = self._rng.uniform(math.radians(70.0), math.radians(150.0))
            step = sweep / (count - 1)
            radius = spacing / (2.0 * math.sin(step / 2.0))
            points = [
                (radius * math.cos(index * step), radius * math.sin(index * step))
                for index in range(count)
            ]
        elif kind == "stagger":
            front = (count + 1) // 2
            # 0.92 spacing between the rows keeps the diagonal neighbours a
            # full spacing apart once the rows are offset by half a step.
            points = [(index * spacing, 0.0) for index in range(front)]
            points += [
                ((index + 0.5) * spacing, spacing * 0.92) for index in range(count - front)
            ]
        else:
            raise ValueError(f"Unknown candidate cloud shape {kind!r}")
        theta = self._rng.uniform(0.0, math.tau) if angle is None else float(angle)
        center_x = sum(item[0] for item in points) / count
        center_y = sum(item[1] for item in points) / count
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        return [
            (
                (x - center_x) * cos_t - (y - center_y) * sin_t,
                (x - center_x) * sin_t + (y - center_y) * cos_t,
            )
            for x, y in points
        ]

    @staticmethod
    def _cloud_min_gap(offsets: Sequence[Tuple[float, float]]) -> float:
        if len(offsets) < 2:
            return float("inf")
        return min(
            math.hypot(offsets[i][0] - offsets[j][0], offsets[i][1] - offsets[j][1])
            for i in range(len(offsets))
            for j in range(i + 1, len(offsets))
        )

    def _cloud_seats_slot(
        self,
        true_point: Point,
        offsets: Sequence[Tuple[float, float]],
        slot: int,
        accept: Optional[Callable[[Point], bool]],
    ) -> Optional[List[Point]]:
        """The markers for one seating, or ``None`` when it does not fit."""

        label = self.option_labels[0]
        anchor = offsets[slot]
        seating: List[Point] = []
        for index, (ox, oy) in enumerate(offsets):
            point = Point(true_point.x + ox - anchor[0], true_point.y + oy - anchor[1])
            if not self._candidate_fits(PointCandidate(x=point.x, y=point.y, label=label)):
                return None
            if index != slot and accept is not None and not accept(point):
                return None
            seating.append(point)
        return seating

    def _answer_rank(self, seating: Sequence[Point], slot: int, key: Callable[[Point], float]) -> int:
        """How many markers ``key`` puts before the answer."""

        value = key(seating[slot])
        return sum(1 for index, point in enumerate(seating) if index != slot and key(point) < value)

    def seat_candidates(
        self,
        true_point: Point,
        *,
        accept: Optional[Callable[[Point], bool]] = None,
        min_separation: Optional[float] = None,
        spacing: Optional[float] = None,
        shapes: Optional[Sequence[str]] = None,
        angle: Optional[float] = None,
        attempts: int = 24,
    ) -> None:
        """Seat the option markers so the answer's slot carries no information.

        ``accept`` vets a *distractor* position -- the room a task allows its
        wrong answers to occupy (inside the triangle, clear of an anchor, not a
        second valid answer).  It is applied to every slot the answer does not
        occupy, in every seating, so it can never single the answer out.

        Preferred path: one uniformly drawn slot, and only arrangements that
        would have fitted at *any* slot.  Feasibility then says nothing about
        where the answer sits, and every reading that survives translation --
        distance to the cloud's centroid, extremeness within the cloud, rank in
        reading order, isolation -- names a slot the shape alone fixed, which
        the answer holds exactly 1/N of the time.

        Fallback path, for a figure with no room for that (a marker cloud
        inside a small triangle needs twice its own width to be seatable at
        both ends): each attempt draws its own slot and is kept if it fits
        there.  Feasibility is then slot-dependent, so the answer's rank by
        distance to the cloud centroid is steered to a uniformly drawn target
        instead of being inherited from the uniform slot.

        Either way the answer's rank by distance to the canvas centre is
        steered the same way, because that reading survives the translation
        that seats the cloud and a figure built around the middle of the frame
        would otherwise leave the answer the innermost marker.
        """

        labels = list(self.option_labels)
        count = len(labels)
        if count == 0:
            raise RuntimeError("option_labels must contain at least one label")
        if not self.point_can_host_candidate(true_point):
            raise RuntimeError("Candidate anchor is too close to the canvas boundary")
        gap = self.minimum_candidate_spacing() if min_separation is None else float(min_separation)
        kinds = tuple(shapes) if shapes else self.CLOUD_SHAPES
        width, height = self.canvas_dimensions
        middle = Point(width / 2.0, height / 2.0)
        rounds = max(1, int(attempts))

        def shrink_to_fit(
            offsets: List[Tuple[float, float]],
            slots: Sequence[int],
        ) -> Optional[Tuple[List[Tuple[float, float]], List[List[Point]]]]:
            # Shrinking is a whole-cloud operation, so it cannot depend on
            # which slot the answer will take; a tight figure gets a tighter
            # cloud rather than an answer pushed back toward the middle.
            for step in range(13):
                shrink = 1.0 - step * 0.04
                layout = [(x * shrink, y * shrink) for x, y in offsets]
                if self._cloud_min_gap(layout) < gap:
                    return None
                seatings = [
                    self._cloud_seats_slot(true_point, layout, slot, accept) for slot in slots
                ]
                if all(seating is not None for seating in seatings):
                    return layout, [seating for seating in seatings if seating is not None]
            return None

        def draw_shape() -> List[Tuple[float, float]]:
            reach = (
                gap * self._rng.uniform(1.04, 1.34) if spacing is None else float(spacing)
            )
            return self.cloud_shape_offsets(self._rng.choice(kinds), count, reach, angle=angle)

        every = list(range(count))
        slot = self._rng.randrange(count)
        pool: List[Tuple[List[Tuple[float, float]], int, List[Point]]] = []
        for _ in range(rounds):
            fitted = shrink_to_fit(draw_shape(), every)
            if fitted is not None:
                layout, seatings = fitted
                pool.append((layout, slot, seatings[slot]))
        steer_centroid = not pool
        if steer_centroid:
            # A figure this tight is where the steering has the least to work
            # with, so it is given the most shapes to choose from: the answer's
            # rank in its own cloud is only as uniform as the pool allows.
            for _ in range(rounds * 4):
                offsets = draw_shape()
                # Every slot of the shape is tried, not one drawn slot: which
                # slots a tight figure can host is exactly what the steering
                # below has to work against, and a pool that saw one slot per
                # shape rarely holds a layout with the answer at the outside
                # of its own cloud.
                for loose in every:
                    fitted = shrink_to_fit(offsets, (loose,))
                    if fitted is not None:
                        layout, seatings = fitted
                        pool.append((layout, loose, seatings[0]))
        if not pool:
            raise RuntimeError("Failed to seat a candidate cloud around the answer")

        keys: List[Callable[[Point], float]] = [lambda point: self.distance(point, middle)]
        if steer_centroid:
            def centroid_key(seating: Sequence[Point]) -> Callable[[Point], float]:
                cx = sum(point.x for point in seating) / len(seating)
                cy = sum(point.y for point in seating) / len(seating)
                return lambda point: self.distance(point, Point(cx, cy))

        targets = [self._rng.randrange(count) for _ in range(2 if steer_centroid else 1)]
        scored: List[Tuple[float, int]] = []
        for index, (_layout, seat, seating) in enumerate(pool):
            miss = abs(self._answer_rank(seating, seat, keys[0]) - targets[0])
            if steer_centroid:
                miss += abs(
                    self._answer_rank(seating, seat, centroid_key(seating)) - targets[1]
                )
            scored.append((miss, index))
        best = min(miss for miss, _ in scored)
        layout, seat, _seating = pool[self._rng.choice([index for miss, index in scored if miss == best])]
        self._commit_candidate_cloud(true_point, layout, labels, gap, seat)

    def _commit_candidate_cloud(
        self,
        true_point: Point,
        layout: Sequence[Tuple[float, float]],
        labels: List[str],
        gap: float,
        slot: int,
    ) -> None:
        anchor = layout[slot]
        self._rng.shuffle(labels)
        correct_label = labels[slot]
        candidates = [
            PointCandidate(
                x=true_point.x + ox - anchor[0],
                y=true_point.y + oy - anchor[1],
                label=labels[index],
            )
            for index, (ox, oy) in enumerate(layout)
        ]
        # Stored in label order: the record must not reveal the answer by
        # keeping it at a fixed index of the candidate list.
        candidates.sort(key=lambda candidate: candidate.label)
        if not self.validate_candidate_layout(candidates, min_distance=gap * 0.95):
            raise RuntimeError("Candidate layout exceeds canvas or becomes too crowded")
        self.candidates, self.correct_label = candidates, correct_label

    def place_candidates_line(self, true_point: Point, angle: float | None = None) -> None:
        """Seat the markers as an evenly spaced rung ladder along ``angle``.

        The answer takes a uniformly random rung.  The rung spacing is cut to
        whatever the canvas allows on *both* sides, so every rung stays
        reachable; picking the widest spacing each rung could individually
        afford is what used to leave only the middle rung feasible near an
        edge, and with it the answer at the middle of its own ladder.
        """

        labels = list(self.option_labels)
        count = len(labels)
        if count == 0:
            raise RuntimeError("option_labels must contain at least one label")
        if angle is None:
            angle = self._rng.uniform(0.0, math.tau)
        pad_x, pad_y = self._candidate_safe_padding(include_edge_buffer=True)
        if not self._point_within_canvas_coords(
            true_point.x, true_point.y, padding_x=pad_x, padding_y=pad_y
        ):
            raise RuntimeError("Candidate anchor is too close to the canvas boundary")
        default_spread = self.minimum_candidate_spacing()
        min_spread = max(default_spread * 0.92, self.point_radius * 1.8)
        forward = self._max_travel_distance(true_point, angle, padding_x=pad_x, padding_y=pad_y)
        backward = self._max_travel_distance(
            true_point, angle + math.pi, padding_x=pad_x, padding_y=pad_y
        )
        room = min(forward, backward) * 0.99
        spread = default_spread if count < 2 else min(default_spread, room / (count - 1))
        if spread < min_spread:
            raise RuntimeError("Failed to fit line candidates within the canvas")
        self.seat_candidates(
            true_point,
            shapes=("ladder",),
            angle=angle,
            spacing=spread,
            min_separation=min_spread * 0.95,
            attempts=1,
        )

    def check_candidates_inside(self)->bool:
        return all(self._candidate_fits(candidate) for candidate in self.candidates)

    def candidate_incidence_is_answer_free(
        self,
        primitives: Sequence[Incidence],
        *,
        reach: Optional[float] = None,
        band: float = 2.0,
    ) -> bool:
        """True when "the marker on the figure" cannot single out the answer.

        Tasks whose answer is *not* on any drawn primitive still draw ink that
        a marker can happen to land on.  When that coincidence hits the correct
        marker alone, the sample answers itself exactly as a designed incidence
        shortcut would, so those generators pass their drawn
        primitives here and redraw the layout when this returns ``False``.

        The default ``reach`` is the marker's radius plus the stroke's half
        width -- the distance at which the ink visibly meets the marker disk.
        ``band`` widens the test in the safe direction on both sides: the answer
        counts as touching a little sooner and a distractor counts as touching a
        little later, so a marker sitting exactly on the threshold can never be
        credited as cover by a hair.
        """

        limit = (
            self.point_radius + self.line_width / 2.0
            if reach is None
            else float(reach)
        )

        def nearest(candidate: PointCandidate) -> float:
            point = Point(candidate.x, candidate.y)
            return min(
                (incidence.distance(point) for incidence in primitives),
                default=float("inf"),
            )

        answer_reach = float("inf")
        distractor_reaches: List[float] = []
        for candidate in self.candidates:
            gap = nearest(candidate)
            if candidate.label == self.correct_label:
                answer_reach = gap
            else:
                distractor_reaches.append(gap)
        if answer_reach > limit + band:
            return True
        return any(gap <= limit - band for gap in distractor_reaches)

    def place_candidates_on_incidence(
        self,
        true_point: Point,
        incidences: Sequence[Incidence],
        *,
        min_offset: Optional[float] = None,
        max_offset: Optional[float] = None,
        min_separation: Optional[float] = None,
        reject: Optional[Callable[[Point], bool]] = None,
        tolerance: float = 2.0,
    ) -> None:
        """Seat the option markers *along the ink the answer sits on*.

        ``incidences`` names every drawn primitive that passes through
        ``true_point``.  Distractors are dealt round-robin across those
        primitives and displaced along each one, so no option can be discarded
        by the answer-free observation "it is not on the figure".  When the answer is the meeting point of two primitives, each
        distractor still satisfies at least one of them while sitting at a
        position the task's rule rejects.

        ``min_offset`` is the perceptual margin of 7.5: no distractor lands
        closer than that along its primitive, and the usual Euclidean
        ``min_separation`` gate plus :meth:`validate_candidate_layout` still
        apply, so tightening incidence never crowds the layout.  ``reject``
        lets a task veto positions that would become a second valid answer.

        How many distractors sit *behind* the answer along the ink is drawn
        first, uniformly, and the markers are then packed outward onto the
        primitive to realise it.  Displacing distractors to either side of the
        answer -- what "scatter them along the primitive" amounts to -- parks
        the answer in the middle of its own row and answers the sample outright
       .  Choosing the rank first also makes the answer's
        position *along* a bounded segment or arc uniform rather than clustered
        at its centre.  The packing walks outward from the near margin and
        takes the first admissible slot, so a layout is constructed rather than
        fished for, and the ranks a primitive cannot host are known before any
        marker is placed instead of after a rejection budget burns out (7.7).
        """

        if not incidences:
            raise ValueError("place_candidates_on_incidence needs at least one incidence")
        for incidence in incidences:
            offset = incidence.distance(true_point)
            if offset > tolerance:
                raise ValueError(
                    f"Declared {incidence.kind} incidence does not pass through the answer "
                    f"({offset:.2f} px away)"
                )
        if not self.point_can_host_candidate(true_point):
            raise RuntimeError("Candidate anchor is too close to the canvas boundary")

        labels = list(self.option_labels)
        count = len(labels)
        gap = self.minimum_candidate_spacing() if min_separation is None else float(min_separation)
        near = gap if min_offset is None else max(gap, float(min_offset))
        ceiling = None if max_offset is None else max(0.0, float(max_offset))
        order = list(incidences)
        self._rng.shuffle(order)
        bases = [incidence.project(true_point) for incidence in order]
        probe_label = labels[0]

        def probe(index: int, offset: float) -> Optional[Point]:
            incidence = order[index]
            low, high = incidence.extent()
            if incidence.closed:
                # Half the circumference is the whole primitive: past that the
                # offset comes back round the other way and stops meaning
                # "further along the ink".
                if abs(offset) > high * 0.5:
                    return None
            else:
                along = bases[index] + offset
                if along < low - 1e-9 or along > high + 1e-9:
                    return None
            if ceiling is not None and abs(offset) > ceiling:
                return None
            point = incidence.point_at(bases[index] + offset)
            if point is None:
                return None
            if not self._candidate_fits(
                PointCandidate(x=point.x, y=point.y, label=probe_label)
            ):
                return None
            if reject is not None and reject(point):
                return None
            return point

        stride = max(1.0, gap / 6.0)
        # A line or a ray reaches past the frame, so the walk outward is
        # bounded by the canvas itself: no admissible marker can sit further
        # than the diagonal, and an unbounded horizon would never terminate.
        reachable = math.hypot(*self.canvas_dimensions)
        span = min(
            reachable,
            max(
                (
                    incidence.extent()[1] * (0.5 if incidence.closed else 1.0)
                    for incidence in order
                ),
                default=0.0,
            ),
        )
        horizon = span if ceiling is None else min(span, ceiling)

        # The row is packed at one spacing throughout, the answer's own
        # clearance included.  Giving the answer a wider berth than its
        # neighbours give each other -- which is what happens when a task
        # declares a ``min_offset`` above the marker spacing -- makes it the
        # most isolated marker of the row and hands the sample over on that
        # alone.  7.5's perceptual margin is whatever
        # ``min_offset`` declares, so the spacing never squeezes past it, nor
        # past the point where markers would start to crowd each other.
        floor = max(
            self.point_radius * 2.4,
            near if min_offset is None else float(min_offset),
        )

        def pack(behind: int, slack: float, squeeze: float = 1.0) -> Optional[List[Point]]:
            """Walk outward on each side and take the first admissible slot."""

            step = first = max(near * squeeze, floor)
            placed: List[Point] = []
            slot = 0
            for sign, wanted in ((-1.0, behind), (1.0, count - 1 - behind)):
                reach = first
                for _ in range(wanted):
                    if slack > 0.0:
                        reach += self._rng.uniform(0.0, step * slack)
                    found: Optional[Point] = None
                    while reach <= horizon + 1e-9:
                        for turn in range(len(order)):
                            point = probe((slot + turn) % len(order), sign * reach)
                            if point is None:
                                continue
                            # The walk lands a marker at exactly the spacing
                            # it asked for, so these have to be compared with
                            # a rounding tolerance: refusing a marker that
                            # misses by a billionth of a pixel pushes the
                            # whole row out by one stride and can cost the
                            # outermost rank the room it needs.
                            if any(
                                self.distance(point, other) < step - 1e-6 for other in placed
                            ):
                                continue
                            if self.distance(point, true_point) < first - 1e-6:
                                continue
                            found = point
                            break
                        if found is not None:
                            break
                        reach += stride
                    if found is None:
                        return None
                    placed.append(found)
                    reach += step
                    slot += 1
            return placed

        # Which ranks the ink can host is a property of the ink, not of the
        # answer, so it is settled before the rank is drawn.  A stretch of ink
        # too short for the row at full spacing can host every rank at a
        # tighter one, and tightening is a whole-row operation that says
        # nothing about which rung the answer takes; keeping the widest
        # spacing instead drops the outer ranks, and with them the answer
        # never being the first or the last marker in its own row

        squeeze = 1.0
        feasible: List[int] = []
        for attempt in range(6):
            squeeze = 1.0 - attempt * 0.05
            feasible = [
                behind for behind in range(count) if pack(behind, slack=0.0, squeeze=squeeze)
            ]
            if len(feasible) == count:
                break
        if not feasible:
            raise RuntimeError("Failed to place candidates along the declared incidences")
        behind = self._rng.choice(feasible)
        # The rungs are not left exactly even: with every gap identical, every
        # marker's nearest neighbour is the same distance away, and "the most
        # isolated marker" becomes a tie that some reading has to break.  The
        # jitter is tried at shrinking widths so a row that only just fits
        # still gets some, instead of dropping straight to an even ladder.
        points = None
        for slack in (0.45, 0.45, 0.30, 0.15, 0.07):
            points = pack(behind, slack=slack, squeeze=squeeze)
            if points is not None:
                break
        if points is None:
            points = pack(behind, slack=0.0, squeeze=squeeze)
        if points is None:
            raise RuntimeError("Failed to place candidates along the declared incidences")

        self._rng.shuffle(labels)
        candidates = [PointCandidate(x=true_point.x, y=true_point.y, label=labels[0])]
        candidates += [
            PointCandidate(x=point.x, y=point.y, label=labels[index + 1])
            for index, point in enumerate(points)
        ]
        if not self.validate_candidate_layout(
            candidates, min_distance=max(near * squeeze, floor) * 0.95,
        ):
            raise RuntimeError("Failed to place candidates along the declared incidences")
        candidates.sort(key=lambda candidate: candidate.label)
        self.candidates = candidates
        self.correct_label = labels[0]

    def place_candidates(
        self,
        true_point: Point,
        *,
        accept: Optional[Callable[[Point], bool]] = None,
        min_separation: Optional[float] = None,
    ) -> None:
        """Seat the markers as one deliberate cloud around the answer's slot."""

        self.seat_candidates(
            true_point,
            accept=accept,
            min_separation=(
                max(self.point_radius * 2.6, self.minimum_candidate_spacing() * 0.95)
                if min_separation is None
                else min_separation
            ),
        )

    @classmethod
    def draw_candidate_markers(
        cls,
        draw: Any,
        candidates: Sequence[PointCandidate],
        *,
        point_radius: float,
        highlight_label: Optional[str] = None,
    ) -> None:
        """Paint candidate overlays from the shared renderer contract.

        Keeping the pixel primitive independent of generator instance state
        lets source generation and independent delivery verification replay the
        same public marker presentation without constructing a task generator.
        """
        if workbench_active():
            draw_survey_markers(cls, draw, candidates, point_radius, highlight_label)
            return
        if ImageDraw is None:
            raise RuntimeError("Pillow is required to draw candidates but is not installed")

        # Resolve supersample factor and raw draw for manual coordinate work.
        if isinstance(draw, ScaledDraw):
            ss = draw.factor
            raw_draw = draw.raw
        else:
            ss = 1
            raw_draw = draw

        font = ImageFont.load_default(round(cls.CANDIDATE_LABEL_FONT_SIZE * ss))
        active_highlights = {
            label.strip().upper()
            for label in str(highlight_label or "")
            if label.strip()
        }

        rendered_radius = round(float(point_radius) * ss)
        for candidate in sorted(candidates, key=lambda c: c.label):
            cx = round(candidate.x * ss)
            cy = round(candidate.y * ss)
            bbox = (
                cx - rendered_radius,
                cy - rendered_radius,
                cx + rendered_radius,
                cy + rendered_radius,
            )
            is_highlight = candidate.label.upper() in active_highlights
            outline = (
                cls.CANDIDATE_HIGHLIGHT_COLOR
                if is_highlight
                else cls.CANDIDATE_OUTLINE_COLOR
            )
            width = round(
                (
                    cls.CANDIDATE_HIGHLIGHT_OUTLINE_WIDTH
                    if is_highlight
                    else cls.CANDIDATE_OUTLINE_WIDTH
                )
                * ss
            )
            fill = (
                cls.CANDIDATE_HIGHLIGHT_FILL
                if is_highlight
                else cls.CANDIDATE_BASE_FILL
            )
            raw_draw.ellipse(bbox, fill=fill, outline=outline, width=width)
            text_bbox = font.getbbox(candidate.label)
            # Center the glyph on (cx, cy). getbbox returns (x0, y0, x1, y1)
            # where (x0, y0) is the top-left of the ink box relative to the
            # text origin — so centering requires offsetting by the bbox
            # midpoint, not by the height alone.
            tx = cx - (text_bbox[0] + text_bbox[2]) // 2
            ty = (
                cy
                - (text_bbox[1] + text_bbox[3]) // 2
                + round(cls.CANDIDATE_LABEL_OFFSET_Y * ss)
            )
            raw_draw.text(
                (tx, ty),
                candidate.label,
                fill=cls.CANDIDATE_TEXT_COLOR,
                font=font,
            )

    def draw_candidates(
        self,
        draw: Any,
        *,
        highlight_label: Optional[str] = None,
    ) -> None:
        if isinstance(draw, DrawingRecorder):
            draw.add_high_level_command("draw_candidates", highlight_label=highlight_label)
            return
        self.draw_candidate_markers(
            draw,
            self.candidates,
            point_radius=self.point_radius,
            highlight_label=highlight_label,
        )

    def draw_line(
        self,
        draw,
        points: List[Point],
        width_factor: float = 1,
        *,
        fill: Optional[Tuple[int, int, int]] = None,
        join_start: bool = False,
        join_end: bool = False,
    ) -> None:
        """Draw a polyline as a thick stroke with butt caps at endpoints.

        Universal vertex-drawing convention:
        - **Interior joints** of the polyline are **mitered**: the stroke is
          rendered as its offset outline (see :func:`stroke_polyline`), so two
          segments meeting at a sharp angle come to a true point instead of the
          blunted step PIL's round `joint="curve"` disc leaves behind.
        - **Endpoints** use butt caps by default — no extra ink at truly
          isolated endpoints (e.g. ray tips). No visible "bump".
        - **Shared-vertex joins** (e.g., a bisector whose start sits on
          another polyline's joint OR on another polyline's butt-capped
          endpoint) are handled by setting `join_start=True` /
          `join_end=True`. When set, a filled disk of radius `line_width/2`
          is painted at that endpoint AFTER the stroke. This disk:
            - covers this line's butt cap, plus
            - covers the other line's butt cap at the same point (for
              endpoint↔endpoint joins like polygon corners), plus
            - fills the triangular gap between the two butt caps — a
              miter-like seal without any visible overshoot beyond the
              natural stroke intersection for angles >= 90°, and only a
              small crescent for acute angles.
          Apply this treatment only at vertices the generator explicitly
          marks as joined, so isolated endpoints stay cap-free.
        - **Closed polylines** (`xy[0] == xy[-1]`) are stroked as a closed
          mitered ring, so the closure vertex is joined exactly like every
          interior vertex and needs no sealing disc.
        """
        width = max(1, round(self.line_width * width_factor))
        color = fill if fill is not None else self.CANDIDATE_OUTLINE_COLOR

        if isinstance(draw, DrawingRecorder):
            recorded = [[int(round(p.x)), int(round(p.y))] for p in points]
            draw.add_high_level_command(
                "draw_line", points=recorded, width_factor=width_factor,
                fill=color, width=width,
                join_start=bool(join_start), join_end=bool(join_end),
            )
            return

        xy = [(int(round(p.x)), int(round(p.y))) for p in points]
        if workbench_active() and draw_survey_rail(draw, xy, color, width):
            return
        stroke_polyline(
            draw, xy, fill=given_colour(color), width=width,
            join_start=join_start, join_end=join_end,
        )

    def draw_dashed_line(
        self,
        draw,
        start: Point,
        end: Point,
        *,
        dash: float = 6.0,
        gap: float = 4.0,
        color: Tuple[int, int, int] = (200, 200, 200),
        width_factor: float = 0.35,
    ) -> None:
        """Draw a light gray dashed line segment in unsupersampled coordinate space.

        Works transparently with both `ScaledDraw` (puzzle/solution images)
        and `DrawingRecorder` (video pipeline). Each dash is emitted as a
        separate `draw_line` high-level command so the video renderer's
        base-vs-solution diff treats the dashes as static base geometry and
        never animates them (they stay gray and stationary across frames).

        - `dash` / `gap` are in the same unsupersampled pixel units as
          `start` / `end`, so values like (6, 4) are directly comparable to
          the canvas size.
        - `width_factor` is applied to `self.line_width` (same semantics as
          `draw_line`), defaulting to ~1/3 so the auxiliary construction
          stays visually subordinate to the 2-3 px black triangle outline
          to keep the construction readable.
        - `color` is a solid RGB; default `(200, 200, 200)` is the spec-mandated
          light gray for reasoning-aid hint lines.
        """
        dx = end.x - start.x
        dy = end.y - start.y
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return
        dash = max(1.0, float(dash))
        gap = max(1.0, float(gap))
        period = dash + gap
        ux = dx / length
        uy = dy / length
        width = max(1, round(self.line_width * max(0.05, width_factor)))

        # Walk the segment in (dash, gap) cycles, emitting a short line per dash.
        t = 0.0
        while t < length - 1e-6:
            t_end = min(length, t + dash)
            p1 = Point(x=start.x + ux * t, y=start.y + uy * t)
            p2 = Point(x=start.x + ux * t_end, y=start.y + uy * t_end)
            if isinstance(draw, DrawingRecorder):
                draw.add_high_level_command(
                    "draw_line",
                    points=[[round(p1.x), round(p1.y)], [round(p2.x), round(p2.y)]],
                    width_factor=width_factor,
                    fill=color,
                    width=width,
                )
            else:
                draw.line(
                    [(round(p1.x), round(p1.y)), (round(p2.x), round(p2.y))],
                    fill=color,
                    width=width,
                )
            t += period

    def draw_circle(
        self,
        draw,
        center: Point,
        radius: int,
        width_factor: float = 1.0,
        *,
        keep_above: Optional[Sequence[Tuple[float, float, float]]] = None,
    ) -> None:
        """Draw a circle stroke (not filled) of radius `radius` at `center`.

        `keep_above` declares a list of circular regions (each tuple is
        `(cx, cy, radius)`) that MUST remain above the circle's stroke
        throughout its drawing. Typical use: mark the positions of
        hollow anchor markers that were painted BEFORE this circle; the
        renderer then keeps those regions frozen to their pre-circle
        state so the circle visually passes BENEATH the markers. This
        is a proper layer compositing mechanism — the circle is never
        painted over marker regions in the first place, so no "patch
        up after the fact" is needed.
        """
        width = max(1, round(self.line_width * width_factor))
        keep_above_list = [
            [float(cx), float(cy), float(r)] for (cx, cy, r) in (keep_above or [])
        ]

        if isinstance(draw, DrawingRecorder):
            cx, cy = round(center.x), round(center.y)
            bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
            draw.add_high_level_command(
                "draw_circle",
                bbox=bbox,
                outline=self.CANDIDATE_OUTLINE_COLOR,
                width=width,
                keep_above=keep_above_list,
            )
            return

        cx, cy = round(center.x), round(center.y)
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(bbox, outline=self.CANDIDATE_OUTLINE_COLOR, width=width)

    def draw_curve(
        self,
        draw,
        points: List[Point],
        *,
        width_factor: float = 1.0,
        fill: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Render a smooth thick curve via stroke-outline polygon fill.

        PIL's `draw.line(xy, width=N, joint="curve")` works well for a
        small number of long segments, but on densely-sampled
        parametric curves (parabola, smooth freeform, etc.) the many
        short segments show visible polyline-joint bumps along the
        stroke edge because each segment is rendered as an independent
        rectangular polygon with its own butt caps.

        This helper instead computes the FULL stroke outline — offset
        curves on either side of the centerline by ±line_width/2 along
        the per-vertex normal — and fills it as a SINGLE closed
        polygon. PIL's polygon fill uses a scan-line algorithm with no
        per-segment joint artifacts, so the curve edge is uniformly
        smooth at any sampling density.

        Per-vertex normals use central differences at interior vertices
        and one-sided differences at endpoints so the offset curves
        remain continuous.
        """
        if len(points) < 2:
            return
        width = max(1, round(self.line_width * width_factor))
        color = fill if fill is not None else self.CANDIDATE_OUTLINE_COLOR
        half = width / 2.0

        n = len(points)
        # Unit tangent at each vertex (central difference in the middle,
        # one-sided at endpoints).
        tangents: List[Tuple[float, float]] = []
        for i, p in enumerate(points):
            if i == 0:
                q = points[1]
                dx, dy = q.x - p.x, q.y - p.y
            elif i == n - 1:
                q = points[-2]
                dx, dy = p.x - q.x, p.y - q.y
            else:
                prev_pt, next_pt = points[i - 1], points[i + 1]
                dx = next_pt.x - prev_pt.x
                dy = next_pt.y - prev_pt.y
            length = math.hypot(dx, dy)
            if length < 1e-9:
                tangents.append((1.0, 0.0))
            else:
                tangents.append((dx / length, dy / length))
        # Per-vertex normals (rotate tangent 90°).
        normals = [(-ty, tx) for (tx, ty) in tangents]
        left = [(p.x + nx * half, p.y + ny * half) for p, (nx, ny) in zip(points, normals)]
        right = [(p.x - nx * half, p.y - ny * half) for p, (nx, ny) in zip(points, normals)]
        outline_xy = left + right[::-1]

        if isinstance(draw, DrawingRecorder):
            draw.polygon(
                [(round(x), round(y)) for (x, y) in outline_xy],
                fill=color,
            )
            return
        draw.polygon(
            [(round(x), round(y)) for (x, y) in outline_xy],
            fill=color,
        )

    def draw_arc(
        self,
        draw,
        center: Point,
        radius: float,
        start_angle: float,
        sweep: float,
        *,
        width_factor: float = 1.0,
        fill: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Smooth circular arc using PIL's native `ImageDraw.arc` primitive.

        `start_angle` and `sweep` are in **radians** using the math
        convention: angles increase counter-clockwise from the positive
        x-axis (standard CCW), matching the rest of the generator code
        that uses `math.cos / math.sin`.

        Because image y is inverted relative to math y, this helper
        internally converts to PIL's CW-from-3-o'clock degrees. It also
        records a high-level `draw_arc` command so the video renderer
        can animate the arc as a smooth reveal in the same direction as
        `sweep` (unlike a polyline approximation, the arc stays
        pixel-smooth regardless of radius or SS factor).
        """
        width = max(1, round(self.line_width * width_factor))
        color = fill if fill is not None else self.CANDIDATE_OUTLINE_COLOR
        cx, cy = round(center.x), round(center.y)
        r = int(round(radius))
        bbox = (cx - r, cy - r, cx + r, cy + r)
        # Math -> PIL degrees. PIL arcs sweep CW from start to end, and
        # the y-axis is flipped, so a math CCW sweep from `start_angle`
        # corresponds to PIL going from `-end_math_deg` to `-start_math_deg`
        # when sweep > 0, and the reverse when sweep < 0.
        start_math_deg = math.degrees(start_angle)
        end_math_deg = math.degrees(start_angle + sweep)
        if sweep >= 0:
            pil_start = -end_math_deg
            pil_end = -start_math_deg
        else:
            pil_start = -start_math_deg
            pil_end = -end_math_deg
        # Normalize so pil_end >= pil_start (PIL arc draws CW from start
        # to end, accepting values outside [0, 360)).
        while pil_end < pil_start:
            pil_end += 360.0

        if isinstance(draw, DrawingRecorder):
            draw.add_high_level_command(
                "draw_arc",
                bbox=bbox,
                start=pil_start,
                end=pil_end,
                outline=color,
                width=width,
            )
            return
        draw.arc(bbox, pil_start, pil_end, fill=color, width=width)

    def draw_anchor_marker(
        self,
        draw,
        center: Point,
        radius: int,
        *,
        fill: Optional[Tuple[int, int, int]] = None,
        outline: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """One anchor disc: hollow by default, solid when ``fill`` is given.

        Passing the fill and outline through lets a task tell two otherwise
        identical anchors apart -- a solid disc against a hollow ring -- when
        its rule has to name one of them.  Both the still path and the recorded
        video command carry the colours, so the two anchors read the same way
        in every artifact.
        """

        fill = self.CANDIDATE_BASE_FILL if fill is None else fill
        outline = self.CANDIDATE_OUTLINE_COLOR if outline is None else outline
        cx, cy = round(center.x), round(center.y)
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        if isinstance(draw, DrawingRecorder):
            draw.add_high_level_command(
                "draw_anchor_marker",
                bbox=bbox,
                fill=fill,
                outline=outline,
                width=max(2, self.line_width),
            )
            return

        draw.ellipse(
            bbox,
            fill=fill,
            outline=outline,
            width=max(2, self.line_width),
        )

    def trim_segment(
        self,
        start: Point,
        end: Point,
        *,
        start_offset: float = 0.0,
        end_offset: float = 0.0,
    ) -> Tuple[Point, Point]:
        dx = end.x - start.x
        dy = end.y - start.y
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return start, end
        ux = dx / length
        uy = dy / length
        trimmed_start = Point(
            x=start.x + ux * max(0.0, start_offset),
            y=start.y + uy * max(0.0, start_offset),
        )
        trimmed_end = Point(
            x=end.x - ux * max(0.0, end_offset),
            y=end.y - uy * max(0.0, end_offset),
        )
        return trimmed_start, trimmed_end

    def _get_candidate_font(self) -> Any:
        if self._candidate_font is None:
            self._candidate_font = ImageFont.load_default(self.CANDIDATE_LABEL_FONT_SIZE)
        return self._candidate_font

    def _video_overlay_extras(self, draw: ImageDraw.ImageDraw) -> None:
        """Hook: draw extra elements on top of animated lines but below candidates.

        Subclasses override this to re-draw static geometry (e.g. anchor circles)
        that must stay visible above animated solution lines in video frames.
        """
        pass

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        raise NotImplementedError("Subclasses must implement _render method")

    def get_draw_base(self) -> Tuple[ImageDraw.ImageDraw, Image.Image]:
        width, height = self.canvas_dimensions
        if self._recording_active:
            if self._recorder is None:
                self._recorder = DrawingRecorder(width, height)
            return self._recorder, self._recorder.base_image

        base, draw = new_scaled_canvas(width, height, self.SUPERSAMPLE)
        if workbench_active():
            draw_workbench(draw, width, height, self.candidates, self.point_radius)
        return draw, base

    def _downscale(self, image: Image.Image) -> Image.Image:
        return finish_scaled_canvas(image, self.canvas_dimensions)

    def build_record_extra(self) -> Dict[str, Any]:
        return {}

    def _task_name_for_contract(self) -> str:
        default_dir = str(self.DEFAULT_OUTPUT_DIR or "")
        if default_dir:
            return Path(default_dir).name
        name = self.__class__.__name__
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name.removesuffix("Generator")).lower()

    def _format_sample_prompt(self, prompt: str) -> str:
        """Apply task-owned, semantic sample values after every prompt rebuild."""

        return str(prompt)

    def _semantic_phase_roles_for_sample(self) -> tuple[str, ...]:
        """Return the exact cumulative image duties for the current sample.

        Most point-target tasks have a fixed semantic construction graph.
        Orthocenter is deliberately sample-dependent: an obtuse triangle can
        require one or two finite-side extensions, and Review explicitly
        requires each such extension to be committed before its altitude.
        Those extension states cannot be hidden inside a static three-image
        role list merely to keep prompt counts constant.
        """

        task_name = self._task_name_for_contract()
        if task_name == "circumcenter" and hasattr(self, "center"):
            vertices = list(getattr(self, "_ordered_vertices")())
            names = self._visible_point_names(vertices)
            sides = [
                f"the side joining the {names[index]} and the {names[(index + 1) % 3]}"
                for index in range(3)
            ]
            return (
                f"the perpendicular bisectors of {sides[0]}, {sides[1]}, and {sides[2]} shown together at their common intersection",
                "all three retained perpendicular bisectors plus the circumcircle through the visible vertices before candidate selection",
            )
        if task_name == "orthocenter" and hasattr(self, "center"):
            altitude_data = getattr(self, "_altitude_data")()
            vertices = list(getattr(self, "_ordered_vertices")())
            names = self._visible_point_names(vertices)
            roles: List[str] = []
            extension_roles: List[str] = []
            for index, item in enumerate(altitude_data):
                side = (
                    f"the side joining the {names[(index + 1) % 3]} and the "
                    f"{names[(index + 2) % 3]}"
                )
                if item.get("extension") is not None:
                    endpoint = (
                        names[(index + 1) % 3]
                        if float(item.get("side_parameter", 0.0)) < 0.0
                        else names[(index + 2) % 3]
                    )
                    extension_roles.append(
                        f"{side} extended beyond the {endpoint} to the exterior perpendicular foot"
                    )
            if extension_roles:
                roles.append(
                    "all required opposite-side extensions shown together and retained: "
                    + "; ".join(extension_roles)
                )
            roles.append(
                "all three vertex altitudes drawn together perpendicular to their opposite-side lines, with every required extension retained"
            )
            return tuple(roles)
        return semantic_phase_roles(task_name)

    def _refresh_geometry_prompts(self) -> None:
        prompts = build_point_prompts(
            task_name=self._task_name_for_contract(), input_description=self._geometry_input_description,
            rule_description=self._geometry_rule_description, solution_description=self._geometry_solution_description,
            video_description=self._geometry_video_description,
        )
        self.solution_image_prompt = self._format_sample_prompt(prompts["solution_image"])
        self.ti2v_prompt = self._format_sample_prompt(prompts["ti2v"])

    def _normalize_construction_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Use the shared cyan only for newly added solution construction.

        Base geometry keeps its task-owned black/gray/colored semantics.  The
        command diff lets us distinguish a solution stroke from an identical
        source stroke without recoloring the entire solved canvas.
        """

        normalized = dict(command)
        cyan = tuple(self.CONSTRUCTION_COLOR)
        command_type = str(normalized.get("type") or "")
        neutral_stroke_types = {
            "draw_line",
            "draw_smooth_polyline",
            "draw_dashed_segments",
            "line",
            "draw_circle",
            "draw_arc",
            "arc",
            "ellipse",
            "rectangle",
            "polygon",
        }

        def is_neutral_stroke(value: Any) -> bool:
            if not isinstance(value, (tuple, list)) or len(value) < 3:
                return False
            channels = tuple(int(channel) for channel in value[:3])
            return max(channels) <= 180 and max(channels) - min(channels) <= 12

        if command_type not in neutral_stroke_types:
            return normalized
        if is_neutral_stroke(normalized.get("fill")):
            normalized["fill"] = cyan
        if is_neutral_stroke(normalized.get("outline")):
            normalized["outline"] = cyan
        return normalized

    def _render_recorded_solution_image(self) -> Image.Image:
        """Replay base plus normalized construction for the terminal image."""

        try:
            geometry_cmds, solution_steps, _final_candidates_cmd = self._extract_solution_diff_commands()
        except (AttributeError, NotImplementedError, TypeError):
            # A small number of task-owned renderers use direct supersampled
            # drawing primitives that cannot be replayed by DrawingRecorder;
            # those renderers already declare their construction palette.
            return self._downscale(self._render(highlight_label=self.correct_label))
        if self._recorder is not None and any(
            extrema != (255, 255) for extrema in self._recorder.base_image.getextrema()
        ):
            # A compatible recorder can still expose a task-owned raster
            # overlay through ``base_image`` without representing it as a
            # replayable command. Keep the task renderer as the terminal-image
            # authority in that case; command harvesting remains available.
            return self._downscale(self._render(highlight_label=self.correct_label))
        if not geometry_cmds and not solution_steps:
            return self._downscale(self._render(highlight_label=self.correct_label))
        width, height = self.canvas_dimensions
        renderer = VideoRenderer(width, height, self)
        renderer.execute_commands(geometry_cmds)
        renderer.execute_commands(solution_steps)
        out = renderer.canvas.copy()
        raw_draw = ImageDraw.Draw(out)
        scaled_draw = ScaledDraw(raw_draw, self.SUPERSAMPLE)
        self._video_overlay_extras(scaled_draw)
        self.draw_candidates(scaled_draw, highlight_label=self.correct_label)
        return self._downscale(out)

    @staticmethod
    def _direction_name(dx: float, dy: float) -> str:
        """8-way screen-space direction name (y grows downward)."""

        names = (
            "right",
            "lower-right",
            "down",
            "lower-left",
            "left",
            "upper-left",
            "up",
            "upper-right",
        )
        angle = math.atan2(dy, dx) % math.tau
        return names[int((angle + math.pi / 8.0) // (math.pi / 4.0)) % 8]

    def _canvas_region(self, point: Point) -> str:
        """Name a canvas location in reader terms (thirds grid)."""

        from data.geometry.core.semantic_contract import canvas_region_phrase

        width, height = self.canvas_dimensions
        return canvas_region_phrase(point.x, point.y, width=float(width), height=float(height))

    @staticmethod
    def _heading_phrase(dx: float, dy: float) -> str:
        """Natural adverbial phrase for a screen-space direction."""

        return {
            "up": "straight up",
            "down": "straight down",
            "left": "to the left",
            "right": "to the right",
            "upper-left": "toward the upper left",
            "upper-right": "toward the upper right",
            "lower-left": "toward the lower left",
            "lower-right": "toward the lower right",
        }[PointTargetPuzzleGenerator._direction_name(dx, dy)]

    @staticmethod
    def _angle_size_phrase(degrees: float) -> str:
        """Coarse visually-assessable size wording for an angle."""

        value = abs(float(degrees)) % 360.0
        if value > 180.0:
            value = 360.0 - value
        if value < 45.0:
            return "a narrow angle"
        if value < 75.0:
            return "a moderate angle"
        if value < 105.0:
            return "roughly a right angle"
        if value < 135.0:
            return "a wide angle"
        return "a very wide angle"

    def _authored_observation(self) -> Optional[str]:

        return None

    def _point_position_label(self, point: Point, peers: Sequence[Point]) -> str:
        if not peers:
            return "selected"
        min_y = min(peer.y for peer in peers)
        max_y = max(peer.y for peer in peers)
        min_x = min(peer.x for peer in peers)
        max_x = max(peer.x for peer in peers)
        spread_x = max(1.0, max_x - min_x)
        spread_y = max(1.0, max_y - min_y)
        vertical = ""
        horizontal = ""
        if point.y <= min_y + spread_y * 0.24:
            vertical = "top"
        elif point.y >= max_y - spread_y * 0.24:
            vertical = "bottom"
        if point.x <= min_x + spread_x * 0.24:
            horizontal = "left"
        elif point.x >= max_x - spread_x * 0.24:
            horizontal = "right"
        if vertical and horizontal:
            return f"{vertical}-{horizontal}"
        return vertical or horizontal or "central"

    def _visible_point_names(self, points: Sequence[Point]) -> List[str]:
        """Give visible, qualitative, unique names without invented indices."""

        values = list(points)
        base = [self._point_position_label(point, values) for point in values]
        names = list(base)
        for label in set(base):
            indices = [index for index, value in enumerate(base) if value == label]
            if len(indices) <= 1:
                continue
            ordered = sorted(indices, key=lambda index: (values[index].x, values[index].y))
            if len(ordered) == 2:
                names[ordered[0]] = f"left {label}"
                names[ordered[1]] = f"right {label}"
            else:
                for rank, index in enumerate(ordered):
                    qualifier = ("leftmost", "middle", "rightmost")[rank]
                    names[index] = f"{qualifier} {label}"
        return [f"{name} visible vertex" for name in names]

    @staticmethod
    def _relative_direction_label(start: Point, end: Point) -> str:
        """Describe a visible displacement without leaking renderer coordinates."""

        dx = float(end.x) - float(start.x)
        dy = float(end.y) - float(start.y)
        magnitude = math.hypot(dx, dy)
        if magnitude <= 1e-9:
            return "nowhere"
        horizontal = ""
        vertical = ""
        if abs(dx) >= magnitude * 0.30:
            horizontal = "right" if dx > 0 else "left"
        if abs(dy) >= magnitude * 0.30:
            vertical = "down" if dy > 0 else "up"
        if vertical and horizontal:
            return f"{vertical}-{horizontal}"
        return vertical or horizontal

    @staticmethod
    def _relative_position_phrase(start: Point, end: Point) -> str:
        direction = PointTargetPuzzleGenerator._relative_direction_label(start, end)
        return {
            "up": "above",
            "down": "below",
            "left": "to the left of",
            "right": "to the right of",
            "up-left": "above and to the left of",
            "up-right": "above and to the right of",
            "down-left": "below and to the left of",
            "down-right": "below and to the right of",
            "nowhere": "at",
        }[direction]

    # ------------------------------------------------------------------
    # Frame composition

    # Seating the cloud kills every reading of where the markers sit
    # *relative to one another*.  One reading survives it: distance to the
    # middle of the frame.  These figures are built around the centre of the
    # canvas and the answer is the figure's focal point -- the incentre, the
    # midpoint of a segment, the vertex of a parabola -- so once the answer
    # lands nearer the centre than the cloud's own radius, no distractor can
    # be closer and "the innermost marker" is the answer for free
    #.  No cloud can undo that: with the answer on the centre
    # itself, every seating of every shape leaves it innermost.

    # So the frame is composed rather than the cloud.  The finished figure
    # and its markers slide off centre together, by one rigid translation.
    # That leaves every distance inside the sample untouched -- the 7.5
    # margins, every declared incidence, the relative marker geometry the
    # seating just made answer-free -- and changes only the relation that was
    # leaking, where the sample sits in its frame.  How far it may slide is
    # read off the ink's own clearance, so the figure stays whole and inside
    # the family's margins; how far it does slide is drawn to land the
    # answer's rank by distance to the frame centre on a uniformly chosen
    # target, and drawn uniformly among the slides that reach that rank, so
    # the figure's own offset carries no residual signal either.
    # ------------------------------------------------------------------

    def _sample_positions(self) -> List[Any]:
        """Every stored position of the current sample, once each.

        ``Point`` and ``PointCandidate`` are mutable, so walking the
        generator's own state is enough to move the whole sample: the tuples,
        lists and records holding them keep holding the same objects.
        """

        found: List[Any] = []
        seen: set = set()

        def visit(value: Any, depth: int) -> None:
            if depth > 8 or id(value) in seen:
                return
            if isinstance(value, (Point, PointCandidate)):
                seen.add(id(value))
                found.append(value)
            elif isinstance(value, (list, tuple, set, frozenset)):
                seen.add(id(value))
                for item in value:
                    visit(item, depth + 1)
            elif isinstance(value, dict):
                seen.add(id(value))
                for item in value.values():
                    visit(item, depth + 1)
            elif is_dataclass(value) and not isinstance(value, type):
                # Covers Incidence and the task-owned records that hold a
                # figure's own points -- a segment, an arc, a pair of rails.
                seen.add(id(value))
                for item in fields(value):
                    visit(getattr(value, item.name, None), depth + 1)

        for value in list(vars(self).values()):
            visit(value, 0)
        return found

    def _slide_sample(self, dx: float, dy: float) -> None:
        for position in self._sample_positions():
            position.x += dx
            position.y += dy

    def _ink_clearance(self, image: Image.Image) -> Optional[Tuple[float, float, float, float]]:
        """Free canvas on each side of the drawn ink: left, right, top, bottom."""

        width, height = self.canvas_dimensions
        pixels = np.asarray(image.convert("L"), dtype=np.int16)
        rows = np.nonzero((pixels < 235).any(axis=1))[0]
        columns = np.nonzero((pixels < 235).any(axis=0))[0]
        if len(rows) == 0 or len(columns) == 0:
            return None
        return (
            float(columns[0]),
            float(width - 1 - columns[-1]),
            float(rows[0]),
            float(height - 1 - rows[-1]),
        )

    def _composition_reach(self, image: Image.Image) -> Optional[Tuple[float, float, float, float]]:
        """How far the sample may slide each way: -x, +x, -y, +y.

        Three things bound it.  The drawn ink's own clearance, measured on the
        plate that carries the most of it -- the terminal one -- so a
        construction reaching past the input figure is inside the budget too.
        The drawable rect, which every stored position keeps to unless it was
        already outside, so no part of the figure is pushed at the frame.  And
        each marker's own canvas-edge rule, which is stricter than both.
        """

        clearance = self._ink_clearance(image)
        if clearance is None:
            return None
        left, right, top, bottom = clearance
        floor = float(self.COMPOSITION_EDGE_FLOOR)
        reach = float(self.COMPOSITION_REACH_RATIO) * self.canvas_short_side
        bounds = [
            -min(reach, max(0.0, left - floor)),
            min(reach, max(0.0, right - floor)),
            -min(reach, max(0.0, top - floor)),
            min(reach, max(0.0, bottom - floor)),
        ]
        low_x, top_y, high_x, bottom_y = self.canvas_bounds()
        for position in self._sample_positions():
            bounds[0] = max(bounds[0], -max(0.0, position.x - low_x))
            bounds[1] = min(bounds[1], max(0.0, high_x - position.x))
            bounds[2] = max(bounds[2], -max(0.0, position.y - top_y))
            bounds[3] = min(bounds[3], max(0.0, bottom_y - position.y))
        for candidate in self.candidates:
            pad_x, pad_y = self._candidate_safe_padding(
                candidate.label, include_edge_buffer=True,
            )
            bounds[0] = max(bounds[0], low_x + pad_x - candidate.x)
            bounds[1] = min(bounds[1], high_x - pad_x - candidate.x)
            bounds[2] = max(bounds[2], top_y + pad_y - candidate.y)
            bounds[3] = min(bounds[3], bottom_y - pad_y - candidate.y)
        if bounds[0] > 0.0 or bounds[1] < 0.0 or bounds[2] > 0.0 or bounds[3] < 0.0:
            # A marker already sits outside its own margin; sliding cannot
            # repair that, and pretending otherwise would move the figure for
            # a reason unrelated to composition.
            return None
        return tuple(bounds)

    def _answer_candidate(self) -> Optional[PointCandidate]:
        """The one marker the answer names, when the answer names one marker.

        A task whose answer is a *pair* of markers has no single marker to
        rank, so it composes its frame without steering (see
        ``_compose_off_centre``).
        """

        for candidate in self.candidates:
            if candidate.label == self.correct_label:
                return candidate
        return None

    def _centre_rank(self, dx: float, dy: float) -> int:
        """Markers nearer the frame centre than the answer, after a slide."""

        width, height = self.canvas_dimensions
        middle_x, middle_y = width / 2.0 - dx, height / 2.0 - dy
        answer = self._answer_candidate()
        reach = math.hypot(answer.x - middle_x, answer.y - middle_y)
        return sum(
            1
            for candidate in self.candidates
            if candidate.label != self.correct_label
            and math.hypot(candidate.x - middle_x, candidate.y - middle_y) < reach
        )

    def _probe_plate(self) -> Image.Image:
        """The terminal plate, drawn without spending the sample's randomness."""

        state = self._rng.getstate()
        try:
            # Framing is part of the frozen logical instance. Presentation
            # props must never change placement or consume its randomness.
            from data.common.prompt_paraphrase import use_prompt_paraphrase
            with use_prompt_paraphrase(None):
                return self._downscale(self._render(highlight_label=self.correct_label))
        finally:
            self._rng.setstate(state)

    def _compose_off_centre(self) -> None:
        """Slide the finished sample off the middle of the frame."""

        if not self.candidates or self._answer_candidate() is None:
            return
        bounds = self._composition_reach(self._probe_plate())
        if bounds is None:
            return
        low_x, high_x, low_y, high_y = bounds
        target = self._rng.randrange(len(self.candidates))
        offsets = [(0.0, 0.0)] + [
            (self._rng.uniform(low_x, high_x), self._rng.uniform(low_y, high_y))
            for _ in range(96)
        ]
        best = min(abs(self._centre_rank(dx, dy) - target) for dx, dy in offsets)
        dx, dy = self._rng.choice(
            [
                offset
                for offset in offsets
                if abs(self._centre_rank(*offset) - target) == best
            ]
        )
        dx, dy = round(dx), round(dy)
        if dx == 0 and dy == 0:
            return
        # A construction with no ends of its own is cut by the frame, so for
        # the slide to be rigid it is the frame that has to travel with the
        # figure.  The plate the slide is checked against is therefore drawn
        # against the frame the slid sample will meet, which is this one
        # shifted the other way.
        self._clip_offset = (-dx, -dy)
        try:
            reference = self._probe_plate()
        finally:
            self._clip_offset = (0.0, 0.0)
        self._slide_sample(dx, dy)
        clearance = self._ink_clearance(slid := self._probe_plate())
        if (
            clearance is None
            or min(clearance) < self.COMPOSITION_EDGE_FLOOR
            or not _is_translation_of(reference, slid, dx, dy)
        ):
            # Either some position of this sample is not reachable from the
            # generator's own state, so the slide would tear the figure apart
            # instead of moving it, or it would push ink at the frame.  Put
            # the sample back.
            self._slide_sample(-dx, -dy)

    def save_puzzle(self) -> PointTargetPuzzleRecord:
        # Compose the frame before anything reads the layout: every artifact,
        # the record's coordinates and the authored trace's own talk of where
        # things sit are all derived below, from the state this moves.
        self._compose_off_centre()
        # Task subclasses historically replaced the shared prompts after
        # ``super().__init__``.  Rebuild them here from the semantic source so
        # renderer dimensions, frame schedules, and CoT checklists cannot leak
        # back into a record.
        self._refresh_geometry_prompts()
        identity = (
            f"{RECORD_NAMESPACE}.geometry:{self._task_name_for_contract()}:"
            f"{self.seed}:{self._sample_index}"
        )
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
        self._sample_index += 1
        self.pid=pid
        puzzle_img = self._downscale(self._render(
            highlight_label=None,
        ))
        solution_img = self._render_recorded_solution_image()

        self.puzzle_path = self.puzzle_dir / f"{pid}_puzzle.png"
        self.solution_path = self.solution_dir / f"{pid}_solution.png"
        puzzle_img.save(self.puzzle_path)
        solution_img.save(self.solution_path)
        video_rel_path: Optional[str] = None
        video_fps: Optional[int] = None
        video_num_frames: Optional[int] = None
        if self.record_video:
            # A failed encode used to be caught, printed, and walked past, so
            # the record below was still built -- with no video -- and the run
            # carried on to publish a row that declared ti2v anyway.  Let the
            # failure propagate: it names this sample and carries the real
            # ffmpeg stderr, which a manifest-stage complaint cannot.
            video_num_frames = self.save_video_solution(pid)
            video_abs = self.solution_dir / f"{pid}_solution.webm"
            video_rel_path = self.relativize_path(video_abs)
            video_fps = 16

        return PointTargetPuzzleRecord(
            id=self.pid,
            ti2v_prompt=self.ti2v_prompt,
            canvas_dimensions=self.canvas_dimensions,
            margin=self.margin,
            candidates=self.candidates,
            correct_option=self.correct_label,
            image=self.relativize_path(self.puzzle_path),
            solution_image_path=self.relativize_path(self.solution_path),
            point_radius=self.point_radius,
            line_width=self.line_width,
            solution_image_prompt=self.solution_image_prompt,
            seed=self.seed,
            extra=self.build_record_extra(),
            reasoning_image=self.relativize_path(self.puzzle_path),
            solution_video_path=video_rel_path,
            video_fps=video_fps,
            video_num_frames=video_num_frames,
        )

    def save_video_solution(self, pid: str) -> Optional[int]:
        # Phase 1: Record trace without highlights
        self._recording_active = True
        self._recorder = None # Reset
        self._render(highlight_label=None) # This populates self._recorder
        trace_base = self._recorder.commands if self._recorder else []

        # Phase 2: Record trace with highlights
        self._recorder = None # Reset
        self._render(highlight_label=self.correct_label)
        trace_solution = self._recorder.commands if self._recorder else []
        self._recording_active = False

        if not trace_base and not trace_solution:
            # Generator likely doesn't use get_draw_base
            return None

        # Diff commands
        # Use simple queue based diffing against the non-candidate base geometry.
        # Any command in 'trace_solution' that matches the head of 'geometry_cmds' is skipped (part of base).
        # Any command that doesn't match is considered new solution geometry or the final highlighted candidates.

        geometry_cmds = [cmd for cmd in trace_base if cmd["type"] != "draw_candidates"]
        base_candidates_cmd = next(
            (
                cmd
                for cmd in trace_base
                if cmd["type"] == "draw_candidates" and cmd.get("highlight_label") is None
            ),
            None,
        )
        base_queue = list(geometry_cmds)
        solution_diff = []

        for cmd in trace_solution:
            if base_queue and cmd == base_queue[0]:
                base_queue.pop(0)
            else:
                solution_diff.append(cmd)

        solution_steps = []
        final_candidates_cmd = None

        for cmd in solution_diff:
            if cmd['type'] == 'draw_candidates':
                final_candidates_cmd = cmd
            else:
                solution_steps.append(self._normalize_construction_command(cmd))

        task_name = self._task_name_for_contract()
        if task_name in SEMANTIC_PHASE_COMMAND_COUNTS:
            phase_groups = semantic_command_groups(task_name, solution_steps)
        elif solution_steps:
            # An un-authored task may have one atomic terminal construction,
            # but it may not manufacture a sequence from renderer commands.
            phase_groups = [solution_steps]
        else:
            phase_groups = []
        # Semantic groups are retained for authoring only. TI2V uses
        # the same command order but assigns time to each physical pen stroke.
        serial_commands = [command for phase in phase_groups for command in phase]

        # Generate Video
        video_path = self.solution_dir / f"{pid}_solution.webm"
        width, height = self.canvas_dimensions
        base_hold = 6
        end_hold = 8
        highlight_frames = 4 if final_candidates_cmd is not None else 0
        command_count = len(serial_commands)
        if command_count <= 3:
            video_frame_budget = SIMPLE_VIDEO_FRAMES
        elif command_count <= 7:
            video_frame_budget = MOTION_VIDEO_FRAMES
        else:
            video_frame_budget = self.MAX_VIDEO_FRAMES
        available = max(
            1,
            video_frame_budget - base_hold - end_hold - highlight_frames,
        )
        command_frames = (
            max(1, min(10, available // command_count))
            if command_count
            else 1
        )

        # Render the recorded commands into an in-memory frame sequence.
        video_renderer = VideoRenderer(
            width,
            height,
            self,
            max_frames=video_frame_budget,
            final_hold_frames=end_hold,
        )

        def candidates_overlay(
            frame: Image.Image,
            highlight: Optional[str] = None,
        ) -> Image.Image:
            out = frame.copy()
            raw_draw = ImageDraw.Draw(out)
            ss = self.SUPERSAMPLE
            scaled_draw = ScaledDraw(raw_draw, ss)
            self._video_overlay_extras(scaled_draw)
            self.draw_candidates(scaled_draw, highlight_label=highlight)
            return out

        def overlay_no_highlight(frame: Image.Image) -> Image.Image:
            return candidates_overlay(frame, highlight=None)

        # 1. Base Frame (Static)
        video_renderer.execute_commands(geometry_cmds)
        # Hold base frame for 1 second
        for _ in range(base_hold):
            if base_candidates_cmd is not None:
                video_renderer.add_pil_frame(candidates_overlay(video_renderer.canvas))
            else:
                video_renderer.write_frame()

        # image, but TI2V always shows the physical drawing process.  Flatten
        # those semantic groups without changing their order and animate one
        # renderer command at a time.  In particular, never fade a completed
        # multi-stroke phase into view: every line, arc, and circle must have a
        # single moving tip from its declared start to its declared end.
        for command in serial_commands:
            video_renderer.animate_command(
                command,
                duration_frames=command_frames,
                overlay_callback=overlay_no_highlight if base_candidates_cmd is not None else None,
            )

        # 3. Animate Answer (Candidates)
        if final_candidates_cmd:
            highlight = final_candidates_cmd.get("highlight_label")
            unselected = candidates_overlay(video_renderer.canvas, highlight=None)
            selected = candidates_overlay(video_renderer.canvas, highlight=highlight)
            # The answer is one localized semantic commit, but it must still be
            # visible as a continuous state change rather than a direct cut
            # from the untouched input to the solved endpoint.
            for step in range(1, 5):
                video_renderer.add_pil_frame(
                    Image.blend(unselected, selected, alpha=step / 4.0)
                )
            for _ in range(end_hold):
                video_renderer.add_pil_frame(selected)

        video_renderer.save(video_path)
        return len(video_renderer.frames)

    @staticmethod
    def strip_video_instruction(prompt: Optional[str]) -> Optional[str]:
        return _strip_video_instruction(prompt)

    @property
    def rng(self) -> random.Random:
        return self._rng

    def canvas_bounds(self) -> Tuple[int, int, int, int]:
        width, height = self.canvas_dimensions
        left = self.margin
        top = self.margin
        right = width - self.margin
        bottom = height - self.margin
        return left, top, right, bottom

    def _extract_solution_diff_commands(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        self._recording_active = True
        self._recorder = None
        try:
            self._render(highlight_label=None)
            trace_base = self._recorder.commands if self._recorder else []

            self._recorder = None
            self._render(highlight_label=self.correct_label)
            trace_solution = self._recorder.commands if self._recorder else []
        finally:
            self._recording_active = False

        geometry_cmds = [cmd for cmd in trace_base if cmd.get("type") != "draw_candidates"]
        base_queue = list(geometry_cmds)
        solution_diff: List[Dict[str, Any]] = []
        for cmd in trace_solution:
            if base_queue and cmd == base_queue[0]:
                base_queue.pop(0)
            else:
                solution_diff.append(cmd)

        solution_steps: List[Dict[str, Any]] = []
        final_candidates_cmd: Optional[Dict[str, Any]] = None
        for cmd in solution_diff:
            if cmd.get("type") == "draw_candidates":
                final_candidates_cmd = cmd
            else:
                solution_steps.append(self._normalize_construction_command(cmd))
        return geometry_cmds, solution_steps, final_candidates_cmd


class DrawingRecorder:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.commands = []
        self.base_image = Image.new("RGB", (width, height), (255, 255, 255))
        # Match the small ScaledDraw surface used by task renderers while
        # keeping recorder commands in their existing 1x coordinate space.
        # Pointing ``raw`` back at the recorder preserves command capture for
        # code paths that deliberately drop to the Pillow-like draw object.
        self.factor = 1
        self.raw = self

    def add_high_level_command(self, type_name, **kwargs):
         self.commands.append({"type": type_name, **kwargs})

    def line(self, xy, fill=None, width=0, joint=None):
        self.commands.append({"type": "line", "xy": xy, "fill": fill, "width": width})

    def ellipse(self, xy, fill=None, outline=None, width=1):
        self.commands.append({"type": "ellipse", "xy": xy, "fill": fill, "outline": outline, "width": width})

    def text(self, xy, text, fill=None, font=None, *_args, **_kwargs):
        self.commands.append({"type": "text", "xy": xy, "text": text, "fill": fill, "font": font})

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.commands.append({"type": "rectangle", "xy": xy, "fill": fill, "outline": outline, "width": width})

    def arc(self, xy, start, end, fill=None, width=1):
        self.commands.append({"type": "arc", "xy": xy, "start": start, "end": end, "fill": fill, "width": width})

    # Pillow Draw methods proxy
    def point(self, xy, fill=None): pass
    def polygon(self, xy, fill=None, outline=None, width=0):
        self.commands.append({
            "type": "polygon",
            "xy": list(xy),
            "fill": fill,
            "outline": outline,
            "width": width,
        })
    def chord(self, xy, start, end, fill=None, outline=None, width=1): pass
    def pieslice(self, xy, start, end, fill=None, outline=None, width=1): pass


class VideoRenderer:
    def __init__(
        self,
        width,
        height,
        generator: PointTargetPuzzleGenerator,
        *,
        max_frames: int = MULTISTEP_VIDEO_FRAMES,
        final_hold_frames: int = 8,
    ):
        self.width = width
        self.height = height
        ss = generator.SUPERSAMPLE
        self.ss = ss
        self._iw = width * ss
        self._ih = height * ss
        self.generator = generator
        self.max_frames = max_frames
        self.final_hold_frames = final_hold_frames
        self.frames = []
        self.canvas = Image.new("RGB", (self._iw, self._ih), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.canvas)
        if workbench_active():
            draw_workbench(ScaledDraw(self.draw, ss), width, height, generator.candidates, generator.point_radius)

    def _sxy(self, xy):
        """Scale coordinate sequence from 1x command to internal resolution."""
        return scale_xy(xy, self.ss)

    def _sw(self, w):
        return scale_stroke_width(w, self.ss)

    def execute_commands(self, commands):
        for cmd in commands:
            self.execute_command_instant(cmd)

    def execute_command_instant(self, cmd):
        ss = self.ss
        t = cmd['type']
        if t == 'draw_candidates':
            scaled_draw = ScaledDraw(self.draw, ss)
            self.generator.draw_candidates(scaled_draw, highlight_label=cmd.get('highlight_label'))
        elif t == 'draw_dashed_segments':
            dash = max(1.0, float(cmd.get("dash", 6.0)))
            gap = max(1.0, float(cmd.get("gap", 4.0)))
            period = dash + gap
            fill = cmd.get("fill") or (200, 200, 200)
            width = self._sw(cmd.get("width", 1))
            for segment in cmd.get("segments", []):
                if len(segment) != 2:
                    continue
                start, end = segment
                x1, y1 = float(start[0]), float(start[1])
                x2, y2 = float(end[0]), float(end[1])
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length <= 1e-6:
                    continue
                ux = dx / length
                uy = dy / length
                t_pos = 0.0
                while t_pos < length - 1e-6:
                    t_end = min(length, t_pos + dash)
                    self.draw.line(
                        [
                            ((x1 + ux * t_pos) * ss, (y1 + uy * t_pos) * ss),
                            ((x1 + ux * t_end) * ss, (y1 + uy * t_end) * ss),
                        ],
                        fill=fill,
                        width=width,
                    )
                    t_pos += period
        elif t == 'draw_line':
            pts = cmd['points']
            xy = [(p[0] * ss, p[1] * ss) for p in pts]
            w = self._sw(cmd['width'])
            if workbench_active() and draw_survey_rail(ScaledDraw(self.draw, ss), pts, cmd["fill"], cmd["width"]):
                return
            # Same mitered stroke as the still-image path (see draw_line):
            # closed rings are joined at every vertex including the closure,
            # open paths keep butt caps plus the declared sealing discs.
            stroke_polyline(
                self.draw, xy, fill=given_colour(cmd['fill']), width=w,
                join_start=bool(cmd.get('join_start')),
                join_end=bool(cmd.get('join_end')),
            )
        elif t == 'draw_circle':
            keep_above = cmd.get('keep_above') or []
            if keep_above:
                # Instant draw with keep_above: snapshot canvas, draw
                # circle, then restore marker regions from snapshot so
                # the circle appears to be on the layer BELOW the
                # markers. Mask radius = marker's outer edge EXACTLY
                # (no pad) so the circle's stroke meets the marker's
                # outline with no visible white halo between them.
                before = self.canvas.copy()
                self.draw.ellipse(
                    self._sxy(cmd['bbox']),
                    outline=cmd['outline'],
                    width=self._sw(cmd['width']),
                )
                ka_mask = Image.new("L", (self._iw, self._ih), 0)
                ka_draw = ImageDraw.Draw(ka_mask)
                for (cx, cy, r) in keep_above:
                    sc = self.ss
                    scx, scy, sr = cx * sc, cy * sc, r * sc
                    ka_draw.ellipse(
                        (scx - sr, scy - sr, scx + sr, scy + sr),
                        fill=255,
                    )
                restored = Image.composite(before, self.canvas, ka_mask)
                self.canvas = restored
                self.draw = ImageDraw.Draw(self.canvas)
            else:
                self.draw.ellipse(
                    self._sxy(cmd['bbox']),
                    outline=cmd['outline'],
                    width=self._sw(cmd['width']),
                )
        elif t == 'draw_arc':
            self.draw.arc(
                self._sxy(cmd['bbox']),
                cmd['start'], cmd['end'],
                fill=cmd['outline'],
                width=self._sw(cmd['width']),
            )
        elif t == 'draw_smooth_polyline':
            self._draw_smooth_polyline(cmd, self.canvas)
            self.draw = ImageDraw.Draw(self.canvas)
        elif t == 'draw_anchor_marker':
            self.draw.ellipse(
                self._sxy(cmd['bbox']),
                fill=cmd.get('fill'),
                outline=cmd.get('outline'),
                width=self._sw(cmd.get('width', 1)),
            )
        elif t == 'line':
            raw_width = cmd.get('width', 0)
            if raw_width:
                stroke_polyline(
                    self.draw, as_point_pairs(cmd["xy"], ss),
                    fill=cmd.get('fill'), width=self._sw(raw_width),
                )
            else:
                self.draw.line(self._sxy(cmd['xy']), fill=cmd.get('fill'), width=0)
        elif t == 'ellipse':
            self.draw.ellipse(self._sxy(cmd['xy']), fill=cmd.get('fill'),
                              outline=cmd.get('outline'), width=self._sw(cmd.get('width', 1)))
        elif t == 'arc':
            # Mirror PIL draw.arc so any generator that calls draw.arc(...)
            # on the ScaledDraw gets its arc replayed into the video base
            # frame. Previously this command type was silently dropped,
            # which made arc_center's input arc vanish from the first
            # frame of the solution video.
            self.draw.arc(
                self._sxy(cmd['xy']),
                cmd['start'], cmd['end'],
                fill=cmd.get('fill'),
                width=self._sw(cmd.get('width', 1)),
            )
        elif t == 'rectangle':
            self.draw.rectangle(
                self._sxy(cmd['xy']),
                fill=cmd.get('fill'),
                outline=cmd.get('outline'),
                width=self._sw(cmd.get('width', 1)) if cmd.get('width') else 1,
            )
        elif t == 'polygon':
            self.draw.polygon(
                self._sxy(cmd['xy']),
                fill=cmd.get('fill'),
                outline=cmd.get('outline'),
                width=self._sw(cmd.get('width', 0)) if cmd.get('width') else 0,
            )

    def animate_command(
        self,
        cmd,
        duration_frames=16,
        overlay_callback: Optional[Callable[[Image.Image], Image.Image]] = None,
    ):
        t = cmd['type']
        # Currently only animating lines and circles for smooth effect
        if t == 'draw_line' or t == 'line':
            self.animate_line(cmd, duration_frames, overlay_callback=overlay_callback)
        elif t == 'draw_smooth_polyline':
            self.animate_smooth_polyline(cmd, duration_frames, overlay_callback=overlay_callback)
        elif t == 'draw_circle' or t == 'ellipse':
             self.animate_circle(cmd, duration_frames, overlay_callback=overlay_callback)
        elif t == 'draw_arc' or t == 'arc':
            self.animate_arc(cmd, duration_frames, overlay_callback=overlay_callback)
        else:
            self.execute_command_instant(cmd)
            for _ in range(duration_frames):
                if overlay_callback is None:
                    self.write_frame()
                else:
                    self.add_pil_frame(overlay_callback(self.canvas))

    def _smooth_polyline_scaled_points(self, cmd) -> np.ndarray:
        return scaled_polyline_points(cmd.get("points", []), self.ss)

    def _draw_smooth_polyline(self, cmd, canvas: Image.Image) -> None:
        draw_smooth_polyline(
            canvas,
            cmd.get("points", []),
            color=cmd.get("fill"),
            width=cmd.get("width", 1),
            factor=self.ss,
        )

    def animate_smooth_polyline(
        self,
        cmd,
        frames,
        overlay_callback: Optional[Callable[[Image.Image], Image.Image]] = None,
    ):
        if frames <= 0:
            self.execute_command_instant(cmd)
            return
        points = self._smooth_polyline_scaled_points(cmd)
        if len(points) < 2:
            self.execute_command_instant(cmd)
            return

        lengths = [0.0]
        for idx in range(1, len(points)):
            prev = points[idx - 1]
            cur = points[idx]
            lengths.append(lengths[-1] + math.hypot(float(cur[0] - prev[0]), float(cur[1] - prev[1])))
        total_len = lengths[-1]
        if total_len <= 1e-6:
            self.execute_command_instant(cmd)
            return

        base_frame = self.canvas.copy()
        final_frame = base_frame.copy()
        self._draw_smooth_polyline(cmd, final_frame)
        last_composite = final_frame

        for frame_index in range(frames):
            target_len = total_len * (frame_index + 1) / frames
            partial: list[list[int]] = [points[0].tolist()]
            for idx in range(1, len(points)):
                if lengths[idx] <= target_len + 1e-6:
                    partial.append(points[idx].tolist())
                    continue
                seg_len = lengths[idx] - lengths[idx - 1]
                if seg_len > 1e-6:
                    ratio = max(0.0, min(1.0, (target_len - lengths[idx - 1]) / seg_len))
                    prev = points[idx - 1]
                    cur = points[idx]
                    interp = [
                        round(float(prev[0]) + float(cur[0] - prev[0]) * ratio),
                        round(float(prev[1]) + float(cur[1] - prev[1]) * ratio),
                    ]
                    partial.append(interp)
                break

            if frame_index == frames - 1:
                temp_canvas = final_frame
            else:
                temp_canvas = base_frame.copy()
                partial_cmd = dict(cmd)
                partial_cmd["points"] = [[x / self.ss, y / self.ss] for x, y in partial]
                self._draw_smooth_polyline(partial_cmd, temp_canvas)
            last_composite = temp_canvas
            frame_to_write = temp_canvas if overlay_callback is None else overlay_callback(temp_canvas)
            self.add_pil_frame(frame_to_write)

        self.canvas = last_composite.copy()
        self.draw = ImageDraw.Draw(self.canvas)

    def animate_arc(
        self,
        cmd,
        frames,
        overlay_callback: Optional[Callable[[Image.Image], Image.Image]] = None,
    ):
        """Reveal a partial arc smoothly, sweeping from `start` to `end`
        at the same direction PIL's draw.arc uses. Uses native PIL arc
        so the stroke stays pixel-smooth regardless of radius / SS."""
        if frames <= 0:
            self.execute_command_instant(cmd)
            return
        ss = self.ss
        if cmd['type'] == 'draw_arc':
            bbox = tuple(v * ss for v in cmd['bbox'])
            width_px = self._sw(cmd['width'])
            outline = cmd['outline']
        else:
            bbox = tuple(v * ss for v in cmd['xy'])
            width_px = self._sw(cmd.get('width', 1))
            outline = cmd.get('fill')
        if outline is None:
            self.execute_command_instant(cmd)
            return
        start = float(cmd['start'])
        end = float(cmd['end'])
        sweep = end - start
        if abs(sweep) < 1e-6:
            self.execute_command_instant(cmd)
            return

        base_frame = self.canvas.copy()
        final_frame = base_frame.copy()
        ImageDraw.Draw(final_frame).arc(bbox, start, end, fill=outline, width=width_px)
        last_composite = final_frame

        for f in range(frames):
            progress = (f + 1) / frames
            current_end = start + sweep * progress
            if f == frames - 1:
                temp_canvas = final_frame
            else:
                temp_canvas = base_frame.copy()
                ImageDraw.Draw(temp_canvas).arc(
                    bbox, start, current_end,
                    fill=outline, width=width_px,
                )
            last_composite = temp_canvas
            frame_to_write = temp_canvas if overlay_callback is None else overlay_callback(temp_canvas)
            self.add_pil_frame(frame_to_write)

        self.canvas = last_composite.copy()
        self.draw = ImageDraw.Draw(self.canvas)

    @staticmethod
    def _count_completed_segments(cumulative_lengths, current_len):
        completed = 0
        while completed < len(cumulative_lengths) and cumulative_lengths[completed] <= current_len + 1e-6:
            completed += 1
        return completed

    def animate_line(
        self,
        cmd,
        frames,
        overlay_callback: Optional[Callable[[Image.Image], Image.Image]] = None,
    ):
        if frames <= 0:
            self.execute_command_instant(cmd)
            return

        ss = self.ss
        # Extract points (1x) and scale to internal resolution.
        if cmd['type'] == 'draw_line':
            points = [(p[0] * ss, p[1] * ss) for p in cmd['points']]
            width = self._sw(cmd['width'])
            fill = cmd['fill']
        else:
            xy = cmd['xy']
            if isinstance(xy[0], (int, float)):
                points = [(xy[i] * ss, xy[i + 1] * ss) for i in range(0, len(xy), 2)]
            else:
                points = [(p[0] * ss, p[1] * ss) for p in xy]
            width = self._sw(cmd.get('width', 1))
            fill = cmd.get('fill')

        if len(points) < 2:
            self.execute_command_instant(cmd)
            return

        total_len = 0
        segments = []
        cumulative_lengths = []
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            segments.append((dist, p1, p2))
            total_len += dist
            cumulative_lengths.append(total_len)

        if total_len == 0:
            self.execute_command_instant(cmd)
            return

        base_frame = self.canvas.copy()
        final_frame = base_frame.copy()
        # Identical mitered stroke as the still-image and base-frame paths, so
        # the animated tip lands on exactly the geometry draw_line would draw.
        stroke_polyline(
            ImageDraw.Draw(final_frame), points, fill=fill, width=width,
            join_start=bool(cmd.get('join_start')),
            join_end=bool(cmd.get('join_end')),
        )
        last_composite = final_frame

        for f in range(frames):
            progress = (f + 1) / frames
            current_len = total_len * progress
            full_segments = self._count_completed_segments(cumulative_lengths, current_len)

            if full_segments >= len(segments):
                current_segment = len(segments) - 1
                current_tip = points[-1]
            else:
                current_segment = full_segments
                prev_len = cumulative_lengths[current_segment - 1] if current_segment > 0 else 0.0
                seg_dist, p1, p2 = segments[current_segment]
                remain = max(0.0, current_len - prev_len)
                if seg_dist > 0 and remain > 0:
                    ratio = min(1.0, remain / seg_dist)
                    current_tip = (
                        round(p1[0] + (p2[0] - p1[0]) * ratio),
                        round(p1[1] + (p2[1] - p1[1]) * ratio),
                    )
                else:
                    current_tip = p1

            revealed_points = list(points[:full_segments + 1]) + [current_tip]
            revealed_mask = Image.new("L", (self._iw, self._ih), 0)
            md = ImageDraw.Draw(revealed_mask)
            if len(revealed_points) >= 2:
                mask_width = width + 2 * ss
                # Reveal each revealed segment as its own butt-capped rectangle:
                # a two-point stroke has no joint to get wrong, and the union of
                # the rectangles is a guaranteed floor on coverage even where the
                # path doubles back on itself.
                for mask_start, mask_end in zip(revealed_points, revealed_points[1:]):
                    md.line([mask_start, mask_end], fill=255, width=mask_width)
                # A round joint disc cannot cover the mitered tip of an acute
                # joint, so the mask also carries the mitered stroke at the same
                # oversized width.  Without it a sharp corner would stay clipped
                # until the final frame and then pop into place.
                stroke_polyline(md, revealed_points, fill=255, width=mask_width)
            # Always reveal start-point area so the junction cap is visible
            # from the very first animation frame.
            sx, sy = points[0]
            mr = width // 2 + ss
            md.ellipse((sx - mr, sy - mr, sx + mr, sy + mr), fill=255)

            if f == frames - 1:
                temp_canvas = final_frame
            else:
                temp_canvas = Image.composite(final_frame, base_frame, revealed_mask)
            last_composite = temp_canvas
            frame_to_write = temp_canvas if overlay_callback is None else overlay_callback(temp_canvas)
            self.add_pil_frame(frame_to_write)
        self.canvas = last_composite.copy()
        self.draw = ImageDraw.Draw(self.canvas)

    def animate_circle(
        self,
        cmd,
        frames,
        overlay_callback: Optional[Callable[[Image.Image], Image.Image]] = None,
    ):
        if frames <= 0:
            self.execute_command_instant(cmd)
            return

        ss = self.ss
        if cmd['type'] == 'draw_circle':
            bbox = tuple(v * ss for v in cmd['bbox'])
            width_px = self._sw(cmd['width'])
            outline = cmd['outline']
        else:
            bbox = tuple(v * ss for v in cmd['xy'])
            width_px = self._sw(cmd.get('width', 1))
            outline = cmd.get('outline')

        if outline is None:
            self.execute_command_instant(cmd)
            return

        base_frame = self.canvas.copy()
        final_frame = base_frame.copy()
        ImageDraw.Draw(final_frame).ellipse(bbox, outline=outline, width=width_px)

        # `keep_above` declares marker regions that MUST stay on top of
        # the circle throughout its animation. Mask radius = marker's
        # outer edge EXACTLY (no pad) so the circle's stroke meets the
        # marker's outline cleanly with no visible white halo between
        # them. Every frame composites base_frame pixels back within
        # this mask — the circle is never painted over marker pixels.
        keep_above = cmd.get('keep_above') or []
        ka_mask = None
        if keep_above:
            ka_mask = Image.new("L", (self._iw, self._ih), 0)
            ka_draw = ImageDraw.Draw(ka_mask)
            for (cx, cy, r) in keep_above:
                scx, scy, sr = cx * ss, cy * ss, r * ss
                ka_draw.ellipse(
                    (scx - sr, scy - sr, scx + sr, scy + sr),
                    fill=255,
                )
            # Also reflect the keep_above in `final_frame` so the
            # end-state preserves markers.
            final_frame = Image.composite(base_frame, final_frame, ka_mask)

        last_composite = final_frame
        mask_width = width_px + 4 * ss

        for f in range(frames):
            end_angle = 360 * (f + 1) / frames
            if f == frames - 1:
                temp_canvas = final_frame
            else:
                frame_mask = Image.new("L", (self._iw, self._ih), 0)
                ImageDraw.Draw(frame_mask).arc(
                    bbox, start=0, end=end_angle,
                    fill=255, width=mask_width,
                )
                temp_canvas = Image.composite(final_frame, base_frame, frame_mask)
                if ka_mask is not None:
                    # Every frame: marker regions frozen to base_frame
                    # (pre-circle) so the circle's progressive stroke
                    # never touches the marker pixels.
                    temp_canvas = Image.composite(base_frame, temp_canvas, ka_mask)
            last_composite = temp_canvas
            frame_to_write = temp_canvas if overlay_callback is None else overlay_callback(temp_canvas)
            self.add_pil_frame(frame_to_write)

        self.canvas = last_composite.copy()
        self.draw = ImageDraw.Draw(self.canvas)

    def write_frame(self):
        self.add_pil_frame(self.canvas)

    def add_pil_frame(self, pil_img):
        # Downscale from internal supersampled to output 1x resolution.
        out = finish_scaled_canvas(pil_img.convert("RGB"), (self.width, self.height))
        self.frames.append(out)

    def save(self, path: Path):
        if not self.frames:
            return

        self.frames = fit_frames_to_video_budget(
            self.frames,
            max_frames=self.max_frames,
            final_hold_frames=self.final_hold_frames,
        )
        encode_rgb_frames_to_webm(self.frames, path, fps=16)


class PointTargetImageEvaluator(AbstractPuzzleEvaluator):
    """Decode terminal point-target images against generated metadata."""


    def image_option_from_frame(
        self,
        frame: np.ndarray,
        record: Dict[str, object],
    ) -> Tuple[Optional[str], int, Optional[Tuple[float, float]]]:
        return self._score_red_point(frame, record)

    def image_option_from_path_against_source(
        self,
        candidate_image: PathLike,
        source_image: PathLike,
        record: Dict[str, object],
    ) -> Tuple[Optional[str], int, Optional[Tuple[float, float]]]:
        """Decode only a newly committed terminal selection.

        Source diagrams may legitimately contain red arcs, rays, or other
        givens.  Scoring all red pixels can therefore mistake an unchanged
        input image for a selected answer.  Terminal-image verification must
        instead identify red support added relative to the original input.
        """

        candidate_bgr = cv2.imread(Path(candidate_image).as_posix(), cv2.IMREAD_COLOR)
        source_bgr = cv2.imread(Path(source_image).as_posix(), cv2.IMREAD_COLOR)
        if (
            candidate_bgr is None
            or source_bgr is None
            or candidate_bgr.shape != source_bgr.shape
        ):
            return None, 0, None
        candidate_rgb = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2RGB)
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        return self.image_option_from_frame_against_source(
            candidate_rgb,
            source_rgb,
            record,
        )

    def image_option_from_frame_against_source(
        self,
        frame: np.ndarray,
        source: np.ndarray,
        record: Dict[str, object],
    ) -> Tuple[Optional[str], int, Optional[Tuple[float, float]]]:
        if frame.shape != source.shape or frame.ndim != 3 or frame.shape[2] < 3:
            return None, 0, None
        height, width = frame.shape[:2]
        scale = self._extract_scale(record.get("canvas_dimensions"), width, height)
        if scale is None:
            return None, 0, None
        scale_x, scale_y = scale
        added_red = self._red_mask(frame).astype(bool) & ~self._red_mask(source).astype(bool)
        red_pixels = np.column_stack(np.nonzero(added_red))
        red_count = int(red_pixels.shape[0])
        if red_count < 20:
            return None, red_count, None
        mean_y = float(red_pixels[:, 0].mean())
        mean_x = float(red_pixels[:, 1].mean())
        centroid = (mean_x, mean_y)
        scaled_candidates: List[Dict[str, object]] = []
        for entry in record.get("candidates") or []:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label")
            x, y = entry.get("x"), entry.get("y")
            if label is None or x is None or y is None:
                continue
            scaled_candidates.append(
                {
                    "label": label,
                    "x": float(x) * scale_x,
                    "y": float(y) * scale_y,
                }
            )
        option = self._select_candidate_by_local_red_support(
            added_red,
            scaled_candidates,
            scale_x=scale_x,
            scale_y=scale_y,
            centroid=centroid,
            point_radius=float(record.get("point_radius", 10) or 10),
        )
        return option, red_count, centroid

    def _score_red_point(
        self,
        frame: np.ndarray,
        record: Dict[str, object],
    ) -> Tuple[Optional[str], int, Optional[Tuple[float, float]]]:
        height = frame.shape[0]
        width = frame.shape[1]
        canvas_dims_obj = record.get("canvas_dimensions")
        scale = self._extract_scale(canvas_dims_obj, width, height)
        if scale is None:
            return None, 0, None
        scale_x, scale_y = scale
        red_mask = self._red_mask(frame)
        red_pixels = np.column_stack(np.nonzero(red_mask > 0.5))
        candidates_raw = record.get("candidates") or []
        scaled_candidates: List[Dict[str, object]] = []
        for entry in candidates_raw:
            label = entry.get("label")
            x, y = entry.get("x"), entry.get("y")

            cx = float(x) * scale_x
            cy = float(y) * scale_y
            scaled_candidates.append({"label": label, "x": cx, "y": cy})
        # if want to exclude red pixels far from candidates:
        # threshold = min(height, width) / 5.0
        # if red_pixels.size and scaled_candidates:
        #     coords = red_pixels.astype(np.float32)
        #     candidate_coords = np.array([[c["y"], c["x"]] for c in scaled_candidates], dtype=np.float32)
        #     if candidate_coords.size:
        #         distance_sq = np.sum((coords[:, None, :] - candidate_coords[None, :, :]) ** 2, axis=2)
        #         mask = np.min(distance_sq, axis=1) <= (threshold * threshold)
        #         red_pixels = red_pixels[mask]
        red_count = int(red_pixels.shape[0])
        if red_count < 20:
            return None, red_count, None
        mean_y = float(red_pixels[:, 0].mean())
        mean_x = float(red_pixels[:, 1].mean())
        red_point = (mean_x, mean_y)

        best_support_label = self._select_candidate_by_local_red_support(
            red_mask.astype(bool),
            scaled_candidates,
            scale_x=scale_x,
            scale_y=scale_y,
            centroid=red_point,
            point_radius=float(record.get("point_radius", 10) or 10),
        )
        if best_support_label is not None:
            return best_support_label, red_count, red_point

        best_label: Optional[str] = None
        best_distance: float = math.inf
        for scaled in scaled_candidates:
            label, cx, cy = scaled['label'], scaled['x'], scaled['y']
            distance = math.hypot(cx - mean_x, cy - mean_y)

            if distance < best_distance:
                best_distance = distance
                best_label = label
        return best_label, red_count, red_point

    def _select_candidate_by_local_red_support(
        self,
        red_mask: np.ndarray,
        scaled_candidates: Sequence[Dict[str, object]],
        *,
        scale_x: float,
        scale_y: float,
        centroid: Tuple[float, float],
        point_radius: float,
    ) -> Optional[str]:
        if not scaled_candidates:
            return None

        height, width = red_mask.shape
        support_radius = max(12.0, point_radius * min(scale_x, scale_y) * 2.2)
        support_scores: List[Tuple[int, float, str]] = []

        for scaled in scaled_candidates:
            label_obj = scaled.get("label")
            cx_obj = scaled.get("x")
            cy_obj = scaled.get("y")
            if label_obj is None or cx_obj is None or cy_obj is None:
                continue
            label = str(label_obj).strip().upper()
            cx = float(cx_obj)
            cy = float(cy_obj)

            y0 = max(0, int(math.floor(cy - support_radius)))
            y1 = min(height, int(math.ceil(cy + support_radius + 1)))
            x0 = max(0, int(math.floor(cx - support_radius)))
            x1 = min(width, int(math.ceil(cx + support_radius + 1)))
            if y0 >= y1 or x0 >= x1:
                continue

            yy, xx = np.ogrid[y0:y1, x0:x1]
            disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (support_radius * support_radius)
            support = int(np.count_nonzero(red_mask[y0:y1, x0:x1] & disk))
            distance = math.hypot(cx - centroid[0], cy - centroid[1])
            support_scores.append((support, distance, label))

        if not support_scores:
            return None

        support_scores.sort(key=lambda item: (-item[0], item[1], item[2]))
        best_support, _, best_label = support_scores[0]
        if best_support <= 0:
            return None

        # A small subset of tasks (for example closest-pair selection) has a
        # typed multi-label answer and intentionally highlights two markers.
        # Count only full-strength local components; low support in a nearby
        # marker window is ordinary overlap leakage from one highlighted disk.
        strong_labels = sorted(
            label
            for support, _distance, label in support_scores
            if support >= max(50, int(math.ceil(best_support * 0.45)))
        )
        if len(strong_labels) > 1:
            return "".join(strong_labels)

        runner_up_support = support_scores[1][0] if len(support_scores) > 1 else 0
        # Nearby candidate disks can overlap the support window around the
        # selected marker.  A small amount of red leakage is therefore not a
        # second highlight: require a dominant local component instead of
        # concatenating every label whose window contains a few red pixels.
        if best_support >= 12 and (
            runner_up_support == 0
            or best_support >= runner_up_support + 8
            or best_support >= runner_up_support * 1.25
        ):
            return best_label
        return None

    def _extract_scale(
        self,
        canvas_dims_obj: object,
        width: int,
        height: int,
    ) -> Optional[Tuple[float, float]]:
        if isinstance(canvas_dims_obj, (list, tuple)) and len(canvas_dims_obj) >= 2:
            raw_width = canvas_dims_obj[0]
            raw_height = canvas_dims_obj[1]
        elif isinstance(canvas_dims_obj, dict) and {"width", "height"} <= set(canvas_dims_obj):
            raw_width = canvas_dims_obj["width"]
            raw_height = canvas_dims_obj["height"]
        else:
            # Geometry's family media contract fixes point-target images at
            # 512x512.  Canonical semantic inputs intentionally omit renderer
            # configuration, so a verifier replaying a canonical manifest
            # must derive this scale from the family contract instead of
            # requiring a copied ``canvas_dimensions`` oracle field.
            raw_width = 512
            raw_height = 512
        if not isinstance(raw_width, (int, float)) or not isinstance(raw_height, (int, float)):
            return None
        canvas_width = float(raw_width)
        canvas_height = float(raw_height)
        if canvas_width <= 0 or canvas_height <= 0:
            return None
        scale_x = width / canvas_width
        scale_y = height / canvas_height
        return scale_x, scale_y

    def _red_mask(self, frame: np.ndarray) -> np.ndarray:
        red = frame[:, :, 0].astype(np.float32)
        green = frame[:, :, 1].astype(np.float32)
        blue = frame[:, :, 2].astype(np.float32)
        dominance = red - np.maximum(green, blue)
        mask = (
            (red >= 140.0) &
            (dominance >= 40.0) &
            (green <= 130.0) &
            (blue <= 130.0)
        )
        return mask.astype(np.float32)


__all__ = [
    "PointCandidate",
    "PointTargetPuzzleGenerator",
    "PointTargetImageEvaluator",
]

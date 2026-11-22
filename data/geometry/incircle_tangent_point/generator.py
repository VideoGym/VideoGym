"""Incircle tangent-point puzzle generator."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.render_primitives import ScaledDraw
from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class IncircleTangentPointGenerator(PointTargetPuzzleGenerator):
    """Find the incircle contact on a marked side from the three bisectors."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A black triangle is shown on a white canvas with five labeled candidate circles near one marked "
        "side; that side is the target side. Rule: The three internal angle bisectors meet at the incenter. Use the "
        "distance from the incenter to the target side as the incircle radius, then identify where the circle touches "
        "that side. Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/incircle_tangent_point"
    EXPECTED_VIDEO_FRAMES = 33
    STATE_TRACE = [
        "input_hold",
        "first_angle_bisector_single_end_growth",
        "second_angle_bisector_single_end_growth",
        "third_angle_bisector_single_end_growth",
        "incenter_marker",
        "incircle_single_end_growth",
        "correct_tangent_point_highlight",
        "final_hold",
    ]

    DEFAULT_TI2V_PROMPT = (
        "Input Image: A white canvas shows a black triangle and five labeled candidate circles near one marked side. "
        "Required Output: Hold the unchanged scene, draw the three complete cyan internal angle-bisector segments "
        "one at a time from their triangle vertices through the common incenter to the opposite sides, place a "
        "hollow marker at that incenter, and draw the cyan incircle using the distance from that center to the "
        "marked side as its radius. Finally highlight the candidate at the contact point on that side. Do not draw "
        "a perpendicular spoke, move any source object, or show a choice-card verdict."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Input Image: A black triangle is shown on a white canvas with five labeled candidate circles near one marked "
        "side. Required Output: Preserve the triangle and candidates. Draw the three complete cyan internal "
        "angle-bisector segments from their triangle vertices through the common incenter to the opposite sides, "
        "add a hollow marker at that incenter, and draw the cyan incircle whose radius is the center-to-side "
        "distance. Highlight only the candidate at the incircle contact on the marked side. Do not draw a "
        "perpendicular spoke."
    )

    # The marked side is kept long so the option row can reach four markers
    # past the contact on either side; the contact's own share of that side is
    # then free to range widely.
    MARKED_SIDE_RANGE = (348.0, 370.0)
    MAX_SIDE = 372.0
    MIN_ANGLE_DEG = 31.0
    MAX_ANGLE_DEG = 105.0

    def _apex_run_bounds(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Apex tangent lengths that keep every interior angle presentable.

        With tangent lengths (x, y, z) the angle at each vertex is
        ``2*atan(r/run)`` for that vertex's own run, and ``r*r = x*y*z/(x+y+z)``.
        Both angle bounds and the canvas-fit bound on the two unmarked sides
        are therefore closed forms in ``z``, so the admissible interval is
        solved rather than searched.
        """

        low = math.tan(math.radians(self.MIN_ANGLE_DEG / 2.0)) ** 2
        high = math.tan(math.radians(self.MAX_ANGLE_DEG / 2.0)) ** 2
        span = x + y
        lower = 1e-6
        upper = self.MAX_SIDE - max(x, y)
        for run, other in ((x, y), (y, x)):
            # angle at this vertex >= MIN: z * (other - low*run) >= low*run*span
            slack = other - low * run
            need = low * run * span
            if slack <= 0.0:
                return None
            lower = max(lower, need / slack)
            # angle at this vertex <= MAX
            slack = other - high * run
            if slack > 0.0:
                upper = min(upper, high * run * span / slack)
        # apex angle bounds: x*y >= low*z*(span+z) and x*y <= high*z*(span+z)
        upper = min(upper, (-span + math.sqrt(span * span + 4.0 * x * y / low)) / 2.0)
        lower = max(lower, (-span + math.sqrt(span * span + 4.0 * x * y / high)) / 2.0)
        return (lower, upper) if lower < upper else None

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        for _ in range(999):
            # The triangle is built from its three tangent lengths rather than
            # from three sampled vertices.  Any positive triple (x, y, z) is
            # realisable -- the sides are y+z, z+x, x+y -- and the incircle
            # meets the marked side x from one end and y from the other.  The
            # split is drawn uniformly across every share the triangle's own
            # angle bounds admit, so the contact is not parked near the middle
            # of its side, where "the marker nearest the side's midpoint" would
            # answer the sample without the incircle.
            side = self._rng.uniform(*self.MARKED_SIDE_RANGE)
            shares = [
                fraction / 200.0
                for fraction in range(40, 161)
                if self._apex_run_bounds(side * fraction / 200.0, side * (1.0 - fraction / 200.0))
                is not None
            ]
            if not shares:
                continue
            share = self._rng.choice(shares) + self._rng.uniform(-0.0025, 0.0025)
            x = side * share
            y = side - x
            bounds = self._apex_run_bounds(x, y)
            if bounds is None:
                continue
            z = self._rng.uniform(*bounds)
            placed = self._place_triangle(x, y, z)
            if placed is None:
                continue
            p1, p2, p3 = placed

            a = self.distance(p2, p3)
            b = self.distance(p1, p3)
            c = self.distance(p1, p2)
            perimeter = a + b + c
            self.triangle_points = (p1, p2, p3)
            self.incenter = Point(
                (a * p1.x + b * p2.x + c * p3.x) / perimeter,
                (a * p1.y + b * p2.y + c * p3.y) / perimeter,
            )

            pa, pb, pc = p1, p2, p3
            target, _parameter = self.project_point_onto_line(self.incenter, pa, pb)
            if not self.point_can_host_candidate(target):
                continue
            try:
                self._place_tangent_candidates(target, pa, pb, pc)
            except RuntimeError:
                continue
            self.side_verts = (pa, pb)
            self.opposite = pc
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate incircle tangent point puzzle.")

    def _place_triangle(
        self,
        x: float,
        y: float,
        z: float,
    ) -> Optional[Tuple[Point, Point, Point]]:
        """Lay the tangent-length triangle on the canvas at a free heading.

        The marked side runs from the first vertex to the second; the apex is
        third.  The whole figure is kept inside the box a marker may occupy, so
        every option on the marked side is automatically on the canvas, and the
        offset within that box is drawn uniformly so the figure is not pinned
        to the middle of the frame.
        """

        side_c = x + y
        side_a = y + z
        side_b = z + x
        # Cosine rule in the triangle's own frame: the marked side on the x
        # axis, the apex above it.
        apex_x = (side_b * side_b + side_c * side_c - side_a * side_a) / (2.0 * side_c)
        apex_y_sq = side_b * side_b - apex_x * apex_x
        if apex_y_sq <= 1.0:
            return None
        local = [(0.0, 0.0), (side_c, 0.0), (apex_x, math.sqrt(apex_y_sq))]
        theta = self._rng.uniform(0.0, math.tau)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rotated = [
            (px * cos_t - py * sin_t, px * sin_t + py * cos_t) for px, py in local
        ]
        left, top, right, bottom = self.canvas_bounds()
        pad_x, pad_y = self._candidate_safe_padding(include_edge_buffer=True)
        min_x = min(point[0] for point in rotated)
        max_x = max(point[0] for point in rotated)
        min_y = min(point[1] for point in rotated)
        max_y = max(point[1] for point in rotated)
        low_x, high_x = left + pad_x - min_x, right - pad_x - max_x
        low_y, high_y = top + pad_y - min_y, bottom - pad_y - max_y
        if low_x > high_x or low_y > high_y:
            return None
        shift_x = self._rng.uniform(low_x, high_x)
        shift_y = self._rng.uniform(low_y, high_y)
        return tuple(  # type: ignore[return-value]
            Point(px + shift_x, py + shift_y) for px, py in rotated
        )

    def _place_tangent_candidates(
        self,
        target: Point,
        pa: Point,
        pb: Point,
        pc: Point,
    ) -> None:
        """Spread A-E along the marked side itself.

        The contact point lies on that side, so every option lies on it: the
        answer-free reading "which marker is on the triangle?" must not narrow
        the field.  A distractor is wrong because the incircle
        would not reach the side there, not because it floats beside the edge.
        The other two edges keep their marker clearance so no option collides
        with a corner.
        """

        if self.distance(pa, pb) < 1e-9:
            raise RuntimeError("Degenerate triangle side")
        marker_clear = max(self.point_radius * 2.4, 28.0)
        corner_clear = self.point_radius * 2.1
        self.place_candidates_on_incidence(
            target,
            [Incidence.segment(pa, pb)],
            min_offset=marker_clear,
            # The pairwise floor sits a hair under the step so a row laid out
            # at exactly the step is not pushed outward by rounding: every
            # extra pixel spent there is a pixel of the side the outermost
            # ranks need, and losing them puts the answer back in the middle
            # of its own row.
            min_separation=marker_clear * 0.98,
            reject=lambda point: (
                self.distance(point, pa) < corner_clear
                or self.distance(point, pb) < corner_clear
            ),
        )

    @staticmethod
    def _line_intersection(
        a: Point,
        b: Point,
        c: Point,
        d: Point,
    ) -> Optional[Point]:
        abx, aby = b.x - a.x, b.y - a.y
        cdx, cdy = d.x - c.x, d.y - c.y
        denominator = abx * cdy - aby * cdx
        if abs(denominator) <= 1e-9:
            return None
        acx, acy = c.x - a.x, c.y - a.y
        t = (acx * cdy - acy * cdx) / denominator
        return Point(a.x + t * abx, a.y + t * aby)

    def _ordered_vertices(self) -> List[Point]:
        """Return the triangle vertices in a stable cyclic image order."""

        vertices = list(self.triangle_points)
        centroid_x = sum(point.x for point in vertices) / 3.0
        centroid_y = sum(point.y for point in vertices) / 3.0
        clockwise = sorted(
            vertices,
            key=lambda point: math.atan2(point.y - centroid_y, point.x - centroid_x),
        )
        start = min(range(3), key=lambda index: (clockwise[index].y, clockwise[index].x))
        return clockwise[start:] + clockwise[:start]

    def _angle_bisector_segments(self) -> List[Tuple[Point, Point]]:
        vertices = list(self._ordered_vertices())
        result: List[Tuple[Point, Point]] = []
        for index, vertex in enumerate(vertices):
            opposite_a = vertices[(index + 1) % 3]
            opposite_b = vertices[(index + 2) % 3]
            endpoint = self._line_intersection(vertex, self.incenter, opposite_a, opposite_b)
            result.append((vertex, endpoint or self.incenter))
        return result

    def inradius(self) -> float:
        vertices = list(self.triangle_points)
        perimeter = sum(
            self.distance(vertices[index], vertices[(index + 1) % 3])
            for index in range(3)
        )
        area = abs(
            sum(
                vertices[index].x * vertices[(index + 1) % 3].y
                - vertices[(index + 1) % 3].x * vertices[index].y
                for index in range(3)
            )
        ) / 2.0
        return area / max(perimeter / 2.0, 1e-9)

    def _authored_observation(self) -> str:
        vertices = list(self.triangle_points)
        names = [
            name.removesuffix(" visible vertex")
            for name in self._visible_point_names(vertices)
        ]
        side_a, side_b = self.side_verts
        index_a = min(range(3), key=lambda index: self.distance(vertices[index], side_a))
        index_b = min(range(3), key=lambda index: self.distance(vertices[index], side_b))
        return (
            f"The labeled markers crowd the side running between the {names[index_a]} and {names[index_b]} "
            "corners, singling it out as the target side. The three internal angle bisectors meet at the "
            f"incenter, {self._canvas_region(self.incenter)}, and the incircle grown from the center-to-side "
            "distance can only touch the target side at the foot of the perpendicular from that center, the "
            f"point of the side nearest it, {self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> Dict[str, object]:
        structured = {
            "triangle_vertices": [p.to_list() for p in self.triangle_points],
            "side_vertices": [p.to_list() for p in self.side_verts],
            "incenter": self.incenter.to_list(),
            "inradius": self.inradius(),
            "correct_option": self.correct_label,
            "state_trace": list(self.STATE_TRACE),
            "answer_schema": {
                "terminal_text_regex": FIVE_CHOICE_CODEC.full_pattern,
                "requires_sample_specific_reasoning_before_terminal_answer": True,
                "terminal_text_contains_exact_answer": True,
                "no_block_after_terminal_image": True,
            },
            "verifier_contract": {
                "profile": "incircle_tangent_point_multimodal_semantic_hard_gate_v1",
                "judged_artifacts": ["text", "final_image", "full_video"],
                "strict_binary_correctness": True,
            },
            "render_geometry": {
                "canvas_dimensions": list(self.canvas_dimensions),
                "point_radius": self.point_radius,
                "line_width": self.line_width,
            },
            "expected_video_frame_count": self.EXPECTED_VIDEO_FRAMES,
            "video_fps": 16,
        }
        return {
            "triangle_vertices": structured["triangle_vertices"],
            "side_vertices": structured["side_vertices"],
            "incenter": structured["incenter"],
            "inradius": structured["inradius"],
            "structured_answer": structured,
        }

    def _semantic_phase_roles_for_sample(self) -> tuple[str, ...]:
        return (
            "the first complete internal angle-bisector segment from a vertex through the incenter to the opposite side",
            "the first two retained complete internal angle-bisector segments",
            "all three retained complete internal angle-bisector segments through their common incenter",
            "the hollow incenter marker and incircle using the center-to-side distance as its radius",
        )

    def _draw_incircle_progress(self, draw: Any, progress: float) -> None:
        progress = max(0.0, min(1.0, float(progress)))
        if progress <= 0.0:
            return
        radius = self.inradius()
        bbox = (
            self.incenter.x - radius,
            self.incenter.y - radius,
            self.incenter.x + radius,
            self.incenter.y + radius,
        )
        start_angle = math.degrees(
            math.atan2(self.target_point.y - self.incenter.y, self.target_point.x - self.incenter.x)
        )
        if progress >= 1.0:
            draw.ellipse(bbox, outline=self.CONSTRUCTION_COLOR, width=self.line_width)
        else:
            draw.arc(
                bbox,
                start=start_angle,
                end=start_angle + 360.0 * progress,
                fill=self.CONSTRUCTION_COLOR,
                width=self.line_width,
            )

    def _render_state(
        self,
        *,
        bisector_progresses: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        marker_progress: float = 0.0,
        circle_progress: float = 0.0,
        highlight_progress: float = 0.0,
    ) -> Image.Image:
        draw, base = self.get_draw_base()
        for (start, end), raw_progress in zip(self._angle_bisector_segments(), bisector_progresses):
            progress = max(0.0, min(1.0, float(raw_progress)))
            if progress <= 0.0:
                continue
            current_end = Point(
                start.x + (end.x - start.x) * progress,
                start.y + (end.y - start.y) * progress,
            )
            self.draw_line(
                draw,
                [start, current_end],
                fill=self.CONSTRUCTION_COLOR,
                width_factor=0.72,
                join_start=True,
            )
        if marker_progress > 0.0:
            self.draw_anchor_marker(draw, self.incenter, 7)
        self._draw_incircle_progress(draw, circle_progress)
        self.draw_line(draw, list(self.triangle_points) + [self.triangle_points[0]])
        highlight_progress = max(0.0, min(1.0, float(highlight_progress)))
        if 0.0 < highlight_progress < 1.0:
            # The candidates are an opaque overlay on the completed
            # construction, so interpolate the two complete overlay states at
            # supersampled resolution.  This preserves one fixed marker
            # footprint and makes the semantic answer commit observable instead
            # of silently treating every partial step as the unselected state.
            unselected = base.copy()
            self.draw_candidates(
                ScaledDraw(ImageDraw.Draw(unselected), self.SUPERSAMPLE),
                highlight_label=None,
            )
            selected = base.copy()
            self.draw_candidates(
                ScaledDraw(ImageDraw.Draw(selected), self.SUPERSAMPLE),
                highlight_label=self.correct_label,
            )
            base = Image.blend(unselected, selected, alpha=highlight_progress)
            draw = ScaledDraw(ImageDraw.Draw(base), self.SUPERSAMPLE)
        else:
            self.draw_candidates(
                draw,
                highlight_label=(
                    self.correct_label if highlight_progress >= 1.0 else None
                ),
            )
        # Keep the hollow incenter marker visible when a candidate occupies
        # the same center point.
        if marker_progress > 0.0:
            self.draw_anchor_marker(draw, self.incenter, 7)
        return base

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        solved = bool(highlight_label)
        return self._render_state(
            bisector_progresses=(1.0, 1.0, 1.0) if solved else (0.0, 0.0, 0.0),
            marker_progress=1.0 if solved else 0.0,
            circle_progress=1.0 if solved else 0.0,
            highlight_progress=1.0 if solved else 0.0,
        )

    def save_video_solution(self, pid: str) -> Optional[int]:
        frames: List[Image.Image] = []

        def frame(**kwargs: float) -> Image.Image:
            return self._downscale(self._render_state(**kwargs))

        initial = frame()
        frames.extend(initial.copy() for _ in range(4))
        progresses = [0.0, 0.0, 0.0]
        for index in range(3):
            for step in range(1, 5):
                current = list(progresses)
                current[index] = step / 4.0
                frames.append(frame(bisector_progresses=tuple(current)))
            progresses[index] = 1.0
        frames.append(frame(bisector_progresses=(1.0, 1.0, 1.0), marker_progress=1.0))
        frames.extend(
            frame(
                bisector_progresses=(1.0, 1.0, 1.0),
                marker_progress=1.0,
                circle_progress=step / 7.0,
            )
            for step in range(1, 8)
        )
        frames.extend(
            frame(
                bisector_progresses=(1.0, 1.0, 1.0),
                marker_progress=1.0,
                circle_progress=1.0,
                highlight_progress=step / 4.0,
            )
            for step in range(1, 5)
        )
        final = frame(
            bisector_progresses=(1.0, 1.0, 1.0),
            marker_progress=1.0,
            circle_progress=1.0,
            highlight_progress=1.0,
        )
        frames.extend(final.copy() for _ in range(5))
        if len(frames) != self.EXPECTED_VIDEO_FRAMES:
            raise RuntimeError(f"incircle_tangent_point produced invalid frame count {len(frames)}")
        output_path = self.solution_dir / f"{pid}_solution.webm"
        arrays = [np.asarray(item.convert("RGB"), dtype=np.uint8) for item in frames]
        if not encode_rgb_frames_to_webm(arrays, output_path, fps=16):
            raise RuntimeError("Failed to encode incircle tangent point semantic video")
        return len(frames)

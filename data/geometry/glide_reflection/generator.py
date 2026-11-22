"""Glide reflection puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class GlideReflectionGenerator(PointTargetPuzzleGenerator):
    """Apply a glide reflection (translate along axis + reflect across axis) to a point."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows a black reflection axis, a blue arrow along that axis with marked endpoints, one "
        "hollow source point, and five labeled candidate circles A-E. Rule: Reflect the source perpendicularly across the "
        "black axis, then translate the reflected point by the blue arrow's tail-to-head vector. Select the candidate at "
        "the final position. Required Output: Answer with one option from A-E."
    )

    GLIDE_ARROW_COLOR = (46, 110, 220)
    DEFAULT_OUTPUT_DIR = "outputs/geometry/glide_reflection"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a black reflection axis segment of medium thickness, a short blue arrow on the axis "
        "whose marked endpoints show the glide direction and distance, "
        "and one small hollow source marker with white fill and a thick black outline on one side of the axis. "
        "Five candidate markers A-E are clustered near the hidden glide-reflected location; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "First reflect the source across the axis, then translate the result along the axis by the distance between the two translation markers. "
        "The video first holds the axis, translation markers, source marker, and candidate markers, then draws one black connector from the source across the axis and along it to the correct candidate. "
        "In the final state, only the glide-reflected candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    @staticmethod
    def _reflect(point: Point, lp1: Point, lp2: Point) -> Point:
        dx, dy = lp2.x - lp1.x, lp2.y - lp1.y
        denom = dx * dx + dy * dy
        if denom < 1e-9:
            return point
        t = ((point.x - lp1.x) * dx + (point.y - lp1.y) * dy) / denom
        proj = Point(lp1.x + t * dx, lp1.y + t * dy)
        return Point(2 * proj.x - point.x, 2 * proj.y - point.y)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            axis_mid = self.pick_target_point(
                0.40, padding=candidate_padding + self.canvas_short_side * 0.16
            )
            axis_angle = self._rng.uniform(0.0, math.pi)
            try:
                ap1, ap2 = self.sample_symmetric_segment(
                    axis_mid,
                    axis_angle,
                    min_half_length=self.canvas_short_side * 0.20,
                    max_half_length=self.canvas_short_side * 0.32,
                    padding=self.line_width,
                )
            except RuntimeError:
                continue

            # Translation markers on axis
            glide_dist = self._rng.uniform(
                self.canvas_short_side * 0.08,
                self.canvas_short_side * 0.16,
            )
            glide_sign = self._rng.choice([-1, 1])
            tm_offset = self._rng.uniform(0.0, self.canvas_short_side * 0.05)
            tm1 = self.point_on_ray(axis_mid, axis_angle, tm_offset)
            tm2 = self.point_on_ray(
                tm1, axis_angle if glide_sign > 0 else axis_angle + math.pi, glide_dist
            )

            # Source point off axis
            normal = axis_angle + math.pi / 2 * self._rng.choice([-1, 1])
            source_offset = self._rng.uniform(
                self.canvas_short_side * 0.14,
                self.canvas_short_side * 0.26,
            )
            source = self.point_on_ray(axis_mid, normal, source_offset)

            # Glide reflection: reflect then translate along axis
            reflected = self._reflect(source, ap1, ap2)
            dx = math.cos(axis_angle) * glide_dist * glide_sign
            dy = math.sin(axis_angle) * glide_dist * glide_sign
            target = Point(reflected.x + dx, reflected.y + dy)

            if not all(
                self.inside_canvas(p, padding=self.line_width + 4)
                for p in [source, target, tm1, tm2]
            ):
                continue
            if not self.point_can_host_candidate(target):
                continue
            if self.distance(source, target) < self.canvas_short_side * 0.20:
                continue

            # Place distractors near the glide-reflected target on the
            # SAME side of the axis as the target, instead of the default
            # `place_candidates` which can scatter them across the canvas.
            try:
                self._place_glide_candidates(target, ap1, ap2, source)
            except RuntimeError:
                continue

            self.axis_points = (ap1, ap2)
            self.trans_markers = (tm1, tm2)
            self.source_point = source
            self.reflected = reflected
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate glide reflection puzzle.")

    def _place_glide_candidates(
        self,
        target: Point,
        ap1: Point,
        ap2: Point,
        source: Point,
    ) -> None:
        """Place 5 markers around the glide-reflected target. The 4
        distractors must lie on the same side of the axis as the target
        (so "same side as source" is no longer a free shortcut), be
        within ~70 px of the answer, and not overlap source/target/each
        other.
        """
        # Sign of the target's signed distance to the axis line, so
        # distractors stay on the same half-plane as the image point.
        ax_dx = ap2.x - ap1.x
        ax_dy = ap2.y - ap1.y
        target_side = (target.x - ap1.x) * ax_dy - (target.y - ap1.y) * ax_dx
        target_sign = 1.0 if target_side >= 0 else -1.0
        axis_clear = max(self.point_radius * 1.4, 16.0)
        marker_clear = max(self.point_radius * 2.4, 28.0)
        source_clear = max(self.point_radius * 2.4, 28.0)

        def same_side(point: Point) -> bool:
            side = (point.x - ap1.x) * ax_dy - (point.y - ap1.y) * ax_dx
            return (
                side * target_sign > 0
                and self.distance_point_to_line(point, ap1, ap2) >= axis_clear
                and self.distance(point, source) >= source_clear
            )

        # Laying the cloud out before choosing which slot the answer takes
        # keeps it off the middle of its own set.
        self.seat_candidates(target, accept=same_side, min_separation=marker_clear)

    def _authored_observation(self) -> str:
        axis_p1, axis_p2 = self.axis_points
        marker_start, marker_end = self.trans_markers
        axis_direction = self._heading_phrase(axis_p2.x - axis_p1.x, axis_p2.y - axis_p1.y)
        glide_direction = self._heading_phrase(
            marker_end.x - marker_start.x, marker_end.y - marker_start.y,
        )
        return (
            f"The black axis runs {axis_direction}, the blue arrow on it points {glide_direction}, and the hollow "
            f"source marker sits {self._canvas_region(self.source_point)} on one side of the axis. Reflecting the "
            f"source perpendicularly across the axis lands at equal distance on the opposite side, "
            f"{self._canvas_region(self.reflected)}; sliding that reflected point along the axis by the blue "
            f"arrow's full length carries it to its final spot, {self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "difficulty_variant": "reflection_then_axis_translation",
            "axis": [p.to_list() for p in self.axis_points],
            "translation_markers": [p.to_list() for p in self.trans_markers],
            "source_point": self.source_point.to_list(),
            "intermediate_reflected": self.reflected.to_list(),
        }

    def _draw_static(self, draw) -> None:
        start, end = self.trans_markers
        angle = math.atan2(end.y - start.y, end.x - start.x)
        head_length = self.canvas_short_side * 0.025
        left = self.point_on_ray(end, angle + math.pi + math.radians(25.0), head_length)
        right = self.point_on_ray(end, angle + math.pi - math.radians(25.0), head_length)
        self.draw_line(draw, [start, end], fill=self.GLIDE_ARROW_COLOR, join_start=True, join_end=True)
        self.draw_line(draw, [left, end, right], fill=self.GLIDE_ARROW_COLOR, join_end=True)
        self.draw_anchor_marker(draw, self.trans_markers[0], 5)
        self.draw_anchor_marker(draw, self.trans_markers[1], 5)
        self.draw_anchor_marker(draw, self.source_point, 7)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_static(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, list(self.axis_points))
        if highlight_label:
            _, te2 = self.trim_segment(
                self.reflected, self.target_point,
                end_offset=float(self.point_radius),
            )
            self.draw_line(
                draw,
                [self.source_point, self.reflected],
                fill=self.CONSTRUCTION_COLOR,
                join_end=True,
            )
            self.draw_line(
                draw,
                [self.reflected, te2],
                fill=self.CONSTRUCTION_COLOR,
                join_start=True,
            )
        self._draw_static(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

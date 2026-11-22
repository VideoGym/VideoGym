"""Circle tangent intersection puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class CircleTangentIntersectionGenerator(PointTargetPuzzleGenerator):
    """Find where tangent lines from two points on a circle intersect externally."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows one black circle, two hollow markers on its circumference, and five labeled "
        "candidate circles A-E. Rule: At each hollow circumference marker, construct the tangent line perpendicular "
        "to the radius through that marker. Extend both tangent lines outside the circle and select the candidate at "
        "their unique intersection. Required Output: Return exactly one uppercase option label in the form "
        "\"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/circle_tangent_intersection"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows an unfilled circle drawn with a black outline of medium thickness and two small hollow white markers on the circumference. "
        "Five candidate markers A-E are clustered near the hidden intersection of the two tangent lines; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the circle, the two circumference markers, and the candidate markers, then draws two black tangent line segments of medium thickness, one from each circumference marker, meeting at the correct candidate. "
        "In the final state, only the intersection marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            R = self._rng.uniform(self.canvas_short_side * 0.12, self.canvas_short_side * 0.22)
            pad = candidate_padding + R + self.canvas_short_side * 0.06
            if pad >= self.canvas_short_side * 0.40:
                continue
            center = self.pick_target_point(0.38, padding=pad)
            if not self.circle_fits(center, R, extra_padding=self.line_width + self.canvas_short_side * 0.06):
                continue

            # Two points on circle with angular separation 60-150 deg
            a1 = self._rng.uniform(0.0, math.tau)
            sep = self._rng.uniform(math.radians(60), math.radians(150))
            a2 = a1 + sep
            cp1 = self.point_on_ray(center, a1, R)
            cp2 = self.point_on_ray(center, a2, R)

            # Tangent at cp1: perpendicular to radius, direction = a1 + pi/2
            # Tangent at cp2: perpendicular to radius, direction = a2 + pi/2
            # Find intersection of the two tangent lines
            mid_a = (a1 + a2) / 2
            half_sep = sep / 2
            cos_half = math.cos(half_sep)
            if cos_half < 1e-6:
                continue
            dist_to_intersection = R / cos_half
            target = self.point_on_ray(center, mid_a, dist_to_intersection)

            if not self.inside_canvas(target, padding=self.line_width + 4):
                continue
            if not self.point_can_host_candidate(target):
                continue
            if self.distance(target, center) < R + self.canvas_short_side * 0.04:
                continue

            try:
                self.place_candidates(target)
            except RuntimeError:
                continue

            self.center = center
            self.R = R
            self.cp1 = cp1
            self.cp2 = cp2
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate circle tangent intersection puzzle.")

    def _authored_observation(self) -> str:
        return (
            f"The circle rests {self._canvas_region(self.center)} with two hollow markers on its rim. The tangent "
            f"at each marker runs perpendicular to its own radius, so the two tangent lines peel away from the "
            f"rim and cross exactly once, outside the circle "
            f"{self._relative_position_phrase(self.center, self.target_point)} it, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "center": self.center.to_list(),
            "radius": self.R,
            "circumference_points": [self.cp1.to_list(), self.cp2.to_list()],
        }

    def _draw_circ_markers(self, draw) -> None:
        self.draw_anchor_marker(draw, self.cp1, 7)
        self.draw_anchor_marker(draw, self.cp2, 7)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_circ_markers(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_circle(draw, self.center, self.R)
        if highlight_label:
            _, te1 = self.trim_segment(self.cp1, self.target_point, end_offset=float(self.point_radius))
            _, te2 = self.trim_segment(self.cp2, self.target_point, end_offset=float(self.point_radius))
            self.draw_line(draw, [self.cp1, te1])
            self.draw_line(draw, [self.cp2, te2])
        self._draw_circ_markers(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

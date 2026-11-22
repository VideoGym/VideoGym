"""Equilateral triangle vertex puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class EquilateralTriangleVertexGenerator(PointTargetPuzzleGenerator):
    """Find the third vertex that completes an equilateral triangle."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows a black line segment with a solid black disc anchor at each endpoint, plus five labeled "
        "candidate circles A-E clustered to one side. Identify which candidate completes a perfect equilateral triangle "
        "with the base segment (all three sides equal, all angles 60 degrees). Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/equilateral_triangle_vertex"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a single thick black line segment (the base) with a small solid black disc anchor at each endpoint. "
        "Five candidate markers A-E are clustered near the hidden apex of the equilateral triangle that would be built on this base; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the base segment, the endpoint discs, and the candidate markers, then draws two black segments of medium thickness from the base endpoints to the correct candidate, completing the equilateral triangle. "
        "In the final state, only the apex marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            mid = self.pick_target_point(0.44, padding=candidate_padding + self.canvas_short_side * 0.16)
            angle = self._rng.uniform(0.0, math.tau)
            try:
                p1, p2 = self.sample_symmetric_segment(
                    mid, angle,
                    min_half_length=self.canvas_short_side * 0.12,
                    max_half_length=self.canvas_short_side * 0.22,
                    padding=self.line_width + 8,
                )
            except RuntimeError:
                continue
            perp = angle + math.pi / 2 * self._rng.choice([-1, 1])
            h = self.distance(p1, p2) * math.sqrt(3) / 2
            apex = Point(
                x=(p1.x + p2.x) / 2 + h * math.cos(perp),
                y=(p1.y + p2.y) / 2 + h * math.sin(perp),
            )
            if not self.point_can_host_candidate(apex):
                continue
            if not self.inside_canvas(apex, padding=self.line_width + 4):
                continue
            try:
                self.place_candidates(apex)
            except RuntimeError:
                continue
            self.base_p1 = p1
            self.base_p2 = p2
            self.target_point = apex
            return self.save_puzzle()
        raise RuntimeError("Failed to generate equilateral triangle vertex puzzle.")

    def _authored_observation(self) -> str:
        base_mid = Point(
            (self.base_p1.x + self.base_p2.x) / 2.0,
            (self.base_p1.y + self.base_p2.y) / 2.0,
        )
        base_run = self._heading_phrase(
            self.base_p2.x - self.base_p1.x,
            self.base_p2.y - self.base_p1.y,
        )
        apex_relation = self._relative_position_phrase(base_mid, self.target_point)
        return (
            f"The thick base runs {base_run} between its two solid end discs, its midpoint "
            f"{self._canvas_region(base_mid)}. The missing corner of an equilateral triangle keeps both new "
            "sides exactly as long as the base, so it lies on the perpendicular through the base's midpoint, "
            f"one full base length from each end disc; here that corner sits {apex_relation} the midpoint, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {"base": [self.base_p1.to_list(), self.base_p2.to_list()]}

    def _draw_endpoint_disc(self, draw, center, radius) -> None:
        cx, cy = round(center.x), round(center.y)
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(
            bbox,
            fill=self.CANDIDATE_OUTLINE_COLOR,
            outline=self.CANDIDATE_OUTLINE_COLOR,
            width=1,
        )

    def _draw_endpoints(self, draw) -> None:
        # Solid black filled discs at each base endpoint anchor the
        # segment more emphatically than the prior hollow markers, so
        # the "line segment with two end-points" reads as the base of an
        # equilateral triangle from the puzzle image alone.
        self._draw_endpoint_disc(draw, self.base_p1, 8)
        self._draw_endpoint_disc(draw, self.base_p2, 8)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_endpoints(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        # Bumped width_factor (1.0 -> 1.25) so the base segment reads as
        # noticeably thicker than the answer-side strokes that get drawn
        # in the solution; the base anchors the puzzle's "this is THE
        # base segment of the equilateral triangle" framing.
        self.draw_line(
            draw, [self.base_p1, self.base_p2],
            width_factor=1.25,
        )
        if highlight_label:
            self.draw_line(draw, [self.base_p1, self.target_point, self.base_p2])
        self._draw_endpoints(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

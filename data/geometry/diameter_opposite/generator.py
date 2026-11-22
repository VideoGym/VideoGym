"""Diameter-opposite point_geometry puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import ImageDraw

from data.geometry.core.point_target_base import (
    Incidence,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class DiameterOppositeGenerator(PointTargetPuzzleGenerator):
    """Find the point opposite a marked circumference point across a circle center."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows a black circle, one hollow source marker on the circle boundary, and five labeled candidate "
        "circles A-E. Determine which candidate is diametrically opposite the source point across the circle center. Answer "
        "with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/diameter_opposite"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows one black circle outline of medium thickness and one small hollow source marker with "
        "white fill and a thick black outline placed on the circle boundary. Five candidate markers A-E are arranged near "
        "the hidden diametrically opposite location; each marker is a small white circle with a thin dark gray outline and "
        "a black uppercase letter. The video first holds the circle, the source marker, and the candidate markers, then "
        "draws one black diameter segment straight through the circle toward the correct opposite candidate. In the final "
        "state, only the candidate at the diametrically opposite position changes to pale red fill with a dark red outline, "
        "while the other markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        tries = 0
        while tries < 999:
            tries += 1
            radius = self._rng.uniform(self.canvas_short_side * 0.18, self.canvas_short_side * 0.28)
            center = self.pick_target_point(
                0.4,
                padding=candidate_padding + radius + self.line_width,
            )
            if not self.circle_fits(center, radius, extra_padding=self.line_width):
                continue
            angle = self._rng.uniform(0.0, math.tau)
            source_point = self.point_on_ray(center, angle, radius)
            target_point = self.point_on_ray(center, angle + math.pi, radius)
            if not self.inside_canvas(source_point, padding=self.line_width):
                continue
            if not self.point_can_host_candidate(target_point):
                continue
            # The antipode is a point of the drawn rim, so every option is a
            # point of that rim.  Spreading the options along
            # the tangent instead used to leave the outermost ones visibly off
            # the circle, which narrowed the field for free.  A distractor is
            # wrong because the chord from the source misses the centre there.
            try:
                self.place_candidates_on_incidence(
                    target_point,
                    [Incidence.circle(center, radius)],
                    reject=lambda point: self.distance(point, source_point)
                    < self.point_radius * 2.6,
                )
            except (RuntimeError, ValueError):
                continue
            self.circle_center = center
            self.circle_radius = int(round(radius))
            self.source_point = source_point
            self.target_point = target_point
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid diameter-opposite puzzle.")

    def _authored_observation(self) -> str:
        side_names = {
            "up": "top", "down": "bottom", "left": "left", "right": "right",
            "upper-left": "upper-left", "upper-right": "upper-right",
            "lower-left": "lower-left", "lower-right": "lower-right",
        }
        source_side = side_names[self._direction_name(
            self.source_point.x - self.circle_center.x,
            self.source_point.y - self.circle_center.y,
        )]
        opposite_side = side_names[self._direction_name(
            self.target_point.x - self.circle_center.x,
            self.target_point.y - self.circle_center.y,
        )]
        return (
            f"The circle sits {self._canvas_region(self.circle_center)}, and the hollow source marker rides its "
            f"rim on the {source_side} side. Its diametric opposite lies on the straight line through the center, "
            f"the same distance beyond it, so the sought point lands on the rim's {opposite_side} side, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "circle_center": self.circle_center.to_list(),
            "circle_radius": self.circle_radius,
            "source_point": self.source_point.to_list(),
        }

    def _draw_source_point(self, draw) -> None:
        self.draw_anchor_marker(draw, self.source_point, 7)

    def _video_overlay_extras(self, draw: ImageDraw.ImageDraw) -> None:
        self._draw_source_point(draw)

    def _render(self, highlight_label: Optional[str]):
        draw, base = self.get_draw_base()
        self.draw_circle(draw, self.circle_center, self.circle_radius)
        if highlight_label:
            line_start, line_end = self.trim_segment(
                self.source_point,
                self.target_point,
                start_offset=0.0,
                end_offset=float(self.point_radius),
            )
            self.draw_line(draw, [line_start, line_end])
        self._draw_source_point(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

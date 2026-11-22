"""Secant intersection puzzle generator.

A circle is drawn with TWO chords visible — neither chord extends beyond
the circle. Each chord has its two endpoints marked with hollow anchors.
The two chord lines, when extended, meet at a single external point P
(both chords share P as their secant origin). Five labeled candidate
markers A-E sit OUTSIDE the circle; identify which marker is exactly P.
"""

from __future__ import annotations

import math
from typing import List, Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Point,
    PointCandidate,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class SecantIntersectionGenerator(PointTargetPuzzleGenerator):
    """Find the external point where two secant chords meet when extended."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows a black circle with two chord segments inside, each connecting two hollow circumference anchors. "
        "Five labeled candidate markers A-E are grouped outside the circle. Identify the candidate that sits at the unique "
        "external point where the two chord lines meet when extended. Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/secant_intersection"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a black circle outline of medium thickness with two black chord segments inside, each connecting two small hollow anchors on the circumference. "
        "Five candidate markers A-E are grouped outside the circle; each marker is a small white circle with a thin dark gray outline and a black uppercase letter, and exactly one marker sits at the point where the two chord lines, extended, meet outside the circle. "
        "The video first holds the circle, the four chord-endpoint anchors, the two chord segments, and the candidate markers, then extends each chord segment outward as a thin black line until both extensions meet at the correct candidate. "
        "In the final state, only the secant intersection candidate changes to pale red fill with a dark red outline, while the other four markers stay white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def _draw_reference(self, draw) -> None:
        self.draw_circle(draw, self.circle_center, self.radius)
        self.draw_line(draw, [self.chord1_p1, self.chord1_p2])
        self.draw_line(draw, [self.chord2_p1, self.chord2_p2])
        for p in (self.chord1_p1, self.chord1_p2, self.chord2_p1, self.chord2_p2):
            self.draw_anchor_marker(draw, p, 6)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_reference(draw)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        cw, ch = self.canvas_dimensions
        for _ in range(999):
            radius = self.canvas_short_side * self._rng.uniform(0.16, 0.22)
            center = Point(
                x=cw * self._rng.uniform(0.40, 0.60),
                y=ch * self._rng.uniform(0.40, 0.60),
            )
            if not self.circle_fits(center, radius, extra_padding=self.line_width + 4):
                continue

            # Pick external point P at distance ratio in (1.6, 2.4) * R.
            ext_ratio = self._rng.uniform(1.7, 2.5)
            ext_angle = self._rng.uniform(0.0, math.tau)
            P = Point(
                x=center.x + radius * ext_ratio * math.cos(ext_angle),
                y=center.y + radius * ext_ratio * math.sin(ext_angle),
            )
            if not self._candidate_fits(PointCandidate(x=P.x, y=P.y, label="A")):
                continue

            # Two secant directions from P, each crossing the circle to make
            # a 2-point chord. Pick directions inside the angular cone from
            # P to circle (~ ±arcsin(R/D)).
            D = self.distance(P, center)
            half_cone = math.asin(min(1.0, radius / D))
            dir_to_center = math.atan2(center.y - P.y, center.x - P.x)
            # Choose two random directions inside the cone, with separation.
            min_sep = math.radians(15.0)
            for _try in range(200):
                d1 = dir_to_center + self._rng.uniform(
                    -half_cone * 0.85, half_cone * 0.85
                )
                d2 = dir_to_center + self._rng.uniform(
                    -half_cone * 0.85, half_cone * 0.85
                )
                if abs(d1 - d2) < min_sep:
                    continue
                break
            else:
                continue

            # Compute the two intersection points of each line with the circle.
            def line_circle_intersect(
                origin: Point, direction: float
            ) -> Optional[List[Point]]:
                # Solve |origin + t*(cos,sin) - center|^2 = R^2
                ox = origin.x - center.x
                oy = origin.y - center.y
                dx = math.cos(direction)
                dy = math.sin(direction)
                b = 2.0 * (ox * dx + oy * dy)
                c = ox * ox + oy * oy - radius * radius
                disc = b * b - 4.0 * c
                if disc < 0:
                    return None
                sqd = math.sqrt(disc)
                t1 = (-b - sqd) / 2.0
                t2 = (-b + sqd) / 2.0
                return [
                    Point(x=origin.x + t1 * dx, y=origin.y + t1 * dy),
                    Point(x=origin.x + t2 * dx, y=origin.y + t2 * dy),
                ]

            ints1 = line_circle_intersect(P, d1)
            ints2 = line_circle_intersect(P, d2)
            if ints1 is None or ints2 is None:
                continue
            chord1_p1, chord1_p2 = ints1[0], ints1[1]
            chord2_p1, chord2_p2 = ints2[0], ints2[1]
            # Chords must be inside canvas.
            for pp in (chord1_p1, chord1_p2, chord2_p1, chord2_p2):
                if not self.inside_canvas(pp, padding=self.line_width):
                    break
            else:
                # Distractor candidates: 4 points outside the circle, NOT
                # at P, NOT on either secant line (otherwise ambiguous).
                self.circle_center = center
                self.radius = radius
                self.external_point = P
                self.chord1_p1 = chord1_p1
                self.chord1_p2 = chord1_p2
                self.chord2_p1 = chord2_p1
                self.chord2_p2 = chord2_p2

                # The markers form one cloud around the external point --
                # scattering them across the canvas would make "the far one is
                # wrong" a free dismissal -- but which slot of that cloud the
                # crossing occupies is drawn only after the layout exists, so
                # the answer is not the marker its neighbours enclose (7.3).
                min_sep_dist = max(self.point_radius * 2.4, 28.0)
                line_clear = max(self.point_radius * 1.6, 18.0)
                circle_clear = max(self.point_radius * 1.4, 14.0)

                def off_the_secants(point: Point) -> bool:
                    return (
                        self.distance(point, center) >= radius + circle_clear
                        and self.distance_point_to_line(point, chord1_p1, chord1_p2)
                        >= line_clear
                        and self.distance_point_to_line(point, chord2_p1, chord2_p2)
                        >= line_clear
                    )

                try:
                    self.seat_candidates(
                        P, accept=off_the_secants, min_separation=min_sep_dist
                    )
                except RuntimeError:
                    continue
                self.target_point = P
                return self.save_puzzle()
        raise RuntimeError("Failed to lay out secant_intersection puzzle")

    def _authored_observation(self) -> str:
        return (
            f"The circle sits {self._canvas_region(self.circle_center)} and carries two chords, each strung "
            f"between its own pair of hollow rim anchors. Prolonging each chord line beyond the rim, the two "
            f"lines converge on one shared crossing outside the circle, "
            f"{self._relative_position_phrase(self.circle_center, self.target_point)} it, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "circle_center": self.circle_center.to_list(),
            "radius": self.radius,
            "external_point": self.external_point.to_list(),
            "chord1_endpoints": [self.chord1_p1.to_list(), self.chord1_p2.to_list()],
            "chord2_endpoints": [self.chord2_p1.to_list(), self.chord2_p2.to_list()],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self._draw_reference(draw)
        if highlight_label:
            # Extend each chord outward to the external point on solution.
            def near_endpoint(endpoint_a: Point, endpoint_b: Point) -> Point:
                return min(
                    (endpoint_a, endpoint_b),
                    key=lambda point: self.distance(point, self.external_point),
                )

            self.draw_line(draw, [near_endpoint(self.chord1_p1, self.chord1_p2), self.external_point])
            self.draw_line(draw, [near_endpoint(self.chord2_p1, self.chord2_p2), self.external_point])
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

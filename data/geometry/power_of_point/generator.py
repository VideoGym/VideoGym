"""External secant distance-product puzzle generator.

A circle plus one external point P (filled black). One full secant
through P is shown with both chord endpoints A (nearer P) and B (farther
from P) marked as hollow anchors. A second secant from P only has its
nearer chord endpoint C marked; the far endpoint D is hidden. By the
equal-distance-product rule for two secants from the same external
anchor, the hidden far endpoint D is determined.

Five labeled candidate markers A-E lie on the circle itself, all NEAR the
(true but unmarked) far endpoint D. Identify the candidate that is
exactly at D. Every option shares D's incidence with the drawn rim, so
"which marker is on the circle?" cannot narrow the field; an option is
wrong because the anchor-to-option product misses the first secant's, not
because it floats off the figure.
"""

from __future__ import annotations

import math
from typing import List, Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointCandidate,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class PowerOfPointGenerator(PointTargetPuzzleGenerator):
    """Identify the hidden chord endpoint from equal external secant products."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows a black circle, one filled black external anchor, one complete secant from that anchor "
        "with both circle endpoints marked as hollow anchors, and one shortened partial secant from the same external "
        "anchor with only its near hollow circle endpoint marked. Five labeled candidate markers A-E are clustered near "
        "the hidden far circle endpoint of the partial secant. Rule: For two secants from the same external anchor, the "
        "distance from the anchor to the near circle intersection times the distance to the far intersection is equal. "
        "Use the complete secant's product, extend the partial secant straight through its marked near intersection, and "
        "select the candidate whose anchor-to-candidate distance gives the same product. Required Output: Answer with one "
        "option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/power_of_point"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a black circle outline of medium thickness, one small filled black external anchor outside the circle, one complete black secant from the external anchor crossing the circle with two small hollow markers on its near and far circle endpoints, and a second partial black secant from the same external anchor that stops shortly after its near hollow circle endpoint. "
        "Five candidate markers A-E are clustered near the hidden far circle endpoint of the partial secant; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the circle, the complete secant, the shortened partial secant, the visible hollow endpoints, and the candidate markers; then extend the partial secant continuation with a black line segment from the visible ray tip to the chosen candidate. "
        "In the final state, only the candidate at the far chord endpoint changes to pale red fill with a dark red outline, while the other four markers stay white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "A square white canvas shows a black circle outline of medium thickness, one small filled black external anchor outside the circle, one complete black secant from the external anchor crossing the circle with two small hollow markers on its near and far circle endpoints, and a second partial black secant from the same external anchor that stops shortly after its near hollow circle endpoint. "
        "Five candidate markers A-E are clustered near the hidden far circle endpoint of the partial secant; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "Extend the partial secant continuation with a black line segment from the visible ray tip to the chosen candidate. "
        "Only the candidate at the far chord endpoint changes to pale red fill with a dark red outline, while the other four markers stay white."
    )

    def _draw_external_anchor(self, draw) -> None:
        cx, cy = round(self.external_point.x), round(self.external_point.y)
        r = max(3, self.point_radius // 2)
        if hasattr(draw, "raw"):
            raw = draw.raw
            ss = draw.factor
            raw.ellipse(
                (cx * ss - r * ss, cy * ss - r * ss, cx * ss + r * ss, cy * ss + r * ss),
                fill=(0, 0, 0),
            )
        else:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0))

    def _draw_reference(self, draw) -> None:
        self.draw_circle(draw, self.circle_center, self.radius)
        # First (full) secant: P -> A -> B
        self.draw_line(draw, [self.external_point, self.full_far])
        # Second (partial) secant: P -> C only.
        self.draw_line(draw, [self.external_point, self.partial_far_anchor])
        self.draw_anchor_marker(draw, self.full_near, 6)
        self.draw_anchor_marker(draw, self.full_far, 6)
        self.draw_anchor_marker(draw, self.partial_near, 6)
        self._draw_external_anchor(draw)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_reference(draw)

    def _place_power_candidates(
        self,
        target: Point,
        center: Point,
        radius: float,
        *,
        anchors: tuple[Point, ...],
    ) -> None:
        """Spread A-E along the circle the hidden chord endpoint lies on.

        The far chord endpoint is a point of the drawn rim, so every option is
        a point of that rim: "which marker is on the circle?" is answer-free
       .  A distractor is wrong because the anchor-to-option
        product misses the first secant's, not because it sits off the figure.
        The three visible hollow chord anchors also ride the rim, so options
        keep a marker's clearance from each of them.
        """

        strict_gap = max(
            self.point_radius * 2.8 + 6.0,
            self.minimum_candidate_spacing() * 1.10,
        )
        anchor_clearance = self.point_radius + 10.0
        self.place_candidates_on_incidence(
            target,
            [Incidence.circle(center, radius)],
            # Matches the runner-up offset the previous along-secant scatter
            # guaranteed, so the near-miss margin does not shrink (7.5).
            min_offset=max(strict_gap, 38.0),
            min_separation=strict_gap,
            reject=lambda point: any(
                self.distance(point, anchor) <= anchor_clearance for anchor in anchors
            ),
        )

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        cw, ch = self.canvas_dimensions
        for _ in range(999):
            radius = self.canvas_short_side * self._rng.uniform(0.16, 0.22)
            center = Point(
                x=cw * self._rng.uniform(0.40, 0.60),
                y=ch * self._rng.uniform(0.40, 0.60),
            )
            if not self.circle_fits(center, radius, extra_padding=self.line_width + 6):
                continue
            ext_ratio = self._rng.uniform(1.6, 2.3)
            D = radius * ext_ratio
            ext_angle = self._rng.uniform(0.0, math.tau)
            P = Point(
                x=center.x + D * math.cos(ext_angle),
                y=center.y + D * math.sin(ext_angle),
            )
            if not self._candidate_fits(PointCandidate(x=P.x, y=P.y, label="A")):
                continue

            half_cone = math.asin(min(1.0, radius / D))
            dir_to_center = math.atan2(center.y - P.y, center.x - P.x)

            def line_circle_intersect(origin: Point, direction: float) -> Optional[List[Point]]:
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

            # First secant line — pick a small angle off center direction.
            d1 = dir_to_center + self._rng.uniform(-half_cone * 0.5, half_cone * 0.5)
            ints1 = line_circle_intersect(P, d1)
            if ints1 is None:
                continue
            full_near, full_far = ints1[0], ints1[1]
            # Second secant — separated by at least 18° from the first.
            for _try in range(200):
                d2 = dir_to_center + self._rng.uniform(-half_cone * 0.85, half_cone * 0.85)
                if abs(d2 - d1) < math.radians(18.0):
                    continue
                break
            else:
                continue
            ints2 = line_circle_intersect(P, d2)
            if ints2 is None:
                continue
            partial_near, partial_far = ints2[0], ints2[1]

            # All four endpoints inside canvas.
            for pp in (full_near, full_far, partial_near, partial_far):
                if not self.inside_canvas(pp, padding=self.line_width):
                    break
            else:
                # Direction unit vector along the secant from P
                ux = math.cos(d2)
                uy = math.sin(d2)
                t_near = (partial_near.x - P.x) * ux + (partial_near.y - P.y) * uy
                t_far = (partial_far.x - P.x) * ux + (partial_far.y - P.y) * uy
                if not self.point_can_host_candidate(partial_far):
                    continue

                chord_span = t_far - t_near
                if chord_span <= self.point_radius * 4.0:
                    continue

                # Keep the second secant visibly partial: it peeks past the
                # near endpoint but stops well before the hidden far endpoint.
                visible_extra = min(chord_span * 0.22, self.canvas_short_side * 0.055)
                visible_t = t_near + visible_extra
                if visible_t >= t_far - self.point_radius * 1.8:
                    continue
                far_ext = self.point_on_ray(P, d2, visible_t)
                if not self.inside_canvas(far_ext, padding=self.line_width):
                    continue

                try:
                    self._place_power_candidates(
                        partial_far,
                        center,
                        radius,
                        anchors=(P, full_near, full_far, partial_near),
                    )
                except (RuntimeError, ValueError):
                    continue
                self.partial_far_anchor = far_ext

                self.circle_center = center
                self.radius = radius
                self.external_point = P
                self.full_near = full_near
                self.full_far = full_far
                self.partial_near = partial_near
                self.partial_far = partial_far

                self.target_point = partial_far
                return self.save_puzzle()
        raise RuntimeError("Failed to lay out power_of_point puzzle")

    def _authored_observation(self) -> str:
        return (
            f"The circle sits {self._canvas_region(self.circle_center)}, and the filled anchor stands outside it "
            f"{self._canvas_region(self.external_point)}. Along the complete secant, the anchor-to-near distance "
            f"times the anchor-to-far distance fixes the product that every secant from this anchor must repeat. "
            f"The shortened secant already shows its near crossing, so continuing it straight on, its hidden far "
            f"crossing must land where that same product is restored — {self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "circle_center": self.circle_center.to_list(),
            "radius": self.radius,
            "external_point": self.external_point.to_list(),
            "full_secant_chord": [self.full_near.to_list(), self.full_far.to_list()],
            "partial_secant_near_chord": self.partial_near.to_list(),
            "partial_secant_far_chord": self.partial_far.to_list(),
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self._draw_reference(draw)
        if highlight_label:
            self.draw_line(draw, [self.partial_far_anchor, self.partial_far])
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

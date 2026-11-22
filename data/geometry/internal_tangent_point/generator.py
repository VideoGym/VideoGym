"""Internal tangent point puzzle generator."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from PIL import Image

from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


def _internal_tangent_points(c1: Point, r1: float, c2: Point, r2: float) -> Optional[Tuple[Point, Point]]:
    """Compute one pair of tangent points on a common internal tangent.

    Uses the internal center of similitude P (divides c1c2 internally in r1:r2).
    """
    dx, dy = c2.x - c1.x, c2.y - c1.y
    d = math.hypot(dx, dy)
    if d <= r1 + r2 + 1e-6 or d < 1e-6:
        return None
    # Internal center of similitude
    px = (r2 * c1.x + r1 * c2.x) / (r1 + r2)
    py = (r2 * c1.y + r1 * c2.y) / (r1 + r2)
    # Distances from P to each center
    dp1 = r1 * d / (r1 + r2)
    dp2 = r2 * d / (r1 + r2)
    # Tangent point on circle 1: angle from c1→P, offset by arccos(r1/dp1)
    phi1 = math.atan2(py - c1.y, px - c1.x)
    cos_val1 = r1 / dp1
    if cos_val1 > 1.0 - 1e-9:
        return None
    delta1 = math.acos(max(-1.0, min(1.0, cos_val1)))
    t1 = Point(c1.x + r1 * math.cos(phi1 + delta1), c1.y + r1 * math.sin(phi1 + delta1))
    # Tangent point on circle 2: angle from c2→P, offset by arccos(r2/dp2)
    phi2 = math.atan2(py - c2.y, px - c2.x)
    cos_val2 = r2 / dp2
    if cos_val2 > 1.0 - 1e-9:
        return None
    delta2 = math.acos(max(-1.0, min(1.0, cos_val2)))
    t2 = Point(c2.x + r2 * math.cos(phi2 + delta2), c2.y + r2 * math.sin(phi2 + delta2))
    return t1, t2


class InternalTangentPointGenerator(PointTargetPuzzleGenerator):
    """Find the tangent point on the common internal tangent of two circles."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows two disjoint black circle outlines of different sizes, plus five labeled candidate circles A-E "
        "clustered near the hidden tangency point on the larger circle. Identify the point on the larger circle where the "
        "common internal tangent line (crossing between the circles) touches it. Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/internal_tangent_point"
    # Both circles fit by construction, so a shape is only redrawn when the
    # heading leaves no room for the required gap; the option scatter is the one
    # random step left and gets its own redraws.
    SHAPE_ATTEMPTS = 64
    SCATTER_ATTEMPTS = 24
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows two disjoint unfilled circles of different sizes drawn with black outlines of medium thickness. "
        "Five candidate markers A-E are clustered near the hidden tangency point on the larger circle; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the two circles and the candidate markers, then draws one black tangent line of medium thickness that crosses between the two circles, touching the larger circle at the correct candidate. "
        "In the final state, only the tangency candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def place_candidates_on_circle(
        self,
        center: Point,
        radius: float,
        true_point: Point,
        mirror_point: Point,
    ) -> bool:
        """Spread A-E along the larger circle's own rim.

        The Prompt states the answer lies on that circle, so every distractor
        lies on it too: "which marker is on a rim?" must carry no signal
       .  A distractor is wrong because the tangent drawn there
        would not slip between the circles, not because it floats in free
        space.  ``mirror_point`` is the *other* internal tangency point on the
        same rim -- equally valid under the rule -- so its neighbourhood is
        excluded rather than turned into a second answer.
        """

        strict_gap = max(
            self.point_radius * 3.0 + 6.0,
            self.minimum_candidate_spacing() * 1.20,
        )
        try:
            self.place_candidates_on_incidence(
                true_point,
                [Incidence.circle(center, radius)],
                min_offset=strict_gap,
                min_separation=strict_gap,
                reject=lambda point: self.distance(point, mirror_point) < strict_gap,
            )
        except (RuntimeError, ValueError):
            return False
        return True

    @staticmethod
    def _reflect_across_centers(point: Point, c1: Point, c2: Point) -> Point:
        dx, dy = c2.x - c1.x, c2.y - c1.y
        norm = dx * dx + dy * dy
        if norm < 1e-12:
            return point
        along = ((point.x - c1.x) * dx + (point.y - c1.y) * dy) / norm
        foot_x, foot_y = c1.x + along * dx, c1.y + along * dy
        return Point(2.0 * foot_x - point.x, 2.0 * foot_y - point.y)

    def _separation_reach(self, *, near_padding: float, far_padding: float, angle: float) -> float:
        """Largest centre separation this heading admits inside the canvas.

        The optimum sits at the corner of the near circle's own box opposite the
        heading, and the two axes are independent, so the per-axis minimum is
        attained by one real centre - a separation at or below this value always
        leaves a non-empty placement box.
        """

        left, top, right, bottom = self.canvas_bounds()
        limits = []
        for low, high, far_low, far_high, component in (
            (left + near_padding, right - near_padding, left + far_padding, right - far_padding, math.cos(angle)),
            (top + near_padding, bottom - near_padding, top + far_padding, bottom - far_padding, math.sin(angle)),
        ):
            if high < low or far_high < far_low:
                return 0.0
            if abs(component) < 1e-9:
                continue
            limits.append((far_high - low) / component if component > 0 else (far_low - high) / component)
        return min(limits) if limits else float("inf")

    def _centre_box_for_offset(
        self,
        *,
        near_padding: float,
        far_padding: float,
        angle: float,
        separation: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        """First-centre box that keeps both circles inside the canvas."""

        left, top, right, bottom = self.canvas_bounds()
        dx = separation * math.cos(angle)
        dy = separation * math.sin(angle)
        x_low = max(left + near_padding, left + far_padding - dx)
        x_high = min(right - near_padding, right - far_padding - dx)
        y_low = max(top + near_padding, top + far_padding - dy)
        y_high = min(bottom - near_padding, bottom - far_padding - dy)
        if x_low > x_high or y_low > y_high:
            return None
        return x_low, y_low, x_high, y_high

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        # Construct the pair; do not fish for one.  This task needs a separation
        # of at least r1 + r2 + 0.10 of the canvas, which from a near-centre
        # first centre puts the second circle off the canvas for all but the
        # near-diagonal headings: sampling the two centres independently
        # accepted only 2-4 of 999 attempts, so ~2% of seeds exhausted the
        # attempt budget outright and killed the whole task run.  Sample
        # the shape, then read the placements the canvas actually admits off the
        # canvas bounds, so every heading is reachable and both circles fit by
        # construction.  Padding the first centre by the candidate anchor
        # allowance plus its own radius also puts the tangency point far enough
        # from the edge to host the option markers, and padding the second by
        # its radius does the same for the far tangency point, so neither needs
        # a rejection test either.
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.04)
        for _ in range(self.SHAPE_ATTEMPTS):
            r1 = self._rng.uniform(self.canvas_short_side * 0.14, self.canvas_short_side * 0.22)
            r2 = self._rng.uniform(self.canvas_short_side * 0.08, r1 * 0.65)
            near_padding = candidate_padding + r1 + self.canvas_short_side * 0.02
            far_padding = r2 + self.line_width
            angle = self._rng.uniform(0.0, math.tau)
            reach = self._separation_reach(
                near_padding=near_padding,
                far_padding=far_padding,
                angle=angle,
            )
            low = r1 + r2 + self.canvas_short_side * 0.10
            high = min(r1 + r2 + self.canvas_short_side * 0.24, reach)
            if high < low:
                continue
            sep = self._rng.uniform(low, high)
            box = self._centre_box_for_offset(
                near_padding=near_padding,
                far_padding=far_padding,
                angle=angle,
                separation=sep,
            )
            if box is None:
                continue
            x_low, y_low, x_high, y_high = box
            c1 = Point(self._rng.uniform(x_low, x_high), self._rng.uniform(y_low, y_high))
            c2 = self.point_on_ray(c1, angle, sep)
            result = _internal_tangent_points(c1, r1, c2, r2)
            if result is None:
                continue
            t1, t2 = result
            # The second internal tangent is the mirror of this one across the
            # line of centres, so its contact on the larger rim satisfies the
            # same rule.  Keep the option scatter clear of it.
            mirror_t1 = self._reflect_across_centers(t1, c1, c2)
            # The option scatter itself is random and its strict spacing check
            # rejects most draws, so redraw it for this shape instead of
            # throwing away a geometry the canvas already accepted.
            if not any(
                self.place_candidates_on_circle(c1, r1, t1, mirror_t1)
                for _ in range(self.SCATTER_ATTEMPTS)
            ):
                continue
            self.c1, self.r1 = c1, r1
            self.c2, self.r2 = c2, r2
            self.tangent_p1, self.tangent_p2 = t1, t2
            self.target_point = t1
            return self.save_puzzle()
        raise RuntimeError("Failed to generate internal tangent point puzzle.")

    def _authored_observation(self) -> str:
        contact_heading = self._heading_phrase(
            self.tangent_p1.x - self.c1.x,
            self.tangent_p1.y - self.c1.y,
        )
        return (
            f"The larger circle sits {self._canvas_region(self.c1)}, and the smaller circle stands fully apart "
            f"{self._relative_position_phrase(self.c1, self.c2)} it. A common internal tangent slips through the "
            f"gap, crossing the line of centers between the two circles, and it can touch each rim only where the "
            f"contact radius meets it at a right angle. On the larger circle that radius points "
            f"{contact_heading}, placing the contact {self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "difficulty_variant": "internal_tangent_crossing_centers",
            "circle_1": {"center": self.c1.to_list(), "radius": self.r1},
            "circle_2": {"center": self.c2.to_list(), "radius": self.r2},
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_circle(draw, self.c1, self.r1)
        self.draw_circle(draw, self.c2, self.r2)
        if highlight_label:
            self.draw_line(draw, [self.tangent_p1, self.tangent_p2])
            self.draw_anchor_marker(draw, self.tangent_p2, 7)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

"""Rhombus vertex puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class RhombusVertexGenerator(PointTargetPuzzleGenerator):
    """Reflect an isosceles-triangle apex across its long side to complete a rhombus.

    The visible triangle supplies two adjacent rhombus sides — its two equal
    sides — and the diagonal between their other endpoints.  The reflected
    apex is the fourth vertex.
    """

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows one black isosceles triangle outline and five labeled candidate circles "
        "A-E on the side opposite its apex. Rule: The two equal sides of the triangle are two sides of a rhombus, and "
        "the longer third side is the diagonal joining their far endpoints. Reflect the apex where the equal sides "
        "meet perpendicularly across that long side; joining the reflected point to both of its endpoints makes all "
        "four rhombus sides equal and opposite sides parallel. Select the candidate at the reflected point. Required "
        "Output: Return exactly one uppercase option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/rhombus_vertex"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows one black isosceles triangle outline whose longer third side will be a diagonal of a rhombus. "
        "Five candidate markers A-E are clustered on the side of that long side opposite the triangle apex; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the triangle and the candidate markers, then draws one cyan construction segment from each end of the long side to the correct candidate, completing a rhombus whose fourth vertex is the reflection of the apex across that long side. "
        "In the final state, only the missing-vertex marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    # Rhombus interior angle at the two diagonal endpoints.  The drawn triangle
    # is isosceles with apex angle 180 - MIN/MAX, and its long side measures
    # 2*cos(angle/2) legs, so the sampled window keeps the long side at least
    # 22% longer than the equal sides at every angle: which vertex is the apex
    # is then readable from the shape itself, not only from where the markers
    # were clustered.
    MIN_RHOMBUS_ANGLE_DEGREES = 52.0
    MAX_RHOMBUS_ANGLE_DEGREES = 104.0

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self._build_configuration()
        return self.save_puzzle()

    def _build_configuration(self) -> None:
        for _ in range(999):
            # Anchor the rhombus's center near the canvas center for
            # composition; build the four vertices around it.
            # Center of the rhombus near the canvas center; the long
            # diagonal (v_left-v_right) is side_len * sqrt(3) ~ 0.50 of
            # canvas, so we need ~0.25 padding from each edge along the
            # long-diagonal axis. Side_len is bumped above 0.28 of
            # canvas so the short diagonal v_top-v_bot is at least 0.14
            # of canvas, large enough that the candidate cluster (~60 px
            # ~ 0.12 of canvas) around v_bot doesn't bleed back into
            # the triangle's interior.
            rhombus_center = self.pick_target_point(0.18, padding=self.canvas_short_side * 0.30)
            side_len = self._rng.uniform(self.canvas_short_side * 0.28, self.canvas_short_side * 0.36)
            base_angle = self._rng.uniform(0.0, math.tau)
            # The rhombus interior angle at v_left and v_right is sampled
            # across a real window instead of being pinned to one shape, so
            # the figure ranges from a wide, flat rhombus to a nearly square
            # one.  The figure rendered as `v_left -> v_top -> v_right ->
            # v_left` is the isosceles triangle formed by two rhombus sides
            # plus the diagonal between their far endpoints; the 4th rhombus
            # vertex v_bot is the mirror of v_top across that diagonal.
            interior_angle_deg = self._rng.uniform(
                self.MIN_RHOMBUS_ANGLE_DEGREES, self.MAX_RHOMBUS_ANGLE_DEGREES
            )
            interior_angle = math.radians(interior_angle_deg)
            half = interior_angle / 2.0
            # Parametrize by side_len + interior_angle: half-diagonals.  The
            # drawn diagonal v_left-v_right measures 2*side_len*cos(half) and
            # the hidden one v_top-v_bot measures 2*side_len*sin(half).
            d_long_half = side_len * math.cos(half)
            d_short_half = side_len * math.sin(half)
            v_left = self.point_on_ray(rhombus_center, base_angle + math.pi, d_long_half)
            v_right = self.point_on_ray(rhombus_center, base_angle, d_long_half)
            v_top = self.point_on_ray(rhombus_center, base_angle + math.pi / 2.0, d_short_half)
            v_bot = self.point_on_ray(rhombus_center, base_angle - math.pi / 2.0, d_short_half)
            target = v_bot
            pts = (v_left, v_top, v_right, v_bot)
            if not all(self.inside_canvas(p, padding=self.line_width + 6) for p in pts):
                continue
            if not self.point_can_host_candidate(target):
                continue
            # Spacing: keep the candidate cluster well separated from
            # the visible triangle (v_left, v_top, v_right). Otherwise
            # candidates clipping into the triangle's interior or onto
            # its edges read as visual noise rather than markers.
            # All three triangle vertices are at distance side_len from
            # v_bot (rhombus side); we want this >= ~0.22 of canvas so
            # the candidate cluster (radius ~60 px) clears the triangle.
            min_sep = self.canvas_short_side * 0.22
            if min(
                self.distance(target, v_left),
                self.distance(target, v_top),
                self.distance(target, v_right),
            ) < min_sep:
                continue
            try:
                self._place_rhombus_candidates(target, v_top, v_left, v_right)
            except RuntimeError:
                continue

            self.v_left = v_left
            self.v_top = v_top
            self.v_right = v_right
            self.v_bot = v_bot
            self.target_point = target
            self.rhombus_angle_degrees = interior_angle_deg
            return
        raise RuntimeError("Failed to generate a valid rhombus vertex puzzle.")

    def _place_rhombus_candidates(
        self,
        target: Point,
        v_top: Point,
        v_left: Point,
        v_right: Point,
    ) -> None:
        """Place 5 markers near v_bot, biased AWAY from the v_left-v_right
        base so distractors don't bleed into the triangle's interior.

        The bias direction is the vector from v_top to v_bot (extended
        outward from the base). Distractors sample angles in a 240° arc
        on that side of the base.
        """
        marker_clear = max(self.point_radius * 2.6, self.minimum_candidate_spacing() * 0.95)
        triangle_clear = max(self.point_radius * 2.0, 22.0)

        def clear_of_triangle(point: Point) -> bool:
            return (
                min(
                    self._point_to_segment_distance(point, v_left, v_top),
                    self._point_to_segment_distance(point, v_top, v_right),
                    self._point_to_segment_distance(point, v_right, v_left),
                )
                >= triangle_clear
            )

        # Keeping distractors off the triangle used to be done by biasing them
        # outward from the base, which left the answer enclosed by its own
        # cloud.  The clearance is now a placement constraint and the layout is
        # drawn before one of its slots becomes the answer.
        self.seat_candidates(
            target, accept=clear_of_triangle, min_separation=marker_clear
        )

    @staticmethod
    def _point_to_segment_distance(P: Point, A: Point, B: Point) -> float:
        ABx, ABy = B.x - A.x, B.y - A.y
        APx, APy = P.x - A.x, P.y - A.y
        ab2 = ABx * ABx + ABy * ABy
        if ab2 < 1e-9:
            return math.hypot(APx, APy)
        t = max(0.0, min(1.0, (APx * ABx + APy * ABy) / ab2))
        cx = A.x + t * ABx
        cy = A.y + t * ABy
        return math.hypot(P.x - cx, P.y - cy)

    def _authored_observation(self) -> str:
        base_mid = Point(
            (self.v_left.x + self.v_right.x) / 2.0,
            (self.v_left.y + self.v_right.y) / 2.0,
        )
        base_direction = self._heading_phrase(
            self.v_right.x - self.v_left.x, self.v_right.y - self.v_left.y,
        )
        return (
            f"The two sides meeting at the apex are equal and the third side is the longer one, running "
            f"{base_direction}, so the apex is a rhombus corner and that long side is the diagonal facing it. The "
            f"apex sits {self._relative_position_phrase(base_mid, self.v_top)} the middle of that diagonal, "
            f"{self._canvas_region(self.v_top)}. Keeping all four rhombus sides equal puts the missing vertex at "
            f"the apex's mirror image straight across the diagonal, the same distance out on the other side, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "shown_vertices": [
                self.v_left.to_list(), self.v_top.to_list(), self.v_right.to_list(),
            ]
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        # The closed isosceles triangle is the stated input primitive:
        # v_left -> v_top -> v_right -> v_left.  The two equal segments are
        # rhombus sides and the longer one is the drawn diagonal.  The 4th
        # rhombus vertex v_bot is the mirror of v_top across that diagonal —
        # that is the answer.
        self.draw_line(
            draw,
            [self.v_left, self.v_top, self.v_right, self.v_left],
        )
        if highlight_label:
            # Solution: complete the rhombus by drawing the two missing
            # sides v_left -> v_bot -> v_right.
            self.draw_line(
                draw,
                [self.v_left, self.v_bot, self.v_right],
                join_start=True, join_end=True,
            )
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

"""Trapezoid median puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class TrapezoidMedianGenerator(PointTargetPuzzleGenerator):
    """Identify the midpoint of the median (midsegment) of a trapezoid.

    A trapezoid's median connects the midpoints of its two non-parallel
    sides (legs) and is parallel to the two bases. The midpoint of THIS
    median is the centroid-like point exactly halfway between the two leg
    midpoints. Equivalently, it is the average of all four trapezoid
    midpoints — and lies inside the trapezoid.
    """

    DEFAULT_RULE_PROMPT = (
        "A black trapezoid outline is shown on a white canvas with two filled black discs marking the midpoints of the two "
        "non-parallel sides (legs), plus five labeled candidate circles A-E clustered near the midpoint of the segment "
        "connecting those two leg discs. Identify which candidate is the midpoint of the segment that joins the two leg "
        "midpoints (the trapezoid's median). Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/trapezoid_median"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a trapezoid drawn as a black outline of medium thickness, with two parallel sides (the bases) and two non-parallel sides (the legs). "
        "Two small filled black discs sit on the two legs at their midpoints. "
        "Five candidate markers A-E are clustered near the midpoint of the line between the two leg midpoints; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the trapezoid, the two leg-midpoint discs, and the candidate markers, then draws one black segment of medium thickness from one disc to the other through the correct candidate. "
        "In the final state, only the median-midpoint marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.04)
        for _ in range(999):
            center = self.pick_target_point(0.40, padding=candidate_padding + self.canvas_short_side * 0.10)
            base_angle = self._rng.uniform(0.0, math.tau)
            axis_angle = base_angle + math.pi / 2.0
            base_long_half = self._rng.uniform(self.canvas_short_side * 0.18, self.canvas_short_side * 0.26)
            base_short_half = base_long_half * self._rng.uniform(0.42, 0.72)
            height_full = self._rng.uniform(self.canvas_short_side * 0.20, self.canvas_short_side * 0.30)
            top_center = self.point_on_ray(center, axis_angle, height_full * 0.5)
            bot_center = self.point_on_ray(center, axis_angle + math.pi, height_full * 0.5)
            # Bottom (long) base.
            b1 = self.point_on_ray(bot_center, base_angle + math.pi, base_long_half)
            b2 = self.point_on_ray(bot_center, base_angle, base_long_half)
            # Top (short) base — generic trapezoid (not necessarily isosceles).
            top_offset = self._rng.uniform(-base_long_half * 0.18, base_long_half * 0.18)
            top_shift = self.point_on_ray(top_center, base_angle, top_offset)
            t1 = self.point_on_ray(top_shift, base_angle + math.pi, base_short_half)
            t2 = self.point_on_ray(top_shift, base_angle, base_short_half)
            # Leg midpoints.
            ml = Point((b1.x + t1.x) / 2.0, (b1.y + t1.y) / 2.0)
            mr = Point((b2.x + t2.x) / 2.0, (b2.y + t2.y) / 2.0)
            target = Point((ml.x + mr.x) / 2.0, (ml.y + mr.y) / 2.0)

            ok = True
            for pt in (b1, b2, t1, t2, ml, mr, target):
                if not self.inside_canvas(pt, padding=self.line_width + 4):
                    ok = False
                    break
            if not ok:
                continue
            if not self.point_can_host_candidate(target):
                continue
            median_len = self.distance(ml, mr)
            # Require median length >= cluster span + leg-disc margin so
            # the 5-marker cluster fits between the two leg midpoints
            # without overlapping the discs.
            min_spacing = self.minimum_candidate_spacing()
            cluster_span = 4.0 * min_spacing
            # disc_margin must clear the leg-midpoint disc (radius 7) plus
            # the candidate marker radius (~14 incl. outline) plus a
            # small visual gap so a candidate never lands on top of the
            # disc.
            disc_margin = 7.0 + self.point_radius + self.CANDIDATE_HIGHLIGHT_OUTLINE_WIDTH + 4.0
            if median_len < cluster_span + 2 * disc_margin:
                continue
            try:
                # 2D-scatter near the median's midpoint so A-E don't sit
                # on a perfect colinear-on-median shortcut row.
                self.place_candidates(target)
            except RuntimeError:
                continue
            strict_gap = max(self.point_radius * 3.0 + 6.0,
                             self.minimum_candidate_spacing() * 1.20)
            cand_pts = [Point(c.x, c.y) for c in self.candidates]
            if not self.points_are_well_spaced(cand_pts, min_distance=strict_gap):
                continue
            # Reject layouts where any candidate is closer to a leg
            # midpoint than the disc safety margin.
            too_close = False
            for cand in self.candidates:
                cp = Point(cand.x, cand.y)
                if (
                    self.distance(cp, ml) < disc_margin
                    or self.distance(cp, mr) < disc_margin
                ):
                    too_close = True
                    break
            if too_close:
                continue
            self.trapezoid = (b1, b2, t2, t1)  # CCW: bot-left, bot-right, top-right, top-left
            self.leg_mids = (ml, mr)
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid trapezoid median puzzle.")

    def _authored_observation(self) -> str:
        mid_left, mid_right = self.leg_mids
        median_direction = self._heading_phrase(
            mid_right.x - mid_left.x, mid_right.y - mid_left.y,
        )
        return (
            f"The two filled discs mark the midpoints of the trapezoid's legs, one "
            f"{self._canvas_region(mid_left)} and the other {self._canvas_region(mid_right)}. The median is the "
            f"segment joining those two discs, running {median_direction}; its own midpoint lies halfway along "
            f"that join, equally far from both discs, {self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "trapezoid_vertices": [p.to_list() for p in self.trapezoid],
            "leg_midpoints": [p.to_list() for p in self.leg_mids],
        }

    def _draw_leg_mid_discs(self, draw) -> None:
        r = 7
        for pt in self.leg_mids:
            cx, cy = round(pt.x), round(pt.y)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0), outline=(0, 0, 0), width=1)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_leg_mid_discs(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        b1, b2, t2, t1 = self.trapezoid
        self.draw_line(draw, [b1, b2, t2, t1, b1])
        if highlight_label:
            ml, mr = self.leg_mids
            self.draw_line(draw, [ml, mr])
        self._draw_leg_mid_discs(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

"""Arc bisector puzzle generator.

Only a partial arc is drawn (no full circle) and the two endpoints of
the arc are marked with hollow anchors. Five labeled candidate markers
A-E lie ON the arc; pick the one that is the arc's midpoint, i.e. the
point that splits the arc into two equal-length sub-arcs.

Only the arc itself is drawn, so the candidate must reason about the
arc's implicit center using only its curvature rather than relying on a
complete circle outline.
"""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Incidence,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class ArcBisectorGenerator(PointTargetPuzzleGenerator):
    """Find the midpoint of an isolated arc (no full circle drawn)."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows a single black arc and two hollow endpoint anchors on its endpoints, plus five labeled candidate "
        "circles A-E lying directly on the arc. Identify the candidate that bisects the arc, splitting it into two equal-length "
        "sub-arcs. Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/arc_bisector"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a single black circular arc of medium thickness with no full circle around it; the two endpoints of the arc are each marked with a small hollow white anchor outlined in black. "
        "Five candidate markers A-E lie directly on the arc itself; each is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video holds the already-complete arc, both endpoint anchors, and all candidate markers unchanged; no construction stroke is added because the full measured arc is source ink. "
        "In the final state, only the candidate at the arc midpoint (where arc length to each endpoint is equal) changes to pale red fill with a dark red outline, while the other four markers stay white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.04)
        for _ in range(999):
            R = self._rng.uniform(self.canvas_short_side * 0.232, self.canvas_short_side * 0.26)
            pad = candidate_padding + R + self.canvas_short_side * 0.02
            if pad >= self.canvas_short_side * 0.36:
                continue
            center = self.pick_target_point(0.30, padding=pad)
            if not self.circle_fits(center, R, extra_padding=self.line_width):
                continue
            a1 = self._rng.uniform(0.0, math.tau)
            # The bisector sits at the arc's middle, so an arc only long enough
            # to hold the five markers holds them symmetrically about it and
            # hands over "the middle marker".  Each half must
            # take all four distractors plus the endpoint anchor's clearance,
            # so the answer can be the first marker along the arc as readily as
            # the third.  The span is derived from that, not resampled until it
            # happens to be long enough (7.7).
            half = 4.0 * self.minimum_candidate_spacing(scale=0.9) + 2.0 * self.minimum_candidate_spacing()
            widest = math.radians(150)
            narrowest = max(math.radians(80), 2.0 * half / R)
            if narrowest > widest:
                continue
            arc_span = self._rng.uniform(narrowest, widest)
            a2 = a1 + arc_span
            mid_angle = a1 + arc_span / 2.0
            ep1 = self.point_on_ray(center, a1, R)
            ep2 = self.point_on_ray(center, a2, R)
            target = self.point_on_ray(center, mid_angle, R)
            if not all(self.inside_canvas(p, padding=self.line_width) for p in [ep1, ep2, target]):
                continue

            # Every marker rides the arc, and how many of them sit before the
            # bisector is drawn before they are placed.  Spreading them
            # symmetrically about the bisector and dropping the seatings whose
            # ends hit an endpoint anchor left only the middle rungs, which
            # made "the middle marker" the answer.
            min_ep_dist = self.minimum_candidate_spacing()
            try:
                self.place_candidates_on_incidence(
                    target,
                    [Incidence.arc(center, R, a1, arc_span)],
                    min_separation=self.minimum_candidate_spacing(scale=0.9),
                    reject=lambda point: (
                        self.distance(point, ep1) < min_ep_dist
                        or self.distance(point, ep2) < min_ep_dist
                    ),
                )
            except (RuntimeError, ValueError):
                continue

            self.arc_center = center
            self.R = R
            self.ep1 = ep1
            self.ep2 = ep2
            self.arc_start = a1
            self.arc_sweep = arc_span
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate arc_bisector puzzle.")

    def _authored_observation(self) -> str:
        bulge = self._heading_phrase(
            self.target_point.x - self.arc_center.x,
            self.target_point.y - self.arc_center.y,
        )
        return (
            f"A single arc bows {bulge} between its two hollow endpoint anchors, and all five candidate markers "
            f"ride on the curve itself. The bisecting point must split the arc into two equal-length halves, so "
            f"walking inward from both anchors at the same pace meets at the crown of the bend, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "arc_center": self.arc_center.to_list(),
            "radius": self.R,
            "arc_endpoints": [self.ep1.to_list(), self.ep2.to_list()],
            "arc_start_deg": math.degrees(self.arc_start),
            "arc_sweep_deg": math.degrees(self.arc_sweep),
        }

    def _draw_arc_endpoints(self, draw) -> None:
        self.draw_anchor_marker(draw, self.ep1, 7)
        self.draw_anchor_marker(draw, self.ep2, 7)

    def _video_overlay_extras(self, draw) -> None:
        # NOTE: draw_arc uses math y-up convention while point_on_ray uses
        # image y-down. To make the arc visually align with endpoint
        # markers placed via point_on_ray, the start angle and sweep are
        # negated (mirror across the x-axis).
        self.draw_arc(draw, self.arc_center, self.R, -self.arc_start, -self.arc_sweep)
        self._draw_arc_endpoints(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_arc(draw, self.arc_center, self.R, -self.arc_start, -self.arc_sweep)
        self._draw_arc_endpoints(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

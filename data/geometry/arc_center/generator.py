"""Arc center puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class ArcCenterGenerator(PointTargetPuzzleGenerator):
    """Find the center of the circle from a partial arc."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows a single black circular arc spanning part of a full circle, with no center marked, plus five "
        "labeled candidate circles A-E. Determine which candidate is the exact center of the full circle that the arc belongs "
        "to. Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/arc_center"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a single black circular arc of medium thickness spanning roughly a quarter to a third of a full circle, with no center or radius marked. "
        "Five candidate markers A-E are clustered near the hidden center of the implied full circle; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the arc and the candidate markers, then extends the given arc in cyan from one of its endpoints around the missing portion of the circle back to the other endpoint, completing a single smooth full circle centered on the correct candidate. "
        "In the final state, only the center marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def __init__(self, *args, arc_span=(90, 130), **kwargs):
        self.arc_span = tuple(arc_span)
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        min_radius = self.canvas_short_side * 0.14
        max_radius = self.canvas_short_side * 0.34
        for _ in range(999):
            center = self.pick_target_point(
                0.42,
                padding=candidate_padding + self.canvas_short_side * 0.10,
            )
            radius = self._rng.uniform(min_radius, max_radius)
            if not self.circle_fits(center, radius, extra_padding=self.line_width):
                continue
            arc_span = math.radians(self._rng.uniform(*self.arc_span))
            arc_start_angle = self._rng.uniform(0.0, math.tau)
            arc_end_angle = arc_start_angle + arc_span
            mid_arc_angle = arc_start_angle + arc_span / 2
            arc_mid = self.point_on_ray(center, mid_arc_angle, radius)
            if not self.inside_canvas(arc_mid, padding=self.line_width):
                continue
            arc_p1 = self.point_on_ray(center, arc_start_angle, radius)
            arc_p2 = self.point_on_ray(center, arc_end_angle, radius)
            if not self.inside_canvas(arc_p1, padding=self.line_width):
                continue
            if not self.inside_canvas(arc_p2, padding=self.line_width):
                continue

            if not self.point_can_host_candidate(center):
                continue

            try:
                self.place_candidates(center)
            except RuntimeError:
                continue

            self.center = center
            self.radius = radius
            self.arc_start_deg = math.degrees(arc_start_angle)
            self.arc_end_deg = math.degrees(arc_end_angle)
            self.target_point = center
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid arc center puzzle.")

    def _authored_observation(self) -> str:
        mid_rad = math.radians((self.arc_start_deg + self.arc_end_deg) / 2.0)
        bulge_dx, bulge_dy = math.cos(mid_rad), -math.sin(mid_rad)
        crown = Point(
            self.center.x + self.radius * bulge_dx,
            self.center.y + self.radius * bulge_dy,
        )
        return (
            f"The lone arc hangs {self._canvas_region(crown)}, bowing outward "
            f"{self._heading_phrase(bulge_dx, bulge_dy)}. Its bend belongs to exactly one full circle, and that "
            f"circle's center must sit at the same distance from every point of the arc, back on the concave side "
            f"{self._canvas_region(self.target_point)} where the candidate markers cluster."
        )

    def build_record_extra(self) -> dict[str, object]:
        start = math.radians(self.arc_start_deg)
        sweep = math.radians(self.arc_end_deg - self.arc_start_deg)
        return {
            "radius": self.radius,
            "arc_start_deg": self.arc_start_deg,
            "arc_end_deg": self.arc_end_deg,
            # Three separated points are sufficient for an evaluator to fit
            # the implied circle independently.  Persisting the hidden center
            # itself would turn the source record into an answer oracle.
            "source_arc_points": [
                self.point_on_ray(self.center, start + sweep * fraction, self.radius).to_list()
                for fraction in (0.0, 0.5, 1.0)
            ],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        # The given arc sweeps CCW (math convention) from arc_start_angle
        # by an angle of `given_sweep`. We draw it via the draw_arc helper
        # so both puzzle and solution phases record the SAME command —
        # the video diff then skips re-animating the given arc.
        start_angle = math.radians(self.arc_start_deg)
        given_sweep = math.radians(self.arc_end_deg - self.arc_start_deg)
        self.draw_arc(draw, self.center, self.radius, start_angle, given_sweep)
        if highlight_label:
            # Solution: draw the COMPLEMENT arc going from the end of the
            # given arc (arc_end_deg) around the missing portion back to
            # the start of the given arc. The video will animate this
            # complement as a smooth reveal starting from the given arc's
            # tail and ending at its head — exactly one continuous pen
            # stroke that closes the circle.
            complement_sweep = math.tau - given_sweep
            complement_start = start_angle + given_sweep
            self.draw_arc(draw, self.center, self.radius, complement_start, complement_sweep)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

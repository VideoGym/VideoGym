"""Dilation between two circles puzzle generator.

Two non-concentric circles of different radii are drawn. The external
center of similitude (homothetic center) lies on the line through both
centers at the position where rays from it through both circles share
the same tangent slopes — geometrically, at parameter t such that
  (c1 - t) / r1 = (c2 - t) / r2
along the centers line, giving t = (r2*c1 - r1*c2) / (r2 - r1) (external
center; r1 != r2). Five labeled candidate markers A-E are scattered;
identify the one at the external center of similitude.
"""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Point,
    PointCandidate,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class DilationToCircleGenerator(PointTargetPuzzleGenerator):
    """Find the external center of similitude of two non-concentric circles."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows two separated black circle outlines of different radii and five labeled "
        "candidate circles A-E. Rule: The external center of similitude is the unique finite point on the line through "
        "the two circle centers from which one positive dilation maps one circle onto the other; equivalently, it is "
        "the intersection of their two common external tangents. Select the candidate at that point. Required Output: "
        "Return exactly one uppercase option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/dilation_to_circle"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows two black circle outlines of medium thickness with different radii and non-overlapping centers. "
        "Five candidate markers A-E are scattered around the canvas; each is a small white circle with a thin dark gray outline and a black uppercase letter, and exactly one candidate sits on the line through both centers at the external center of similitude (the external homothety center, where a positive dilation maps the smaller circle onto the larger). "
        "The video first holds the two circles and the candidate markers, then draws one black common external tangent line of medium thickness touching both circles on the same side; this line passes through the correct candidate. "
        "In the final state, only the dilation-center candidate changes to pale red fill with a dark red outline, while the other four markers stay white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def _draw_reference(self, draw) -> None:
        self.draw_circle(draw, self.c1, self.r1)
        self.draw_circle(draw, self.c2, self.r2)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_reference(draw)

    def _external_tangent_line(self) -> tuple[Point, Point]:
        # One of the two external tangents passes through the external
        # similitude center. Compute it.
        dx = self.c2.x - self.c1.x
        dy = self.c2.y - self.c1.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return self.c1, self.c2
        theta = math.atan2(dy, dx)
        sin_a = (self.r1 - self.r2) / d
        if abs(sin_a) > 1.0 - 1e-9:
            # No external tangent; degenerate.
            return self.c1, self.c2
        alpha = math.asin(max(-1.0, min(1.0, sin_a)))
        tangent_angle = theta + math.pi / 2 - alpha
        t1 = Point(
            x=self.c1.x + self.r1 * math.cos(tangent_angle),
            y=self.c1.y + self.r1 * math.sin(tangent_angle),
        )
        t2 = Point(
            x=self.c2.x + self.r2 * math.cos(tangent_angle),
            y=self.c2.y + self.r2 * math.sin(tangent_angle),
        )
        return t1, t2

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        cw, ch = self.canvas_dimensions
        for _ in range(999):
            r1 = self.canvas_short_side * self._rng.uniform(0.12, 0.16)
            r2 = self.canvas_short_side * self._rng.uniform(0.05, 0.09)
            if abs(r1 - r2) < self.canvas_short_side * 0.04:
                continue
            # Ensure non-overlapping circles: sep must exceed r1 + r2 + buffer.
            sep_min = r1 + r2 + self.canvas_short_side * 0.04
            sep_max = r1 + r2 + self.canvas_short_side * 0.16
            if sep_max > self.canvas_short_side * 0.45:
                sep_max = self.canvas_short_side * 0.45
            if sep_min >= sep_max:
                continue
            sep = self._rng.uniform(sep_min, sep_max)
            angle = self._rng.uniform(0.0, math.tau)
            # Place the larger circle away from the side where the
            # similitude center will land, so the center fits canvas.
            mid = Point(x=cw * 0.55, y=ch * 0.55)
            c1 = Point(
                x=mid.x - 0.5 * sep * math.cos(angle),
                y=mid.y - 0.5 * sep * math.sin(angle),
            )
            c2 = Point(
                x=mid.x + 0.5 * sep * math.cos(angle),
                y=mid.y + 0.5 * sep * math.sin(angle),
            )
            if r1 < r2:
                c1, c2 = c2, c1
                r1, r2 = r2, r1
            if not self.circle_fits(c1, r1, extra_padding=self.line_width):
                continue
            if not self.circle_fits(c2, r2, extra_padding=self.line_width):
                continue

            # External center of similitude.
            denom = r1 - r2
            t = (r1 * c2.x - r2 * c1.x) / denom, (r1 * c2.y - r2 * c1.y) / denom
            ext_sim = Point(x=t[0], y=t[1])
            if not self._candidate_fits(
                PointCandidate(x=ext_sim.x, y=ext_sim.y, label="A")
            ):
                continue

            self.c1 = c1
            self.r1 = r1
            self.c2 = c2
            self.r2 = r2
            self.ext_sim = ext_sim

            # A distractor may sit anywhere clear of the two discs -- the line
            # through the centres included, where the internal centre of
            # similitude and every other wrong point on that line live.  The
            # markers used to be drawn one at a time from a box centred on the
            # canvas while the answer sat out beyond the smaller circle, so
            # the answer was the outlier of its own set: 0.75 furthest in x
            # and 0.69 in y from the middle of the frame, 0.60 furthest from
            # the cloud's centroid, against 0.200 chance.  Barring the line
            # through the centres was itself a shortcut, since only the answer
            # was left on it.  The shared placer lays the arrangement out and
            # then takes one of its slots for the answer.
            clear = self.canvas_short_side * 0.02

            def outside_both(point: Point) -> bool:
                return (
                    self.distance(point, c1) >= r1 + clear
                    and self.distance(point, c2) >= r2 + clear
                )

            try:
                self.seat_candidates(
                    ext_sim,
                    accept=outside_both,
                    min_separation=self.canvas_short_side * 0.13,
                )
            except RuntimeError:
                continue
            self.target_point = ext_sim
            return self.save_puzzle()
        raise RuntimeError("Failed to lay out dilation_to_circle puzzle")

    def _authored_observation(self) -> str:
        return (
            f"The larger circle sits {self._canvas_region(self.c1)}, and the clearly smaller one lies "
            f"{self._relative_position_phrase(self.c1, self.c2)} it. A positive dilation carrying one circle onto "
            f"the other must be centered on the line through both centers, and because the sizes differ, that "
            f"external center of similitude falls beyond the smaller circle on the side away from the larger one "
            f"— {self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "circle_1": {"center": self.c1.to_list(), "radius": self.r1},
            "circle_2": {"center": self.c2.to_list(), "radius": self.r2},
            "external_center_of_similitude": self.ext_sim.to_list(),
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self._draw_reference(draw)
        if highlight_label:
            t1, t2 = self._external_tangent_line()
            # Extend the tangent past both circles to the similitude center.
            self.draw_line(draw, [self.ext_sim, t1])
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

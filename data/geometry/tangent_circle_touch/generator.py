"""Tangent circle touch puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Incidence,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class TangentCircleTouchGenerator(PointTargetPuzzleGenerator):
    """Find the kissing point where two externally tangent circles meet."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows two black circles of different sizes that are externally tangent, plus five "
        "labeled candidate circles A-E spaced out along one of the two circles. Rule: For external tangency, the two circle "
        "centers and the touch point are collinear; the touch point lies between the centers, one radius from each center. "
        "Select the candidate at that point. Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/tangent_circle_touch"
    # A rim only hosts the option row once it is long enough for four
    # distractors to sit on either side of the kiss, so the answer's place in
    # the row can be drawn uniformly.
    HOST_RADIUS_RATIO = 0.17
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows two unfilled circles of different sizes drawn with black outlines of medium thickness, touching each other externally at exactly one point. "
        "Five candidate markers A-E are spaced out along one of the two circles; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the two circles and the candidate markers. "
        "In the final state, only the tangent-point marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def __init__(self, *args, internal=False, **kwargs):
        self.internal = bool(internal)
        if self.internal:
            for field in (
                "DEFAULT_RULE_PROMPT",
                "DEFAULT_TI2V_PROMPT",
                "DEFAULT_SOLUTION_IMAGE_PROMPT",
            ):
                setattr(
                    self,
                    field,
                    getattr(self, field)
                    .replace("externally tangent", "internally tangent")
                    .replace("touch externally", "touch internally")
                    .replace("externally at exactly", "internally at exactly")
                    .replace("For external tangency", "For internal tangency")
                    .replace(
                        "lies between the centers",
                        "lies beyond the smaller circle centre on the ray from the larger centre",
                    )
                    .replace("from the outside", "from the inside"),
                )
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        host_floor = self.canvas_short_side * self.HOST_RADIUS_RATIO
        for _ in range(999):
            r1 = self._rng.uniform(host_floor, self.canvas_short_side * 0.27)
            r2 = self._rng.uniform(self.canvas_short_side * 0.075, r1 * 0.72)
            angle = self._rng.uniform(0.0, math.tau)
            sep = r1 - r2 if self.internal else r1 + r2

            c1 = self.pick_target_point(
                0.36, padding=candidate_padding + r1 + self.canvas_short_side * 0.02
            )
            c2 = self.point_on_ray(c1, angle, sep)

            if not self.circle_fits(c1, r1, extra_padding=self.line_width):
                continue
            if not self.circle_fits(c2, r2, extra_padding=self.line_width):
                continue

            # Tangent point: on the line c1→c2 at distance r1 from c1
            target = self.point_on_ray(c1, angle, r1)

            if not self.point_can_host_candidate(target):
                continue

            # The kissing point belongs to both rims, so seating the row on
            # either one keeps every option on the figure.  It
            # is seated on *one* rim because the two rims curve to opposite
            # sides of their common tangent: a row dealt across both straddles
            # the kiss whatever rank it is given, and hands "the marker nearest
            # the cloud centroid" the answer.  A single rim is a long open run
            # -- half its circumference each way -- so all five ranks are
            # reachable and the answer's place in the row is drawn uniformly.
            hosts = [
                (centre, radius, other_centre, other_radius)
                for centre, radius, other_centre, other_radius in (
                    (c1, r1, c2, r2),
                    (c2, r2, c1, r1),
                )
                if radius >= host_floor
                and self.circle_fits(centre, radius, extra_padding=candidate_padding)
            ]
            if not hosts:
                continue
            host_centre, host_radius, other_centre, other_radius = self._rng.choice(
                hosts
            )
            # A distractor must read as clearly parted from the other rim, not
            # as a second kiss.  The widened step is applied to
            # every neighbouring pair, not only to the answer's own
            # neighbours: a row that leaves extra room around the answer alone
            # hands "the most isolated marker" the answer for free.
            rim_gap = self.point_radius * 0.9
            step = self.minimum_candidate_spacing() * 1.3
            try:
                self.place_candidates_on_incidence(
                    target,
                    [Incidence.circle(host_centre, host_radius)],
                    min_offset=step,
                    min_separation=step,
                    reject=lambda point: (
                        abs(self.distance(point, other_centre) - other_radius) < rim_gap
                    ),
                )
            except (RuntimeError, ValueError):
                continue

            self.c1 = c1
            self.r1 = r1
            self.c2 = c2
            self.r2 = r2
            self.host_is_first = host_centre is c1
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate tangent circle touch puzzle.")

    def _authored_observation(self) -> str:
        if self.internal:
            return "The small circle lies inside the large one and meets its rim at one point. The contact lies on the ray from the large centre through the small centre, one large radius away."
        host = "larger" if self.host_is_first else "smaller"
        return (
            f"The larger circle sits {self._canvas_region(self.c1)}, and the smaller circle presses against it "
            f"from the outside, {self._relative_position_phrase(self.c1, self.c2)} it. The five markers are strung "
            f"out along the {host} circle, so the question is which of them sits where the rims meet. Externally "
            f"tangent circles meet at exactly one point on the straight line joining their centers, one radius out "
            f"from each, so the kissing point sits where the two rims squeeze together, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "circle_1": {"center": self.c1.to_list(), "radius": self.r1},
            "circle_2": {"center": self.c2.to_list(), "radius": self.r2},
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_circle(draw, self.c1, self.r1)
        self.draw_circle(draw, self.c2, self.r2)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

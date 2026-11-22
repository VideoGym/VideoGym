"""Segment trisection puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class SegmentTrisectionGenerator(PointTargetPuzzleGenerator):
    """Find the point that divides a segment in ratio 1:2 from the marked end."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows a black line segment with a solid black disc at one endpoint and a small "
        "hollow circle at the other, plus five labeled candidate circles A-E spaced out along the segment. Rule: "
        "Split the segment into three equal parts. Select the candidate at the division point one third of the way "
        "from the solid black endpoint towards the hollow endpoint. Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/segment_trisection"
    # Big enough that the hollow ring keeps a clearly open centre against the
    # solid disc: which endpoint is which decides the answer.
    ANCHOR_RADIUS = 10
    MARKED_ANCHOR_FILL = (0, 0, 0)
    # Half-lengths: the option row has to reach four markers between the
    # answer and the marked end, and the answer sits only a third of the way
    # along, so the whole segment is drawn as long as its heading allows.  A
    # heading that cannot carry that length would leave the outermost rank
    # unreachable and push the answer back toward the middle of its own row,
    # so the heading is drawn from the ones that can.
    MIN_HALF_LENGTH = 196.0
    MAX_HALF_LENGTH = 232.0
    MIDPOINT_OFFSET = 55.0
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a line segment of medium thickness with a solid black disc at one endpoint and a small hollow white circle outlined in dark gray at the other. "
        "Five candidate markers A-E are spaced out along the segment; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the segment, the two endpoint markers, and the candidate markers. "
        "In the final state, only the marker one third of the way from the solid black endpoint to the hollow endpoint changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        padding = self.line_width + 8
        width, height = self.canvas_dimensions
        for _ in range(999):
            # The midpoint is offset across the segment rather than along it:
            # sliding it sideways varies where the figure sits in the frame
            # without spending any of the length the option row needs.
            offset = self._rng.uniform(-self.MIDPOINT_OFFSET, self.MIDPOINT_OFFSET)

            def midpoint_for(heading: float) -> Point:
                return Point(
                    width / 2.0 - math.sin(heading) * offset,
                    height / 2.0 + math.cos(heading) * offset,
                )

            def room_for(heading: float) -> tuple[Point, float, float]:
                centre = midpoint_for(heading)
                ahead = (
                    self._max_travel_distance(centre, heading, padding_x=padding) * 0.98
                )
                behind = (
                    self._max_travel_distance(
                        centre, heading + math.pi, padding_x=padding
                    )
                    * 0.98
                )
                return centre, ahead, behind

            headings = [
                math.radians(step)
                for step in range(360)
                if sum(room_for(math.radians(step))[1:]) >= 2.0 * self.MIN_HALF_LENGTH
            ]
            if not headings:
                continue
            angle = self._rng.choice(headings) + self._rng.uniform(
                -math.radians(0.5), math.radians(0.5)
            )
            anchor, ahead, behind = room_for(angle)
            total = ahead + behind
            if total < 2.0 * self.MIN_HALF_LENGTH:
                continue
            # Length first, then where the segment sits along its own heading:
            # a segment always centred on the frame would put the canvas centre
            # at its own midpoint, and "the marker nearest the middle of the
            # frame" would then track the answer's distance from that midpoint.
            length = self._rng.uniform(
                2.0 * self.MIN_HALF_LENGTH, min(total, 2.0 * self.MAX_HALF_LENGTH)
            )
            back = self._rng.uniform(max(0.0, length - ahead), min(behind, length))
            first = self.point_on_ray(anchor, angle + math.pi, back)
            second = self.point_on_ray(anchor, angle, length - back)
            # Which end carries the solid disc is drawn freely; the rule names
            # that end, so the answer is a third of the way out from it and the
            # other trisection point is never a second valid answer (7.2).
            marked, free = (
                (first, second) if self._rng.random() < 0.5 else (second, first)
            )
            target = Point(
                x=marked.x + (free.x - marked.x) / 3.0,
                y=marked.y + (free.y - marked.y) / 3.0,
            )
            if not self.point_can_host_candidate(target):
                continue
            try:
                self._place_collinear_candidates(target, marked, free)
            except RuntimeError:
                continue
            self.marked_endpoint = marked
            self.open_endpoint = free
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid segment trisection puzzle.")

    def _place_collinear_candidates(
        self,
        target: Point,
        marked: Point,
        free: Point,
    ) -> None:
        """Spread A-E along the whole segment, the answer at a free rank.

        Every option sits on the drawn segment, so "which marker is on the
        line?" carries no signal, and how many of them lie
        between the answer and the marked end is drawn uniformly rather than
        inherited from a row centred on the answer.  The row is free to cover
        both trisection points because the solid disc names which end the
        third is measured from.
        """

        step = max(self.point_radius * 2.4, 28.0)
        endpoint_clear = self.point_radius + self.ANCHOR_RADIUS + 4.0
        self.place_candidates_on_incidence(
            target,
            [Incidence.segment(marked, free)],
            # The answer's own neighbours are held to the same step as every
            # other neighbouring pair: a row that keeps extra room around the
            # answer alone hands "the most isolated marker" the answer.
            min_separation=step,
            reject=lambda point: (
                self.distance(point, marked) < endpoint_clear
                or self.distance(point, free) < endpoint_clear
            ),
        )

    def _authored_observation(self) -> str:
        run_heading = self._heading_phrase(
            self.open_endpoint.x - self.marked_endpoint.x,
            self.open_endpoint.y - self.marked_endpoint.y,
        )
        span_x = self.open_endpoint.x - self.marked_endpoint.x
        span_y = self.open_endpoint.y - self.marked_endpoint.y
        span_sq = span_x * span_x + span_y * span_y
        target_t = (
            (self.target_point.x - self.marked_endpoint.x) * span_x
            + (self.target_point.y - self.marked_endpoint.y) * span_y
        ) / span_sq
        closer_count = sum(
            1
            for candidate in self.candidates
            if (
                (candidate.x - self.marked_endpoint.x) * span_x
                + (candidate.y - self.marked_endpoint.y) * span_y
            )
            / span_sq
            < target_t - 1e-6
        )
        closer_phrase = {
            0: "no other center sits between it and that disc",
            1: "one other center sits between it and that disc",
            2: "two other centers sit between it and that disc",
            3: "three other centers sit between it and that disc",
            4: "four other centers sit between it and that disc",
        }[closer_count]
        return (
            f"The solid black disc sits {self._canvas_region(self.marked_endpoint)} and the segment runs "
            f"{run_heading} from it to the hollow endpoint {self._canvas_region(self.open_endpoint)}, with every "
            "marker sitting on the segment itself. Cutting the full length into three equal parts and stepping one "
            "part out from the solid disc places the required point one third of the way along the span; "
            f"{closer_phrase}, and the rest lie farther out toward the hollow end."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "marked_endpoint": self.marked_endpoint.to_list(),
            "open_endpoint": self.open_endpoint.to_list(),
        }

    def _draw_endpoints(self, draw) -> None:
        self.draw_anchor_marker(
            draw,
            self.marked_endpoint,
            self.ANCHOR_RADIUS,
            fill=self.MARKED_ANCHOR_FILL,
            outline=self.MARKED_ANCHOR_FILL,
        )
        self.draw_anchor_marker(draw, self.open_endpoint, self.ANCHOR_RADIUS)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_endpoints(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, [self.marked_endpoint, self.open_endpoint])
        self._draw_endpoints(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

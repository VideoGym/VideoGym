"""Midline (midsegment) of triangle puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class MidlineOfTriangleGenerator(PointTargetPuzzleGenerator):
    """Identify the OTHER endpoint of the midsegment of a triangle.

    The puzzle marks ONE side's midpoint with a small filled black disc
    and shows a short segment emanating from it parallel to the third
    side (a hint that the midsegment connects midpoints of two sides).
    Solvers identify the candidate that lies on the second side at its
    midpoint.

    Distinctively NOT the midpoint task: this asks for the OTHER endpoint
    of the implied midsegment, requiring the solver to pick a midpoint on
    a DIFFERENT side from the visibly-marked one.
    """

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows one black triangle, one filled black disc on a side, and five labeled "
        "candidate circles A-E along one different side. Rule: The filled disc is the midpoint of its side. A triangle "
        "midsegment joins that point to the midpoint of the candidate-bearing side and is parallel to the remaining "
        "side. Select the candidate at this other endpoint. Required Output: Return exactly one uppercase option label "
        "in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/midline_of_triangle"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a triangle drawn as a black outline of medium thickness with one filled black disc inside one of the sides marking the midpoint of that side. "
        "Five candidate markers A-E are clustered along a different side of the triangle near its midpoint; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the triangle, the disc, and the candidate markers, then draws one solid cyan segment of medium thickness from the marked midpoint to the correct candidate, parallel to the remaining triangle side. "
        "In the final state, only the correct midpoint marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        for _ in range(999):
            try:
                p1, p2, p3 = self.sample_triangle_vertices(
                    jitter_ratio=0.50,
                    min_side_ratio=0.30,
                    min_area_ratio=0.060,
                    min_altitude_ratio=0.18,
                    min_angle_deg=46.0,
                    max_angle_deg=92.0,
                )
            except RuntimeError:
                continue
            sides = [(p1, p2, p3), (p2, p3, p1), (p3, p1, p2)]
            idx = self._rng.randint(0, 2)
            sa, sb, opp = sides[idx]
            # Marked midpoint = midpoint of sa-sb.
            marked_mid = Point((sa.x + sb.x) / 2.0, (sa.y + sb.y) / 2.0)
            # Pick which OTHER side hosts the target — randomly between
            # sa-opp and sb-opp.
            other_side_choice = self._rng.choice([(sa, opp), (sb, opp)])
            ta, tb = other_side_choice
            target = Point((ta.x + tb.x) / 2.0, (ta.y + tb.y) / 2.0)
            target_side_len = self.distance(ta, tb)
            min_spacing = self.minimum_candidate_spacing()
            if target_side_len < min_spacing * 4.5:
                continue
            if not self.point_can_host_candidate(target):
                continue

            angle = math.atan2(tb.y - ta.y, tb.x - ta.x)
            try:
                self.place_candidates_line(target, angle=angle)
            except RuntimeError:
                continue

            self.triangle_points = (p1, p2, p3)
            self.marked_mid = marked_mid
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid midline-of-triangle puzzle.")

    def _authored_observation(self) -> str:
        verts = list(self.triangle_points)
        names = self._visible_point_names(verts)

        def _side_midpoint(pair: tuple[int, int]) -> Point:
            first, second = verts[pair[0]], verts[pair[1]]
            return Point((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)

        pairs = ((0, 1), (1, 2), (2, 0))
        marked_pair = min(pairs, key=lambda pair: self.distance(_side_midpoint(pair), self.marked_mid))
        target_pair = min(pairs, key=lambda pair: self.distance(_side_midpoint(pair), self.target_point))
        third_pair = next(pair for pair in pairs if pair not in (marked_pair, target_pair))
        heading = self._heading_phrase(
            self.target_point.x - self.marked_mid.x,
            self.target_point.y - self.marked_mid.y,
        )
        return (
            f"The black disc {self._canvas_region(self.marked_mid)} marks the midpoint of the side joining "
            f"the {names[marked_pair[0]]} and the {names[marked_pair[1]]}, while the markers line up along "
            f"the side joining the {names[target_pair[0]]} and the {names[target_pair[1]]}. A midsegment "
            "joins the midpoints of two sides and runs parallel to the third side, here the one joining the "
            f"{names[third_pair[0]]} and the {names[third_pair[1]]}. Starting at the disc and heading "
            f"{heading} parallel to that side, I reach the marker-bearing side exactly at its halfway point, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "triangle_vertices": [p.to_list() for p in self.triangle_points],
            "marked_midpoint": self.marked_mid.to_list(),
        }

    def _draw_marked_disc(self, draw) -> None:
        cx, cy = round(self.marked_mid.x), round(self.marked_mid.y)
        r = 7
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0), outline=(0, 0, 0), width=1)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_marked_disc(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        p1, p2, p3 = self.triangle_points
        self.draw_line(draw, [p1, p2, p3, p1])
        if highlight_label:
            # Draw the midsegment (solid black) from marked midpoint to
            # target — this is the answer.
            self.draw_line(draw, [self.marked_mid, self.target_point])
        self._draw_marked_disc(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

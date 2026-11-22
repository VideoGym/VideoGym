"""Diagonal intersection puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


def _line_intersection(p1: Point, p2: Point, p3: Point, p4: Point):
    """Intersection of line p1-p2 and line p3-p4. Returns None if parallel."""
    d1x, d1y = p2.x - p1.x, p2.y - p1.y
    d2x, d2y = p4.x - p3.x, p4.y - p3.y
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-9:
        return None
    t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / denom
    return Point(p1.x + t * d1x, p1.y + t * d1y)


class DiagonalIntersectionGenerator(PointTargetPuzzleGenerator):
    """Find where the diagonals of a convex quadrilateral cross."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows a convex quadrilateral as a black outline, plus five labeled candidate circles A-E inside it. "
        "Identify which candidate is the intersection point of the two diagonals. Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/diagonal_intersection"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a convex quadrilateral drawn as a black outline of medium thickness. "
        "Five candidate markers A-E are clustered near the hidden diagonal intersection; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the quadrilateral and the candidate markers, then draws two black diagonal segments of medium thickness connecting opposite vertices, crossing at the correct candidate. "
        "In the final state, only the intersection marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def __init__(self, *args, concave=False, **kwargs):
        self.concave = bool(concave)
        if self.concave:
            self.DEFAULT_RULE_PROMPT = "Input Image: A simple concave quadrilateral and five candidate markers are shown. Rule: Intersect the two infinite lines through opposite vertex pairs, extending the diagonals where necessary. Required Output: Answer with one option from A-E."
            self.DEFAULT_TI2V_PROMPT = "Input Image: A simple concave quadrilateral and five candidate markers are shown. Rule: Find the intersection of the supporting lines of its two diagonals. Required Output: Draw each diagonal in cyan and extend it as needed until the two lines meet at the correct candidate; highlight only that candidate. Preserve the outline and all markers."
            self.DEFAULT_SOLUTION_IMAGE_PROMPT = self.strip_video_instruction(self.DEFAULT_TI2V_PROMPT)
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            center = self.pick_target_point(0.44, padding=candidate_padding + self.canvas_short_side * 0.10)
            verts = []
            base_a = self._rng.uniform(0.0, math.tau)
            for i in range(4):
                a = base_a + math.tau / 4 * i + self._rng.uniform(-0.35, 0.35)
                d = self._rng.uniform(self.canvas_short_side * 0.14, self.canvas_short_side * 0.26)
                verts.append(self.point_on_ray(center, a, d))
            if not all(self.inside_canvas(v, padding=self.line_width + 4) for v in verts):
                continue
            if self.concave:
                # Place vertex 2 inside triangle 0-1-3, between vertex 0 and
                # the opposite side. The 0-2 diagonal must extend past 2.
                t = self._rng.uniform(0.35, 0.65)
                crossing = Point(verts[1].x * t + verts[3].x * (1-t), verts[1].y * t + verts[3].y * (1-t))
                q = self._rng.uniform(0.30, 0.50)
                verts[2] = Point(verts[0].x + q * (crossing.x-verts[0].x), verts[0].y + q * (crossing.y-verts[0].y))
                if self.distance(verts[2], crossing) < 45:
                    continue
            cross = _line_intersection(verts[0], verts[2], verts[1], verts[3])
            if cross is None:
                continue
            if not self.point_can_host_candidate(cross):
                continue
            area = 0.0
            for i in range(4):
                j = (i + 1) % 4
                area += verts[i].x * verts[j].y - verts[j].x * verts[i].y
            if abs(area) < (self.canvas_short_side ** 2) * 0.04:
                continue
            try:
                self.place_candidates(cross)
            except RuntimeError:
                continue
            self.quad = tuple(verts)
            self.target_point = cross
            return self.save_puzzle()
        raise RuntimeError("Failed to generate diagonal intersection puzzle.")

    def _authored_observation(self) -> str:
        if self.concave:
            return "One vertex is reflex. One diagonal ends before the other diagonal's supporting line; extend it beyond the reflex vertex to find their unique intersection outside the polygon."
        corners = list(self.quad)
        names = [self._point_position_label(corner, corners) for corner in corners]
        return (
            "The scene contains just the convex quadrilateral's black outline. Each diagonal joins one pair of "
            f"opposite corners — the {names[0]} corner with the {names[2]} corner, and the {names[1]} corner "
            f"with the {names[3]} corner — and since the shape is convex those two chords cross exactly once, "
            f"{self._canvas_region(self.target_point)}. The required point is that crossing, so I take the "
            "marker centered where the two diagonals meet."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {"quadrilateral": [v.to_list() for v in self.quad]}

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, list(self.quad) + [self.quad[0]])
        if highlight_label:
            self.draw_line(draw, [self.quad[0], self.target_point if self.concave else self.quad[2]])
            self.draw_line(draw, [self.quad[1], self.quad[3]])
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

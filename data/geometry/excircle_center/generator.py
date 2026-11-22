"""Excircle center puzzle generator."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from PIL import Image

from data.geometry.core.point_target_base import (
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


def _compute_excircle(p1: Point, p2: Point, p3: Point) -> Tuple[Point, float]:
    """Compute the excircle opposite p1.

    The excircle is tangent externally to side p2-p3 and tangent to the
    extensions of sides p1-p2 and p1-p3.

    Returns (center, radius).  Returns (Point(0,0), 0) for degenerate cases.
    """
    a = math.hypot(p2.x - p3.x, p2.y - p3.y)  # side opposite p1
    b = math.hypot(p1.x - p3.x, p1.y - p3.y)  # side opposite p2
    c = math.hypot(p1.x - p2.x, p1.y - p2.y)  # side opposite p3
    denom = -a + b + c
    if abs(denom) < 1e-9:
        return Point(0, 0), 0.0
    ix = (-a * p1.x + b * p2.x + c * p3.x) / denom
    iy = (-a * p1.y + b * p2.y + c * p3.y) / denom
    s = (a + b + c) / 2.0
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0:
        return Point(0, 0), 0.0
    area = math.sqrt(area_sq)
    r = area / (s - a)
    return Point(ix, iy), r


class ExcircleCenterGenerator(PointTargetPuzzleGenerator):
    """Find the center of the excircle opposite one vertex of a triangle."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows a black triangle outline with one vertex marked by a filled dot, plus five "
        "labeled candidate circles A-E outside the triangle. Rule: Use the excircle opposite the marked vertex: it is "
        "tangent to the side opposite that vertex and to the extensions of the two adjacent sides. Select the candidate "
        "that is the center of this excircle. Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/excircle_center"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a triangle drawn as a black outline of medium thickness with one vertex highlighted by a small filled black circle. "
        "Five candidate markers A-E are clustered near the hidden excircle center outside the triangle; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the triangle, the highlighted vertex, and the candidate markers, then draws two cyan construction extension segments of medium thickness extending the two sides adjacent to the highlighted vertex beyond the triangle, "
        "and finally draws one cyan construction excircle of medium thickness centered on the correct option, tangent to the opposite side and the two extension lines. "
        "In the final state, only the excircle center marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        for _ in range(999):
            p1, p2, p3 = self.sample_triangle_vertices(
                jitter_ratio=0.50,
                min_side_ratio=0.16,
                min_area_ratio=0.030,
                min_altitude_ratio=0.12,
                min_angle_deg=40.0,
                max_angle_deg=100.0,
            )
            vertices = [p1, p2, p3]
            idx = self._rng.randint(0, 2)
            va = vertices[idx]
            vb = vertices[(idx + 1) % 3]
            vc = vertices[(idx + 2) % 3]

            center, r = _compute_excircle(va, vb, vc)
            if r < self.canvas_short_side * 0.06:
                continue
            if not self.circle_fits(center, r, extra_padding=self.line_width):
                continue
            if not self.point_can_host_candidate(center):
                continue

            ext_len = self.canvas_short_side * 0.22
            d_ab = max(1e-9, self.distance(va, vb))
            d_ac = max(1e-9, self.distance(va, vc))
            ext_b = Point(
                x=vb.x + (vb.x - va.x) / d_ab * ext_len,
                y=vb.y + (vb.y - va.y) / d_ab * ext_len,
            )
            ext_c = Point(
                x=vc.x + (vc.x - va.x) / d_ac * ext_len,
                y=vc.y + (vc.y - va.y) / d_ac * ext_len,
            )
            if not self.inside_canvas(ext_b, padding=self.line_width):
                continue
            if not self.inside_canvas(ext_c, padding=self.line_width):
                continue

            # Place distractors near the excircle center, scattered within
            # a small disc around it instead of using the default
            # `place_candidates` (which spreads them apart, often into a
            # tight cluster in dead canvas far from the answer).
            try:
                self._place_excircle_candidates(center, r)
            except RuntimeError:
                continue

            self.triangle_points = (va, vb, vc)
            self.marked_vertex = va
            self.ext_b = ext_b
            self.ext_c = ext_c
            self.target_point = center
            self.r = r
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid excircle center puzzle.")

    def _place_excircle_candidates(self, target: Point, radius: float) -> None:
        """Place candidates: 1 at the excircle center, 4 distractors in a
        small disc around it. Distractors stay close enough that "the one
        further from the triangle" is not a freebie shortcut.
        """
        # Distractors stay within reach of the excircle center, so they all
        # sit "where an excircle center could plausibly be" and neither the
        # far one nor the near one gives the answer away.  Growing that reach
        # around the answer would put it at the middle of its own cloud, so
        # the cloud is laid out first and one of its slots taken for the
        # answer.
        marker_clear = max(self.point_radius * 2.4, 28.0)
        self.seat_candidates(target, min_separation=marker_clear)

    def _authored_observation(self) -> str:
        marked, far_one, far_two = self.triangle_points
        names = self._visible_point_names([marked, far_one, far_two])
        opposite_mid = Point((far_one.x + far_two.x) / 2.0, (far_one.y + far_two.y) / 2.0)
        outward = self._relative_position_phrase(opposite_mid, self.target_point)
        return (
            f"The marked corner is the {names[0]}, so the excircle in play leans against the side joining the "
            f"{names[1]} and the {names[2]}, on the far side from that corner. Prolonging the marked corner's "
            "two sides onward past those other corners, the circle must touch the opposite side and both "
            f"prolongations at one common distance, which places its center {outward} that side, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "triangle_vertices": [p.to_list() for p in self.triangle_points],
            "marked_vertex_index": 0,
            "excircle_radius": self.r,
        }

    def _draw_marked_vertex(self, draw) -> None:
        self.draw_anchor_marker(draw, self.marked_vertex, 7)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_marked_vertex(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        va, vb, vc = self.triangle_points
        self.draw_line(draw, [va, vb, vc, va])
        if highlight_label:
            # Extension lines (solution only)
            self.draw_line(draw, [vb, self.ext_b])
            self.draw_line(draw, [vc, self.ext_c])
            # Excircle
            self.draw_circle(draw, self.target_point, self.r)
        self._draw_marked_vertex(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

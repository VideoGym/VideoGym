"""Napoleon vertex puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


def _equilateral_apex_outward(pa: Point, pb: Point, pc: Point) -> Point:
    """Equilateral triangle apex on side pa-pb, outward from pc."""
    mx = (pa.x + pb.x) / 2
    my = (pa.y + pb.y) / 2
    dx, dy = pb.x - pa.x, pb.y - pa.y
    s = math.hypot(dx, dy)
    if s < 1e-9:
        return Point(mx, my)
    ux, uy = -dy / s, dx / s
    h = s * math.sqrt(3) / 2
    a1 = Point(mx + h * ux, my + h * uy)
    a2 = Point(mx - h * ux, my - h * uy)
    return a1 if math.hypot(a1.x - pc.x, a1.y - pc.y) > math.hypot(a2.x - pc.x, a2.y - pc.y) else a2


class NapoleonVertexGenerator(PointTargetPuzzleGenerator):
    """Find the centroid of an outward equilateral triangle on one side (Napoleon triangle vertex)."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows one black triangle. One of its sides is also the blue base of a complete "
        "blue equilateral triangle constructed outward from the black triangle. Five labeled candidate circles A-E "
        "are visible near the blue triangle. Rule: Use the visible blue equilateral triangle only. Its centroid is "
        "the common intersection of its three medians, where each median joins a vertex to the midpoint of the "
        "opposite side. Select the candidate at that intersection. Required Output: Return exactly one uppercase "
        "option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/napoleon_vertex"
    SELECTED_EQUILATERAL_COLOR = (45, 96, 176)
    DEFAULT_TI2V_PROMPT = (
        "Input Image: A white canvas shows one black triangle. One of its sides is also the blue base of a complete "
        "blue equilateral triangle constructed outward from the black triangle. Five labeled candidate circles A-E "
        "are visible near the blue triangle. Rule: Use the visible blue equilateral triangle only. Its centroid is "
        "the common intersection of its three medians, where each median joins a vertex to the midpoint of the "
        "opposite side. Select the candidate at that intersection. Required Output: Hold the complete input, then "
        "draw the three medians of the blue triangle in cyan one at a time, retaining each completed stroke in the "
        "cumulative construction. Once their common intersection "
        "is established, highlight only the selected candidate. Keep both triangle outlines, all unselected "
        "candidates, and the background fixed."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Input Image: A white canvas shows one black triangle, one complete blue outward equilateral triangle sharing "
        "a side with it, and five labeled candidate circles A-E. Rule: The selected candidate is the centroid of "
        "the visible blue equilateral triangle, at the common intersection of its three medians. Required Output: "
        "Preserve the complete input, add all three median segments of the blue triangle in cyan, and highlight only the "
        "candidate at their common intersection."
    )

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        for _ in range(999):
            p1, p2, p3 = self.sample_triangle_vertices(
                jitter_ratio=0.50,
                min_side_ratio=0.16,
                min_area_ratio=0.030,
                min_altitude_ratio=0.12,
                min_angle_deg=40.0,
                max_angle_deg=108.0,
            )
            sides = [(p1, p2, p3), (p2, p3, p1), (p3, p1, p2)]
            idx = self._rng.randint(0, 2)
            pa, pb, pc = sides[idx]
            apex = _equilateral_apex_outward(pa, pb, pc)
            centroid = Point(
                x=(pa.x + pb.x + apex.x) / 3,
                y=(pa.y + pb.y + apex.y) / 3,
            )
            if not self.inside_canvas(apex, padding=self.line_width):
                continue
            if not self.point_can_host_candidate(centroid):
                continue
            # Stricter min-pairwise-spacing check so the cluster doesn't
            # collapse into touching disks (E/B and A/C were colliding
            # in earlier samples). Retry placement up to 8 times.
            strict_gap = max(self.point_radius * 2.8 + 6.0,
                             self.minimum_candidate_spacing() * 1.10)
            placed_ok = False
            for _ in range(8):
                try:
                    self.place_candidates(centroid)
                except RuntimeError:
                    continue
                cand_pts = [Point(c.x, c.y) for c in self.candidates]
                if self.points_are_well_spaced(cand_pts, min_distance=strict_gap):
                    placed_ok = True
                    break
            if not placed_ok:
                continue
            self.triangle_points = (p1, p2, p3)
            self.eq_apex = apex
            self.eq_side = (pa, pb)
            self.target_point = centroid
            return self.save_puzzle()
        raise RuntimeError("Failed to generate napoleon vertex puzzle.")

    def _authored_observation(self) -> str:
        base_a, base_b = self.eq_side
        base_mid = Point((base_a.x + base_b.x) / 2.0, (base_a.y + base_b.y) / 2.0)
        outward = self._heading_phrase(
            self.eq_apex.x - base_mid.x,
            self.eq_apex.y - base_mid.y,
        )
        return (
            f"The blue equilateral triangle stands on its shared side with the black triangle and points "
            f"{outward}, its apex sitting {self._canvas_region(self.eq_apex)}. Only the blue construction "
            "matters here: its centroid is where its three medians cross, each median joining one blue corner "
            "to the midpoint of the opposite blue side. Those medians meet "
            f"{self._canvas_region(self.target_point)}, in the middle of the blue outline."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "triangle_vertices": [p.to_list() for p in self.triangle_points],
            # These three vertices are all visible in the blue input
            # construction.  The canonical verifier deliberately derives the
            # centroid from them and never receives the authoring side index.
            "constructed_equilateral_vertices": [
                self.eq_side[0].to_list(),
                self.eq_side[1].to_list(),
                self.eq_apex.to_list(),
            ],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, list(self.triangle_points) + [self.triangle_points[0]])
        pa, pb = self.eq_side
        # The selected side and its entire outward equilateral construction
        # are source primitives, not a hidden solution overlay.
        self.draw_line(
            draw,
            [pa, self.eq_apex, pb, pa],
            fill=self.SELECTED_EQUILATERAL_COLOR,
        )
        if highlight_label:
            for vertex, first, second in (
                (pa, pb, self.eq_apex),
                (pb, pa, self.eq_apex),
                (self.eq_apex, pa, pb),
            ):
                midpoint = Point((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)
                self.draw_line(
                    draw,
                    [vertex, midpoint],
                    width_factor=0.55,
                    fill=self.CONSTRUCTION_COLOR,
                )
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

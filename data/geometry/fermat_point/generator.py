"""Fermat point puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


def _point_in_triangle(p: Point, a: Point, b: Point, c: Point) -> bool:
    """Inclusive point-in-triangle test via signed area sign matching."""
    def _cross(o: Point, q: Point, r: Point) -> float:
        return (q.x - o.x) * (r.y - o.y) - (q.y - o.y) * (r.x - o.x)

    d1 = _cross(p, a, b)
    d2 = _cross(p, b, c)
    d3 = _cross(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _compute_fermat_point(p1: Point, p2: Point, p3: Point) -> Optional[Point]:
    """Compute the Fermat point using Weiszfeld iteration.

    Returns None if any triangle angle >= 120 degrees.
    """
    def _angle(a: Point, vertex: Point, b: Point) -> float:
        v1x, v1y = a.x - vertex.x, a.y - vertex.y
        v2x, v2y = b.x - vertex.x, b.y - vertex.y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 < 1e-9 or mag2 < 1e-9:
            return 0.0
        cos_val = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (mag1 * mag2)))
        return math.degrees(math.acos(cos_val))

    angles = [_angle(p2, p1, p3), _angle(p1, p2, p3), _angle(p1, p3, p2)]
    if max(angles) >= 120.0:
        return None

    x = (p1.x + p2.x + p3.x) / 3.0
    y = (p1.y + p2.y + p3.y) / 3.0
    pts = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
    for _ in range(500):
        wx, wy, ws = 0.0, 0.0, 0.0
        for px, py in pts:
            d = math.hypot(x - px, y - py)
            if d < 1e-12:
                return Point(px, py)
            w = 1.0 / d
            wx += w * px
            wy += w * py
            ws += w
        nx, ny = wx / ws, wy / ws
        if math.hypot(nx - x, ny - y) < 1e-9:
            break
        x, y = nx, ny
    return Point(x, y)


def _equilateral_apex(pa: Point, pb: Point, pc: Point) -> Point:
    """Build equilateral triangle on side pa-pb, on the outside relative to pc."""
    mx = (pa.x + pb.x) / 2
    my = (pa.y + pb.y) / 2
    dx = pb.x - pa.x
    dy = pb.y - pa.y
    side_len = math.hypot(dx, dy)
    if side_len < 1e-9:
        return Point(mx, my)
    ux, uy = -dy / side_len, dx / side_len
    h = side_len * math.sqrt(3) / 2
    a1 = Point(mx + h * ux, my + h * uy)
    a2 = Point(mx - h * ux, my - h * uy)
    d1 = math.hypot(a1.x - pc.x, a1.y - pc.y)
    d2 = math.hypot(a2.x - pc.x, a2.y - pc.y)
    return a1 if d1 > d2 else a2


class FermatPointGenerator(PointTargetPuzzleGenerator):
    """Find the Fermat point of a triangle (minimizes total distance to vertices)."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A black triangle outline is shown on a white canvas with five labeled candidate circles A-E inside "
        "it. Rule: The Fermat point is the interior point that minimizes the sum of its distances to the three vertices. "
        "For the shown triangle, it is equivalently the point at which each pair of vertex rays meets at 120 degrees. "
        "Select the candidate at that point. Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/fermat_point"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a triangle drawn only as a black outline of medium thickness, with the interior left empty. "
        "Five candidate markers A-E are clustered near the hidden Fermat point inside the triangle; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the triangle and the candidate markers, then constructs three outward equilateral triangles on the three sides as black outlines of medium thickness, "
        "and draws three black connector segments from each equilateral apex to the opposite original vertex, all converging at the correct candidate. "
        "In the final state, only the Fermat point marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        for _ in range(999):
            p1, p2, p3 = self.sample_triangle_vertices(
                jitter_ratio=0.56,
                min_side_ratio=0.18,
                min_area_ratio=0.035,
                min_altitude_ratio=0.13,
                min_angle_deg=38.0,
                max_angle_deg=108.0,
            )
            fermat = _compute_fermat_point(p1, p2, p3)
            if fermat is None:
                continue

            # Check equilateral apexes fit in canvas
            apexes = [
                _equilateral_apex(p2, p3, p1),
                _equilateral_apex(p1, p3, p2),
                _equilateral_apex(p1, p2, p3),
            ]
            if not all(self.inside_canvas(a, padding=self.line_width) for a in apexes):
                continue

            if not self.point_can_host_candidate(fermat):
                continue
            # Place 4 distractors INSIDE the triangle within a small disc
            # of the Fermat point. The default `place_candidates` ignores
            # the triangle interior constraint and routinely scatters
            # markers along an edge / outside the triangle.
            try:
                self._place_fermat_candidates(fermat, p1, p2, p3)
            except RuntimeError:
                continue

            self.triangle_points = (p1, p2, p3)
            self.apexes = apexes
            self.target_point = fermat
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid Fermat point puzzle.")

    def _place_fermat_candidates(
        self,
        target: Point,
        a: Point,
        b: Point,
        c: Point,
    ) -> None:
        """Place 5 markers; correct one at the Fermat point. The other 4
        must each lie strictly inside the triangle and within ~70 px of
        the answer so that "is it inside the triangle" is no longer a
        free disambiguator.
        """
        # Every distractor is strictly inside the triangle and clear of its
        # edges, so "is it inside" stops disambiguating.  The cloud is laid
        # out before one of its slots is taken for the answer, so the answer
        # is not the marker its four neighbours surround.
        edge_clear = max(self.point_radius * 1.4, 16.0)
        marker_clear = max(self.point_radius * 2.4, 28.0)

        def inside(point: Point) -> bool:
            return _point_in_triangle(point, a, b, c) and all(
                self.distance_point_to_line(point, first, second) >= edge_clear
                for first, second in ((a, b), (b, c), (c, a))
            )

        self.seat_candidates(target, accept=inside, min_separation=marker_clear)

    def _authored_observation(self) -> str:
        vertices = list(self.triangle_points)
        names = self._visible_point_names(vertices)
        return (
            f"The triangle joins the {names[0]}, the {names[1]}, and the {names[2]}, and all the labeled "
            "markers sit inside it. The point minimizing the summed distance to the three corners is the one "
            "that views each side under 120 degrees, and it can be built by raising an outward equilateral "
            "triangle on each side and joining each new apex to the corner opposite it. Those three "
            f"connectors converge {self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "triangle_vertices": [p.to_list() for p in self.triangle_points],
            "equilateral_apexes": [a.to_list() for a in self.apexes],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        p1, p2, p3 = self.triangle_points
        self.draw_line(draw, [p1, p2, p3, p1])
        if highlight_label:
            sides = [(p2, p3, p1), (p1, p3, p2), (p1, p2, p3)]
            for (pa, pb, pc), apex in zip(sides, self.apexes):
                # Equilateral triangle outline (two new sides)
                self.draw_line(draw, [pa, apex, pb])
                # Line from apex to opposite original vertex
                self.draw_line(draw, [apex, pc])
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

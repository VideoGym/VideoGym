"""TriangleCenter puzzle generator.
"""

from __future__ import annotations
from typing import Optional
from PIL import Image
from data.geometry.core.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class TriangleCenterGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the center of a triangle."""

    DEFAULT_RULE_PROMPT=(
        "A black triangle outline is shown on a white canvas with five labeled candidate circles A-E near its interior. "
        "Determine which candidate is the triangle's center of mass, namely the centroid — the unique point where the "
        "three medians (each running from a vertex to the midpoint of the opposite side) meet. "
        "Answer with one option from A-E."
    )
    DEFAULT_OUTPUT_DIR = "outputs/geometry/triangle_center"
    DEFAULT_TI2V_PROMPT=(
        "A square white canvas shows a triangle drawn only as a black outline of medium thickness, with no midpoint marks at first. "
        "Five candidate markers A-E are clustered near the hidden centroid; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the triangle and the five markers, then draws three black medians of medium thickness, each running from one triangle vertex to the midpoint of the opposite side so all three meet at one point. "
        "In the final state, only the centroid marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        p1, p2, p3 = self.sample_triangle_vertices(
            jitter_ratio=0.56,
            min_side_ratio=0.18,
            min_area_ratio=0.035,
            min_altitude_ratio=0.13,
            min_angle_deg=38.0,
            max_angle_deg=108.0,
        )
        center=Point(
            x=(p1.x + p2.x + p3.x) / 3,
            y=(p1.y + p2.y + p3.y) / 3,
        )
        p12=Point(x=(p1.x + p2.x)/2, y=(p1.y + p2.y)/2)
        p23=Point(x=(p2.x + p3.x)/2, y=(p2.y + p3.y)/2)
        p31=Point(x=(p3.x + p1.x)/2, y=(p3.y + p1.y)/2)
        self.triangle_points = (p1, p2, p3)
        self.midpoints = (p23, p31, p12)
        self.target_point = center
        self.place_candidates(center)
        return self.save_puzzle()

    def _authored_observation(self) -> str:
        vertices = list(self.triangle_points)
        names = self._visible_point_names(vertices)
        return (
            f"The triangle's outline joins the {names[0]}, the {names[1]}, and the {names[2]}, with the "
            "labeled markers clustered around its middle. A median runs from each corner to the midpoint of "
            "the side opposite it, and the three medians always share one crossing, the balance point sitting "
            "a third of the way up each median from its side; here that crossing lands "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "triangle_vertices": [point.to_list() for point in self.triangle_points],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()

        self.draw_line(draw,list(self.triangle_points)+[self.triangle_points[0]])

        if highlight_label:
            # Medians run from each triangle vertex to the opposite-side
            # midpoint. The vertex-side endpoint shares ink with the
            # triangle outline's curve joint — inset via join_start so the
            # butt cap hides inside the joint (no vertex "bump"). The
            # midpoint-side endpoint lies in the middle of an edge, so the
            # edge's stroke also covers that butt cap (join_end).
            for vertex, midpoint in zip(self.triangle_points, self.midpoints):
                self.draw_line(
                    draw, [vertex, midpoint],
                    join_start=True, join_end=True,
                )

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

"""Triangle-from-midpoints puzzle generator."""

from __future__ import annotations

from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class TriangleFromMidpointsGenerator(PointTargetPuzzleGenerator):
    """Given the midpoints of a triangle's three sides, find one original vertex."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows three hollow circular markers connected as a black triangle, plus five "
        "labeled candidate circles A-E. Rule: Treat the three hollow markers as the side midpoints of one hidden "
        "triangle. Each original vertex equals the sum of its two adjacent midpoint positions minus the opposite "
        "midpoint position. Select the unique candidate matching any reconstructed original vertex; the instance "
        "contains exactly one such candidate. Required Output: Return exactly one uppercase option label in the form "
        "\"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/triangle_from_midpoints"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows three small hollow circular markers outlined in black representing the midpoints of the three sides of a hidden triangle, connected by black line segments of medium thickness forming the medial triangle. "
        "Five candidate markers A-E are clustered near the hidden original vertex; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the medial triangle and the candidate markers, then draws the three recovered original-triangle sides in cyan one at a time around the triangle, completing each side from one vertex to the next before starting another. "
        "In the final state, only the original-vertex marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            m1 = self.pick_target_point(0.48, padding=candidate_padding + self.canvas_short_side * 0.08)
            m2 = self.pick_target_point(0.48, padding=candidate_padding + self.canvas_short_side * 0.08)
            m3 = self.pick_target_point(0.48, padding=candidate_padding + self.canvas_short_side * 0.08)
            if self.triangle_area(m1, m2, m3) < (self.canvas_short_side ** 2) * 0.015:
                continue
            sides = self.triangle_side_lengths(m1, m2, m3)
            if min(sides) < self.canvas_short_side * 0.10:
                continue

            # Original vertices: A = M_AB + M_AC - M_BC, etc.
            # Let m1=M_BC, m2=M_AC, m3=M_AB
            vA = Point(x=m3.x + m2.x - m1.x, y=m3.y + m2.y - m1.y)
            vB = Point(x=m3.x + m1.x - m2.x, y=m3.y + m1.y - m2.y)
            vC = Point(x=m2.x + m1.x - m3.x, y=m2.y + m1.y - m3.y)

            # Pick one vertex as target
            idx = self._rng.randint(0, 2)
            target = [vA, vB, vC][idx]

            if not self.inside_canvas(target, padding=self.line_width + 4):
                continue
            if not all(self.inside_canvas(v, padding=self.line_width) for v in [vA, vB, vC]):
                continue
            if not self.point_can_host_candidate(target):
                continue

            try:
                self.place_candidates(target)
            except RuntimeError:
                continue

            self.midpoints = (m1, m2, m3)
            self.original_verts = (vA, vB, vC)
            self.target_idx = idx
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate triangle-from-midpoints puzzle.")

    def _authored_observation(self) -> str:
        midpoint_points = list(self.midpoints)
        names = [
            name.removesuffix(" visible vertex")
            for name in self._visible_point_names(midpoint_points)
        ]
        opposite_name = names[self.target_idx]
        adjacent_names = [name for index, name in enumerate(names) if index != self.target_idx]
        opposite_mid = midpoint_points[self.target_idx]
        heading = self._heading_phrase(
            self.target_point.x - opposite_mid.x,
            self.target_point.y - opposite_mid.y,
        )
        return (
            "The three hollow markers are the side midpoints of a hidden triangle, and the black outline "
            "joining them is its medial triangle. Undoing that relation doubles the figure, so every "
            "recovered corner falls outside the small outline. For the corner opposite the "
            f"{opposite_name} midpoint, I start at that midpoint, pass through the point halfway between the "
            f"{adjacent_names[0]} and {adjacent_names[1]} midpoints, and continue the same distance again, "
            f"heading {heading} and landing {self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "midpoints": [m.to_list() for m in self.midpoints],
            "original_vertices": [v.to_list() for v in self.original_verts],
            "target_vertex_index": self.target_idx,
        }

    def _draw_midpoint_markers(self, draw) -> None:
        for m in self.midpoints:
            cx, cy = round(m.x), round(m.y)
            radius = 7
            width = max(2, self.line_width)
            inner_radius = max(1, radius - width)
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=None,
                outline=self.CANDIDATE_OUTLINE_COLOR,
                width=width,
            )
            draw.ellipse(
                (
                    cx - inner_radius,
                    cy - inner_radius,
                    cx + inner_radius,
                    cy + inner_radius,
                ),
                fill=self.CANDIDATE_BASE_FILL,
                outline=None,
            )

    def _video_overlay_extras(self, draw) -> None:
        self._draw_midpoint_markers(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, list(self.midpoints) + [self.midpoints[0]])
        if highlight_label:
            vA, vB, vC = self.original_verts
            self.draw_line(draw, [vA, vB], fill=self.CONSTRUCTION_COLOR, join_end=True)
            self.draw_line(
                draw,
                [vB, vC],
                fill=self.CONSTRUCTION_COLOR,
                join_start=True,
                join_end=True,
            )
            self.draw_line(draw, [vC, vA], fill=self.CONSTRUCTION_COLOR, join_start=True)
        self._draw_midpoint_markers(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

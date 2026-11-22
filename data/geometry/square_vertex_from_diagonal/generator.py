"""Square vertex from diagonal puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class SquareVertexFromDiagonalGenerator(PointTargetPuzzleGenerator):
    """Given two opposite (diagonal) vertices of a square, identify which
    candidate is one of the other two vertices.

    Distinctively different from `inscribed_square_vertex` (which inscribes
    a square inside another shape): here we are given the DIAGONAL of the
    square explicitly as two anchor markers, and must rotate by 90 degrees
    around the diagonal's midpoint by half its length to find the missing
    vertices.
    """

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows two normal-size outlined anchor circles connected by a black segment representing the diagonal of a "
        "square, plus five labeled candidate circles A-E. Identify which candidate is one of the other two vertices of the "
        "square. Each missing vertex sits on a perpendicular through the diagonal's midpoint, at a distance equal to half the "
        "diagonal length. Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/square_vertex_from_diagonal"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows two normal-size hollow anchor circles outlined in dark gray and connected by one black line segment of medium thickness; the two circles mark two opposite corners of a square (its diagonal). "
        "Five candidate markers A-E are clustered near one of the two missing square vertices; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the diagonal segment, the two anchor circles, and the candidate markers, then draws two black segments of medium thickness from each anchor to the correct candidate, completing one half of the square. "
        "In the final state, only the missing-vertex marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            center = self.pick_target_point(0.36, padding=candidate_padding + self.canvas_short_side * 0.10)
            half_diag = self._rng.uniform(self.canvas_short_side * 0.16, self.canvas_short_side * 0.26)
            angle = self._rng.uniform(0.0, math.tau)
            # Two diagonal vertices.
            d1 = self.point_on_ray(center, angle, half_diag)
            d2 = self.point_on_ray(center, angle + math.pi, half_diag)
            # Two perpendicular vertices.
            perp = angle + math.pi / 2.0
            v1 = self.point_on_ray(center, perp, half_diag)
            v2 = self.point_on_ray(center, perp + math.pi, half_diag)

            anchor_marker_r = self.point_radius
            for pt in (d1, d2, v1, v2):
                if not self.inside_canvas(pt, padding=anchor_marker_r + 4):
                    break
            else:
                # Pick which missing vertex is the target.
                target = self._rng.choice([v1, v2])
                if not self.point_can_host_candidate(target):
                    continue
                try:
                    self.place_candidates(target)
                except RuntimeError:
                    continue
                self.diagonal = (d1, d2)
                self.both_missing = (v1, v2)
                self.target_point = target
                return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid square-vertex-from-diagonal puzzle.")

    def _authored_observation(self) -> str:
        diag_start, diag_end = self.diagonal
        diagonal_direction = self._heading_phrase(
            diag_end.x - diag_start.x, diag_end.y - diag_start.y,
        )
        diagonal_mid = Point(
            (diag_start.x + diag_end.x) / 2.0,
            (diag_start.y + diag_end.y) / 2.0,
        )
        return (
            f"The segment between the two hollow anchors runs {diagonal_direction} and is the square's full "
            f"diagonal, so the anchors are opposite corners. The two missing corners lie on the perpendicular "
            f"through the diagonal's midpoint, each half a diagonal away from it; the one the candidates cluster "
            f"around is {self._relative_position_phrase(diagonal_mid, self.target_point)} that midpoint, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "diagonal_endpoints": [d.to_list() for d in self.diagonal],
            "missing_vertices": [v.to_list() for v in self.both_missing],
        }

    def _draw_anchor_circles(self, draw) -> None:
        anchor_r = self.point_radius
        for pt in self.diagonal:
            self.draw_anchor_marker(draw, pt, anchor_r)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_anchor_circles(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        d1, d2 = self.diagonal
        # Draw the diagonal segment first; anchors paint on top of it.
        self.draw_line(draw, [d1, d2])
        if highlight_label:
            for anchor in (d1, d2):
                self.draw_line(draw, [anchor, self.target_point],
                               join_start=True, join_end=True)
        self._draw_anchor_circles(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

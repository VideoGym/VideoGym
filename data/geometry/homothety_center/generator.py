"""Homothety center puzzle generator."""

from __future__ import annotations

import math
from typing import List, Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class HomothetyCenterGenerator(PointTargetPuzzleGenerator):
    """Find the center of dilation between two similar parallel triangles."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows two triangle outlines of different sizes whose corresponding sides are parallel, plus five "
        "labeled candidate circles A-E. Determine which candidate is the center of dilation: the unique point where lines "
        "connecting each pair of corresponding vertices all intersect. Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/homothety_center"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows two triangle outlines of different sizes drawn in black of medium thickness, with their corresponding sides parallel. "
        "Five candidate markers A-E are clustered near the hidden center of dilation; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the two triangles and the candidate markers, then draws three black lines of medium thickness, each connecting a pair of corresponding vertices and converging at the correct candidate. "
        "In the final state, only the homothety center marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def __init__(self, *args, opposite=False, **kwargs):
        self.opposite = bool(opposite)
        if self.opposite:
            self.DEFAULT_RULE_PROMPT = "Input Image: Two similar triangles have parallel corresponding sides and five candidate markers A-E. Rule: Find the centre of the negative homothety mapping corresponding vertices onto opposite rays at one constant distance ratio. Required Output: Answer with one option from A-E."
            self.DEFAULT_TI2V_PROMPT = "Input Image: Two similar triangles have parallel corresponding sides and five candidate markers A-E. Rule: Find the centre of the negative homothety mapping corresponding vertices onto opposite rays at one constant distance ratio. Required Output: Draw three complete cyan joins, each from a large vertex through the centre to its corresponding small vertex, in sequence, preserving earlier joins, and highlight only the correct marker in pale red. Static camera."
            self.DEFAULT_SOLUTION_IMAGE_PROMPT = self.strip_video_instruction(self.DEFAULT_TI2V_PROMPT)
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            center = self.pick_target_point(
                0.44,
                padding=candidate_padding + self.canvas_short_side * 0.06,
            )
            scale = self._rng.uniform(0.38, 0.62) * (-1 if self.opposite else 1)
            base_angle = self._rng.uniform(0.0, math.tau)
            angles = [
                base_angle + self._rng.uniform(math.radians(55), math.radians(65)) * i
                for i in range(3)
            ]
            dist_large = self._rng.uniform(
                self.canvas_short_side * 0.18, self.canvas_short_side * 0.28
            )
            large_verts = [
                self.point_on_ray(center, a, dist_large * self._rng.uniform(0.85, 1.15))
                for a in angles
            ]
            small_verts = [
                Point(
                    x=center.x + scale * (v.x - center.x),
                    y=center.y + scale * (v.y - center.y),
                )
                for v in large_verts
            ]

            all_pts = large_verts + small_verts + [center]
            if not all(
                self.inside_canvas(p, padding=self.line_width + 4) for p in all_pts
            ):
                continue

            large_area = self.triangle_area(*large_verts)
            small_area = self.triangle_area(*small_verts)
            if large_area < (self.canvas_short_side**2) * 0.02:
                continue
            if small_area < (self.canvas_short_side**2) * 0.005:
                continue

            min_sep = self.canvas_short_side * 0.06
            too_close = False
            for lv in large_verts:
                for sv in small_verts:
                    if self.distance(lv, sv) < min_sep:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                continue

            if not self.point_can_host_candidate(center):
                continue

            # Place distractors near the homothety center instead of
            # using the default `place_candidates` (which would scatter
            # them across the canvas, leaving the answer the unique
            # candidate at the convergence point of the three connectors).
            try:
                self._place_homothety_candidates(
                    center,
                    large_verts,
                    small_verts,
                )
            except RuntimeError:
                continue

            self.large_triangle = tuple(large_verts)
            self.small_triangle = tuple(small_verts)
            self.target_point = center
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid homothety center puzzle.")

    def _place_homothety_candidates(
        self,
        target: Point,
        large_verts: List[Point],
        small_verts: List[Point],
    ) -> None:
        """Place 5 markers around the homothety center. Distractors must
        not coincide with triangle vertices and must be within ~70 px of
        the answer so the model has to actually trace the converging
        connectors rather than dismiss outliers.
        """
        # Distractors keep clear of both triangles' corners and stay near
        # enough that the converging connectors have to be traced rather than
        # outliers dismissed.  Laying the cloud out before choosing which slot
        # the answer takes keeps it off the middle of its own set (7.3).
        marker_clear = max(self.point_radius * 2.4, 28.0)
        vertex_clear = max(self.point_radius * 2.0, 24.0)
        corners = list(large_verts) + list(small_verts)

        def clear_of_corners(point: Point) -> bool:
            return all(self.distance(point, corner) >= vertex_clear for corner in corners)

        self.seat_candidates(
            target, accept=clear_of_corners, min_separation=marker_clear
        )

    def _authored_observation(self) -> str:
        if self.opposite:
            return "Corresponding vertices lie on opposite rays through one common centre, with the same negative scale factor. Intersect their joining lines to locate that centre."
        large = list(self.large_triangle)
        small = list(self.small_triangle)
        names = [
            name.removesuffix(" visible vertex")
            for name in self._visible_point_names(large)
        ]
        ratios = [
            self.distance(self.target_point, small[index])
            / max(self.distance(self.target_point, large[index]), 1e-9)
            for index in range(3)
        ]
        percent = int(round(sum(ratios) / 3.0 * 20.0)) * 5
        pairings = ", ".join(
            f"the {names[index]} corner with the small corner "
            f"{self._relative_position_phrase(large[index], small[index])} it"
            for index in range(3)
        )
        small_centroid = Point(
            x=sum(point.x for point in small) / 3.0,
            y=sum(point.y for point in small) / 3.0,
        )
        return (
            "Both outlines keep the same orientation and every side of the smaller one runs parallel to a side of "
            f"the larger, so each large corner has one partner in the small triangle: {pairings}. Each small side "
            f"measures about {percent}% of the parallel side it copies, and that one fraction holds for all three "
            "pairs, so the whole picture is a single shrink about one fixed point. On every join the small corner "
            "stands at that same fraction of the way out from the fixed point, which puts the point beyond the "
            "small corner, away from the large one. Running the three joins on in that direction, they meet at one "
            f"point {self._relative_position_phrase(small_centroid, self.target_point)} the small triangle and "
            f"clear of both outlines, {self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "large_triangle": [p.to_list() for p in self.large_triangle],
            "small_triangle": [p.to_list() for p in self.small_triangle],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, list(self.large_triangle) + [self.large_triangle[0]])
        self.draw_line(draw, list(self.small_triangle) + [self.small_triangle[0]])
        if highlight_label:
            for lv, sv in zip(self.large_triangle, self.small_triangle):
                if self.opposite:
                    self.draw_line(draw, [lv, sv])
                    continue
                _, trimmed_end = self.trim_segment(
                    lv, self.target_point,
                    start_offset=0.0,
                    end_offset=float(self.point_radius),
                )
                self.draw_line(draw, [lv, trimmed_end])
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

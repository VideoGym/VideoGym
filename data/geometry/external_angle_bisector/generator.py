"""External angle bisector puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class ExternalAngleBisectorGenerator(PointTargetPuzzleGenerator):
    """Find a point on the external bisector of an angle."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows two black rays meeting at one vertex and five labeled candidate circles "
        "A-E. Rule: First take the internal bisector of the smaller non-reflex angle between the rays. The external "
        "angle-bisector line passes through the same vertex and is perpendicular to that internal bisector; either "
        "opposite ray of this line has the same locus. Select the unique candidate on that line. Required Output: "
        "Return exactly one uppercase option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/external_angle_bisector"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows two black rays of medium thickness meeting at a vertex. "
        "Five candidate markers A-E are clustered near the hidden external angle bisector; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the two rays and candidate markers, draws the smaller angle's internal bisector in cyan, then retains it while drawing the cyan perpendicular external angle-bisector line through the correct candidate. "
        "In the final state, only the bisector candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def __init__(self, *args, opening_range=(50.0, 140.0), **kwargs):
        if not 0 < opening_range[0] <= opening_range[1] < 180:
            raise ValueError("angles must lie strictly between zero and 180 degrees")
        self.opening_range = opening_range
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            vertex = self.pick_target_point(0.44, padding=candidate_padding + self.canvas_short_side * 0.14)
            a1 = self._rng.uniform(0.0, math.tau)
            angle_between = math.radians(self._rng.uniform(*self.opening_range))
            a2 = a1 + angle_between
            # External bisector is perpendicular to internal bisector
            internal_bisector_angle = (a1 + a2) / 2
            external_bisector_angle = internal_bisector_angle + math.pi / 2

            ray_len = self._rng.uniform(self.canvas_short_side * 0.18, self.canvas_short_side * 0.30)
            r1_end = self.point_on_ray(vertex, a1, ray_len)
            r2_end = self.point_on_ray(vertex, a2, ray_len)

            target_dist = self._rng.uniform(self.canvas_short_side * 0.14, self.canvas_short_side * 0.26)
            target = self.point_on_ray(vertex, external_bisector_angle, target_dist)

            if not all(self.inside_canvas(p, padding=self.line_width + 4) for p in [r1_end, r2_end, target]):
                continue
            if not self.point_can_host_candidate(target):
                continue
            if self.distance(target, vertex) < self.canvas_short_side * 0.10:
                continue

            try:
                self.place_candidates(target)
            except RuntimeError:
                continue

            # ``place_candidates`` guarantees spatial separation between
            # marker disks, but a distant distractor can still be almost
            # collinear with the external-bisector line.  Such a marker is a
            # visually ambiguous second solution even though its floating
            # point coordinates are not exactly equal to the constructed
            # target.  Enforce the same angular-residual margin used by the
            # independent coordinate replay before accepting the sample.
            direction_x = math.cos(external_bisector_angle)
            direction_y = math.sin(external_bisector_angle)
            candidate_residuals = []
            for candidate in self.candidates:
                offset_x = candidate.x - vertex.x
                offset_y = candidate.y - vertex.y
                offset_norm = math.hypot(offset_x, offset_y)
                if offset_norm <= 1e-9:
                    candidate_residuals = []
                    break
                candidate_residuals.append(
                    abs(offset_x * direction_y - offset_y * direction_x)
                    / offset_norm
                )
            candidate_residuals.sort()
            if (
                len(candidate_residuals) != len(self.candidates)
                or candidate_residuals[0] > 1e-6
                or candidate_residuals[1] < 0.08
            ):
                continue

            internal_end = self.point_on_ray(vertex, internal_bisector_angle, ray_len * 0.72)
            if not self.inside_canvas(internal_end, padding=self.line_width):
                continue

            self.vertex = vertex
            self.ray1_end = r1_end
            self.ray2_end = r2_end
            self.internal_bisector_end = internal_end
            self.ext_bisector_angle = external_bisector_angle
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate external angle bisector puzzle.")

    def _authored_observation(self) -> str:
        first_heading = self._heading_phrase(
            self.ray1_end.x - self.vertex.x, self.ray1_end.y - self.vertex.y
        )
        second_heading = self._heading_phrase(
            self.ray2_end.x - self.vertex.x, self.ray2_end.y - self.vertex.y
        )
        opening_size = self._angle_size_phrase(
            self.angle_at_vertex(self.ray1_end, self.vertex, self.ray2_end)
        )
        internal_heading = self._heading_phrase(
            self.internal_bisector_end.x - self.vertex.x,
            self.internal_bisector_end.y - self.vertex.y,
        )
        external_heading = self._heading_phrase(
            math.cos(self.ext_bisector_angle), math.sin(self.ext_bisector_angle)
        )
        return (
            f"Two rays share the vertex {self._canvas_region(self.vertex)}: one heads {first_heading}, the other "
            f"{second_heading}, enclosing {opening_size}. The internal bisector of that smaller angle leaves the "
            f"vertex {internal_heading}, and the external bisector is the line through the same vertex "
            f"perpendicular to it, running {external_heading} into the cluster of markers; only a center lying on "
            "that perpendicular line qualifies."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "vertex": self.vertex.to_list(),
            "ray1_end": self.ray1_end.to_list(),
            "ray2_end": self.ray2_end.to_list(),
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, [self.ray1_end, self.vertex, self.ray2_end])
        if highlight_label:
            # The external line is defined from the internal bisector, so the
            # authored visual process establishes and retains that dependency.
            self.draw_line(draw, [self.vertex, self.internal_bisector_end], join_start=True)
            # External bisector: clip to the canvas edge so the line
            # reaches all the way to the boundary (minus a small padding
            # for the line width). This gives a clearly visible bisector
            # that extends well past the answer marker.
            forward_dist = self._max_travel_distance(
                self.vertex,
                self.ext_bisector_angle,
                padding_x=self.line_width + 8,
                padding_y=self.line_width + 8,
            )
            bisector_end = self.point_on_ray(
                self.vertex, self.ext_bisector_angle,
                max(self.distance(self.vertex, self.target_point) * 1.2, forward_dist * 0.98),
            )
            self.draw_line(draw, [self.vertex, bisector_end], join_start=True)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

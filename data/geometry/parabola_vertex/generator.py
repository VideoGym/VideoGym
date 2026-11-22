"""Parabola vertex puzzle generator."""

from __future__ import annotations

import math
from typing import List, Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class ParabolaVertexGenerator(PointTargetPuzzleGenerator):
    """Find the vertex (turning point) of a parabolic arc."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows a black parabolic curve (a smooth U-shaped arc at some rotation), plus five labeled candidate "
        "circles A-E near the curve's turning point. Identify the vertex, the point where the parabola changes direction "
        "(the tip of the U). Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/parabola_vertex"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a black parabolic curve of medium thickness (a smooth U-shaped arc at an arbitrary rotation). "
        "Five candidate markers A-E are clustered near the hidden vertex; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the parabolic curve and the candidate markers, then draws a short cyan line segment of medium thickness marking the axis of symmetry through the correct candidate. "
        "In the final state, only the vertex marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            vertex = self.pick_target_point(
                0.44, padding=candidate_padding + self.canvas_short_side * 0.12
            )
            axis_angle = self._rng.uniform(0.0, math.tau)
            # Curvature is bounded from below so the turning point stays
            # sharply localized: on a flatter arc a marker one runner-up margin
            # up the arm would be visually indistinguishable from the tip.  The
            # arm is then capped so the deeper curve still fits the canvas.
            a_coeff = self._rng.uniform(0.008, 0.016)  # curvature in pixels
            arm_ceiling = min(
                self.canvas_short_side * 0.28,
                math.sqrt(self.canvas_short_side * 0.30 / a_coeff),
            )
            arm_floor = self.canvas_short_side * 0.18
            if arm_ceiling <= arm_floor:
                continue
            arm_len = self._rng.uniform(arm_floor, arm_ceiling)

            # Build parabola polyline: y = a*x^2 in local coords, rotated.
            # Sample ~1 px per segment at the 4x-supersampled internal
            # resolution to keep the rendered curve visually smooth; at
            # ~280 px arm length that means n ~ 240 is safe.
            n = 240
            pts: List[Point] = []
            ok = True
            for i in range(n + 1):
                t = -arm_len + 2 * arm_len * i / n
                local_x = t
                local_y = a_coeff * t * t
                cos_a, sin_a = math.cos(axis_angle), math.sin(axis_angle)
                wx = vertex.x + local_x * cos_a - local_y * sin_a
                wy = vertex.y + local_x * sin_a + local_y * cos_a
                p = Point(wx, wy)
                if not self.inside_canvas(p, padding=self.line_width):
                    ok = False
                    break
                pts.append(p)
            if not ok or len(pts) < n + 1:
                continue

            if not self.point_can_host_candidate(vertex):
                continue
            # Every option rides the drawn curve, so "which marker is on the
            # parabola?" is answer-free; a distractor is wrong
            # because the arc still slopes there, not because it floats beside
            # the ink.  The minimum arc-length offset is the distance at which
            # the curve has already dropped a full marker radius plus air away
            # from the tip along the axis, keeping the near-miss visible (7.5).
            axis_depth = self.point_radius + 6.0
            min_arc = max(
                self.minimum_candidate_spacing(),
                math.sqrt(axis_depth / a_coeff),
            )
            try:
                self.place_candidates_on_incidence(
                    vertex,
                    [Incidence.polyline(pts)],
                    min_offset=min_arc,
                )
            except (RuntimeError, ValueError):
                continue

            self.parabola_pts = pts
            self.axis_angle = axis_angle
            self.target_point = vertex
            return self.save_puzzle()
        raise RuntimeError("Failed to generate parabola vertex puzzle.")

    def _authored_observation(self) -> str:
        open_dx, open_dy = -math.sin(self.axis_angle), math.cos(self.axis_angle)
        return (
            f"The parabolic curve opens {self._heading_phrase(open_dx, open_dy)}, its two arms sweeping out that "
            f"way while the bend tightens in the opposite direction. The vertex is the unique turning point at "
            f"the tip of the U — where the curve is at its tightest and its symmetry axis crosses it — and on "
            f"this curve that tip sits {self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        sample_indices = (0, 30, 60, 90, 120, 150, 180, 210, 240)
        return {
            "axis_angle_deg": math.degrees(self.axis_angle),
            # A bounded sampling of the visible curve lets the verifier fit
            # the turning point from input geometry instead of trusting a
            # saved vertex/correct option.
            "source_curve_points": [self.parabola_pts[index].to_list() for index in sample_indices],
        }

    def _axis_segment(self) -> tuple[Point, Point]:
        """Short symmetry-axis segment through the vertex, into the opening."""

        open_x, open_y = -math.sin(self.axis_angle), math.cos(self.axis_angle)
        vertex = self.target_point
        back = self.canvas_short_side * 0.05
        forward = self.canvas_short_side * 0.16
        return (
            Point(vertex.x - open_x * back, vertex.y - open_y * back),
            Point(vertex.x + open_x * forward, vertex.y + open_y * forward),
        )

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        # draw_curve renders the thick stroke as a filled polygon so
        # the curve edge stays uniformly smooth along the whole arc —
        # no polyline-joint bumps even at dense sampling.
        self.draw_curve(draw, self.parabola_pts)
        if highlight_label:
            # The prompt-declared symmetry-axis mark: one short cyan segment
            # through the vertex along the axis, committed with the answer.
            axis_start, axis_end = self._axis_segment()
            self.draw_line(
                draw,
                [axis_start, axis_end],
                fill=self.CONSTRUCTION_COLOR,
            )
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

"""Angle trisection puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


_REFERENCE_RAY_COLOR = (50, 112, 196)


class AngleTrisectionGenerator(PointTargetPuzzleGenerator):
    """Find the point on an arc that marks one-third of a given angle."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows two black rays meeting at a vertex and five labeled candidate circles A-E; a blue ring marks the endpoint of the reference ray. "
        "Rule: Starting at that marked ray and sweeping through the smaller interior angle toward the other ray, identify the candidate at exactly one third of the sweep. "
        "Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/angle_trisection"
    DEFAULT_TI2V_PROMPT = (
        "Input Image: A square white canvas shows two black rays meeting at a vertex; a blue ring marks the endpoint of the reference ray. "
        "Five candidate markers A-E are clustered near the hidden trisection direction; each is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "Required Output: "
        "The video first holds the two rays and the candidate markers, then draws one cyan construction line from the vertex toward the correct candidate, stopping at the candidate circle boundary rather than entering its interior. "
        "In the final state, only the trisection marker changes to pale red fill with a dark red outline, while the other four markers remain white."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def __init__(self, *args, opening_range=(60.0, 150.0), **kwargs):
        if not 0 < opening_range[0] <= opening_range[1] < 180:
            raise ValueError("angles must lie strictly between zero and 180 degrees")
        self.opening_range = opening_range
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        for _ in range(999):
            vertex = self.pick_target_point(0.44, padding=candidate_padding + self.canvas_short_side * 0.14)
            a1 = self._rng.uniform(0.0, math.tau)
            angle_span = math.radians(self._rng.uniform(*self.opening_range))
            a2 = a1 + angle_span
            trisect_angle = a1 + angle_span / 3

            ray_len = self._rng.uniform(self.canvas_short_side * 0.20, self.canvas_short_side * 0.32)
            arc_r = self._rng.uniform(self.canvas_short_side * 0.12, self.canvas_short_side * 0.20)
            r1_end = self.point_on_ray(vertex, a1, ray_len)
            r2_end = self.point_on_ray(vertex, a2, ray_len)
            target = self.point_on_ray(vertex, trisect_angle, arc_r)

            if not all(self.inside_canvas(p, padding=self.line_width + 4) for p in [r1_end, r2_end, target]):
                continue
            # Check all arc points fit
            arc_ok = True
            for i in range(11):
                t = i / 10
                a = a1 + angle_span * t
                ap = self.point_on_ray(vertex, a, arc_r)
                if not self.inside_canvas(ap, padding=self.line_width):
                    arc_ok = False
                    break
            if not arc_ok:
                continue

            # 2D-scatter candidates near the trisection point so A-E
            # don't form a perfect arc-shortcut column (the visual
            # shortcut from the previous arc-only placement).
            if not self.point_can_host_candidate(target):
                continue
            try:
                self.place_candidates(target)
            except RuntimeError:
                continue
            strict_gap = max(self.point_radius * 3.0 + 6.0,
                             self.minimum_candidate_spacing() * 1.20)
            cand_pts = [Point(c.x, c.y) for c in self.candidates]
            if not self.points_are_well_spaced(cand_pts, min_distance=strict_gap):
                continue
            angular_errors = sorted(
                abs(
                    (
                        math.atan2(point.y - vertex.y, point.x - vertex.x)
                        - trisect_angle
                        + math.pi
                    )
                    % math.tau
                    - math.pi
                )
                for point in cand_pts
            )
            # The source replay requires a clear angular winner.  Euclidean
            # candidate spacing alone allowed a radially displaced distractor
            # to sit on almost the same ray as the true trisector.
            if len(angular_errors) < 2 or angular_errors[0] > 1e-6 or angular_errors[1] < 0.06:
                continue

            # Bound hidden stretches for every candidate's prospective ray,
            # so observability cannot identify the correct marker by itself.
            # Brief crossings are fine; a long invisible stroke is not.
            radius = self.point_radius + self.line_width / 2 + 1
            occluded = False
            for endpoint in self.candidates:
                length = math.hypot(endpoint.x - vertex.x, endpoint.y - vertex.y)
                stroke_length = length - self.point_radius
                if stroke_length <= 0:
                    occluded = True
                    break
                ux, uy = (endpoint.x - vertex.x) / length, (endpoint.y - vertex.y) / length
                intervals = []
                for other in self.candidates:
                    if other.label == endpoint.label:
                        continue
                    along = (other.x - vertex.x) * ux + (other.y - vertex.y) * uy
                    across = abs((other.x - vertex.x) * uy - (other.y - vertex.y) * ux)
                    if across >= radius:
                        continue
                    half = math.sqrt(radius * radius - across * across)
                    left, right = max(0.0, along - half), min(stroke_length, along + half)
                    if right > left:
                        intervals.append((left, right))
                merged = []
                for left, right in sorted(intervals):
                    if merged and left <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
                    else:
                        merged.append((left, right))
                if any(right - left > 0.20 * stroke_length for left, right in merged):
                    occluded = True
                    break
            if occluded:
                continue

            self.vertex = vertex
            self.ray1_end = r1_end
            self.ray2_end = r2_end
            self.arc_r = arc_r
            self.a1 = a1
            self.a2 = a2
            self.trisect_angle = trisect_angle
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate angle trisection puzzle.")

    def _authored_observation(self) -> str:
        reference_heading = self._heading_phrase(
            self.ray1_end.x - self.vertex.x, self.ray1_end.y - self.vertex.y
        )
        other_heading = self._heading_phrase(
            self.ray2_end.x - self.vertex.x, self.ray2_end.y - self.vertex.y
        )
        span_size = self._angle_size_phrase(math.degrees(self.a2 - self.a1))
        trisector_heading = self._heading_phrase(
            math.cos(self.trisect_angle), math.sin(self.trisect_angle)
        )
        return (
            f"From the vertex {self._canvas_region(self.vertex)}, the blue-ringed reference ray heads "
            f"{reference_heading} and the other ray heads {other_heading}, with the smaller opening between them "
            f"spanning {span_size}. Sweeping from the marked ray one third of the way across that opening fixes a "
            f"direction leaving the vertex {trisector_heading}, so I follow that direction out of the vertex and "
            "keep the marker centered on it; the scattered neighbors all sit off that ray."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "vertex": self.vertex.to_list(),
            "reference_ray_endpoint": self.ray1_end.to_list(),
            "other_ray_endpoint": self.ray2_end.to_list(),
            "angle_span_deg": math.degrees(self.a2 - self.a1),
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, [self.ray1_end, self.vertex, self.ray2_end])
        marker_radius = max(5, self.point_radius // 2)
        cx, cy = round(self.ray1_end.x), round(self.ray1_end.y)
        draw.ellipse(
            (cx - marker_radius, cy - marker_radius, cx + marker_radius, cy + marker_radius),
            fill=(255, 255, 255),
            outline=_REFERENCE_RAY_COLOR,
            width=max(3, self.line_width // 2),
        )
        if highlight_label:
            candidate = next(
                item for item in self.candidates if item.label == highlight_label
            )
            trisect_end = Point(float(candidate.x), float(candidate.y))
            _start, trisect_end = self.trim_segment(
                self.vertex,
                trisect_end,
                end_offset=float(self.point_radius),
            )
            # Trisector shares the apex vertex with the two-ray polyline —
            # inset the start so the butt cap is hidden inside the joint.
            self.draw_line(
                draw,
                [self.vertex, trisect_end],
                fill=self.CONSTRUCTION_COLOR,
                join_start=True,
            )
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

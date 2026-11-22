"""Symmetric completion puzzle generator."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class SymmetricCompletionGenerator(PointTargetPuzzleGenerator):
    """Given a mirror line and an incomplete polygon, find the missing vertex to complete symmetry."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows one thin reflection axis, five vertices and four edges of an incomplete "
        "convex hexagon, and five labeled candidate circles A-E. Rule: Complete a hexagon that is mirror-symmetric "
        "about the shown axis. Reflect the unmatched visible vertex perpendicularly across the axis and select the "
        "candidate at that reflected location; all existing vertices and edges stay fixed. Required Output: Return "
        "exactly one uppercase option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/symmetric_completion"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a thin solid gray reflection axis and an incomplete convex hexagon outline of medium thickness missing one vertex on the reflected side. "
        "The five visible vertices lie symmetrically about the axis, so the intended hexagon is mirror-symmetric. "
        "Five candidate markers A-E are clustered near the hidden missing vertex; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the axis, the incomplete hexagon, and the candidate markers, then draws the two missing hexagon edges in cyan from the adjacent visible vertices to the correct candidate. "
        "In the final state, only the missing-vertex candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    @staticmethod
    def _reflect_point(pt: Point, axis_p1: Point, axis_p2: Point) -> Point:
        dx = axis_p2.x - axis_p1.x
        dy = axis_p2.y - axis_p1.y
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return pt
        t = ((pt.x - axis_p1.x) * dx + (pt.y - axis_p1.y) * dy) / denom
        proj = Point(axis_p1.x + t * dx, axis_p1.y + t * dy)
        return Point(2.0 * proj.x - pt.x, 2.0 * proj.y - pt.y)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        short = self.canvas_short_side
        for _ in range(999):
            # Axis of symmetry (random orientation, centered-ish).
            axis_angle = self._rng.uniform(0.0, math.pi)
            axis_mid = self.pick_target_point(0.18, padding=candidate_padding + short * 0.18)
            try:
                axis_p1, axis_p2 = self.sample_symmetric_segment(
                    axis_mid, axis_angle,
                    min_half_length=short * 0.26,
                    max_half_length=short * 0.34,
                    padding=self.line_width,
                )
            except RuntimeError:
                continue

            # Build a symmetric, strictly convex hexagon by choosing three
            # monotonically ordered "along-axis" positions t1 < t2 < t3 and
            # three strictly positive "off-axis" distances h1, h2, h3. The
            # middle distance must exceed the average of h1 and h3 so the
            # upper polyline peaks outward (guaranteeing convexity of the
            # full hexagon once mirrored across the axis).
            span = short * self._rng.uniform(0.22, 0.30)
            t1 = -span * self._rng.uniform(0.9, 1.05)
            t3 = span * self._rng.uniform(0.9, 1.05)
            t2 = self._rng.uniform(t1 * 0.35 + t3 * 0.15, t1 * 0.15 + t3 * 0.35)
            # Keep t2 comfortably inside (t1, t3)
            t2 = max(t1 + span * 0.15, min(t3 - span * 0.15, t2))
            h1 = short * self._rng.uniform(0.14, 0.22)
            h3 = short * self._rng.uniform(0.14, 0.22)
            h2 = short * self._rng.uniform(0.24, 0.30)
            # Enforce strict convex peak: h2 strictly > (h1 + h3) / 2.
            if h2 <= (h1 + h3) / 2 + short * 0.02:
                continue

            cos_a = math.cos(axis_angle)
            sin_a = math.sin(axis_angle)
            nx = -sin_a
            ny = cos_a

            def _build(t: float, h: float) -> Point:
                return Point(
                    axis_mid.x + t * cos_a + h * nx,
                    axis_mid.y + t * sin_a + h * ny,
                )

            upper = [_build(t1, h1), _build(t2, h2), _build(t3, h3)]
            lower = [_build(t3, -h3), _build(t2, -h2), _build(t1, -h1)]
            full_polygon = upper + lower  # 6 vertices in CCW order

            if not all(self.inside_canvas(v, padding=self.line_width + 6) for v in full_polygon):
                continue

            # Remove one vertex from the LOWER (reflected) side. Removing an
            # interior lower vertex (index 4) keeps the visible polyline in
            # one piece; removing a corner lower vertex (indices 3 or 5)
            # yields a still-natural "opening" on the reflected side.
            missing_full_idx = self._rng.choice([3, 4, 5])
            target = full_polygon[missing_full_idx]

            if not self.point_can_host_candidate(target):
                continue
            # Keep the missing vertex visibly off the axis so the mirror
            # relationship to its source-side counterpart is easy to see.
            if self.distance_point_to_line(target, axis_p1, axis_p2) < short * 0.08:
                continue

            try:
                self.place_candidates(target)
            except RuntimeError:
                continue

            visible_edges: List[Tuple[Point, Point]] = []
            n = len(full_polygon)
            for start_idx in range(n):
                end_idx = (start_idx + 1) % n
                if start_idx == missing_full_idx or end_idx == missing_full_idx:
                    continue
                visible_edges.append((full_polygon[start_idx], full_polygon[end_idx]))
            prev_idx = (missing_full_idx - 1) % n
            next_idx = (missing_full_idx + 1) % n
            self.visible_edges = visible_edges
            self.completion_neighbors = (full_polygon[prev_idx], full_polygon[next_idx])
            self.axis_points = (axis_p1, axis_p2)
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate symmetric completion puzzle.")

    def _authored_observation(self) -> str:
        axis_p1, axis_p2 = self.axis_points
        axis_direction = self._heading_phrase(axis_p2.x - axis_p1.x, axis_p2.y - axis_p1.y)
        unmatched_corner = self._reflect_point(self.target_point, axis_p1, axis_p2)
        return (
            f"The thin gray axis runs {axis_direction}, and five corners of the intended hexagon are already "
            f"joined by its visible edges. Every corner must pair with a mirror partner across that axis, and the "
            f"one visible corner still without a partner sits {self._canvas_region(unmatched_corner)}; reflecting "
            f"it perpendicularly across the axis, equally far on the other side, places the missing corner "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "axis": [p.to_list() for p in self.axis_points],
            "visible_edges": [[start.to_list(), end.to_list()] for start, end in self.visible_edges],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        # Axis of symmetry: thin solid gray line, drawn first so polygon sits
        # on top. No dashing — the axis is self-explanatory as a thin ruler.
        self.draw_line(draw, list(self.axis_points), width_factor=0.5, fill=(140, 140, 140))
        # Group consecutive visible edges into contiguous polylines so that
        # internal joints use PIL's `joint="curve"` (no butt-cap gaps, no
        # extra round "bumps" at shared polygon vertices).
        for polyline in self._contiguous_visible_polylines():
            self.draw_line(draw, polyline)
        if highlight_label:
            n1, n2 = self.completion_neighbors
            # Single continuous stroke: shape-endpoint → answer-marker →
            # other shape-endpoint. join_start / join_end inset the two
            # outer endpoints by line_width/2 so their butt caps hide
            # inside the adjacent visible-edge ink that already ends at
            # n1 and n2 (no bumps beyond the natural joint).
            self.draw_line(
                draw, [n1, self.target_point, n2],
                join_start=True, join_end=True,
            )
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

    def _contiguous_visible_polylines(self) -> List[List[Point]]:
        """Merge self.visible_edges into maximal contiguous polylines.

        Each adjacent edge sharing a vertex is folded into the same
        polyline so that interior polygon vertices are drawn as polyline
        joints (clean curve joint) rather than independent butt-capped
        line endpoints (visible bumps + gaps).
        """
        edges = list(self.visible_edges)
        if not edges:
            return []
        polylines: List[List[Point]] = []
        used = [False] * len(edges)
        for i, (s, e) in enumerate(edges):
            if used[i]:
                continue
            used[i] = True
            chain = [s, e]
            # Extend forward.
            while True:
                for j, (sj, ej) in enumerate(edges):
                    if used[j]:
                        continue
                    if chain[-1] is sj or (chain[-1].x == sj.x and chain[-1].y == sj.y):
                        chain.append(ej)
                        used[j] = True
                        break
                    if chain[-1] is ej or (chain[-1].x == ej.x and chain[-1].y == ej.y):
                        chain.append(sj)
                        used[j] = True
                        break
                else:
                    break
            # Extend backward.
            while True:
                for j, (sj, ej) in enumerate(edges):
                    if used[j]:
                        continue
                    if chain[0] is ej or (chain[0].x == ej.x and chain[0].y == ej.y):
                        chain.insert(0, sj)
                        used[j] = True
                        break
                    if chain[0] is sj or (chain[0].x == sj.x and chain[0].y == sj.y):
                        chain.insert(0, ej)
                        used[j] = True
                        break
                else:
                    break
            polylines.append(chain)
        return polylines

"""Nine-point center puzzle generator."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


def _foot_of_altitude(vertex: Point, opp1: Point, opp2: Point) -> Point:
    """Foot of altitude from vertex to line opp1-opp2."""
    dx, dy = opp2.x - opp1.x, opp2.y - opp1.y
    denom = dx * dx + dy * dy
    if denom < 1e-9:
        return opp1
    t = ((vertex.x - opp1.x) * dx + (vertex.y - opp1.y) * dy) / denom
    return Point(opp1.x + t * dx, opp1.y + t * dy)


def _nine_point_center(p1: Point, p2: Point, p3: Point) -> Tuple[Point, float]:
    """Compute nine-point center and radius.

    The nine-point center is the midpoint of the segment from orthocenter to circumcenter.
    The nine-point circle passes through: midpoints of sides, feet of altitudes,
    and midpoints of vertex-to-orthocenter segments.
    """
    # Circumcenter
    ax, ay = p1.x, p1.y
    bx, by = p2.x, p2.y
    cx, cy = p3.x, p3.y
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-9:
        return Point(0, 0), 0
    sq_a = ax * ax + ay * ay
    sq_b = bx * bx + by * by
    sq_c = cx * cx + cy * cy
    Ux = (sq_a * (by - cy) + sq_b * (cy - ay) + sq_c * (ay - by)) / D
    Uy = (sq_a * (cx - bx) + sq_b * (ax - cx) + sq_c * (bx - ax)) / D
    circumradius = math.hypot(ax - Ux, ay - Uy)

    # Orthocenter: O = A + B + C - 2*circumcenter
    ortho = Point(ax + bx + cx - 2 * Ux, ay + by + cy - 2 * Uy)

    # Nine-point center = midpoint of orthocenter and circumcenter
    npc = Point((ortho.x + Ux) / 2, (ortho.y + Uy) / 2)
    npr = circumradius / 2
    return npc, npr


class NinePointCenterGenerator(PointTargetPuzzleGenerator):
    """Find the nine-point center of a triangle with full construction."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A black triangle outline is shown on a white canvas with five labeled candidate circles A-E inside it. "
        "Rule: Construct the three side midpoints and the three altitude feet, where each altitude foot is the perpendicular "
        "projection of a vertex onto the opposite side. These six points lie on the triangle's nine-point circle. Select "
        "the candidate at that circle's center. Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/nine_point_center"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a triangle drawn as a black outline of medium thickness. "
        "Five candidate markers A-E are clustered near the hidden nine-point center; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the triangle and the candidate markers, then draws the three altitudes as black segments of medium thickness from each vertex straight to the foot on the opposite side (each altitude meets its foot exactly on the triangle edge, with no overshoot), then marks the six construction points — three side midpoints and three altitude feet — with small white hollow circles, and finally draws the nine-point circle of medium thickness centered on the correct candidate as a single smooth stroke that passes BENEATH the six already-marked hollow circles so that the markers remain fully visible throughout the drawing. "
        "In the final state, only the nine-point center marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        marker_radius = 7  # same radius used when drawing hollow markers
        min_marker_sep = marker_radius * 2.2  # buffer so no two markers touch
        for _ in range(999):
            # Push the marker-separation check INTO the sampler's validator so
            # the sampler itself keeps retrying until it yields a triangle
            # whose six construction markers (side midpoints + altitude feet)
            # are all pairwise well-spaced. Doing it outside the sampler wastes
            # the sampler's own 999-attempt budget on near-equilateral triangles
            # that have clustered feet/midpoints.
            def _construction_points_spread(a: Point, b: Point, c: Point) -> bool:
                verts_local = [a, b, c]
                mids = [
                    Point((verts_local[(i + 1) % 3].x + verts_local[(i + 2) % 3].x) / 2,
                          (verts_local[(i + 1) % 3].y + verts_local[(i + 2) % 3].y) / 2)
                    for i in range(3)
                ]
                feet = [
                    _foot_of_altitude(verts_local[i], verts_local[(i + 1) % 3], verts_local[(i + 2) % 3])
                    for i in range(3)
                ]
                pts = mids + feet
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        if self.distance(pts[i], pts[j]) < min_marker_sep:
                            return False
                return True

            # The joint validator (six construction points pairwise separated)
            # rejects most raw triangles, so the default 999-attempt budget can
            # exhaust on real seeds (e.g. 888) and abort family-level runs.
            p1, p2, p3 = self.sample_triangle_vertices(
                jitter_ratio=0.56, min_side_ratio=0.22, min_area_ratio=0.05,
                min_altitude_ratio=0.17, min_angle_deg=42.0, max_angle_deg=100.0,
                validator=_construction_points_spread,
                max_attempts=20000,
            )
            npc, npr = _nine_point_center(p1, p2, p3)
            if npr < self.canvas_short_side * 0.08:
                continue
            if not self.circle_fits(npc, npr, extra_padding=self.line_width):
                continue
            if not self.point_can_host_candidate(npc):
                continue

            # Precompute construction points
            verts = [p1, p2, p3]
            midpoints = [
                Point((verts[(i + 1) % 3].x + verts[(i + 2) % 3].x) / 2,
                      (verts[(i + 1) % 3].y + verts[(i + 2) % 3].y) / 2)
                for i in range(3)
            ]
            altitude_feet = [
                _foot_of_altitude(verts[i], verts[(i + 1) % 3], verts[(i + 2) % 3])
                for i in range(3)
            ]

            all_construction = midpoints + altitude_feet
            if not all(self.inside_canvas(p, padding=self.line_width) for p in all_construction):
                continue

            try:
                self.place_candidates(npc)
            except RuntimeError:
                continue

            self.triangle_points = (p1, p2, p3)
            self.midpoints = midpoints
            self.altitude_feet = altitude_feet
            self.np_radius = npr
            self.marker_radius = marker_radius
            self.target_point = npc
            return self.save_puzzle()
        raise RuntimeError("Failed to generate nine-point center puzzle.")

    def _authored_observation(self) -> str:
        vertices = list(self.triangle_points)
        names = self._visible_point_names(vertices)
        return (
            f"The triangle joins the {names[0]}, the {names[1]}, and the {names[2]}. Marking the midpoint of "
            "each side and dropping each altitude onto its perpendicular foot yields six construction points, "
            "and a single circle passes through all six of them; its center is the one point equally distant "
            f"from those six, which pins it {self._canvas_region(self.target_point)}, inside the marker "
            "cluster."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "triangle_vertices": [p.to_list() for p in self.triangle_points],
            "side_midpoints": [p.to_list() for p in self.midpoints],
            "altitude_feet": [p.to_list() for p in self.altitude_feet],
            "nine_point_radius": self.np_radius,
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        p1, p2, p3 = self.triangle_points
        self.draw_line(draw, [p1, p2, p3, p1])
        if highlight_label:
            verts = [p1, p2, p3]
            # Step 1: Draw the three altitudes, vertex → foot. Drop the
            # join_end disk at the foot — the foot is ON the triangle
            # edge, and a disk of radius line_width/2 there extends
            # perpendicular to the edge, half outside the triangle.
            # The altitude's butt cap
            # at the foot is parallel to the edge (altitudes are
            # perpendicular to edges), so it already meets the edge's
            # stroke band cleanly with no gap and no overshoot.
            for i in range(3):
                v = verts[i]
                foot = self.altitude_feet[i]
                self.draw_line(
                    draw, [v, foot],
                    join_start=True,
                )
            # Step 2: Mark the six construction points (3 midpoints +
            # 3 altitude feet). These are painted BEFORE the circle so
            # they appear first in the video's animation sequence.
            marker_positions: List[Tuple[float, float, float]] = []
            for m in self.midpoints:
                self.draw_anchor_marker(draw, m, self.marker_radius)
                marker_positions.append((m.x, m.y, float(self.marker_radius)))
            for foot in self.altitude_feet:
                self.draw_anchor_marker(draw, foot, self.marker_radius)
                marker_positions.append((foot.x, foot.y, float(self.marker_radius)))
            # Step 3: Nine-point circle with keep_above pointing at all
            # six marker regions. The VideoRenderer respects this and
            # composites the circle BENEATH the marker regions in every
            # animation frame, so the circle visually passes under the
            # hollow markers instead of painting over them. No post-hoc
            # "redraw markers on top" — the layer ordering is structural.
            # (In the static-PNG render path the circle is drawn
            # normally; the static scene's final layers are set by
            # drawing markers again after the circle — but the default
            # non-recording draw_circle ignores keep_above, so we
            # follow that up with an explicit final marker pass below.)
            self.draw_circle(
                draw, self.target_point, self.np_radius,
                keep_above=marker_positions,
            )
            # Static PNG: for the non-recording path, the keep_above
            # mechanism is a no-op on the actual pixel buffer, so we
            # re-draw the six markers once more here to restore them on
            # top of the circle. During video tracing (recording) this
            # second pass is absorbed into the trace as redundant
            # draw_anchor_marker commands executed instantly at the end
            # — the visual outcome is identical to the layer-composited
            # video frames, so no flicker.
            if not self._recording_active:
                for m in self.midpoints:
                    self.draw_anchor_marker(draw, m, self.marker_radius)
                for foot in self.altitude_feet:
                    self.draw_anchor_marker(draw, foot, self.marker_radius)
        # Step 4: Candidate markers last so the highlighted answer marker
        # sits on top of EVERY construction element.
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

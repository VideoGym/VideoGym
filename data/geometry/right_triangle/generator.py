"""Right angle triangle puzzle generator.
"""

from __future__ import annotations
from typing import Any, Optional
from PIL import Image
import math
import itertools
from data.geometry.core.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point, PointCandidate

class RightTriangleGenerator(PointTargetPuzzleGenerator):
    """
    Generate puzzles where 5 points are shown, but only 3 form a right-angled triangle.
    The goal is to identify the vertex with the right angle.
    """

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows five labeled candidate circles A-E and no connecting lines. Rule: Exactly "
        "one three-point subset forms a right triangle. Select the labeled point at that subset's 90-degree vertex; "
        "all other three-point subsets are non-right. Required Output: Return exactly one uppercase option label in "
        "the form \"Answer: X.\""
    )
    DEFAULT_OUTPUT_DIR = "outputs/geometry/right_triangle"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas initially shows only five candidate markers A-E and no connecting geometry at all. "
        "Each candidate marker is a small white circle with a thin dark gray outline and a black uppercase letter, and exactly three of these five points form one right triangle. "
        "The video first holds the five markers alone, then draws a cyan triangle outline of medium thickness connecting the three special points in the order that closes one triangle and adds a small cyan square corner mark inside its right-angle vertex. "
        "In the final state, only the marker at the 90 degree corner changes to pale red fill with a dark red outline, while the other four markers stay white. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def _cos_angle(self, p_a: Point, p_b: Point, p_c: Point) -> float:
        """Calculates the cosine of the angle at vertex p_b for triangle p_a, p_b, p_c."""
        v_ba = Point(x=p_a.x - p_b.x, y=p_a.y - p_b.y)
        v_bc = Point(x=p_c.x - p_b.x, y=p_c.y - p_b.y)

        dot_product = v_ba.x * v_bc.x + v_ba.y * v_bc.y
        mag_ba = math.sqrt(v_ba.x**2 + v_ba.y**2)
        mag_bc = math.sqrt(v_bc.x**2 + v_bc.y**2)

        if mag_ba == 0 or mag_bc == 0:
            return 1.0

        return dot_product / (mag_ba * mag_bc)

    def _draw_right_angle_marker(self, draw: Any, vertex: Point) -> None:
        size = self.canvas_short_side * 0.045
        # right_triangle_points is (right-angle vertex, arm endpoint, arm
        # endpoint).  The marker arms must run along the two triangle sides
        # that meet at the vertex, so aim them at the two ARM endpoints —
        # never at the vertex itself (atan2(0, 0) degenerates to a fixed
        # screen-right arm regardless of the triangle's orientation).
        _vertex_entry, arm_one, arm_two = self.right_triangle_points
        angle_a = math.atan2(arm_one.y - vertex.y, arm_one.x - vertex.x)
        angle_b = math.atan2(arm_two.y - vertex.y, arm_two.x - vertex.x)
        ux1, uy1 = math.cos(angle_a), math.sin(angle_a)
        ux2, uy2 = math.cos(angle_b), math.sin(angle_b)
        p1 = Point(vertex.x + ux1 * size, vertex.y + uy1 * size)
        corner = Point(
            vertex.x + (ux1 + ux2) * size,
            vertex.y + (uy1 + uy2) * size,
        )
        p2 = Point(vertex.x + ux2 * size, vertex.y + uy2 * size)
        self.draw_line(draw, [p1, corner, p2], width_factor=0.7)

    def _authored_observation(self) -> str:
        right_vertex, arm_one, arm_two = self.right_triangle_points
        peers = [Point(float(candidate.x), float(candidate.y)) for candidate in self.candidates]
        names = [
            name.removesuffix(" visible vertex")
            for name in self._visible_point_names(peers)
        ]

        def marker_name(point: Point) -> str:
            return names[min(range(len(peers)), key=lambda index: self.distance(peers[index], point))]

        arm_one_heading = self._heading_phrase(arm_one.x - right_vertex.x, arm_one.y - right_vertex.y)
        arm_two_heading = self._heading_phrase(arm_two.x - right_vertex.x, arm_two.y - right_vertex.y)
        return (
            "No lines are given, only loose markers, so the right angle must be read from the directions "
            f"between them. From the {marker_name(right_vertex)} marker, {self._canvas_region(right_vertex)}, "
            f"one arm runs {arm_one_heading}, reaching the {marker_name(arm_one)} marker, while another runs "
            f"{arm_two_heading}, reaching the {marker_name(arm_two)} marker, and those two arms are "
            "perpendicular; no other marker sees two companions at a square corner, so the right angle sits "
            "at that shared origin."
        )

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        min_side_len = self.canvas_short_side * 0.24
        max_side_len = self.canvas_short_side * 0.34
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        min_dist_between_points = max(self.canvas_short_side * 0.16, self.minimum_candidate_spacing(scale=1.05))
        ambiguity_threshold = 0.28

        tries = 0
        while tries < 999999:
            # 1. Generate the right-angled triangle
            p_right = self.pick_target_point(0.5, padding=candidate_padding + self.canvas_short_side * 0.12)

            angle = self._rng.uniform(0, 2 * math.pi)
            len1 = self._rng.uniform(min_side_len, max_side_len)
            len2 = max(min_side_len, min(max_side_len, len1 * self._rng.uniform(0.8, 1.25)))

            p1 = Point(
                x=p_right.x + len1 * math.cos(angle),
                y=p_right.y + len1 * math.sin(angle),
            )
            p2 = Point(
                x=p_right.x + len2 * math.cos(angle + math.pi / 2),
                y=p_right.y + len2 * math.sin(angle + math.pi / 2),
            )

            right_triangle_pts = [p_right, p1, p2]

            if not all(self.point_can_host_candidate(p) for p in right_triangle_pts):
                tries += 1
                continue
            if self.distance(p1, p2) < self.canvas_short_side * 0.34:
                tries += 1
                continue

            # 2. Generate 2 distractor points NEAR the right-triangle
            # bounding box. The default `pick_target_point(0.5)` would
            # plant distractors anywhere on the canvas, so the answer was
            # often the visually obvious "vertex closest to other markers"
            # while distractors floated in dead space — a shortcut leak.
            # Constrain distractors to a disc around the triangle centroid
            # so all 5 markers visually belong to the same figure region.
            tri_cx = (p_right.x + p1.x + p2.x) / 3.0
            tri_cy = (p_right.y + p1.y + p2.y) / 3.0
            # Radius spans roughly from the centroid out to one of the
            # triangle vertices, with a small inflation so distractors can
            # also land just outside the triangle.
            tri_radius = max(
                self.distance(Point(tri_cx, tri_cy), p_right),
                self.distance(Point(tri_cx, tri_cy), p1),
                self.distance(Point(tri_cx, tri_cy), p2),
            )
            distractor_radius = tri_radius * 1.15
            all_points = list(right_triangle_pts)
            distractors_ok = True
            for _ in range(2):
                for _d_try in range(200):
                    angle = self._rng.uniform(0.0, 2 * math.pi)
                    rho = self._rng.uniform(0.15, 1.0) * distractor_radius
                    cx = tri_cx + math.cos(angle) * rho
                    cy = tri_cy + math.sin(angle) * rho
                    d = Point(cx, cy)
                    cand_check = PointCandidate(x=cx, y=cy, label="A")
                    if not self._candidate_fits(cand_check):
                        continue
                    if all(self.distance(d, p) > min_dist_between_points for p in all_points):
                        all_points.append(d)
                        break
                else:
                    distractors_ok = False
                    break

            if not distractors_ok:
                tries += 1
                continue

            # 3. Check for uniqueness of the right angle
            is_ambiguous = False

            for combo in itertools.combinations(all_points, 3):
                p_a, p_b, p_c = combo

                if (p_a in right_triangle_pts and p_b in right_triangle_pts and p_c in right_triangle_pts):
                    continue

                cosines = [
                    abs(self._cos_angle(p_b, p_a, p_c)),
                    abs(self._cos_angle(p_a, p_b, p_c)),
                    abs(self._cos_angle(p_b, p_c, p_a)),
                ]

                if any(c < ambiguity_threshold for c in cosines):
                    is_ambiguous = True
                    break

            if is_ambiguous:
                tries += 1
                continue
            if not self.points_are_well_spaced(all_points, min_distance=min_dist_between_points):
                tries += 1
                continue

            # 4. Recenter the five-point cloud on the canvas.  The triangle is
            # sampled from a random anchor, so without this step the whole
            # figure regularly hugs one corner and leaves a large dead zone.
            width, height = self.canvas_dimensions
            min_x = min(p.x for p in all_points)
            max_x = max(p.x for p in all_points)
            min_y = min(p.y for p in all_points)
            max_y = max(p.y for p in all_points)
            desired_dx = (width / 2.0) - (min_x + max_x) / 2.0
            desired_dy = (height / 2.0) - (min_y + max_y) / 2.0
            # Clamp the shift so every point keeps the candidate padding.
            pad = candidate_padding
            dx = max(pad - min_x, min(desired_dx, (width - pad) - max_x))
            dy = max(pad - min_y, min(desired_dy, (height - pad) - max_y))
            all_points = [Point(p.x + dx, p.y + dy) for p in all_points]
            shifted_triangle = [Point(p.x + dx, p.y + dy) for p in right_triangle_pts]
            p_right, p1, p2 = shifted_triangle
            self.target_point = p_right
            self.right_triangle_points = (p_right, p1, p2)

            # 5. Manually create candidates and set the correct label
            self._rng.shuffle(all_points)
            self.candidates = []
            labels = "ABCDE"
            for i, p in enumerate(all_points):
                label = labels[i]
                self.candidates.append(PointCandidate(x=p.x, y=p.y, label=label))
                if abs(p.x - self.target_point.x) < 1e-9 and abs(p.y - self.target_point.y) < 1e-9:
                    self.correct_label = label
            if not self.validate_candidate_layout(self.candidates, min_distance=min_dist_between_points * 0.95):
                tries += 1
                continue

            record = self.save_puzzle()
            record.right_triangle_points = self.right_triangle_points
            return record

        raise RuntimeError("Failed to generate a valid puzzle after many tries.")

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )

        if highlight_label:
            p_right, p1, p2 = self.right_triangle_points
            self.draw_line(draw, [p1, p_right, p2, p1])
            self._draw_right_angle_marker(draw, p_right)

        return base

"""Kite vertex puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image, ImageFont

from data.geometry.core.point_target_base import (
    Point,
    PointCandidate,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class KiteVertexGenerator(PointTargetPuzzleGenerator):
    """Complete the 4th vertex of a kite (two pairs of adjacent equal sides)."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows three of the four kite vertices connected by two line segments: one apex on a thin gray "
        "dashed symmetry axis, one wing vertex on one side of that axis, and the opposite-axis apex. Five labeled "
        "candidate circles A-E sit near the hidden fourth vertex. Identify the missing mirror-wing vertex, which is the "
        "reflection of the visible wing across the symmetry axis. Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/kite_vertex"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows three connected black line segments of medium thickness forming an open path through three of the four kite vertices: an apex on the symmetry axis, a wing vertex on one side, and the opposite-axis apex. A thin gray dashed line between the two apexes marks the kite's symmetry axis. "
        "Five candidate markers A-E are clustered near the hidden fourth (mirror-wing) vertex; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the open path and the candidate markers, then draws two black closing segments of medium thickness from the apex endpoints to the correct candidate, completing the kite. "
        "In the final state, only the correct vertex marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            # A kite has a symmetry axis through two opposite vertices.
            # Pick axis direction + two arm lengths. The kite's four
            # vertices: top + bottom on the axis, left + right
            # symmetric across the axis at axis_mid.
            axis_mid = self.pick_target_point(0.36, padding=candidate_padding + self.canvas_short_side * 0.16)
            axis_angle = self._rng.uniform(0.0, math.tau)
            d1 = self._rng.uniform(self.canvas_short_side * 0.10, self.canvas_short_side * 0.18)
            d2 = self._rng.uniform(self.canvas_short_side * 0.16, self.canvas_short_side * 0.26)
            # Make d1 and d2 deliberately different so the upper and
            # lower equal-side pairs have visibly different lengths
            # (otherwise the kite collapses toward a rhombus).
            if abs(d2 - d1) < self.canvas_short_side * 0.06:
                continue
            top = self.point_on_ray(axis_mid, axis_angle, d1)
            bottom = self.point_on_ray(axis_mid, axis_angle + math.pi, d2)
            # Wing vertices perpendicular to the axis.
            perp = axis_angle + math.pi / 2
            wing = self._rng.uniform(self.canvas_short_side * 0.09, self.canvas_short_side * 0.16)
            left = self.point_on_ray(axis_mid, perp, wing)
            right = self.point_on_ray(axis_mid, perp + math.pi, wing)

            # Show top -> visible_wing -> bottom (3 vertices, 2 sides).
            # The 2 visible sides are: top-visible_wing (an upper-pair
            # side) and visible_wing-bottom (a lower-pair side). Target =
            # the OTHER (mirror) wing.
            visible_wing = self._rng.choice([left, right])
            target = right if visible_wing is left else left
            shown = [top, visible_wing, bottom]

            all_pts = shown + [target]
            if not all(self.inside_canvas(p, padding=self.line_width + 6) for p in all_pts):
                continue
            if not self.point_can_host_candidate(target):
                continue
            # Check area
            area = abs(
                (right.x - left.x) * (top.y - bottom.y) - (top.x - bottom.x) * (right.y - left.y)
            ) / 2
            if area < (self.canvas_short_side ** 2) * 0.03:
                continue
            # The candidate cluster must not collide with the visible
            # axis vertices (top, bottom, visible_wing) — keep a min
            # separation so the puzzle reads as "find the mirror wing"
            # not "click an existing path vertex".
            min_sep = self.canvas_short_side * 0.10
            if min(
                self.distance(target, top),
                self.distance(target, bottom),
                self.distance(target, visible_wing),
            ) < min_sep:
                continue
            try:
                self._place_kite_candidates(
                    target, top, bottom, visible_wing,
                )
            except RuntimeError:
                continue
            self.shown_path = tuple(shown)
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate kite vertex puzzle.")

    def _place_kite_candidates(
        self,
        target: Point,
        top: Point,
        bottom: Point,
        visible_wing: Point,
    ) -> None:
        """Place 5 markers near the missing wing, biased AWAY from the
        top-bottom axis so distractors stay on the mirror-wing side and
        do not bleed into the visible kite interior.
        """
        marker_clear = max(self.point_radius * 2.6, self.minimum_candidate_spacing() * 0.95)
        path_clear = max(self.point_radius * 2.0, 22.0)

        def clear_of_kite(point: Point) -> bool:
            return (
                min(
                    self._point_to_segment_distance(point, top, visible_wing),
                    self._point_to_segment_distance(point, visible_wing, bottom),
                )
                >= path_clear
            )

        # Distractors stay off the drawn wing by constraint rather than by an
        # outward bias, so the answer no longer sits at the middle of the
        # cloud its own distractors form.
        self.seat_candidates(target, accept=clear_of_kite, min_separation=marker_clear)

    @staticmethod
    def _point_to_segment_distance(P: Point, A: Point, B: Point) -> float:
        ABx, ABy = B.x - A.x, B.y - A.y
        APx, APy = P.x - A.x, P.y - A.y
        ab2 = ABx * ABx + ABy * ABy
        if ab2 < 1e-9:
            return math.hypot(APx, APy)
        t = max(0.0, min(1.0, (APx * ABx + APy * ABy) / ab2))
        cx = A.x + t * ABx
        cy = A.y + t * ABy
        return math.hypot(P.x - cx, P.y - cy)

    def _authored_observation(self) -> str:
        apex_one, visible_wing, apex_two = self.shown_path
        axis_direction = self._heading_phrase(apex_two.x - apex_one.x, apex_two.y - apex_one.y)
        axis_mid = Point((apex_one.x + apex_two.x) / 2.0, (apex_one.y + apex_two.y) / 2.0)
        return (
            f"The dashed axis joins the kite's two apexes, running {axis_direction}, and the single visible wing "
            f"bulges out {self._relative_position_phrase(axis_mid, visible_wing)} it, "
            f"{self._canvas_region(visible_wing)}. A kite is mirror-symmetric about that axis, so the missing "
            f"wing is the visible wing's reflection, equally far out on the opposite side, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {"shown_vertices": [p.to_list() for p in self.shown_path]}

    def _draw_candidate_label_only(
        self,
        draw,
        candidate: PointCandidate,
        *,
        fill: tuple[int, int, int] | None = None,
    ) -> None:
        """Redraw one candidate label after solution strokes cross its marker."""
        if hasattr(draw, "add_high_level_command"):
            return
        ss = getattr(draw, "factor", 1)
        raw_draw = getattr(draw, "raw", draw)
        font = ImageFont.load_default(round(self.CANDIDATE_LABEL_FONT_SIZE * ss))
        cx = round(candidate.x * ss)
        cy = round(candidate.y * ss)
        text_bbox = font.getbbox(candidate.label)
        tx = cx - (text_bbox[0] + text_bbox[2]) // 2
        ty = cy - (text_bbox[1] + text_bbox[3]) // 2 + round(self.CANDIDATE_LABEL_OFFSET_Y * ss)
        raw_draw.text((tx, ty), candidate.label, fill=fill or self.CANDIDATE_TEXT_COLOR, font=font)

    def _draw_target_join_disk(self, draw) -> None:
        if hasattr(draw, "add_high_level_command"):
            return
        ss = getattr(draw, "factor", 1)
        raw_draw = getattr(draw, "raw", draw)
        cx = round(self.target_point.x * ss)
        cy = round(self.target_point.y * ss)
        radius = max(1, round(self.line_width * 1.2 * ss))
        raw_draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=self.CANDIDATE_OUTLINE_COLOR,
        )

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        top, visible_wing, bottom = self.shown_path
        # Draw the symmetry axis as a thin gray dashed line so the kite's
        # symmetry (and hence the location of the missing mirror wing) is
        # legible from the static frame alone.
        self.draw_dashed_line(draw, top, bottom, dash=8.0, gap=5.0)
        # Draw the visible sides as one joined polyline so the known wing
        # vertex is sealed instead of two butt-capped segments meeting.
        self.draw_line(draw, [top, visible_wing, bottom])
        if highlight_label:
            # Single continuous closing stroke; outer endpoints inset so
            # butt caps hide inside the shown-path butt caps (no bump at
            # shape corners; the target joint in the middle is handled by
            # PIL's joint="curve" and then covered by the answer marker).
            self.draw_line(
                draw,
                [bottom, self.target_point, top],
                join_start=True,
                join_end=True,
            )
        self.draw_candidates(draw, highlight_label=highlight_label)
        if highlight_label:
            # The highlighted marker sits exactly on the completed vertex.
            # Redraw the closing stroke above the marker so the kite outline
            # remains visibly continuous, then restore the option label.
            self.draw_line(
                draw,
                [bottom, self.target_point, top],
                join_start=True,
                join_end=True,
            )
            self._draw_target_join_disk(draw)
            correct_candidate = next(
                candidate for candidate in self.candidates
                if candidate.label == highlight_label
            )
            self._draw_candidate_label_only(draw, correct_candidate)
        return base

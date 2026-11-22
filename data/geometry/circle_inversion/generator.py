"""Circle inversion puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class CircleInversionGenerator(PointTargetPuzzleGenerator):
    """Find the image of a point under inversion in a given circle."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows a large black inversion circle with its center marked by a small hollow marker, "
        "one hollow source point outside the circle, and five labeled candidate circles A-E inside the circle. "
        "Rule: Let O be the marked center, r the circle radius, and P the source. Its inverse P' lies on the ray from O "
        "through P and satisfies OP times OP' equals r squared; equivalently, OP' = r squared divided by OP. Select the "
        "candidate whose center is P'. Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/circle_inversion"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows one large unfilled inversion circle drawn with a black outline of medium thickness with its center marked by a small hollow white marker outlined in black, "
        "and one small hollow source marker with white fill and a thick black outline positioned outside the circle. "
        "Five candidate markers A-E are clustered near the hidden inverted image inside the circle; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the inversion circle, the center marker O, the source marker P, and the candidate markers, then draws one complete radial segment from O to whichever of P and its inverse P' is farther from O. The segment passes through all three points: O, P, and the correct candidate P'. "
        "In the final state, only the inverted-image candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def __init__(self, *args, source_region="outside", source_start=False, **kwargs):
        if source_region not in ("inside", "outside"):
            raise ValueError("source_region must be inside or outside")
        self.source_region = source_region
        self.source_start = source_start
        if source_region == "inside":
            # Only the location description changes; the equation and action do not.
            self.DEFAULT_RULE_PROMPT = self.DEFAULT_RULE_PROMPT.replace("source point outside", "source point inside").replace("candidate circles A-E inside the circle", "candidate circles A-E near the inverted image")
            self.DEFAULT_TI2V_PROMPT = self.DEFAULT_TI2V_PROMPT.replace("positioned outside", "positioned inside").replace("image inside", "image outside")
            self.DEFAULT_SOLUTION_IMAGE_PROMPT = self.strip_video_instruction(self.DEFAULT_TI2V_PROMPT)
        elif source_start:
            self.DEFAULT_TI2V_PROMPT = self.DEFAULT_TI2V_PROMPT.replace(
                "then draws one complete radial segment from O to whichever of P and its inverse P' is farther from O. The segment passes through all three points: O, P, and the correct candidate P'.",
                "then draws one cyan radial segment starting from the source P and ending at the center O, passing through the correct inverse candidate P'.",
            )
            self.DEFAULT_SOLUTION_IMAGE_PROMPT = self.strip_video_instruction(self.DEFAULT_TI2V_PROMPT)
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        min_R = self.canvas_short_side * 0.16
        max_R = self.canvas_short_side * 0.26
        # Keep the target (and every other candidate) clear of the
        # center marker: center marker has radius 7, candidate markers
        # have radius self.point_radius (≈ 11), plus a small visual gap.
        center_clear_distance = self.point_radius + 7 + 6  # ≈ 24 px
        for _ in range(999):
            R = self._rng.uniform(min_R, max_R)
            center_padding = candidate_padding + R + self.canvas_short_side * 0.04
            if center_padding >= self.canvas_short_side * 0.42:
                continue
            center = self.pick_target_point(0.36, padding=center_padding)
            if not self.circle_fits(center, R, extra_padding=self.line_width):
                continue

            source_angle = self._rng.uniform(0.0, math.tau)
            # Constrain source_dist so the INVERTED target sits
            # comfortably away from the center (inv_dist = R² / source_dist
            # needs to be ≥ center_clear_distance + some extra room for
            # candidate spread). Also keep target inside the circle
            # (inv_dist ≤ R * 0.85) so it doesn't brush the circle edge.
            min_inv = center_clear_distance + self.canvas_short_side * 0.02
            max_inv = R * 0.85
            if min_inv >= max_inv:
                continue
            source_dist_max = R * R / min_inv
            source_dist_min = max(R * 1.4, R * R / max_inv)
            if source_dist_min >= source_dist_max:
                continue
            source_dist = (
                R / self._rng.uniform(1.4, 1.9)
                if self.source_region == "inside"
                else self._rng.uniform(source_dist_min, min(source_dist_max, R * 2.4))
            )
            source = self.point_on_ray(center, source_angle, source_dist)
            if not self.inside_canvas(source, padding=self.line_width + 8):
                continue

            inv_dist = R * R / source_dist
            target = self.point_on_ray(center, source_angle, inv_dist)

            if not self.point_can_host_candidate(target):
                continue
            if self.distance(center, target) < center_clear_distance:
                continue

            # Stricter min-pairwise-spacing check than the base validator
            # (which accepts down to ~0.9 * minimum_candidate_spacing) to
            # avoid the visible B/C/D/E touching observed in earlier
            # samples. Retry placement up to 8 times to find a layout
            # whose minimum pairwise gap clears the visual threshold.
            strict_gap = max(self.point_radius * 2.8 + 6.0,
                             self.minimum_candidate_spacing() * 1.10)
            placed_ok = False
            for _ in range(8):
                try:
                    self.place_candidates(target)
                except RuntimeError:
                    continue
                cand_pts = [Point(c.x, c.y) for c in self.candidates]
                if self.points_are_well_spaced(cand_pts, min_distance=strict_gap):
                    placed_ok = True
                    break
            if not placed_ok:
                continue
            # Also reject layouts where any NON-target candidate would
            # overlap the center marker — they cluster around target so
            # a candidate on the center-side of target can intrude.
            if any(self.distance(Point(c.x, c.y), center) < center_clear_distance
                   for c in self.candidates):
                continue

            if self.source_region == "inside" and any(
                self.distance(Point(c.x, c.y), source) < center_clear_distance
                for c in self.candidates
            ):
                continue
            self.center = center
            self.R = R
            self.source_point = source
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid circle inversion puzzle.")

    def _authored_observation(self) -> str:
        side_names = {
            "up": "top", "down": "bottom", "left": "left", "right": "right",
            "upper-left": "upper-left", "upper-right": "upper-right",
            "lower-left": "lower-left", "lower-right": "lower-right",
        }
        source_side = side_names[self._direction_name(
            self.source_point.x - self.center.x,
            self.source_point.y - self.center.y,
        )]
        return (
            f"The inversion circle sits {self._canvas_region(self.center)}, and the hollow source marker stands "
            f"{self.source_region} the rim, off its {source_side} side. Inversion keeps the point on the ray from the "
            f"marked center through the source and sets its distance to the squared radius divided by the "
            f"source distance. Its inverse lies {'outside' if self.source_region == 'inside' else 'inside'} the circle — here "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "inversion_center": self.center.to_list(),
            "inversion_radius": self.R,
            "source_point": self.source_point.to_list(),
        }

    def _draw_center_marker(self, draw) -> None:
        self.draw_anchor_marker(draw, self.center, 7)

    def _draw_source(self, draw) -> None:
        self.draw_anchor_marker(draw, self.source_point, 7)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_center_marker(draw)
        self._draw_source(draw)

    def _radial_segment(self) -> tuple[Point, Point]:
        if self.source_start and self.source_region == "outside":
            return self.source_point, self.center
        far = max(
            (self.source_point, self.target_point),
            key=lambda point: self.distance(self.center, point),
        )
        return self.center, far

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_circle(draw, self.center, self.R)
        if highlight_label:
            self.draw_line(draw, list(self._radial_segment()))
        self._draw_center_marker(draw)
        self._draw_source(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

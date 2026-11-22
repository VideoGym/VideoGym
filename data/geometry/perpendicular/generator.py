"""Perpendicular puzzle generator.
"""

from __future__ import annotations
from typing import Optional
from PIL import Image, ImageDraw
import math
from data.geometry.core.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord

class PerpendicularGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the center of a triangle."""

    DEFAULT_RULE_PROMPT=(
        "A white canvas shows one black reference line, one small hollow white marker outlined in black marking a through-point, and five labeled "
        "candidate circles A-E. Determine which candidate lies on the line through the small circle that is perpendicular to "
        "the reference line. Answer with one option from A-E."
    )
    DEFAULT_OUTPUT_DIR = "outputs/geometry/perpendicular"
    DEFAULT_TI2V_PROMPT=(
        "A square white canvas shows one black reference segment of medium thickness and one small hollow marker with white fill and a thick black outline centered on the marked through-point. "
        "Five candidate markers A-E sit near the hidden perpendicular direction on a short straight row; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the reference segment, the marked through-point marker, and the candidate row, then draws one black segment of medium thickness starting just outside that hollow marker and extending toward the correct candidate at a right angle to the reference segment. "
        "In the final state, only the candidate on this perpendicular segment changes to pale red fill with a dark red outline, while the other markers stay white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin = 64
        anchor_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            line_angle = self._rng.uniform(0.0, math.tau)
            normal_angle = line_angle + math.pi / 2 * self._rng.choice([-1, 1])
            through_point = self.pick_target_point(0.46, padding=anchor_padding + self.canvas_short_side * 0.16)
            try:
                reference_mid = self.sample_point_along_direction(
                    through_point,
                    normal_angle,
                    min_distance=self.canvas_short_side * 0.2,
                    max_distance=self.canvas_short_side * 0.3,
                    padding=self.line_width,
                )
                p_ref1 = self.sample_point_along_direction(
                    reference_mid,
                    line_angle,
                    min_distance=self.canvas_short_side * 0.14,
                    max_distance=self.canvas_short_side * 0.22,
                    padding=self.line_width,
                )
                p_ref2 = self.sample_point_along_direction(
                    reference_mid,
                    line_angle + math.pi,
                    min_distance=self.canvas_short_side * 0.14,
                    max_distance=self.canvas_short_side * 0.22,
                    padding=self.line_width,
                )
                # The reference segment lies between the through-point and the
                # candidate row along this same normal, so the row has to clear
                # it: an answer landing on the segment would be the only marker
                # touching the drawing and would also make the
                # perpendicular's far end unreadable.
                reference_reach = self.distance(through_point, reference_mid)
                target = self.sample_point_along_direction(
                    through_point,
                    normal_angle,
                    min_distance=reference_reach + self.point_radius * 2.6,
                    max_distance=max(
                        reference_reach + self.point_radius * 2.6 + 1.0,
                        min(
                            reference_reach + self.canvas_short_side * 0.16,
                            self.canvas_short_side * 0.40,
                        ),
                    ),
                    padding=anchor_padding,
                )
                self.place_candidates_line(target, line_angle + self._rng.uniform(-0.05, 0.05))
            except RuntimeError:
                continue
            if self.distance(p_ref1, p_ref2) < self.canvas_short_side * 0.28:
                continue
            self.points = (through_point, p_ref1, p_ref2)
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to find valid perpendicular geometry after 999 tries")

    def _authored_observation(self) -> str:
        through_point, p_ref1, p_ref2 = self.points
        reference_heading = self._heading_phrase(p_ref2.x - p_ref1.x, p_ref2.y - p_ref1.y)
        perpendicular_heading = self._heading_phrase(
            self.target_point.x - through_point.x, self.target_point.y - through_point.y
        )
        return (
            f"The black reference segment runs {reference_heading}, and the hollow through-marker floats "
            f"{self._canvas_region(through_point)}, clear of it. The required line must pass through that marker "
            f"at a right angle to the reference direction, which sends it {perpendicular_heading} from the marker "
            "into the short row of markers; since the row runs roughly parallel to the reference, exactly one "
            "center can lie on the perpendicular."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "source_line_endpoints": [self.points[1].to_list(), self.points[2].to_list()],
            "through_point": self.points[0].to_list(),
        }

    def _draw_through_point(self, draw) -> None:
        self.draw_anchor_marker(draw, self.points[0], 7)

    def _video_overlay_extras(self, draw: ImageDraw.ImageDraw) -> None:
        self._draw_through_point(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()

        self.draw_line(draw,self.points[1:])
        if highlight_label:
            line_start, line_end = self.trim_segment(
                self.points[0],
                self.target_point,
                start_offset=0.0,
                end_offset=float(self.point_radius),
            )
            self.draw_line(draw,[line_start, line_end])
        self._draw_through_point(draw)

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

"""Orthogonal projection puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class OrthogonalProjectionGenerator(PointTargetPuzzleGenerator):
    """Find the foot of the perpendicular from a floating point onto a line."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows a long black line segment and one hollow source point floating off the line, plus five labeled "
        "candidate circles A-E on the line. Identify which candidate is the foot of the perpendicular dropped from the "
        "source point onto the line, i.e. the nearest point on the line to the source. Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/orthogonal_projection"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows one long black line segment of medium thickness crossing the canvas and one small hollow source marker with white fill and a thick black outline floating away from the line. "
        "Five candidate markers A-E are clustered on the line near the hidden projection foot; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the line segment, the source marker, and the candidate markers, then draws one black perpendicular connector of medium thickness from the source marker straight down to the correct candidate on the line. "
        "In the final state, only the projection-foot candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        for _ in range(999):
            line_angle = self._rng.uniform(0.0, math.pi)
            line_mid = self.pick_target_point(
                0.42,
                padding=candidate_padding + self.canvas_short_side * 0.15,
            )
            try:
                p1, p2 = self.sample_symmetric_segment(
                    line_mid,
                    line_angle,
                    min_half_length=self.canvas_short_side * 0.22,
                    max_half_length=self.canvas_short_side * 0.36,
                    padding=self.line_width,
                )
            except RuntimeError:
                continue

            normal_angle = line_angle + math.pi / 2 * self._rng.choice([-1, 1])
            source_dist = self._rng.uniform(
                self.canvas_short_side * 0.18,
                self.canvas_short_side * 0.32,
            )
            foot_offset = self._rng.uniform(-0.3, 0.3)
            foot_point = Point(
                x=line_mid.x + foot_offset * (p2.x - p1.x),
                y=line_mid.y + foot_offset * (p2.y - p1.y),
            )
            source_point = self.point_on_ray(foot_point, normal_angle, source_dist)

            if not self.inside_canvas(source_point, padding=self.line_width + 8):
                continue
            if not self.point_can_host_candidate(foot_point):
                continue
            _, t = self.project_point_onto_line(foot_point, p1, p2)
            if not 0.15 <= t <= 0.85:
                continue
            if self.distance(source_point, foot_point) < self.canvas_short_side * 0.16:
                continue

            # The Prompt promises the options sit on the line, and the answer
            # does; so must the distractors.  A distractor is
            # wrong because the connector to the source would not be
            # perpendicular there, not because it floats beside the ink.  The
            # minimum offset grows with the source distance so the tilt away
            # from a right angle stays at least ~14 degrees (7.5).
            source_dist = self.distance(source_point, foot_point)
            try:
                self.place_candidates_on_incidence(
                    foot_point,
                    [Incidence.segment(p1, p2)],
                    min_offset=max(
                        self.minimum_candidate_spacing(),
                        source_dist * math.tan(math.radians(14.0)),
                    ),
                    max_offset=None,
                    reject=lambda point: (
                        min(self.distance(point, p1), self.distance(point, p2))
                        < self.point_radius * 1.6
                    ),
                )
            except (RuntimeError, ValueError):
                continue

            self.line_points = (p1, p2)
            self.source_point = source_point
            self.target_point = foot_point
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid orthogonal projection puzzle.")

    def _authored_observation(self) -> str:
        p1, p2 = self.line_points
        line_heading = self._heading_phrase(p2.x - p1.x, p2.y - p1.y)
        drop_heading = self._heading_phrase(
            self.target_point.x - self.source_point.x,
            self.target_point.y - self.source_point.y,
        )
        return (
            f"The long line runs {line_heading} across the canvas, while the hollow source marker floats "
            f"{self._canvas_region(self.source_point)}, off the line. The required point is the perpendicular "
            f"foot — the line's closest point to the source — so I drop from the source {drop_heading}, meeting "
            f"the line squarely {self._canvas_region(self.target_point)}; any center displaced along the line "
            "from that foot lies farther from the source."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "line_points": [p.to_list() for p in self.line_points],
            "source_point": self.source_point.to_list(),
        }

    def _draw_source_point(self, draw) -> None:
        self.draw_anchor_marker(draw, self.source_point, 7)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_source_point(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, list(self.line_points))
        if highlight_label:
            _, trimmed_end = self.trim_segment(
                self.source_point, self.target_point,
                start_offset=0.0,
                end_offset=float(self.point_radius),
            )
            self.draw_line(draw, [self.source_point, trimmed_end])
        self._draw_source_point(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

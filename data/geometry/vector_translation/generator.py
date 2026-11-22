"""Vector translation puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class VectorTranslationGenerator(PointTargetPuzzleGenerator):
    """Given a translation vector arrow and a source point, find the translated image."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows one black tail-to-head arrow, a separate hollow source point, and five labeled "
        "candidate circles A-E. Rule: Translate the source by exactly the arrow's tail-to-head displacement, preserving "
        "both direction and length. Select the candidate at the translated point. Required Output: Answer with one option "
        "from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/vector_translation"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a black arrow segment of medium thickness representing a translation vector from one hollow marker to another, "
        "and a separate small hollow source marker with white fill and a thick black outline. "
        "Five candidate markers A-E are clustered near the hidden translated position; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the translation arrow, the source marker, and the candidate markers, then draws one solid cyan connector of medium thickness from just outside the source marker to the correct candidate, parallel to the translation arrow. "
        "In the final state, only the translated-position candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        for _ in range(999):
            vec_length = self._rng.uniform(self.canvas_short_side * 0.18, self.canvas_short_side * 0.32)
            vec_angle = self._rng.uniform(0.0, math.tau)
            dx = vec_length * math.cos(vec_angle)
            dy = vec_length * math.sin(vec_angle)

            # Vector start point
            vec_start = self.pick_target_point(0.50, padding=candidate_padding)
            vec_end = Point(vec_start.x + dx, vec_start.y + dy)

            if not self.inside_canvas(vec_start, padding=self.line_width + 4):
                continue
            if not self.inside_canvas(vec_end, padding=self.line_width + 4):
                continue

            # Source point — separate from the vector
            source = self.pick_target_point(0.50, padding=candidate_padding)
            # Make sure source is not too close to the vector line
            if self.distance_point_to_line(source, vec_start, vec_end) < self.canvas_short_side * 0.08:
                continue
            if self.distance(source, vec_start) < self.canvas_short_side * 0.10:
                continue

            target = Point(source.x + dx, source.y + dy)

            if not self.point_can_host_candidate(target):
                continue
            if not self.inside_canvas(source, padding=self.line_width + 8):
                continue
            if self.distance(source, target) < self.canvas_short_side * 0.15:
                continue

            try:
                self.place_candidates(target)
            except RuntimeError:
                continue

            self.vec_start = vec_start
            self.vec_end = vec_end
            self.source_point = source
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate vector translation puzzle.")

    def _authored_observation(self) -> str:
        arrow_direction = self._heading_phrase(
            self.vec_end.x - self.vec_start.x, self.vec_end.y - self.vec_start.y,
        )
        return (
            f"The arrow points {arrow_direction}, fixing both the direction and the distance of the slide, and the "
            f"separate hollow source marker sits {self._canvas_region(self.source_point)}. Carrying the source "
            f"through exactly that tail-to-head displacement, with no turn and no stretch, lands "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "difficulty_variant": "single_vector_translation",
            "vector_start": self.vec_start.to_list(),
            "vector_end": self.vec_end.to_list(),
            "source_point": self.source_point.to_list(),
        }

    def _draw_arrow(self, draw) -> None:
        """Draw the translation vector with an arrowhead as a single polyline."""
        angle = math.atan2(self.vec_end.y - self.vec_start.y, self.vec_end.x - self.vec_start.x)
        head_len = self.canvas_short_side * 0.03
        left = self.point_on_ray(self.vec_end, angle + math.pi + math.radians(25), head_len)
        right = self.point_on_ray(self.vec_end, angle + math.pi - math.radians(25), head_len)
        # Single polyline: left wing → tip → right wing, then shaft.
        # Shaft's vec_start endpoint is shared with the hollow anchor
        # marker at vec_start, and vec_end endpoint is shared with the
        # arrowhead polyline's joint. Both need join-disk seals to fill
        # the tiny gap between butt cap and adjacent ink.
        self.draw_line(draw, [left, self.vec_end, right])
        self.draw_line(
            draw, [self.vec_start, self.vec_end],
            join_start=True, join_end=True,
        )

    def _draw_source(self, draw) -> None:
        self.draw_anchor_marker(draw, self.source_point, 7)

    def _draw_vec_endpoints(self, draw) -> None:
        self.draw_anchor_marker(draw, self.vec_start, 5)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_source(draw)
        self._draw_vec_endpoints(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self._draw_arrow(draw)
        if highlight_label:
            # Draw the connector from source marker to target candidate.
            # join_start=True seals the butt cap under the hollow source
            # marker (previously the old 8-px trim left a visible gap);
            # the target-side trim stops short of the answer marker's
            # disk so the candidate outline covers the line end cleanly.
            _, trimmed_end = self.trim_segment(
                self.source_point, self.target_point,
                start_offset=0.0,
                end_offset=float(self.point_radius),
            )
            self.draw_line(
                draw, [self.source_point, trimmed_end],
                join_start=True,
            )
        self._draw_vec_endpoints(draw)
        self._draw_source(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

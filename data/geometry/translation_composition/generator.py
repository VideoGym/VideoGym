"""Translation composition puzzle generator."""

from __future__ import annotations

import math

from PIL import Image

from data.common.render_primitives import arrow_stroke_points
from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class TranslationCompositionGenerator(PointTargetPuzzleGenerator):
    """Apply two translation vectors to a source point and find the final position."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows two black tail-to-head arrows representing translation vectors, one hollow "
        "source point, and five labeled candidate circles A-E. Rule: Add each arrow's tail-to-head displacement to the "
        "source point. The two translations may be applied in either order because their displacement vectors add to the "
        "same result. Select the candidate at that final point. Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/translation_composition"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows two black arrow segments of medium thickness representing two different translation vectors, "
        "and one small hollow source marker with white fill and a thick black outline. "
        "Five candidate markers A-E are clustered near the hidden final position; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the two translation arrows, the source marker, and the candidate markers. It may apply either visible vector first, drawing its cyan connector from the source to the corresponding intermediate point, then applies the remaining vector from there, drawing the second cyan connector to the correct candidate; both orders are equivalent. "
        "In the final state, only the final-position candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def _arrow_strokes(self, start: Point, end: Point) -> list[list[Point]]:
        """The two polylines one vector arrow is stroked from: head, then shaft.

        Shared with the option-layout parity check so that check measures the
        ink the renderer actually paints, arrowhead wings included.
        """

        head, shaft = arrow_stroke_points(
            (start.x, start.y), (end.x, end.y),
            head_length=self.canvas_short_side * 0.025,
        )
        return [[Point(x, y) for x, y in points] for points in (head, shaft)]

    def _draw_arrow(self, draw, start: Point, end: Point) -> None:
        head, shaft = self._arrow_strokes(start, end)
        self.draw_line(draw, head)
        # Shaft joins the hollow anchor at `start` and the arrowhead
        # polyline at `end`; seal both butt caps with join disks so
        # there is no gap between the shaft and the hollow marker /
        # arrowhead joint.
        self.draw_line(
            draw, shaft,
            join_start=True, join_end=True,
        )

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        for _ in range(999):
            # Two translation vectors
            len1 = self._rng.uniform(self.canvas_short_side * 0.12, self.canvas_short_side * 0.24)
            len2 = self._rng.uniform(self.canvas_short_side * 0.12, self.canvas_short_side * 0.24)
            angle1 = self._rng.uniform(0.0, math.tau)
            angle2 = self._rng.uniform(0.0, math.tau)
            # Ensure vectors are not too parallel
            angle_diff = abs((angle2 - angle1 + math.pi) % math.tau - math.pi)
            if angle_diff < math.radians(30) or angle_diff > math.radians(150):
                continue

            dx1, dy1 = len1 * math.cos(angle1), len1 * math.sin(angle1)
            dx2, dy2 = len2 * math.cos(angle2), len2 * math.sin(angle2)

            padding_needed = candidate_padding

            # Place vector arrows and source
            v1_start = self.pick_target_point(0.46, padding=padding_needed)
            v1_end = Point(v1_start.x + dx1, v1_start.y + dy1)

            v2_start = self.pick_target_point(0.46, padding=padding_needed)
            v2_end = Point(v2_start.x + dx2, v2_start.y + dy2)

            source = self.pick_target_point(0.40, padding=padding_needed)
            intermediate = Point(source.x + dx1, source.y + dy1)
            target = Point(intermediate.x + dx2, intermediate.y + dy2)

            all_pts = [v1_start, v1_end, v2_start, v2_end, source, intermediate, target]
            if not all(self.inside_canvas(p, padding=self.line_width + 4) for p in all_pts):
                continue
            if not self.point_can_host_candidate(target):
                continue
            if self.distance(source, target) < self.canvas_short_side * 0.12:
                continue
            # Vectors should not overlap with source area
            if self.distance(v1_start, source) < self.canvas_short_side * 0.08:
                continue
            if self.distance(v2_start, source) < self.canvas_short_side * 0.08:
                continue

            try:
                self.place_candidates(target)
            except RuntimeError:
                continue
            # The composed image point is not on either arrow, but a marker can
            # land on one.  When the correct marker is the only one touching ink
            # the sample answers itself exactly as a designed incidence shortcut
            # would, so redraw the scatter instead.
            if not self.candidate_incidence_is_answer_free(
                [
                    Incidence.polyline(stroke)
                    for arrow in ((v1_start, v1_end), (v2_start, v2_end))
                    for stroke in self._arrow_strokes(*arrow)
                ]
            ):
                continue

            self.v1_start = v1_start
            self.v1_end = v1_end
            self.v2_start = v2_start
            self.v2_end = v2_end
            self.source_point = source
            self.intermediate_point = intermediate
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate translation composition puzzle.")

    def _authored_observation(self) -> str:
        first_direction = self._heading_phrase(
            self.v1_end.x - self.v1_start.x, self.v1_end.y - self.v1_start.y,
        )
        second_direction = self._heading_phrase(
            self.v2_end.x - self.v2_start.x, self.v2_end.y - self.v2_start.y,
        )
        combined_direction = self._heading_phrase(
            self.target_point.x - self.source_point.x,
            self.target_point.y - self.source_point.y,
        )
        return (
            f"One arrow points {first_direction} and the other points {second_direction}, while the hollow source "
            f"marker sits {self._canvas_region(self.source_point)}. Carrying the source through both tail-to-head "
            f"displacements, one after the other in either order, moves it {combined_direction} overall and lands "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "difficulty_variant": "two_vector_composition",
            "vector1": [self.v1_start.to_list(), self.v1_end.to_list()],
            "vector2": [self.v2_start.to_list(), self.v2_end.to_list()],
            "source_point": self.source_point.to_list(),
            "intermediate_point": self.intermediate_point.to_list(),
        }

    def _draw_arrows(self, draw) -> None:
        self._draw_arrow(draw, self.v1_start, self.v1_end)
        self._draw_arrow(draw, self.v2_start, self.v2_end)

    def _draw_source(self, draw) -> None:
        self.draw_anchor_marker(draw, self.source_point, 7)

    def _draw_vec_markers(self, draw) -> None:
        self.draw_anchor_marker(draw, self.v1_start, 5)
        self.draw_anchor_marker(draw, self.v2_start, 5)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_source(draw)
        self._draw_vec_markers(draw)

    def _render(self, highlight_label: str | None) -> Image.Image:
        draw, base = self.get_draw_base()
        self._draw_arrows(draw)
        if highlight_label:
            _, trimmed_e2 = self.trim_segment(
                self.intermediate_point, self.target_point,
                start_offset=0.0, end_offset=float(self.point_radius),
            )
            self.draw_line(
                draw,
                [self.source_point, self.intermediate_point],
                fill=self.CONSTRUCTION_COLOR,
                join_start=True,
                join_end=True,
            )
            self.draw_line(
                draw,
                [self.intermediate_point, trimmed_e2],
                fill=self.CONSTRUCTION_COLOR,
                join_start=True,
            )
        self._draw_vec_markers(draw)
        self._draw_source(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

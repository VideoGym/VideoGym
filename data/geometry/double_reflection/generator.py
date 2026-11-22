"""Double-reflection point_geometry puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class DoubleReflectionGenerator(PointTargetPuzzleGenerator):
    """Reflect a point across two mirror axes in sequence."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows a blue first reflection axis, a green second reflection axis, one hollow source "
        "marker, and five labeled candidate circles A-E. Rule: Reflect the source perpendicularly across the blue axis, "
        "then reflect that intermediate point perpendicularly across the green axis. Select the candidate at the final "
        "location. Required Output: Answer with one option from A-E."
    )

    FIRST_AXIS_COLOR = (46, 110, 220)
    SECOND_AXIS_COLOR = (34, 145, 88)
    DEFAULT_OUTPUT_DIR = "outputs/geometry/double_reflection"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a blue first reflection-axis segment and a green second reflection-axis segment that "
        "cross near the middle, plus one small hollow "
        "source marker with white fill and a thick black outline. Five candidate markers A-E are clustered near the hidden "
        "final reflected location; each marker is a small white circle with a thin dark gray outline and a black uppercase "
        "letter. The video first holds the two axes, the source marker, and the candidate markers, then draws one black "
        "connector from the source point to its first reflected location and a second black connector from there to the "
        "final twice-reflected candidate. In the final state, only the final reflected candidate changes to pale red fill "
        "with a dark red outline, while the other markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    @staticmethod
    def _reflect_point(point_to_reflect: Point, line_p1: Point, line_p2: Point) -> Point:
        dx = line_p2.x - line_p1.x
        dy = line_p2.y - line_p1.y
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return point_to_reflect
        t = ((point_to_reflect.x - line_p1.x) * dx + (point_to_reflect.y - line_p1.y) * dy) / denom
        projection = Point(x=line_p1.x + t * dx, y=line_p1.y + t * dy)
        return Point(
            x=2.0 * projection.x - point_to_reflect.x,
            y=2.0 * projection.y - point_to_reflect.y,
        )

    def __init__(self, *args, parallel_axes=False, **kwargs):
        self.parallel_axes = bool(parallel_axes)
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        tries = 0
        while tries < 999:
            tries += 1
            pivot = self.pick_target_point(
                0.4,
                padding=candidate_padding + self.canvas_short_side * 0.18,
            )
            axis1_angle = self._rng.uniform(0.0, math.pi)
            delta = self._rng.uniform(math.radians(40.0), math.radians(125.0))
            axis2_angle = axis1_angle if self.parallel_axes else axis1_angle + delta

            try:
                axis1_p1, axis1_p2 = self.sample_symmetric_segment(
                    pivot,
                    axis1_angle,
                    min_half_length=self.canvas_short_side * 0.18,
                    max_half_length=self.canvas_short_side * 0.28,
                    padding=self.line_width,
                )
                second_pivot = (
                    self.point_on_ray(
                        pivot, axis1_angle + math.pi / 2, self.canvas_short_side * 0.14
                    )
                    if self.parallel_axes
                    else pivot
                )
                axis2_p1, axis2_p2 = self.sample_symmetric_segment(
                    second_pivot,
                    axis2_angle,
                    min_half_length=self.canvas_short_side * 0.16,
                    max_half_length=self.canvas_short_side * 0.24,
                    padding=self.line_width,
                )
            except RuntimeError:
                continue

            source_point = self.pick_target_point(0.52, padding=candidate_padding)
            if not self.inside_canvas(source_point, padding=self.line_width):
                continue
            if self.distance(source_point, pivot) < self.canvas_short_side * 0.18:
                continue
            if (
                self.distance_point_to_line(source_point, axis1_p1, axis1_p2)
                < self.canvas_short_side * 0.12
            ):
                continue
            if (
                self.distance_point_to_line(source_point, axis2_p1, axis2_p2)
                < self.canvas_short_side * 0.12
            ):
                continue

            intermediate_point = self._reflect_point(source_point, axis1_p1, axis1_p2)
            target_point = self._reflect_point(intermediate_point, axis2_p1, axis2_p2)
            if not self.inside_canvas(intermediate_point, padding=self.line_width):
                continue
            if not self.point_can_host_candidate(target_point):
                continue
            if (
                self.distance(source_point, target_point)
                < self.canvas_short_side * 0.24
            ):
                continue
            if (
                self.distance(intermediate_point, target_point)
                < self.canvas_short_side * 0.12
            ):
                continue

            try:
                self.place_candidates(target_point)
            except RuntimeError:
                continue

            self.axis1_points = (axis1_p1, axis1_p2)
            self.axis2_points = (axis2_p1, axis2_p2)
            self.source_point = source_point
            self.intermediate_point = intermediate_point
            self.target_point = target_point
            return self.save_puzzle()

        raise RuntimeError("Failed to generate a valid double reflection puzzle.")

    def _authored_observation(self) -> str:
        axis1_p1, axis1_p2 = self.axis1_points
        axis2_p1, axis2_p2 = self.axis2_points
        first_direction = self._heading_phrase(axis1_p2.x - axis1_p1.x, axis1_p2.y - axis1_p1.y)
        second_direction = self._heading_phrase(axis2_p2.x - axis2_p1.x, axis2_p2.y - axis2_p1.y)
        return (
            f"The blue axis runs {first_direction} and the green axis runs {second_direction}, while the hollow "
            f"source marker sits {self._canvas_region(self.source_point)}. Reflecting the source perpendicularly "
            f"across the blue axis drops it at equal distance on the opposite side, "
            f"{self._canvas_region(self.intermediate_point)}; reflecting that intermediate point across the green "
            f"axis the same way places the final point {self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "difficulty_variant": "two_axis_reflection_composition",
            "reflection_axis_1": [point.to_list() for point in self.axis1_points],
            "reflection_axis_2": [point.to_list() for point in self.axis2_points],
            "source_point": self.source_point.to_list(),
            "intermediate_point": self.intermediate_point.to_list(),
        }

    def _draw_source_point(self, draw) -> None:
        self.draw_anchor_marker(draw, self.source_point, 7)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_source_point(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, list(self.axis1_points), fill=self.FIRST_AXIS_COLOR)
        self.draw_line(draw, list(self.axis2_points), fill=self.SECOND_AXIS_COLOR)
        if highlight_label:
            _, trimmed_end = self.trim_segment(
                self.intermediate_point,
                self.target_point,
                start_offset=0.0,
                end_offset=float(self.point_radius),
            )
            self.draw_line(
                draw,
                [self.source_point, self.intermediate_point],
                fill=self.CONSTRUCTION_COLOR,
                join_end=True,
            )
            self.draw_line(
                draw,
                [self.intermediate_point, trimmed_end],
                fill=self.CONSTRUCTION_COLOR,
                join_start=True,
            )
        self._draw_source_point(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

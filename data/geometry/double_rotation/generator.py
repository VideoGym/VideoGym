"""Double rotation puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)
from data.common.typed_answers import FIVE_CHOICE_CODEC


def _signed_angle(radians: float) -> float:
    """The same turn, reported in (-pi, pi]."""

    return (radians + math.pi) % math.tau - math.pi


class DoubleRotationGenerator(PointTargetPuzzleGenerator):
    """Apply two successive rotations to a source point and find the final position."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows two hollow pivot markers, a separate solid black source dot that is distinctly smaller "
        "than the pivots, and five labeled candidate circles A-E. A blue arrowed arc around one pivot shows the first "
        "rotation amount and turn direction, with its arrowhead showing which way to turn; a green arrowed arc around "
        "the other pivot shows the second rotation amount and turn direction. Rule: Rotate the source around the blue-arc "
        "pivot through exactly the blue arc's directed angle. Then rotate that intermediate point around the green-arc "
        "pivot through exactly the green arc's directed angle. Select the candidate at the final position. Required "
        "Output: Answer with one option from A-E."
    )

    FIRST_ROTATION_COLOR = (46, 110, 220)
    SECOND_ROTATION_COLOR = (34, 145, 88)
    DEFAULT_OUTPUT_DIR = "outputs/geometry/double_rotation"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows two hollow pivot markers (white fill, thick black outlines) that are visibly LARGER than the other dot, together with one small solid black source dot placed away from both pivots. "
        "A short blue arrowed arc around the first pivot indicates the first rotation amount and turn direction; its arrowhead shows which way to turn. A short green arrowed arc around the second pivot indicates the second rotation amount and turn direction; its arrowhead shows which way to turn. "
        "Five candidate markers A-E sit on the same invisible circle centered on the second pivot at the intermediate-to-second-pivot radius; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "First rotate the source dot around the blue-arc pivot by following the blue arrowed arc to get an intermediate position. Then rotate that intermediate point around the green-arc pivot by following the green arrowed arc to the final answer. "
        "The video first holds the two pivots, their indicator arcs, the source dot, and the candidate markers, then draws one big blue arc of medium thickness from the source dot around the first pivot to the intermediate position, followed by one big green arc of medium thickness from the intermediate position around the second pivot to the correct candidate. "
        "In the final state, only the final-position candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    @staticmethod
    def _point_at_math_angle(center: Point, angle: float, radius: float) -> Point:
        return Point(
            x=center.x + radius * math.cos(angle),
            y=center.y - radius * math.sin(angle),
        )

    @staticmethod
    def _math_angle(center: Point, point: Point) -> float:
        return math.atan2(-(point.y - center.y), point.x - center.x)

    @staticmethod
    def _rotate_point(point: Point, pivot: Point, angle: float) -> Point:
        dx, dy = point.x - pivot.x, point.y - pivot.y
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return Point(
            pivot.x + dx * cos_a + dy * sin_a,
            pivot.y - dx * sin_a + dy * cos_a,
        )

    def _arc_points(
        self,
        center: Point,
        radius: float,
        start_angle: float,
        sweep: float,
        *,
        step_degrees: float,
    ) -> list[Point]:
        steps = max(8, math.ceil(abs(math.degrees(sweep)) / max(1.0, step_degrees)))
        return [
            self._point_at_math_angle(center, start_angle + sweep * index / steps, radius)
            for index in range(steps + 1)
        ]

    def _arc_fits(
        self,
        center: Point,
        radius: float,
        start_angle: float,
        sweep: float,
        *,
        padding: float,
    ) -> bool:
        return all(
            self.inside_canvas(point, padding=padding)
            for point in self._arc_points(center, radius, start_angle, sweep, step_degrees=2.0)
        )

    def _draw_cv2_polyline(
        self,
        base: Image.Image,
        points: list[Point],
        *,
        width_factor: float,
        fill: tuple[int, int, int],
    ) -> None:
        scale_x = base.size[0] / self.canvas_dimensions[0]
        scale_y = base.size[1] / self.canvas_dimensions[1]
        scaled = np.array(
            [[round(point.x * scale_x), round(point.y * scale_y)] for point in points],
            dtype=np.int32,
        )
        arr = np.array(base)
        thickness = max(1, round(self.line_width * width_factor * max(scale_x, scale_y)))
        cv2.polylines(arr, [scaled], isClosed=False, color=fill, thickness=thickness, lineType=cv2.LINE_AA)
        base.paste(Image.fromarray(arr))

    @staticmethod
    def _arrowhead_points(tip: Point, previous: Point) -> list[Point]:
        """The filled triangle one indicator arrowhead occupies.

        Shared with the option-layout parity check so that check measures the
        ink the renderer actually paints, arrowhead included.
        """

        direction_angle = math.atan2(tip.y - previous.y, tip.x - previous.x)
        arrow_size = 9.0
        ux = math.cos(direction_angle)
        uy = math.sin(direction_angle)
        px = -uy
        py = ux
        base_back = arrow_size * 1.15
        half_w = arrow_size * 0.72
        bx = tip.x - ux * base_back
        by = tip.y - uy * base_back
        return [
            tip,
            Point(bx + px * half_w, by + py * half_w),
            Point(bx - px * half_w, by - py * half_w),
        ]

    def _draw_arrowhead(self, draw, tip: Point, previous: Point, *, fill: tuple[int, int, int]) -> None:
        points = [
            (round(point.x), round(point.y))
            for point in self._arrowhead_points(tip, previous)
        ]
        draw.polygon(points, fill=fill, outline=fill, width=1)

    def _draw_arrowed_arc(
        self,
        draw,
        base: Image.Image,
        center: Point,
        radius: float,
        start_angle: float,
        sweep: float,
        *,
        width_factor: float,
        fill: tuple[int, int, int],
        arrowhead: bool,
        step_degrees: float = 0.75,
    ) -> None:
        if arrowhead:
            arrow_backoff = min(abs(sweep) * 0.22, 8.0 / max(radius, 1.0))
            stroke_sweep = sweep - math.copysign(arrow_backoff, sweep)
        else:
            stroke_sweep = sweep
        points = self._arc_points(center, radius, start_angle, stroke_sweep, step_degrees=step_degrees)
        if hasattr(draw, "add_high_level_command"):
            draw.add_high_level_command(
                "draw_smooth_polyline",
                points=[[point.x, point.y] for point in points],
                width=max(1, round(self.line_width * width_factor)),
                fill=fill,
            )
        else:
            self._draw_cv2_polyline(base, points, width_factor=width_factor, fill=fill)

        if arrowhead:
            arrow_points = self._arc_points(center, radius, start_angle, sweep, step_degrees=step_degrees)
            self._draw_arrowhead(draw, arrow_points[-1], arrow_points[-2], fill=fill)

    def _place_second_radius_candidates(self, target_angle: float) -> None:
        """Seat the five markers along the second rotation's own circle.

        Every option sits on the circle the second turn preserves, so a
        distractor is wrong only by how far round that turn it stops.  All
        four of them used to be dealt on the *far* side of the answer, at
        fixed multiples of the minimum angular separation: the answer was
        then always the marker the second sweep reaches first, an end of its
        own arc row, and read 0.34 farthest-from-the-cloud-centroid and 0.39
        bottom-most against 0.200 chance.  The shared placer deals them to
        both sides and draws how many fall behind the answer before any of
        them is placed.
        """

        radius = self.distance(self.pivot2, self.intermediate_point)
        min_distance = max(self.point_radius * 2.6, self.minimum_candidate_spacing(scale=0.62))
        if radius <= min_distance:
            raise RuntimeError("Second rotation radius is too small for deceptive candidates")

        protected = (self.pivot1, self.pivot2, self.source_point)
        self.place_candidates_on_incidence(
            self.target_point,
            [Incidence.circle(self.pivot2, radius)],
            min_separation=min_distance,
            reject=lambda point: any(
                self.distance(point, anchor) < min_distance * 0.75 for anchor in protected
            ),
        )
        self.candidate_angle_offsets_degrees = [
            round(
                math.degrees(
                    _signed_angle(
                        math.atan2(self.pivot2.y - candidate.y, candidate.x - self.pivot2.x)
                        - target_angle
                    )
                ),
                6,
            )
            for candidate in self.candidates
        ]

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            pivot1 = self.pick_target_point(0.36, padding=candidate_padding + self.canvas_short_side * 0.14)
            pivot2 = self.pick_target_point(0.36, padding=candidate_padding + self.canvas_short_side * 0.14)
            if self.distance(pivot1, pivot2) < self.canvas_short_side * 0.10:
                continue

            source = self.pick_target_point(0.46, padding=candidate_padding)

            sign1 = self._rng.choice([-1, 1])
            sign2 = self._rng.choice([-1, 1])
            deg1 = self._rng.randint(4, 18) * 10  # 40..180
            deg2 = self._rng.randint(4, 18) * 10
            angle1 = sign1 * math.radians(deg1)
            angle2 = sign2 * math.radians(deg2)

            intermediate = self._rotate_point(source, pivot1, angle1)
            target = self._rotate_point(intermediate, pivot2, angle2)
            start_a1 = self._math_angle(pivot1, source)
            start_a2 = self._math_angle(pivot2, intermediate)

            all_pts = [pivot1, pivot2, source, intermediate, target]
            if not all(self.inside_canvas(p, padding=self.line_width + 8) for p in all_pts):
                continue
            if not self.point_can_host_candidate(target):
                continue
            if self.distance(source, target) < self.canvas_short_side * 0.12:
                continue
            if not self._arc_fits(pivot1, self.distance(pivot1, source), start_a1, angle1, padding=self.line_width + 8):
                continue
            if not self._arc_fits(pivot2, self.distance(pivot2, intermediate), start_a2, angle2, padding=self.line_width + 8):
                continue

            # Indicator arcs around the two pivots must not overlap
            indicator_r = self.canvas_short_side * 0.05
            pivot_dist = self.distance(pivot1, pivot2)
            if pivot_dist < indicator_r * 2 + self.canvas_short_side * 0.02:
                continue

            self.pivot1 = pivot1
            self.pivot2 = pivot2
            self.source_point = source
            self.intermediate_point = intermediate
            self.target_point = target
            self.angle1 = angle1
            self.angle2 = angle2
            self.deg1 = deg1
            self.deg2 = deg2
            self.sign1 = sign1
            self.sign2 = sign2

            try:
                self._place_second_radius_candidates(start_a2 + angle2)
            except RuntimeError:
                continue
            # The answer is not on any drawn primitive, but a marker can still
            # land on a pivot's rotation-indicator arc.  When the correct marker
            # is the only one touching that ink the sample answers itself
            # exactly as a designed incidence shortcut would, so
            # redraw instead of shipping it.
            # Built from the stroked points rather than a centre-and-sweep arc:
            # this generator measures angles with a flipped y axis, and the
            # arrowhead sits just past the stroke's end, so the polyline the
            # renderer actually walks is the convention-free description of the
            # ink.
            indicator_strokes = []
            for pivot, start, sweep in (
                (pivot1, start_a1, angle1),
                (pivot2, start_a2, angle2),
            ):
                stroke = self._arc_points(pivot, indicator_r, start, sweep, step_degrees=2.0)
                head = self._arrowhead_points(stroke[-1], stroke[-2])
                indicator_strokes.append(Incidence.polyline(stroke))
                indicator_strokes.append(Incidence.polyline(head + [head[0]]))
            if not self.candidate_incidence_is_answer_free(indicator_strokes):
                continue

            return self.save_puzzle()
        raise RuntimeError("Failed to generate double rotation puzzle.")

    def build_record_extra(self) -> dict[str, object]:
        return {
            "difficulty_variant": "two_pivot_rotation_composition",
            "pivot1": self.pivot1.to_list(),
            "pivot2": self.pivot2.to_list(),
            "source_point": self.source_point.to_list(),
            "intermediate_point": self.intermediate_point.to_list(),
            "rotation1_degrees": self.deg1 * (1 if self.sign1 > 0 else -1),
            "rotation2_degrees": self.deg2 * (1 if self.sign2 > 0 else -1),
            "candidate_angle_offsets_degrees": list(self.candidate_angle_offsets_degrees),
        }

    def _reference_reasoning_text(self) -> str:
        selected = next(candidate for candidate in self.candidates if candidate.label == self.correct_label)
        selected_point = Point(float(selected.x), float(selected.y))
        peers = [Point(float(candidate.x), float(candidate.y)) for candidate in self.candidates]
        first_turn = "counterclockwise" if self.sign1 > 0 else "clockwise"
        second_turn = "counterclockwise" if self.sign2 > 0 else "clockwise"
        source_from_blue = self._relative_position_phrase(self.pivot1, self.source_point)
        intermediate_from_blue = self._relative_position_phrase(self.pivot1, self.intermediate_point)
        intermediate_from_green = self._relative_position_phrase(self.pivot2, self.intermediate_point)
        target_from_green = self._relative_position_phrase(self.pivot2, selected_point)
        candidate_position = self._point_position_label(selected_point, peers)
        return (
            f"The blue-arc pivot sits {self._canvas_region(self.pivot1)} and the green-arc pivot "
            f"{self._canvas_region(self.pivot2)}; the source begins {source_from_blue} the blue-arc pivot. "
            f"Following the blue arc {first_turn} through {self._angle_size_phrase(self.deg1)} at "
            f"constant radius puts the intermediate point {intermediate_from_blue} that pivot and "
            f"{intermediate_from_green} the green-arc pivot. The "
            f"green arc then turns {second_turn} through {self._angle_size_phrase(self.deg2)}, again preserving "
            f"its radius, and ends {target_from_green} the "
            f"green pivot at the {candidate_position} "
            f"candidate {self.correct_label}.\n\n{FIVE_CHOICE_CODEC.format(self.correct_label)}"
        )

    def _draw_pivots(self, draw) -> None:
        # Pivots: large hollow rings (radius 9) so they read as the
        # "rotation centers" and are visibly larger than the source.
        self.draw_anchor_marker(draw, self.pivot1, 9)
        self.draw_anchor_marker(draw, self.pivot2, 9)

    def _draw_source(self, draw) -> None:
        # Source: small solid black dot (radius 5) — clearly different
        # from the hollow pivot rings so the solver knows which point
        # gets rotated vs which points are the centers.
        cx, cy = round(self.source_point.x), round(self.source_point.y)
        r = 5
        draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                     fill=self.CANDIDATE_OUTLINE_COLOR)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_pivots(draw)
        self._draw_source(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        indicator_r = self.canvas_short_side * 0.05
        start_a1 = self._math_angle(self.pivot1, self.source_point)
        start_a2 = self._math_angle(self.pivot2, self.intermediate_point)
        self._draw_arrowed_arc(
            draw, base, self.pivot1, indicator_r, start_a1, self.angle1,
            width_factor=0.7, fill=self.FIRST_ROTATION_COLOR, arrowhead=True,
        )
        self._draw_arrowed_arc(
            draw, base, self.pivot2, indicator_r, start_a2, self.angle2,
            width_factor=0.7, fill=self.SECOND_ROTATION_COLOR, arrowhead=True,
        )
        if highlight_label:
            r1 = self.distance(self.pivot1, self.source_point)
            self._draw_arrowed_arc(
                draw, base, self.pivot1, r1, start_a1, self.angle1,
                width_factor=1.0, fill=self.FIRST_ROTATION_COLOR, arrowhead=False,
            )
            r2 = self.distance(self.pivot2, self.intermediate_point)
            self._draw_arrowed_arc(
                draw, base, self.pivot2, r2, start_a2, self.angle2,
                width_factor=1.0, fill=self.SECOND_ROTATION_COLOR, arrowhead=False,
            )
        self._draw_pivots(draw)
        self._draw_source(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

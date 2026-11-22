"""Point rotation puzzle generator."""

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


class PointRotationGenerator(PointTargetPuzzleGenerator):
    """Rotate a source point around a pivot by an angle stated in the prompt."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows one hollow pivot marker, one solid black source dot, one arrowed indicator "
        "arc centered on the pivot, and five labeled candidate circles A-E. Rule: Rotate the source around the hollow "
        "pivot through exactly the sweep shown by the indicator arc, in the arrowhead's direction, while preserving "
        "the source-to-pivot distance. Select the candidate at the rotated position. Required Output: Return exactly "
        "one uppercase option label in the form \"Answer: X.\""
    )

    SUPERSAMPLE = 8
    DEFAULT_OUTPUT_DIR = "outputs/geometry/point_rotation"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a hollow pivot marker (white fill, thick black outline), a solid black source dot, and a prominent black indicator arc centered on the pivot. "
        "The arc span defines the rotation amount, and the triangular arrowhead at the end of the arc defines the rotation direction. "
        "Five candidate markers A-E all sit on the same invisible circle centered on the pivot at the source-to-pivot radius; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "Rotate the source dot around the hollow pivot by following the arrowed arc to exactly one candidate. "
        "The video first holds the pivot with its indicator arc, the source dot, and the candidate markers, then draws one black arc of medium thickness from the source dot along the rotation path to the correct candidate, keeping the same radius from the pivot throughout. "
        "In the final state, only the rotated-position candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    @staticmethod
    def _point_at_math_angle(center: Point, angle: float, radius: float) -> Point:
        return Point(
            x=center.x + radius * math.cos(angle),
            y=center.y - radius * math.sin(angle),
        )

    def _place_same_radius_candidates(self, target_angle: float) -> None:
        """Seat the five markers along the turning circle itself.

        Every option sits on the circle the rotation preserves, so "which
        marker could the source even reach?" carries no signal, and a
        distractor is wrong only by how far round the sweep it stops.  The
        markers used to be dealt at fixed angles measured off the source and
        off the target, two of each: the row then had the answer at its middle
        rung by construction, and read 0.33 nearest-the-cloud-centroid and
        0.52 most-isolated against 0.200 chance.  The shared placer deals them
        along the circle instead and draws how many fall behind the answer
        before any of them is placed.
        """

        min_distance = self.minimum_candidate_spacing(scale=0.9)
        self.place_candidates_on_incidence(
            self.target_point,
            [Incidence.circle(self.pivot, self.radius)],
            min_separation=min_distance,
            reject=lambda point: self.distance(point, self.source_point) < min_distance,
        )
        self.candidate_angle_offsets_degrees = [
            round(
                math.degrees(
                    _signed_angle(
                        math.atan2(self.pivot.y - candidate.y, candidate.x - self.pivot.x)
                        - target_angle
                    )
                ),
                6,
            )
            for candidate in self.candidates
        ]

    def __init__(self, *args, rotation_degrees=None, **kwargs):
        if rotation_degrees is not None and not 20 <= rotation_degrees <= 300:
            raise ValueError("rotation_degrees must lie in [20, 300]")
        self.rotation_degrees = rotation_degrees
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        candidate_host_padding = self.candidate_anchor_padding(extra=self.line_width)
        for _ in range(999):
            pivot = self.pick_target_point(
                0.18, padding=candidate_padding + self.canvas_short_side * 0.30,
            )
            radius = self._rng.uniform(
                self.canvas_short_side * 0.16,
                self.canvas_short_side * 0.27,
            )
            source_angle = self._rng.uniform(0.0, math.tau)
            rotation_sign = self._rng.choice([-1, 1])
            rotation_deg = self.rotation_degrees if self.rotation_degrees is not None else self._rng.randint(5, 16) * 10
            rotation_amount = math.radians(rotation_deg)
            target_angle = source_angle + rotation_sign * rotation_amount

            source_point = self._point_at_math_angle(pivot, source_angle, radius)
            target_point = self._point_at_math_angle(pivot, target_angle, radius)

            if not self.inside_canvas(source_point, padding=self.line_width + 8):
                continue
            if not self.point_can_host_candidate(target_point):
                continue
            if not self.circle_fits(pivot, radius, extra_padding=candidate_host_padding):
                continue
            if self.distance(source_point, target_point) < self.canvas_short_side * 0.15:
                continue

            self.pivot = pivot
            self.radius = radius
            self.source_point = source_point
            self.source_angle = source_angle
            self.target_angle = target_angle
            self.rotation_sign = rotation_sign
            self.rotation_amount = rotation_amount
            self.rotation_deg = rotation_deg
            self.target_point = target_point

            try:
                self._place_same_radius_candidates(target_angle)
            except RuntimeError:
                continue

            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid rotation puzzle.")

    def build_record_extra(self) -> dict[str, object]:
        return {
            "difficulty_variant": "single_forward_rotation",
            "pivot": self.pivot.to_list(),
            "source_point": self.source_point.to_list(),
            "rotation_degrees": self.rotation_deg * (1 if self.rotation_sign > 0 else -1),
            "radius": self.radius,
            "source_angle_degrees": (math.degrees(self.source_angle) % 360.0),
            "target_angle_degrees": (math.degrees(self.target_angle) % 360.0),
            "candidate_angle_offsets_degrees": list(self.candidate_angle_offsets_degrees),
        }

    def _reference_reasoning_text(self) -> str:
        selected = next(candidate for candidate in self.candidates if candidate.label == self.correct_label)
        selected_point = Point(float(selected.x), float(selected.y))
        peers = [Point(float(candidate.x), float(candidate.y)) for candidate in self.candidates]
        turn = "counterclockwise" if self.rotation_sign > 0 else "clockwise"
        source_position = self._relative_position_phrase(self.pivot, self.source_point)
        target_position = self._relative_position_phrase(self.pivot, selected_point)
        candidate_position = self._point_position_label(selected_point, peers)
        return (
            f"The hollow pivot sits {self._canvas_region(self.pivot)}, and the solid source begins "
            f"{source_position} it on the turning circle. The indicator "
            f"arrow turns {turn}, sweeping {self._angle_size_phrase(self.rotation_deg)}; following that complete "
            f"visible sweep while keeping the pivot distance fixed "
            f"carries the point to a position {target_position} the pivot. That endpoint coincides with "
            f"the {candidate_position} candidate, "
            f"labeled {self.correct_label}; the other candidates stop at different angular positions.\n\n"
            f"{FIVE_CHOICE_CODEC.format(self.correct_label)}"
        )

    def _draw_pivot(self, draw) -> None:
        # Pivot: hollow ring (radius 9) so it's visibly LARGER than the
        # source dot and obviously the center of rotation.
        self.draw_anchor_marker(draw, self.pivot, 9)

    def _draw_source(self, draw) -> None:
        # Source: small solid black dot (radius 5). Visually distinct
        # from the hollow pivot marker so the solver can tell at a
        # glance which point gets rotated vs which point is the center.
        cx, cy = round(self.source_point.x), round(self.source_point.y)
        r = 5
        draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                     fill=self.CANDIDATE_OUTLINE_COLOR)

    def _draw_indicator_arc(self, draw, base: Image.Image) -> None:
        """Readable angle arc with an arrowhead showing rotation direction."""
        arc_r = self.radius * 0.45
        sweep = self.rotation_sign * self.rotation_amount
        arrow_size = max(8.0, min(12.0, arc_r * 0.24))
        arrow_backoff = min(abs(sweep) * 0.22, arrow_size * 0.9 / arc_r)
        stroke_sweep = sweep - math.copysign(arrow_backoff, sweep)
        self._draw_arc_with_recording_direction(
            draw,
            base,
            radius=arc_r,
            width_factor=0.85,
            step_degrees=0.5,
            sweep=stroke_sweep,
        )
        points = self._arc_points(
            self.pivot,
            arc_r,
            self.source_angle,
            sweep,
            step_degrees=0.5,
        )
        tip = points[-1]
        previous = points[-2]
        tangent_dx = tip.x - previous.x
        tangent_dy = tip.y - previous.y
        self._draw_arrowhead(draw, tip, math.atan2(tangent_dy, tangent_dx), size=arrow_size)

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

    def _draw_arc_with_recording_direction(
        self,
        draw,
        base: Image.Image,
        *,
        radius: float,
        width_factor: float,
        step_degrees: float,
        sweep: Optional[float] = None,
    ) -> None:
        sweep = self.rotation_sign * self.rotation_amount if sweep is None else sweep
        points = self._arc_points(
            self.pivot,
            radius,
            self.source_angle,
            sweep,
            step_degrees=step_degrees,
        )
        if hasattr(draw, "add_high_level_command"):
            draw.add_high_level_command(
                "draw_smooth_polyline",
                points=[[point.x, point.y] for point in points],
                width=max(1, round(self.line_width * width_factor)),
                fill=self.CANDIDATE_OUTLINE_COLOR,
            )
            return
        self._draw_cv2_polyline(base, points, width_factor=width_factor)

    def _draw_rotation_arc(self, draw, base: Image.Image, *, radius: float, width_factor: float) -> None:
        self._draw_arc_with_recording_direction(
            draw,
            base,
            radius=radius,
            width_factor=width_factor,
            step_degrees=0.5,
            sweep=None,
        )

    def _draw_cv2_polyline(self, base: Image.Image, points: list[Point], *, width_factor: float) -> None:
        scale_x = base.size[0] / self.canvas_dimensions[0]
        scale_y = base.size[1] / self.canvas_dimensions[1]
        scaled = np.array(
            [[round(point.x * scale_x), round(point.y * scale_y)] for point in points],
            dtype=np.int32,
        )
        arr = np.array(base)
        thickness = max(1, round(self.line_width * width_factor * max(scale_x, scale_y)))
        cv2.polylines(
            arr,
            [scaled],
            isClosed=False,
            color=self.CANDIDATE_OUTLINE_COLOR,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        base.paste(Image.fromarray(arr))

    def _draw_arrowhead(self, draw, tip: Point, direction_angle: float, size: float) -> None:
        ux = math.cos(direction_angle)
        uy = math.sin(direction_angle)
        px = -uy
        py = ux
        base_back = size * 1.15
        half_w = size * 0.75
        bx = tip.x - ux * base_back
        by = tip.y - uy * base_back
        points = [
            (round(tip.x), round(tip.y)),
            (round(bx + px * half_w), round(by + py * half_w)),
            (round(bx - px * half_w), round(by - py * half_w)),
        ]
        draw.polygon(
            points,
            fill=self.CANDIDATE_OUTLINE_COLOR,
            outline=self.CANDIDATE_OUTLINE_COLOR,
            width=1,
        )

    def _video_overlay_extras(self, draw) -> None:
        self._draw_pivot(draw)
        self._draw_source(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        # Indicator arc near the pivot (always visible) — shows the
        # rotation angle and direction visually, so the solver doesn't
        # have to rely on the prompt's textual angle description.
        self._draw_indicator_arc(draw, base)
        if highlight_label:
            # Solution: big rotation arc from source to target, same
            # radius as the pivot-to-source distance.
            self._draw_rotation_arc(draw, base, radius=self.radius, width_factor=1.0)
        self._draw_pivot(draw)
        self._draw_source(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

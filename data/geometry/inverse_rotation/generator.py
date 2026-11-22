"""Inverse rotation puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from data.geometry.core.point_target_base import Point, PointCandidate, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class InverseRotationGenerator(PointTargetPuzzleGenerator):
    """Given a rotated point, pivot, and angle, find the original pre-image point."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows a solid black filled-disc pivot point, a hollow image point (the result of rotation), "
        "a prominent indicator arc centered on the pivot, and five labeled candidate circles A-E. The arc span defines "
        "the rotation amount, and the arrowhead defines the reverse rotation direction from the hollow image point back "
        "to the hidden original point. All candidates lie on the same invisible circle centered on the pivot at the "
        "image-to-pivot radius. Find the original point by following the arrowed reverse rotation from the hollow image point. Answer with one "
        "option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/inverse_rotation"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a small solid black filled-disc pivot marker, "
        "a small hollow image marker with white fill and a thick black outline representing the rotated result, "
        "and a prominent black indicator arc centered on the pivot. The arc span defines the rotation amount, and the triangular arrowhead at the end of the arc defines the reverse rotation direction from the hollow image marker back to the hidden original point. "
        "Five candidate markers A-E all sit on the same invisible circle centered on the pivot at the image-to-pivot radius; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "Find the original point by following the arrowed reverse rotation from the hollow image marker back along the same circle. "
        "The video first holds the pivot, the image marker, the angle arc with arrowhead, and the candidate markers, then draws one black arc of medium thickness tracing the reverse rotation path from the image marker back to the correct candidate. "
        "In the final state, only the original-position candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    @staticmethod
    def _point_at_math_angle(center: Point, angle: float, radius: float) -> Point:
        return Point(
            x=center.x + radius * math.cos(angle),
            y=center.y - radius * math.sin(angle),
        )

    def _arc_points(
        self,
        center: Point,
        radius: float,
        start_angle: float,
        sweep: float,
        *,
        step_degrees: float = 0.5,
    ) -> list[Point]:
        steps = max(8, math.ceil(abs(math.degrees(sweep)) / max(1.0, step_degrees)))
        return [
            self._point_at_math_angle(center, start_angle + sweep * index / steps, radius)
            for index in range(steps + 1)
        ]

    def _draw_smooth_polyline(
        self,
        draw,
        base: Image.Image,
        points: list[Point],
        *,
        width_factor: float,
    ) -> None:
        width = max(1, round(self.line_width * width_factor))
        if hasattr(draw, "add_high_level_command"):
            draw.add_high_level_command(
                "draw_smooth_polyline",
                points=[[point.x, point.y] for point in points],
                width=width,
                fill=self.CANDIDATE_OUTLINE_COLOR,
            )
            return

        scale_x = base.size[0] / self.canvas_dimensions[0]
        scale_y = base.size[1] / self.canvas_dimensions[1]
        scaled = np.array(
            [[round(point.x * scale_x), round(point.y * scale_y)] for point in points],
            dtype=np.int32,
        )
        arr = np.array(base)
        cv2.polylines(
            arr,
            [scaled],
            isClosed=False,
            color=self.CANDIDATE_OUTLINE_COLOR,
            thickness=max(1, round(width * max(scale_x, scale_y))),
            lineType=cv2.LINE_AA,
        )
        base.paste(Image.fromarray(arr))

    def _place_same_radius_candidates(self, original_angle: float) -> None:
        labels = list(self.option_labels)
        self._rng.shuffle(labels)
        min_distance = self.minimum_candidate_spacing(scale=0.9)
        # How many markers sit before the original along the rim is drawn
        # first and the rest laid out around it.  Two markers short of the
        # original and two past the image put the answer between the groups
        # in every sample, which makes it the middle marker of its own arc
        # and answers the puzzle without undoing the turn.
        step = math.degrees(min_distance / max(1.0, self.radius)) * 1.2
        for _ in range(160):
            behind = self._rng.randrange(len(labels))
            degrees: list[float] = []
            for index in range(behind, 0, -1):
                degrees.append(-(step * index + self._rng.uniform(0.0, step * 0.4)))
            degrees.append(0.0)
            for index in range(1, len(labels) - behind):
                degrees.append(step * index + self._rng.uniform(0.0, step * 0.4))
            offsets = [math.radians(value * self.rotation_sign) for value in degrees]
            candidates: list[PointCandidate] = []
            for label, offset in zip(labels, offsets):
                point = self._point_at_math_angle(self.pivot, original_angle + offset, self.radius)
                candidates.append(PointCandidate(x=point.x, y=point.y, label=label))
            if not self.validate_candidate_layout(candidates, min_distance=min_distance):
                continue
            if any(
                self.distance(Point(candidate.x, candidate.y), self.image_point) < min_distance
                for candidate in candidates
            ):
                continue
            self.candidates = candidates
            self.correct_label = labels[behind]
            self.candidate_angle_offsets_degrees = [
                round(math.degrees(offset), 6) for offset in offsets
            ]
            return

        raise RuntimeError("Failed to place same-radius inverse-rotation candidates")

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
                0.40, padding=candidate_padding + self.canvas_short_side * 0.20,
            )
            radius = self._rng.uniform(
                self.canvas_short_side * 0.18,
                self.canvas_short_side * 0.30,
            )
            original_angle = self._rng.uniform(0.0, math.tau)
            rotation_sign = self._rng.choice([-1, 1])
            # Keep the visual indicator on the shorter circular arc. Larger
            # rotations are geometrically valid but visually read as the
            # opposite shorter turn, which makes inverse-rotation direction
            # ambiguous from the image alone.
            rotation_deg = self.rotation_degrees if self.rotation_degrees is not None else self._rng.randint(5, 16) * 10
            rotation_amount = math.radians(rotation_deg)
            image_angle = original_angle + rotation_sign * rotation_amount

            original_point = self._point_at_math_angle(pivot, original_angle, radius)
            image_point = self._point_at_math_angle(pivot, image_angle, radius)

            if not self.inside_canvas(image_point, padding=self.line_width + 8):
                continue
            if not self.point_can_host_candidate(original_point):
                continue
            if not self.circle_fits(pivot, radius, extra_padding=candidate_host_padding):
                continue
            if self.distance(original_point, image_point) < self.canvas_short_side * 0.15:
                continue
            # Indicator-arc radius is roughly half the rotation radius.
            # That's ~3x the prior `0.06 * canvas_short_side` so the
            # rotation gesture is genuinely readable from the puzzle
            # image alone, and still cleanly inside the rotation circle
            # so it does not collide with the pivot/image markers.
            indicator_radius = radius * 0.5
            # Defensive: ensure arc fits inside the canvas without
            # bumping the canvas edge.
            if not self.circle_fits(pivot, indicator_radius, extra_padding=self.line_width + 4):
                continue

            self.pivot = pivot
            self.radius = radius
            self.indicator_radius = indicator_radius
            self.original_point = original_point
            self.image_point = image_point
            self.original_angle = original_angle
            self.image_angle = image_angle
            self.rotation_sign = rotation_sign
            self.rotation_amount = rotation_amount
            self.rotation_deg = rotation_deg
            self.target_point = original_point

            try:
                self._place_same_radius_candidates(original_angle)
            except RuntimeError:
                continue

            return self.save_puzzle()
        raise RuntimeError("Failed to generate inverse rotation puzzle.")

    def _authored_observation(self) -> str:
        # In this generator's math-angle convention (screen y grows downward
        # but angles are laid out with a y-flip), a positive rotation_sign is
        # a counterclockwise forward turn on screen, so the arrowed reverse
        # sweep drawn from the hollow marker runs clockwise on screen.
        reverse_turn = "clockwise" if self.rotation_sign > 0 else "counterclockwise"
        return (
            f"The solid disc {self._canvas_region(self.pivot)} marks the turning center, and the hollow marker "
            f"{self._relative_position_phrase(self.pivot, self.image_point)} it shows where the point ended up "
            f"after the turn. The arrowed arc runs {reverse_turn}, sweeping "
            f"{self._angle_size_phrase(self.rotation_deg)} back the way the point came; walking the hollow marker "
            f"through that same sweep at unchanged distance from the disc lands the starting point "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "difficulty_variant": "inverse_source_recovery",
            "pivot": self.pivot.to_list(),
            "original_point": self.original_point.to_list(),
            "image_point": self.image_point.to_list(),
            "rotation_degrees": self.rotation_deg * (1 if self.rotation_sign > 0 else -1),
            "radius": self.radius,
            "original_angle_degrees": math.degrees(self.original_angle) % 360.0,
            "image_angle_degrees": math.degrees(self.image_angle) % 360.0,
            "candidate_angle_offsets_degrees": list(self.candidate_angle_offsets_degrees),
        }

    def _draw_pivot(self, draw) -> None:
        # Pivot is a solid filled disc — clearly distinguishable from
        # the hollow image marker. The pivot is the rotation center, and
        # a filled point is the conventional symbol for that.
        cx, cy = round(self.pivot.x), round(self.pivot.y)
        r = 8
        draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=self.CANDIDATE_OUTLINE_COLOR,
            outline=self.CANDIDATE_OUTLINE_COLOR,
            width=1,
        )

    def _draw_image_point(self, draw) -> None:
        self.draw_anchor_marker(draw, self.image_point, 8)

    def _draw_arrowhead(self, draw, tip: Point, direction_angle: float, size: float) -> None:
        ux = math.cos(direction_angle)
        uy = math.sin(direction_angle)
        px = -uy
        py = ux
        base_back = size * 1.15
        half_w = size * 0.75
        bx = tip.x - ux * base_back
        by = tip.y - uy * base_back
        p_left = (round(bx + px * half_w), round(by + py * half_w))
        p_right = (round(bx - px * half_w), round(by - py * half_w))
        p_tip = (round(tip.x), round(tip.y))
        draw.polygon(
            [p_tip, p_left, p_right],
            fill=self.CANDIDATE_OUTLINE_COLOR,
            outline=self.CANDIDATE_OUTLINE_COLOR,
            width=1,
        )

    def _video_overlay_extras(self, draw) -> None:
        self._draw_pivot(draw)
        self._draw_image_point(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        arc_r = self.indicator_radius
        sweep = -self.rotation_sign * self.rotation_amount
        indicator_points = self._arc_points(self.pivot, arc_r, self.image_angle, sweep)
        arrow_size = max(8.0, min(12.0, arc_r * 0.24))
        arrow_backoff = min(abs(sweep) * 0.22, arrow_size * 0.9 / arc_r)
        stroke_sweep = sweep - math.copysign(arrow_backoff, sweep)
        stroke_points = self._arc_points(self.pivot, arc_r, self.image_angle, stroke_sweep)
        self._draw_smooth_polyline(draw, base, stroke_points, width_factor=0.85)
        tip = indicator_points[-1]
        previous = indicator_points[-2]
        self._draw_arrowhead(
            draw,
            tip,
            math.atan2(tip.y - previous.y, tip.x - previous.x),
            size=arrow_size,
        )
        if highlight_label:
            solution_points = self._arc_points(
                self.pivot,
                self.radius,
                self.image_angle,
                -self.rotation_sign * self.rotation_amount,
            )
            self._draw_smooth_polyline(draw, base, solution_points, width_factor=1.0)
        self._draw_pivot(draw)
        self._draw_image_point(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

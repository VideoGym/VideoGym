"""External tangent point puzzle generator."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


def _external_tangent_points(
    c1: Point, r1: float, c2: Point, r2: float
) -> Optional[Tuple[Point, Point]]:
    """Compute one pair of tangent points on the upper external tangent.

    Returns (point_on_circle_1, point_on_circle_2) or None if circles overlap.
    """
    dx = c2.x - c1.x
    dy = c2.y - c1.y
    d = math.hypot(dx, dy)
    if d < abs(r1 - r2) + 1e-6 or d < 1e-6:
        return None
    theta = math.atan2(dy, dx)
    sin_a = (r1 - r2) / d
    if abs(sin_a) > 1.0 - 1e-9:
        return None
    alpha = math.asin(max(-1.0, min(1.0, sin_a)))
    tangent_angle = theta + math.pi / 2 - alpha
    t1 = Point(
        x=c1.x + r1 * math.cos(tangent_angle),
        y=c1.y + r1 * math.sin(tangent_angle),
    )
    t2 = Point(
        x=c2.x + r2 * math.cos(tangent_angle),
        y=c2.y + r2 * math.sin(tangent_angle),
    )
    return t1, t2


class ExternalTangentPointGenerator(PointTargetPuzzleGenerator):
    """Find the tangent point on the common external tangent of two circles."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows two black circle outlines of different sizes, plus five labeled candidate circles A-E on the "
        "larger circle. Identify the point on the larger circle where the common external tangent line (a straight line "
        "touching both circles on the same side without crossing between them) touches it. "
        "Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/external_tangent_point"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows two unfilled circles of different sizes drawn with black outlines of medium thickness. "
        "Five candidate markers A-E are clustered on the larger circle near the hidden tangency point; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the two circles and the candidate markers, then draws one black tangent line of medium thickness that rests on top of both circles, touching the larger circle at the correct candidate and the smaller circle at a small hollow marker. "
        "In the final state, only the tangency candidate on the larger circle changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def place_candidates_on_circle(self, center: Point, R: float, true_point: Point) -> bool:
        labels = list(self.option_labels)
        correct_index = self._rng.randint(0, len(labels) - 1)
        self.correct_label = labels[correct_index]
        theta_t = math.atan2(true_point.y - center.y, true_point.x - center.x)
        angular_spread = math.radians(self._rng.uniform(80.0, 120.0))
        angular_step = angular_spread / (len(labels) - 1)
        theta_start = theta_t - correct_index * angular_step
        from data.geometry.core.point_target_base import PointCandidate
        candidates = []
        for i in range(len(labels)):
            theta = theta_start + i * angular_step
            cx = center.x + R * math.cos(theta)
            cy = center.y + R * math.sin(theta)
            p = Point(cx, cy) if i != correct_index else true_point
            candidates.append(PointCandidate(x=p.x, y=p.y, label=labels[i]))
        if not self.validate_candidate_layout(candidates, min_distance=self.minimum_candidate_spacing(scale=0.9)):
            return False
        self.candidates = candidates
        return True

    def __init__(self, *args, overlapping=False, **kwargs):
        self.overlapping = bool(overlapping)
        if self.overlapping:
            for field in (
                "DEFAULT_RULE_PROMPT",
                "DEFAULT_TI2V_PROMPT",
                "DEFAULT_SOLUTION_IMAGE_PROMPT",
            ):
                setattr(
                    self,
                    field,
                    getattr(self, field)
                    .replace("non-overlapping", "overlapping")
                    .replace("separate circles", "overlapping circles"),
                )
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.04)
        for _ in range(999):
            r1 = self._rng.uniform(
                self.canvas_short_side * 0.16, self.canvas_short_side * 0.26
            )
            r2 = self._rng.uniform(self.canvas_short_side * 0.08, r1 * 0.65)
            sep = (
                self._rng.uniform(r1 - r2 + 0.65 * r2, r1 + r2 - 0.35 * r2)
                if self.overlapping
                else self._rng.uniform(
                    r1 + r2 + self.canvas_short_side * 0.06,
                    r1 + r2 + self.canvas_short_side * 0.20,
                )
            )
            angle = self._rng.uniform(0.0, math.tau)
            c1 = self.pick_target_point(
                0.30, padding=candidate_padding + r1 + self.canvas_short_side * 0.02
            )
            c2 = self.point_on_ray(c1, angle, sep)

            if not self.circle_fits(c1, r1, extra_padding=self.line_width):
                continue
            if not self.circle_fits(c2, r2, extra_padding=self.line_width):
                continue

            result = _external_tangent_points(c1, r1, c2, r2)
            if result is None:
                continue
            t1, t2 = result
            if not self.inside_canvas(t1, padding=self.line_width):
                continue
            if not self.inside_canvas(t2, padding=self.line_width):
                continue

            if not self.place_candidates_on_circle(c1, r1, t1):
                continue

            self.c1 = c1
            self.r1 = r1
            self.c2 = c2
            self.r2 = r2
            self.tangent_p1 = t1
            self.tangent_p2 = t2
            self.target_point = t1
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid external tangent point puzzle.")

    def _authored_observation(self) -> str:
        contact_heading = self._heading_phrase(
            self.tangent_p1.x - self.c1.x,
            self.tangent_p1.y - self.c1.y,
        )
        return (
            f"The larger circle sits {self._canvas_region(self.c1)}, with the smaller circle located "
            f"{self._relative_position_phrase(self.c1, self.c2)} it. A common external tangent rests against both "
            f"rims on the same side without cutting between them, and each contact radius meets that support line "
            f"at a right angle. On the larger circle that radius points {contact_heading} to reach the contact, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "difficulty_variant": "external_tangent_same_side",
            "circle_1": {"center": self.c1.to_list(), "radius": self.r1},
            "circle_2": {"center": self.c2.to_list(), "radius": self.r2},
            "tangent_point_on_c1": self.tangent_p1.to_list(),
            "tangent_point_on_c2": self.tangent_p2.to_list(),
        }

    def _draw_tangent_p2_marker(self, draw) -> None:
        self.draw_anchor_marker(draw, self.tangent_p2, 7)

    def _video_overlay_extras(self, draw) -> None:
        pass

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_circle(draw, self.c1, self.r1)
        self.draw_circle(draw, self.c2, self.r2)
        if highlight_label:
            self.draw_line(draw, [self.tangent_p1, self.tangent_p2])
            # Marker on top of circle so it's not pierced
            self._draw_tangent_p2_marker(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

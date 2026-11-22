"""Tangent line pair angle puzzle generator.

The input shows only the circle and one external anchor.  The two tangents
are *not* drawn: drawing them would put the required angle on the canvas and
reduce the task to matching one visible wedge against five others, with the
circle and the tangency condition playing no part.

The opening between the two tangents from an external point P follows from
the tangency condition alone: each contact radius meets its tangent at a
right angle, so the half-opening satisfies sin(theta / 2) = R / |PC| and

    theta = 2 * arcsin(R / |PC|).

The solver therefore has to read the radius and the centre-to-anchor
distance off the figure.  The five candidate openings are built as one
ordered set and the answer is drawn from it uniformly, so the target's
opening is exchangeable with the distractors' and no learned band narrows
the field.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PIL import Image

from data.geometry.core.point_target_base import (
    Point,
    PointCandidate,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class TangentLinePairAngleGenerator(PointTargetPuzzleGenerator):
    """Identify the candidate equal to the angle between two tangents from an external point."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows a black circle and one filled black anchor above it, outside the circle, "
        "plus five candidate wedges A-E in a row in the lower half showing different openings. "
        "Rule: Two straight lines run from the anchor and each touches the circle at exactly one point, where it "
        "meets the radius to that point at a right angle. Those two lines make one angle at the anchor. Identify the "
        "candidate whose wedge equals that angle. "
        "Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/tangent_line_pair_angle"
    # Openings the wedge row has to separate, and the floor on their spacing.
    # 13 degrees clears the 12-degree sibling floor at the wedge ray length and
    # keeps the radius-to-distance ratio the solver must read separated by at
    # least ~9% across the whole range.
    MIN_OPENING_DEG = 34.0
    MAX_OPENING_DEG = 100.0
    MIN_OPENING_GAP_DEG = 13.0
    MIN_RATIO_MARGIN = 0.08
    # Radius band that fits the tallest layout (the widest anchor distance) for
    # every opening, so the radius carries no hint about the opening.
    MIN_RADIUS = 52.0
    MAX_RADIUS = 68.0
    DEFAULT_TI2V_PROMPT = (
        "Input Image: A white canvas shows a black circle outline of medium thickness in the upper half and one "
        "small filled black anchor above it, outside the circle. Five candidate wedges A-E sit in a row in the lower "
        "half, each opening upward; at the apex of each wedge sits a small white candidate marker with a thin dark "
        "gray outline and a black uppercase letter. "
        "Required Output: Hold the unchanged scene, then draw the two cyan tangent segments from the anchor to the "
        "circle one at a time, each running from the anchor to the point where it grazes the rim, and trace a thin "
        "cyan arc between them near the anchor to mark the opening they make. Finally recolor only the matching "
        "candidate marker to pale red fill with a dark red outline, leaving the other four white. Keep the circle, "
        "the anchor, the wedge row, the background, and the camera fixed. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Input Image: A white canvas shows a black circle, one filled black anchor above it outside the circle, and "
        "five candidate wedges A-E in a row in the lower half. "
        "Required Output: Preserve the circle, the anchor, and every wedge. Add the two cyan tangent segments from "
        "the anchor to their contacts on the rim and a thin cyan arc between them near the anchor, and recolor only "
        "the matching candidate marker to pale red fill with a dark red outline."
    )

    def _draw_wedge(self, draw, vertex: Point, angle_deg: float, ray_len: float, base_dir: float) -> None:
        half = math.radians(angle_deg) / 2.0
        a1 = base_dir - half
        a2 = base_dir + half
        p1 = self.point_on_ray(vertex, a1, ray_len)
        p2 = self.point_on_ray(vertex, a2, ray_len)
        s1, _ = self.trim_segment(vertex, p1, start_offset=float(self.point_radius))
        s2, _ = self.trim_segment(vertex, p2, start_offset=float(self.point_radius))
        self.draw_line(draw, [s1, p1])
        self.draw_line(draw, [s2, p2])

    def _draw_external_anchor(self, draw) -> None:
        cx, cy = round(self.external_point.x), round(self.external_point.y)
        r = max(3, self.point_radius // 2)
        if hasattr(draw, "raw"):
            raw = draw.raw
            ss = draw.factor
            raw.ellipse(
                (cx * ss - r * ss, cy * ss - r * ss, cx * ss + r * ss, cy * ss + r * ss),
                fill=(0, 0, 0),
            )
        else:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0))

    def _draw_reference(self, draw) -> None:
        """Source ink only: the circle and the external anchor."""

        self.draw_circle(draw, self.circle_center, self.radius)
        self._draw_external_anchor(draw)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_reference(draw)

    def _calc_tangent_point(self, center: Point, R: float, ext: Point, side: int) -> Point:
        D = self.distance(center, ext)
        d_CK = R * R / D
        ux = (ext.x - center.x) / D
        uy = (ext.y - center.y) / D
        Kx = center.x + d_CK * ux
        Ky = center.y + d_CK * uy
        h = math.sqrt(max(0.0, R * R - d_CK * d_CK))
        vx = -uy
        vy = ux
        return Point(x=Kx + side * h * vx, y=Ky + side * h * vy)

    def _sample_openings(self) -> List[float]:
        """Five openings spanning the declared range with the declared floor.

        Built by shifting sorted uniforms rather than resampled until they
        happen to separate, so every draw is a legal set.
        """

        count = len(self.option_labels)
        span = self.MAX_OPENING_DEG - self.MIN_OPENING_DEG - self.MIN_OPENING_GAP_DEG * (count - 1)
        base = sorted(self._rng.uniform(0.0, span) for _ in range(count))
        return [
            self.MIN_OPENING_DEG + base[index] + self.MIN_OPENING_GAP_DEG * index
            for index in range(count)
        ]

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        cw, ch = self.canvas_dimensions
        wedge_ray_len = self.canvas_short_side * 0.085
        cell_y = ch * 0.835
        for _ in range(999):
            openings = self._sample_openings()
            # The answer is one of the five openings chosen uniformly, so its
            # marginal distribution is exactly the distractors' and no learned
            # band or rank narrows the field.
            target_index = self._rng.randrange(len(openings))
            target_deg = openings[target_index]
            half = math.radians(target_deg) / 2.0
            radius = self._rng.uniform(self.MIN_RADIUS, self.MAX_RADIUS)
            distance = radius / math.sin(half)

            # Every distractor must demand a visibly different radius-to-anchor
            # ratio, which is the quantity the solver actually reads (7.5).
            ratio_margin = min(
                abs(math.sin(math.radians(value) / 2.0) - math.sin(half))
                for index, value in enumerate(openings)
                if index != target_index
            ) / math.sin(half)
            if ratio_margin < self.MIN_RATIO_MARGIN:
                continue

            ext_angle = math.radians(self._rng.uniform(-104.0, -76.0))
            top_limit = self.margin + self.point_radius * 0.6
            bottom_limit = cell_y - wedge_ray_len - self.point_radius - 8.0
            top = -distance * math.sin(ext_angle) if math.sin(ext_angle) < 0 else 0.0
            span = top + radius
            if bottom_limit - top_limit < span:
                continue
            center = Point(
                x=cw * self._rng.uniform(0.40, 0.60),
                y=self._rng.uniform(top_limit + top, bottom_limit - radius),
            )
            external = Point(
                x=center.x + distance * math.cos(ext_angle),
                y=center.y + distance * math.sin(ext_angle),
            )
            if not self.circle_fits(center, radius, extra_padding=self.line_width + 4):
                continue
            if not self.inside_canvas(external, padding=self.line_width + 10):
                continue

            self.circle_center = center
            self.radius = radius
            self.external_point = external
            self.tangent_p1 = self._calc_tangent_point(center, radius, external, +1)
            self.tangent_p2 = self._calc_tangent_point(center, radius, external, -1)
            self.target_deg = target_deg

            # The row is laid out inside the exact rect a marker may occupy, so
            # the outer two wedges always fit instead of being fished for

            slots: List[Tuple[float, float]] = []
            pad_x, _pad_y = self._candidate_safe_padding(include_edge_buffer=True)
            left, _top, right, _bottom = self.canvas_bounds()
            jitter_x = 3.0
            jitter_y = ch * 0.02
            first_x = left + pad_x + jitter_x
            cell_w = ((right - pad_x - jitter_x) - first_x) / 4.0
            for index in range(5):
                slots.append((
                    first_x + index * cell_w + self._rng.uniform(-jitter_x, jitter_x),
                    cell_y + self._rng.uniform(-jitter_y, jitter_y),
                ))

            # Slot order and label order are both shuffled, so the row never
            # reads as a monotone ladder of openings.
            order = list(range(5))
            self._rng.shuffle(order)
            labels = list(self.option_labels)
            self._rng.shuffle(labels)
            self.wedge_angles = [openings[index] for index in order]
            self.wedge_dirs = [
                math.radians(self._rng.uniform(265.0, 275.0)) for _ in range(5)
            ]
            self.wedge_ray_len = wedge_ray_len

            candidates: List[PointCandidate] = [
                PointCandidate(x=slot[0], y=slot[1], label=labels[index])
                for index, slot in enumerate(slots)
            ]
            if not self.validate_candidate_layout(candidates, min_distance=self.canvas_short_side * 0.16):
                continue
            slot_index = order.index(target_index)
            self.candidates = candidates
            self.correct_label = labels[slot_index]
            self.target_point = Point(slots[slot_index][0], slots[slot_index][1])
            return self.save_puzzle()
        raise RuntimeError("Failed to lay out tangent_line_pair_angle puzzle")

    def _authored_observation(self) -> str:
        count_words = ("none", "one", "two", "three", "four")
        narrower = sum(1 for value in self.wedge_angles if value < self.target_deg - 1e-6)
        wider = sum(1 for value in self.wedge_angles if value > self.target_deg + 1e-6)
        parts = []
        if narrower:
            parts.append(f"{count_words[narrower]} {'is' if narrower == 1 else 'are'} visibly narrower")
        if wider:
            parts.append(f"{count_words[wider]} {'is' if wider == 1 else 'are'} visibly wider")
        comparison = " and ".join(parts)
        distance = self.distance(self.circle_center, self.external_point)
        ratio = distance / max(self.radius, 1e-9)
        # Round the half-opening and double it, so the two figures the trace
        # quotes agree with each other and stay inside a degree of the truth.
        half_deg = round(math.degrees(math.asin(min(1.0, self.radius / max(distance, 1e-9)))))
        return (
            f"The circle hangs in the upper half of the canvas with the filled anchor above it, and the anchor "
            f"stands about {ratio:.1f} times the circle's radius from its center. Each line from the anchor that "
            "just grazes the circle meets the radius to its contact at a right angle, so that contact, the center "
            "and the anchor form a right triangle whose hypotenuse is the center-to-anchor distance and whose "
            "opposite leg is the radius. The half-opening at the anchor therefore has sine equal to the radius over "
            f"that distance, about {half_deg}°, and the two tangents are mirror images across "
            f"the center line, so the full opening is about {2 * half_deg}°. Of the five "
            f"upward-opening wedges in the bottom row, {comparison} than that spread, leaving a single wedge with "
            "the matching opening."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "circle_center": self.circle_center.to_list(),
            "radius": self.radius,
            "external_point": self.external_point.to_list(),
            "tangent_pair_angle_deg": self.target_deg,
            "candidate_angles_deg": list(self.wedge_angles),
        }

    def _tangent_angle_arc(self) -> Tuple[float, float, float]:
        """Math-convention (start, sweep, radius) for the angle mark at P."""

        p = self.external_point
        screen_1 = math.atan2(self.tangent_p1.y - p.y, self.tangent_p1.x - p.x)
        screen_2 = math.atan2(self.tangent_p2.y - p.y, self.tangent_p2.x - p.x)
        math_1, math_2 = -screen_1, -screen_2
        sweep = (math_2 - math_1 + math.pi) % math.tau - math.pi
        radius = max(
            18.0,
            min(
                0.45 * self.distance(p, self.tangent_p1),
                0.14 * self.canvas_short_side,
            ),
        )
        return math_1, sweep, radius

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        if highlight_label:
            # Solution-only construction: the two tangents the rule describes,
            # then the opening they make at the anchor.
            self.draw_line(draw, [self.external_point, self.tangent_p1])
            self.draw_line(draw, [self.external_point, self.tangent_p2])
        self._draw_reference(draw)
        for cand, ang_deg, ang_dir in zip(self.candidates, self.wedge_angles, self.wedge_dirs):
            v = Point(cand.x, cand.y)
            self._draw_wedge(draw, v, ang_deg, self.wedge_ray_len, ang_dir)
        if highlight_label:
            start, sweep, radius = self._tangent_angle_arc()
            self.draw_arc(
                draw,
                self.external_point,
                radius,
                start,
                sweep,
                width_factor=0.7,
                fill=self.CONSTRUCTION_COLOR,
            )
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

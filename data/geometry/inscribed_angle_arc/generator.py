"""Inscribed angle to central angle puzzle generator.

A circle is drawn in the upper half with an inscribed angle: a vertex on
the circle and two chord endpoints that subtend a marked arc. The central
angle on that arc equals 2 * inscribed (inscribed angle theorem). Five
candidate wedges A-E are drawn in a row in the lower half; pick the one
whose opening equals the central angle (= 2 * inscribed).
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


class InscribedAngleArcGenerator(PointTargetPuzzleGenerator):
    """Identify the candidate equal to 2 * inscribed angle (= central angle on the same arc)."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows a black circle outline with one filled black anchor and two hollow markers on the circumference. "
        "The two chord segments from the anchor to each hollow marker form an inscribed angle, and the highlighted arc between "
        "the hollow markers is a blue arc showing what that angle subtends. Five candidate wedges A-E in a row in the lower half show different "
        "openings. Identify the candidate whose wedge equals the central angle on the blue highlighted arc, that is twice the "
        "inscribed angle. Answer with one option from A-E."
    )

    MARKED_ARC_COLOR = (37, 99, 235)
    DEFAULT_OUTPUT_DIR = "outputs/geometry/inscribed_angle_arc"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a black circle outline of medium thickness in the upper half, with one small filled black anchor on the circumference and two small hollow markers also on the circumference connected by two black chord segments forming an inscribed angle at the filled anchor; a medium-width blue arc segment between the two hollow markers (on the far side of the circle) marks the arc that the inscribed angle subtends. "
        "Five candidate wedges A-E sit in a row in the lower half, each opening upward; at the apex of each wedge sits a small white candidate marker with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the circle, the inscribed angle, the marked arc, and the candidate wedges, then draws one radius from the circle center to the first hollow marker and, only after it is complete, draws the other radius from the center to the second hollow marker so the central angle subtending the same arc becomes visible. "
        "In the final state, only the candidate whose wedge equals the central angle (twice the inscribed) changes to pale red fill with a dark red outline, while the other four markers stay white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def __init__(self, *args, inscribed_range=(20.0, 75.0), **kwargs) -> None:
        if not 0 < inscribed_range[0] <= inscribed_range[1] < 90:
            raise ValueError("inscribed angles must lie between zero and 90 degrees")
        self.inscribed_range = inscribed_range
        super().__init__(*args, **kwargs)
        self._geometry_solution_description = (
            "Preserve the circle, inscribed angle, arc, and option wedges. First mark the circle center with a hollow "
            "black marker; then draw the two cyan radii to the arc endpoints and mark only the matching option."
        )
        self._geometry_video_description = (
            "Hold the unchanged input. Mark the circle center first, draw one cyan radius from the center to its "
            "arc endpoint, then draw the other cyan radius from the center to the other endpoint. Finish each "
            "radius before starting the next, then mark the matching candidate. Preserve each completed "
            "construction."
        )

    def _draw_wedge(self, draw, vertex: Point, angle_deg: float, ray_len: float, base_dir: float) -> None:
        half = math.radians(angle_deg) / 2.0
        a1 = base_dir - half
        a2 = base_dir + half
        p1 = self.point_on_ray(vertex, a1, ray_len)
        p2 = self.point_on_ray(vertex, a2, ray_len)
        self.draw_line(draw, [vertex, p1])
        self.draw_line(draw, [vertex, p2])

    def _draw_filled_marker(self, draw, center: Point) -> None:
        cx, cy = round(center.x), round(center.y)
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

    def _draw_hollow_center(self, draw) -> None:
        self.draw_anchor_marker(draw, self.circle_center, 6)

    def _draw_reference_markers(self, draw) -> None:
        self.draw_anchor_marker(draw, self.arc_p1, 6)
        self.draw_anchor_marker(draw, self.arc_p2, 6)
        self._draw_filled_marker(draw, self.apex)

    def _draw_reference_static(self, draw) -> None:
        # Circle
        self.draw_circle(draw, self.circle_center, self.radius)
        # Marked arc: color distinguishes it without changing stroke width.
        # NOTE: draw_arc uses math y-up convention while point_on_ray uses
        # image y-down; negate start and sweep to align with endpoint
        # markers placed via point_on_ray.
        self.draw_arc(
            draw,
            self.circle_center,
            self.radius,
            -self.arc_start_angle,
            -self.arc_sweep,
            fill=self.MARKED_ARC_COLOR,
        )
        # Two chords from inscribed apex to the two arc endpoints.
        self.draw_line(draw, [self.arc_p1, self.apex])
        self.draw_line(draw, [self.apex, self.arc_p2])
        self._draw_reference_markers(draw)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_reference_static(draw)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        cw, ch = self.canvas_dimensions
        for _ in range(999):
            inscribed_deg = self._rng.uniform(*self.inscribed_range)
            target_deg = inscribed_deg * 2.0  # central angle
            min_gap = 12.0
            choices = [target_deg]
            attempt = 0
            while len(choices) < 5 and attempt < 600:
                attempt += 1
                cand = self._rng.uniform(25.0, 160.0)
                if any(abs(cand - c) < min_gap for c in choices):
                    continue
                choices.append(cand)
            if len(choices) < 5:
                continue
            self._rng.shuffle(choices)
            target_idx = choices.index(target_deg)

            # Place circle in upper half.
            radius = self.canvas_short_side * self._rng.uniform(0.13, 0.17)
            cx_center = cw * self._rng.uniform(0.42, 0.58)
            cy_center = ch * self._rng.uniform(0.22, 0.30)
            center = Point(cx_center, cy_center)

            # Pick two arc endpoints subtending angle = 2 * inscribed at center.
            arc_sweep = math.radians(target_deg)
            base_angle = self._rng.uniform(0.0, math.tau)
            arc_p1_angle = base_angle - arc_sweep / 2.0
            arc_p2_angle = base_angle + arc_sweep / 2.0
            arc_p1 = self.point_on_ray(center, arc_p1_angle, radius)
            arc_p2 = self.point_on_ray(center, arc_p2_angle, radius)

            # Inscribed apex on the arc OPPOSITE the marked arc (so the
            # inscribed angle's arc is the one between arc_p1 and arc_p2 on
            # the FAR side from apex). The inscribed apex angle equals
            # half the central angle on the SAME (far) side as the marked
            # arc — which is the inscribed angle theorem.
            apex_angle = base_angle + math.pi + self._rng.uniform(-0.4, 0.4)
            apex = self.point_on_ray(center, apex_angle, radius)

            self.circle_center = center
            self.radius = radius
            self.arc_p1 = arc_p1
            self.arc_p2 = arc_p2
            self.apex = apex
            self.arc_start_angle = arc_p1_angle
            self.arc_sweep = arc_sweep
            self.inscribed_deg = inscribed_deg
            self.central_deg = target_deg

            slots: List[Tuple[float, float]] = []
            left, top, right, bottom = self.canvas_bounds()
            # Keep all five labeled wedge apices inside the shared candidate
            # safe area.  The old `left + 15 ... right - 15` span sat on the
            # label padding boundary, so some seeds exhausted all retries.
            slot_left = left + 32.0
            slot_right = right - 32.0
            cell_w = (slot_right - slot_left) / 4.0
            cell_y = ch * 0.78
            jitter_x = cell_w * 0.025
            jitter_y = ch * 0.03
            for i in range(5):
                cx = slot_left + i * cell_w + self._rng.uniform(-jitter_x, jitter_x)
                cy = cell_y + self._rng.uniform(-jitter_y, jitter_y)
                slots.append((cx, cy))

            labels = list(self.option_labels)
            self._rng.shuffle(labels)

            self.wedge_angles = choices
            self.wedge_dirs = [
                math.radians(self._rng.uniform(265.0, 275.0)) for _ in range(5)
            ]
            self.wedge_ray_len = self.canvas_short_side * 0.09

            candidates: List[PointCandidate] = []
            for i, (xc, yc) in enumerate(slots):
                candidates.append(PointCandidate(x=xc, y=yc, label=labels[i]))
            if not self.validate_candidate_layout(candidates, min_distance=self.canvas_short_side * 0.145):
                continue
            # Verify circle fits and the apex/arc endpoints are inside.
            if not self.circle_fits(center, radius, extra_padding=self.line_width + 4):
                continue
            for p in (arc_p1, arc_p2, apex):
                if not self.inside_canvas(p, padding=self.line_width):
                    break
            else:
                self.candidates = candidates
                self.correct_label = labels[target_idx]
                self.target_point = Point(slots[target_idx][0], slots[target_idx][1])
                return self.save_puzzle()
        raise RuntimeError("Failed to lay out inscribed_angle_arc puzzle")

    def _authored_observation(self) -> str:
        count_words = ("none", "one", "two", "three", "four")
        narrower = sum(1 for value in self.wedge_angles if value < self.central_deg - 1e-6)
        wider = sum(1 for value in self.wedge_angles if value > self.central_deg + 1e-6)
        parts = []
        if narrower:
            parts.append(f"{count_words[narrower]} {'is' if narrower == 1 else 'are'} visibly narrower")
        if wider:
            parts.append(f"{count_words[wider]} {'is' if wider == 1 else 'are'} visibly wider")
        comparison = " and ".join(parts)
        return (
            f"The circle sits in the upper part of the canvas with the blue arc strung between its two hollow "
            f"markers, and the filled apex on the opposite side of the rim, {self._canvas_region(self.apex)}, "
            f"opens {self._angle_size_phrase(self.inscribed_deg)} toward that arc. The central angle standing on "
            f"the same blue arc is twice the inscribed one, so the wedge to find must open "
            f"{self._angle_size_phrase(self.central_deg)}. Of the five upward-opening wedges in the bottom row, "
            f"{comparison} than that doubled opening."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "difficulty_variant": "inscribed_to_central_with_center_reveal",
            "circle_center": self.circle_center.to_list(),
            "radius": self.radius,
            "arc_endpoints": [self.arc_p1.to_list(), self.arc_p2.to_list()],
            "inscribed_apex": self.apex.to_list(),
            "inscribed_deg": self.inscribed_deg,
            "central_deg_target": self.central_deg,
            "candidate_angles_deg": list(self.wedge_angles),
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self._draw_reference_static(draw)
        if highlight_label:
            self._draw_hollow_center(draw)
            first_start, first_end = self.trim_segment(
                self.circle_center,
                self.arc_p1,
                start_offset=6.0,
                end_offset=6.0,
            )
            second_start, second_end = self.trim_segment(
                self.circle_center,
                self.arc_p2,
                start_offset=6.0,
                end_offset=6.0,
            )
            self.draw_line(draw, [first_start, first_end], fill=self.CONSTRUCTION_COLOR)
            self.draw_line(draw, [second_start, second_end], fill=self.CONSTRUCTION_COLOR)
        for cand, ang_deg, ang_dir in zip(self.candidates, self.wedge_angles, self.wedge_dirs):
            v = Point(cand.x, cand.y)
            self._draw_wedge(draw, v, ang_deg, self.wedge_ray_len, ang_dir)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

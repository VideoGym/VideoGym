"""Angle supplement puzzle generator.

A reference angle alpha is drawn in the upper half of the canvas. Five
candidate wedges A-E are drawn in a row in the lower half. Exactly one
candidate equals 180 - alpha (the supplement); the other four are
distractors with at least 12 deg separation.
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


class AngleSupplementGenerator(PointTargetPuzzleGenerator):
    """Find which candidate wedge equals 180 - alpha."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows one black reference angle drawn at a marked anchor vertex in the upper half, plus five small "
        "candidate wedges A-E in a row in the lower half. Identify the candidate whose wedge equals the supplement of the "
        "reference angle, that is 180 degrees minus the reference. Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/angle_supplement"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows one black reference angle in the upper half: two solid line segments of medium thickness meeting at a small filled black anchor vertex. "
        "Five small wedges of medium thickness sit in a row in the lower half, each opening upward; at the apex of each wedge sits a small white candidate marker A-E with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the reference angle and the row of candidate wedges unchanged. "
        "In the final state, only the candidate whose wedge equals 180 minus the reference changes to pale red fill with a dark red outline, while the other four markers stay white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

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

    def _draw_reference(self, draw) -> None:
        v = self.ref_vertex
        ray_len = self.ref_ray_len
        half = math.radians(self.ref_angle_deg) / 2.0
        a1 = self.ref_direction - half
        a2 = self.ref_direction + half
        p1 = self.point_on_ray(v, a1, ray_len)
        p2 = self.point_on_ray(v, a2, ray_len)
        self.draw_line(draw, [p1, v, p2])
        cx, cy = round(v.x), round(v.y)
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

    def _video_overlay_extras(self, draw) -> None:
        self._draw_reference(draw)

    def __init__(self, *args, opening_range=(40.0, 130.0), **kwargs):
        if not 0 < opening_range[0] <= opening_range[1] < 180:
            raise ValueError("angles must lie strictly between zero and 180 degrees")
        self.opening_range = opening_range
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        cw, ch = self.canvas_dimensions
        for _ in range(999):
            ref_angle = self._rng.uniform(*self.opening_range)
            target_angle = 180.0 - ref_angle
            min_gap = 12.0
            choices = [target_angle]
            attempt = 0
            # Both reference and supplement live in (40, 140); pick 4
            # distractors in the same band so visual scale matches.
            while len(choices) < 5 and attempt < 600:
                attempt += 1
                cand = self._rng.uniform(35.0, 145.0)
                if any(abs(cand - c) < min_gap for c in choices):
                    continue
                choices.append(cand)
            if len(choices) < 5:
                continue
            self._rng.shuffle(choices)
            target_idx = choices.index(target_angle)

            self.ref_angle_deg = ref_angle
            self.ref_vertex = Point(
                x=cw * self._rng.uniform(0.42, 0.58),
                y=ch * self._rng.uniform(0.20, 0.28),
            )
            self.ref_direction = math.radians(self._rng.uniform(80.0, 100.0))
            self.ref_ray_len = self.canvas_short_side * 0.18

            slots: List[Tuple[float, float]] = []
            left, top, right, bottom = self.canvas_bounds()
            usable = right - left - 30
            cell_w = usable / 4.0
            cell_y = ch * 0.74
            jitter_x = cell_w * 0.05
            jitter_y = ch * 0.04
            base_x = left + 15
            for i in range(5):
                cx = base_x + i * cell_w + self._rng.uniform(-jitter_x, jitter_x)
                cy = cell_y + self._rng.uniform(-jitter_y, jitter_y)
                slots.append((cx, cy))

            labels = list(self.option_labels)
            self._rng.shuffle(labels)

            wedge_angles = choices
            wedge_dirs = [
                math.radians(self._rng.uniform(265.0, 275.0)) for _ in range(5)
            ]
            wedge_ray_len = self.canvas_short_side * 0.10

            candidates: List[PointCandidate] = []
            for i, (cx, cy) in enumerate(slots):
                candidates.append(PointCandidate(x=cx, y=cy, label=labels[i]))
            if not self.validate_candidate_layout(candidates, min_distance=self.canvas_short_side * 0.16):
                continue

            self.candidates = candidates
            self.correct_label = labels[target_idx]
            self.target_point = Point(slots[target_idx][0], slots[target_idx][1])
            self.wedge_angles = wedge_angles
            self.wedge_dirs = wedge_dirs
            self.wedge_ray_len = wedge_ray_len
            return self.save_puzzle()
        raise RuntimeError("Failed to lay out angle_supplement puzzle")

    def _authored_observation(self) -> str:
        arms_heading = self._heading_phrase(
            math.cos(self.ref_direction), math.sin(self.ref_direction)
        )
        reference_size = self._angle_size_phrase(self.ref_angle_deg)
        supplement_size = self._angle_size_phrase(180.0 - self.ref_angle_deg)
        return (
            f"The reference angle sits at its filled anchor vertex {self._canvas_region(self.ref_vertex)}, "
            f"its two arms opening {arms_heading} and spanning {reference_size}. A supplement is whatever remains "
            f"of a straight line — 180 degrees minus the reference — so the matching wedge must open "
            f"{supplement_size}. Scanning the row of upward-opening wedges across the lower half, I look for the "
            "one whose spread completes the straight line, rejecting openings clearly tighter or wider than that "
            "remainder."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "difficulty_variant": "linear_pair_acute_or_obtuse",
            # ``reference_*`` is reserved for authored delivery artifacts by
            # the canonical scrubber.  These are puzzle-input measurements,
            # so keep them under an unambiguous source-geometry namespace.
            "source_angle_deg": self.ref_angle_deg,
            "source_vertex": self.ref_vertex.to_list(),
            "candidate_angles_deg": list(self.wedge_angles),
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self._draw_reference(draw)
        for cand, ang_deg, ang_dir in zip(self.candidates, self.wedge_angles, self.wedge_dirs):
            v = Point(cand.x, cand.y)
            self._draw_wedge(draw, v, ang_deg, self.wedge_ray_len, ang_dir)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

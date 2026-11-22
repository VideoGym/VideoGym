"""Regular octagon vertex puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord


class RegularOctagonVertexGenerator(PointTargetPuzzleGenerator):
    """Given consecutive vertices of a regular octagon, identify the next
    vertex.

    A regular octagon has equal sides and 135-degree interior angles. Given
    five consecutive sides (six visible vertices), the next vertex is uniquely
    determined.
    """

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows five connected line segments forming five consecutive sides of a regular octagon, plus five "
        "labeled candidate circles A-E. Identify which candidate is the next vertex that continues the regular octagon "
        "(all sides equal, all interior angles 135 degrees). Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/regular_octagon_vertex"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows five connected black line segments of medium thickness forming five consecutive sides of a regular octagon (equal sides, 135-degree interior angles), so six of the octagon's eight vertices are visible. "
        "Five candidate markers A-E are clustered near the hidden seventh vertex needed to continue the octagon; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the open path and the candidate markers, then draws three more black segments of medium thickness to extend the chain, the first ending at the correct candidate. "
        "In the final state, only the next-vertex marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        for _ in range(999):
            center = self.pick_target_point(0.32, padding=candidate_padding + self.canvas_short_side * 0.16)
            R = self._rng.uniform(self.canvas_short_side * 0.13, self.canvas_short_side * 0.22)
            base_angle = self._rng.uniform(0.0, math.tau)

            verts = [self.point_on_ray(center, base_angle + math.tau / 8 * i, R) for i in range(8)]
            if not all(self.inside_canvas(v, padding=self.line_width + 4) for v in verts):
                continue
            # Show 5 consecutive sides (6 vertices). Target = 7th vertex.
            start_idx = self._rng.randint(0, 7)
            shown = [verts[(start_idx + i) % 8] for i in range(6)]
            target = verts[(start_idx + 6) % 8]
            if not self.point_can_host_candidate(target):
                continue
            try:
                self.place_candidates(target)
            except RuntimeError:
                continue

            self.shown_path = tuple(shown)
            self.all_verts = tuple(verts)
            self.start_idx = start_idx
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid regular octagon vertex puzzle.")

    def _authored_observation(self) -> str:
        chain_end = self.shown_path[-1]
        next_direction = self._heading_phrase(
            self.target_point.x - chain_end.x, self.target_point.y - chain_end.y,
        )
        return (
            f"Five equal sides of the regular octagon are already drawn, and the open chain currently ends "
            f"{self._canvas_region(chain_end)}. Every corner of the octagon bends by the same gentle amount "
            f"toward the same side, so the chain continues with one more side of the shared length, heading "
            f"{next_direction} to the next corner, {self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "shown_vertices": [p.to_list() for p in self.shown_path],
            "all_vertices": [p.to_list() for p in self.all_verts],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, list(self.shown_path))
        if highlight_label:
            # Single continuous closing chain: shown_path[-1] -> target ->
            # next vertex -> shown_path[0]. (The last vertex before
            # closing becomes verts[(start_idx + 7) % 8].)
            v_after_target = self.all_verts[(self.start_idx + 7) % 8]
            remaining = [
                self.shown_path[-1],
                self.target_point,
                v_after_target,
                self.shown_path[0],
            ]
            self.draw_line(draw, remaining, join_start=True, join_end=True)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

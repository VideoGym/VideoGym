"""Spiral continuation puzzle generator."""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord


# Directions: right, down, left, up (90-degree turns clockwise)
_DX = [1, 0, -1, 0]
_DY = [0, 1, 0, -1]


class SpiralContinuationGenerator(PointTargetPuzzleGenerator):
    """Find the next endpoint in a growing square spiral."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows one black square-spiral polyline and five labeled candidate circles A-E. "
        "Rule: Continue from the visible tip with one more segment. It must turn 90 degrees in the same turn direction "
        "as every preceding corner and exceed the preceding segment by the same constant length increment seen "
        "throughout the spiral. Select the candidate at that next endpoint. Required Output: Return exactly one "
        "uppercase option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/spiral_continuation"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a black polyline of medium thickness forming a growing square spiral: each successive segment turns ninety degrees and is longer than the previous one by a constant increment. "
        "Five candidate markers A-E are clustered near the hidden endpoint of the next segment; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the spiral and the candidate markers, then draws one black extension segment of medium thickness from the spiral tip to the correct candidate, continuing the spiral pattern. "
        "In the final state, only the next-endpoint marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def __init__(self, *args, contracting=False, **kwargs):
        self.contracting = bool(contracting)
        if self.contracting:
            for field in (
                "DEFAULT_RULE_PROMPT",
                "DEFAULT_TI2V_PROMPT",
                "DEFAULT_SOLUTION_IMAGE_PROMPT",
            ):
                value = (
                    getattr(self, field)
                    .replace("outward", "inward")
                    .replace("longer", "shorter")
                    .replace("increasing", "decreasing")
                    .replace("increase", "decrease")
                    .replace("growing", "shrinking")
                    .replace(
                        "exceed the preceding segment",
                        "be shorter than the preceding segment",
                    )
                )
                setattr(self, field, value)
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            num_segments = self._rng.randint(5, 7)
            increment = self._rng.uniform(
                self.canvas_short_side * 0.030,
                self.canvas_short_side * 0.050,
            )
            base_len = self._rng.uniform(
                self.canvas_short_side * 0.035,
                self.canvas_short_side * 0.055,
            )
            if self.contracting:
                base_len += increment * (num_segments + 1)
                increment = -increment
            start = self.pick_target_point(
                0.30, padding=candidate_padding + self.canvas_short_side * 0.15
            )
            direction_start = self._rng.randint(0, 3)

            points: List[Point] = [start]
            valid = True
            for i in range(num_segments + 1):
                seg_len = base_len + increment * i
                d = (direction_start + i) % 4
                prev = points[-1]
                nxt = Point(
                    x=prev.x + _DX[d] * seg_len,
                    y=prev.y + _DY[d] * seg_len,
                )
                if not self.inside_canvas(nxt, padding=self.line_width + 4):
                    valid = False
                    break
                points.append(nxt)
            if not valid or len(points) < num_segments + 2:
                continue

            spiral_points = points[:-1]
            target_point = points[-1]

            if not self.point_can_host_candidate(target_point):
                continue

            try:
                self.place_candidates(target_point)
            except RuntimeError:
                continue

            self.spiral_points = spiral_points
            self.target_point = target_point
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid spiral continuation puzzle.")

    def _authored_observation(self) -> str:
        if self.contracting:
            return "The square spiral turns in the same direction at each corner while lengths decrease by one constant positive amount. Subtract that amount once more and follow the next turn."
        tip = self.spiral_points[-1]
        next_heading = self._heading_phrase(
            self.target_point.x - tip.x,
            self.target_point.y - tip.y,
        )
        return (
            f"The polyline winds outward as a square spiral: every corner turns the same way, and each leg runs a "
            f"bit longer than the one before. From the visible tip {self._canvas_region(tip)}, the next leg must "
            f"repeat that pattern — one more turn in the same rotational sense, one more constant increment of "
            f"length — carrying the path {next_heading}, out to its endpoint "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "spiral_points": [p.to_list() for p in self.spiral_points],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, self.spiral_points)
        if highlight_label:
            tip = self.spiral_points[-1]
            _, trimmed_end = self.trim_segment(
                tip, self.target_point,
                start_offset=0.0,
                end_offset=float(self.point_radius),
            )
            # Extension joins the spiral polyline at its endpoint tip
            # (butt cap + 90° turn = visible triangular gap without join).
            # join_start=True seals it with a disk at the tip.
            self.draw_line(draw, [tip, trimmed_end], join_start=True)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

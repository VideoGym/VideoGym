"""External ratio point puzzle generator."""

from __future__ import annotations

import math
from typing import Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class ExternalRatioPointGenerator(PointTargetPuzzleGenerator):
    """Find a point beyond a through-anchor from an explicit distance ratio."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows one filled black start anchor, one hollow through anchor, and five labeled "
        "candidate circles A-E. Rule: The target lies on the ray from the filled start anchor through and beyond the "
        "hollow anchor. It satisfies distance(start, target) = {R} times distance(through, target); equivalently, "
        "distance(through, target) equals distance(start, through) divided by {R_MINUS_1}. Select the candidate at that "
        "target. Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/external_ratio_point"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows one small filled black start anchor and one small hollow through anchor; these two anchors define a ray that starts at the filled anchor and passes through the hollow anchor. "
        "Five candidate markers A-E are clustered near the hidden target point beyond the hollow anchor; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The target point lies on that ray beyond the hollow anchor, and it satisfies distance(start anchor, target) = {R} * distance(through anchor, target). "
        "Equivalently, the distance from the hollow through anchor to the target equals the start-to-through anchor distance divided by {R_MINUS_1}. "
        "The video first holds the two anchors and the candidate markers, then draws a black segment of medium thickness from the filled start anchor through the hollow anchor to the correct candidate. "
        "In the final state, only the correct target marker changes to pale red fill with a dark red outline, while the other four markers remain white. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    START_ANCHOR_RADIUS = 7
    THROUGH_ANCHOR_RADIUS = 7
    RATIOS = (2, 3, 4, 5)
    HALF_LENGTH_RANGES = {
        2: (0.27, 0.31),
        3: (0.22, 0.28),
        4: (0.27, 0.31),
        5: (0.32, 0.37),
    }

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        segment_padding = max(candidate_padding, self.line_width + self.START_ANCHOR_RADIUS + 3)
        for _ in range(999):
            ratio = self._rng.choice(self.RATIOS)
            midpoint = self.pick_target_point(
                0.22 if ratio == 5 else 0.34, padding=segment_padding
            )
            angle = self._rng.uniform(0.0, math.tau)
            min_half_ratio, max_half_ratio = self.HALF_LENGTH_RANGES[ratio]
            min_half_length = self.canvas_short_side * min_half_ratio
            max_half_length = self.canvas_short_side * max_half_ratio

            try:
                start_anchor, target = self.sample_symmetric_segment(
                    midpoint,
                    angle,
                    min_half_length=min_half_length,
                    max_half_length=max_half_length,
                    padding=segment_padding,
                )
            except RuntimeError:
                continue

            through_anchor = Point(
                x=start_anchor.x + (target.x - start_anchor.x) * ((ratio - 1) / ratio),
                y=start_anchor.y + (target.y - start_anchor.y) * ((ratio - 1) / ratio),
            )
            if not self.point_can_host_candidate(target):
                continue
            if self.distance(through_anchor, target) < 64.0:
                continue
            if self.distance(start_anchor, through_anchor) < self.canvas_short_side * 0.25:
                continue

            try:
                self._place_ratio_candidates(start_anchor, through_anchor, target)
            except RuntimeError:
                continue

            self.start_anchor = start_anchor
            self.through_anchor = through_anchor
            self.ratio = ratio
            self.target_point = target

            return self.save_puzzle()
        raise RuntimeError("Failed to generate external ratio point puzzle.")

    def _format_sample_prompt(self, prompt: str) -> str:
        ratio = getattr(self, "ratio", None)
        if ratio is None:
            return str(prompt)
        return (
            str(prompt)
            .replace("{R_MINUS_1}", f"{ratio - 1:g}")
            .replace("{R}", f"{ratio:g}")
        )

    def _place_ratio_candidates(
        self,
        start_anchor: Point,
        through_anchor: Point,
        target: Point,
    ) -> None:
        # Every distractor sits on the ray the two anchors define, so no
        # marker can be dropped for being off it, and how many of them fall
        # behind the answer is drawn before they are placed -- a row built
        # symmetrically around the answer would hand over the answer as its
        # middle marker.
        anchor_clear = 32.0
        self.place_candidates_on_incidence(
            target,
            # The ray from the hollow anchor onward, not the whole line: a
            # marker behind the anchors is dismissible without the ratio.
            [Incidence.ray(through_anchor, target)],
            min_offset=32.0,
            min_separation=28.0,
            reject=lambda point: (
                self.distance(point, start_anchor) <= anchor_clear
                or self.distance(point, through_anchor) <= anchor_clear
            ),
        )

    def _authored_observation(self) -> str:
        ray_heading = self._heading_phrase(
            self.target_point.x - self.through_anchor.x,
            self.target_point.y - self.through_anchor.y,
        )
        if self.ratio == 2:
            span_relation = "must match the full anchor-to-anchor span"
        else:
            span_relation = (
                f"must equal the anchor-to-anchor span divided by {self.ratio - 1}"
            )
        return (
            f"The ray sets out from the filled anchor {self._canvas_region(self.start_anchor)}, passes through "
            f"the hollow anchor {self._canvas_region(self.through_anchor)}, and continues {ray_heading} beyond "
            f"it. With the stated factor {self.ratio}, the stretch from the hollow anchor out to the "
            f"required point {span_relation}, so I carry that length past the hollow anchor along the same line; "
            "every other center on that ray stops shorter or overshoots, breaking the stated ratio."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "start_anchor": self.start_anchor.to_list(),
            "through_anchor": self.through_anchor.to_list(),
            "ratio": self.ratio,
            "target_point": self.target_point.to_list(),
        }

    def _draw_start_anchor(self, draw) -> None:
        r = self.START_ANCHOR_RADIUS
        cx, cy = round(self.start_anchor.x), round(self.start_anchor.y)
        draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=(32, 32, 32),
            outline=(32, 32, 32),
            width=max(2, self.line_width),
        )

    def _draw_through_anchor(self, draw) -> None:
        self.draw_anchor_marker(draw, self.through_anchor, self.THROUGH_ANCHOR_RADIUS)

    def _draw_anchors(self, draw) -> None:
        self._draw_start_anchor(draw)
        self._draw_through_anchor(draw)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_anchors(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        if highlight_label:
            self.draw_line(draw, [self.start_anchor, self.target_point])
        self._draw_anchors(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

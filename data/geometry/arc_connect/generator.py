"""Generate masked circular-arc continuation puzzles."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from data.common.prompts import soften_precise_video_timing
from data.common.base import PathLike
from data.geometry.core.point_target_base import (
    PointCandidate,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.typed_answers import FIVE_CHOICE_CODEC

@dataclass
class CircleSpec:
    cx: float
    cy: float
    r: float

    def bbox(self) -> Tuple[int, int, int, int]:
        return (int(round(self.cx - self.r)), int(round(self.cy - self.r)),
                int(round(self.cx + self.r)), int(round(self.cy + self.r)))

    def to_dict(self) -> Dict[str, float]:
        return {"cx": self.cx, "cy": self.cy, "radius": self.r}

class ArcConnectGenerator(PointTargetPuzzleGenerator):
    DEFAULT_RULE_PROMPT = (
        "Input Image: The input is white image. An opaque light-gray vertical band with two gray edge lines hides "
        "the middle of the construction. One thick near-black circular-arc fragment is visible on the left. Five "
        "thick charcoal circular-arc fragments are visible on the right, each paired with exactly one fixed "
        "circular marker labeled A, B, C, D, or E. A marker identifies the adjacent right fragment only; use the "
        "marker center, outline, fill, and letter only as an identifier, not as part of a curve. Rule: Mentally "
        "continue the left fragment and each right fragment through the band as circular arcs. The correct option "
        "is the unique right fragment whose continuation is on the same mathematical circle as the left fragment, "
        "so position, tangent direction, and curvature all remain continuous across the hidden region. A fragment "
        "that merely points roughly toward the left one but has a displaced centerline, mismatched tangent, or "
        "different curvature is not correct. The generator guarantees exactly one match. Required Output: Return "
        "exactly one uppercase option label in the form \"Answer: X.\""
    )
    DEFAULT_OUTPUT_DIR = "outputs/geometry/arc_connect"
    EXPECTED_VIDEO_FRAMES = 29
    MASK_FILL = (226, 229, 233)
    MASK_EDGE = (132, 139, 148)
    ARC_COLOR = (38, 38, 42)
    LEFT_ARC_COLOR = (20, 20, 23)
    STATE_TRACE = [
        "input_hold",
        "mask_narrowing_and_arc_reveal",
        "revealed_geometry_hold",
        "correct_candidate_highlight",
        "final_hold",
    ]
    DEFAULT_TI2V_PROMPT = (
        "Input Image: The input is white image. An opaque light-gray vertical band with two gray edge lines hides "
        "the middle of the construction. One thick near-black circular-arc fragment is visible on the left. Five "
        "thick charcoal circular-arc fragments are visible on the right, each paired with exactly one fixed "
        "circular marker labeled A, B, C, D, or E. A marker identifies the adjacent right fragment only; use the "
        "marker center, outline, fill, and letter only as an identifier, not as part of a curve. Rule: Mentally "
        "continue the left fragment and each right fragment through the band as circular arcs. The correct option "
        "is the unique right fragment whose continuation is on the same mathematical circle as the left fragment, "
        "so position, tangent direction, and curvature all remain continuous across the hidden region. A fragment "
        "that merely points roughly toward the left one but has a displaced centerline, mismatched tangent, or "
        "different curvature is not correct. The generator guarantees exactly one match. Required Output: "
        "Preserve the input scene, then commit the solved construction and selected candidate. Use a fixed "
        "camera, preserve every unchanged object, and show each semantic construction or commit in rule order "
        "without teleportation, hard cuts, full-frame crossfades, flicker, disappearance, or identity changes."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Use the supplied input image as the fixed base and produce exactly final solution image. Remove the "
        "entire light-gray central mask band and both of its gray edge lines, restoring a clean white background "
        "in their former locations. Reveal the hidden portions of all five right circular arcs from the former "
        "right mask edge to the canvas center, and reveal the hidden portion of the one left arc from the former "
        "left mask edge to the center. Preserve every visible arc's original stroke width, color, curvature, "
        "position, and identity. The unique matching left and right portions must meet without a gap, kink, "
        "overlap, or thickness change to form one continuous circle arc. The other four right portions must end "
        "at the center and must not acquire left-side continuations. Change only the matching candidate marker to "
        "pale-red fill with a dark-red outline while keeping its black letter readable. Keep the other four "
        "markers white with their original dark outlines and black letters. Preserve all marker centers, sizes, "
        "labels, and all already visible arc portions, the white background outside the former mask, scale, "
        "framing, and camera. No object moves, falls, compacts, refills, reorders, disappears, duplicates, or "
        "changes identity, and no extra line, curve, highlight, legend, answer banner, or text may appear."
    )

    def __init__(
        self,
        output_dir: PathLike = None,
        *,
        canvas_width: int = 512,
        aspect: Optional[float] = None,
        seed: Optional[int] = None,
        ti2v_prompt: Optional[str] = None,
        record_video: bool = False,
        point_radius: Optional[int] = None,
        line_width: Optional[int] = None,
    ) -> None:
        custom_ti2v_prompt = ti2v_prompt
        super().__init__(
            output_dir=output_dir,
            canvas_width=canvas_width,
            aspect=aspect,
            seed=seed,
            ti2v_prompt=ti2v_prompt,
            record_video=record_video,
            point_radius=point_radius,
            line_width=line_width,
        )
        self.mask_fraction = 0.35
        self.arc_span_deg = 20.0
        self.circles: List[CircleSpec] = []
        self.mask_x_center: float = 0.0
        self.mask_width: float = 0.0
        self.mask_top: float = 0.0
        self.mask_bottom: float = 0.0
        self.left_circle: Optional[CircleSpec] = None
        self.solution_image_prompt = soften_precise_video_timing(self.DEFAULT_SOLUTION_IMAGE_PROMPT)
        self.ti2v_prompt = soften_precise_video_timing(
            str(custom_ti2v_prompt or self.DEFAULT_TI2V_PROMPT)
        )
        self._geometry_input_description = (
            "A vertical opaque band hides the middle of one left circular-arc fragment and five labeled right "
            "arc fragments. Each nearby marker identifies its right fragment but is not part of the curve."
        )
        self._geometry_rule_description = (
            "Select the unique right fragment whose continuation through the hidden band lies on the same circle "
            "as the left fragment, preserving position, tangent direction, and curvature."
        )
        self._geometry_solution_description = (
            "Preserve every visible arc and marker. Narrow the occluding band symmetrically so the already-existing "
            "hidden continuations are revealed from both sides, then mark only the right fragment that joins the "
            "left fragment without a gap, kink, or curvature change."
        )
        self._geometry_video_description = (
            "Hold the unchanged input. Narrow the gray band symmetrically toward its center; as the two edges move "
            "inward, reveal the already-existing left and right arc portions from their respective mask edges. "
            "Never clear the mask first or draw hidden arcs serially. Hold the completed curves, then smoothly mark "
            "the unique matching fragment."
        )

    def _arc_stays_inside(
        self,
        circle: CircleSpec,
        theta_start: float,
        theta_end: float,
        *,
        padding: float,
        samples: int = 21,
    ) -> bool:
        width, height = self.canvas_dimensions
        for idx in range(samples + 1):
            ratio = idx / samples
            theta = theta_start + (theta_end - theta_start) * ratio
            px, py = self._point_on_circle(circle, theta)
            if not (padding <= px <= width - padding and padding <= py <= height - padding):
                return False
        return True

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        width, height = self.canvas_dimensions
        self.mask_width = width * self.mask_fraction
        self.mask_x_center = width / 2.0
        self.mask_top = 0.0
        self.mask_bottom = float(height)
        span_rad = math.radians(self.arc_span_deg)
        margin = self.margin
        candidate_pad_x, candidate_pad_y = self._candidate_safe_padding()
        candidate_gap = max(self.minimum_candidate_spacing(scale=0.96), height * 0.09)
        stroke = max(3, int(round(min(width, height) * 0.015)))

        for _ in range(1000):
            # Restore original radius logic: uniform(0.38, 0.55) * min_dimension
            radius = self.rng.uniform(0.4, 0.52) * min(width, height)

            mask_left = self.mask_x_center - self.mask_width/2
            mask_right = self.mask_x_center + self.mask_width/2

            cx = self.rng.uniform(self.mask_x_center - 0.4 * radius, self.mask_x_center + 0.4 * radius)

            # Validate strict crossing: check if valid Y exists at mask boundaries
            check_x_left = mask_left
            check_x_right = mask_right
            if abs(check_x_left - cx) >= radius or abs(check_x_right - cx) >= radius:
                continue

            gap = self.rng.uniform(candidate_gap, candidate_gap * 1.15)

            # We want 5 candidate arcs. Let's space their crossings at mask_right evenly.
            # Pick center crossing Y for the middle candidate (index 2)
            base_crossing_y = self.rng.uniform(
                self.mask_top + 2.2 * gap,
                self.mask_bottom - 2.2 * gap,
            )

            # Generate 5 crossing Ys
            crossing_ys = [base_crossing_y + (i - 2) * gap for i in range(5)]

            temp_circles = []
            temp_candidates_points = []
            valid_group = True

            sign = 1 if self.rng.choice([True, False]) else -1

            for y_cross in crossing_ys:
                dx = mask_right - cx
                dy_term = math.sqrt(max(0, radius**2 - dx**2))

                cy = y_cross + sign * dy_term

                c = CircleSpec(cx, cy, radius)

                # Now validate endpoints
                # Requirement: "visible part has 20 degrees"
                theta_base = math.atan2(y_cross - cy, mask_right - cx)

                test_pt = self._point_on_circle(c, theta_base + 0.01)
                direction = 1.0 if test_pt[0] > mask_right else -1.0

                theta_end = theta_base + direction * span_rad

                # Validate endpoints y-coordinates
                p_start = (mask_right, y_cross)
                p_end = self._point_on_circle(c, theta_end)

                # Check right arc containment
                if not (self.mask_top < p_start[1] < self.mask_bottom): valid_group = False
                if not (self.mask_top < p_end[1] < self.mask_bottom): valid_group = False
                if p_end[0] <= mask_right: valid_group = False
                if p_end[0] + candidate_pad_x + 8 >= width - margin: valid_group = False

                cross_ys_center = self._get_y_at_x(c, self.mask_x_center)
                if not cross_ys_center:
                    valid_group = False
                else:
                    y_cross_center = min(cross_ys_center, key=lambda y: abs(y - y_cross))
                    theta_center = math.atan2(y_cross_center - c.cy, self.mask_x_center - c.cx)
                    if not self._arc_stays_inside(c, theta_center, theta_end, padding=stroke + 2):
                        valid_group = False

                # Check left arc containment (theoretical left arc for this circle)
                cross_ys_left = self._get_y_at_x(c, mask_left)
                if not cross_ys_left:
                    valid_group = False
                else:
                    y_cross_left = min(cross_ys_left, key=lambda y: abs(y - y_cross))
                    theta_left_base = math.atan2(y_cross_left - c.cy, mask_left - c.cx)

                    test_pt_left = self._point_on_circle(c, theta_left_base - 0.01)
                    dir_left = -1.0 if test_pt_left[0] < mask_left else 1.0
                    theta_left_end = theta_left_base + dir_left * span_rad

                    p_left_start = (mask_left, y_cross_left)
                    p_left_end = self._point_on_circle(c, theta_left_end)

                    if not (self.mask_top < p_left_start[1] < self.mask_bottom): valid_group = False
                    if not (self.mask_top < p_left_end[1] < self.mask_bottom): valid_group = False
                    if p_left_end[0] >= mask_left: valid_group = False
                    if valid_group and not self._arc_stays_inside(c, theta_center, theta_left_end, padding=stroke + 2):
                        valid_group = False

                if not valid_group: break

                temp_circles.append(c)
                temp_candidates_points.append(p_end)


            if valid_group:
                self.circles = temp_circles
                labels = ["A", "B", "C", "D", "E"]
                label_offset = max(candidate_pad_x + 6.0, 22.0)
                self.candidates = []
                for pt, label in zip(temp_candidates_points, labels):
                    candidate_x = pt[0] + label_offset
                    if not self._point_within_canvas_coords(
                        candidate_x,
                        pt[1],
                        padding_x=candidate_pad_x,
                        padding_y=candidate_pad_y,
                    ):
                        valid_group = False
                        break
                    candidate = PointCandidate(
                        x=candidate_x,
                        y=pt[1],
                        label=label,
                    )
                    self.candidates.append(candidate)
                if not valid_group:
                    continue
                if not self.validate_candidate_layout(
                    self.candidates, min_distance=self.minimum_candidate_spacing(scale=0.92),
                ):
                    continue
                self.correct_label = labels[self.rng.randint(0, 4)]
                correct_index = labels.index(self.correct_label)
                self.left_circle = self.circles[correct_index]
                break
        else:
             raise RuntimeError("Failed to generate valid geometry")

        # Create unique ID and save
        return self.save_puzzle()

    def _get_y_at_x(self, c: CircleSpec, x: float) -> Optional[List[float]]:
        dist_x = abs(x - c.cx)
        if dist_x > c.r:
            return None
        dy = math.sqrt(c.r**2 - dist_x**2)
        return [c.cy - dy, c.cy + dy]

    def _point_on_circle(self, c: CircleSpec, theta: float) -> Tuple[float, float]:
        return (c.cx + c.r * math.cos(theta), c.cy + c.r * math.sin(theta))


    def _continuity_measurements(self) -> List[Dict[str, Any]]:
        if self.left_circle is None:
            raise RuntimeError("arc_connect left circle is not initialized")
        source = self.left_circle
        measurements: List[Dict[str, Any]] = []
        ordered_candidates = sorted(self.candidates, key=lambda item: item.label)
        for candidate, circle in zip(ordered_candidates, self.circles):
            center_offset = math.hypot(circle.cx - source.cx, circle.cy - source.cy)
            radius_offset = abs(circle.r - source.r)
            residual = math.hypot(center_offset, radius_offset)
            measurements.append(
                {
                    "label": candidate.label,
                    "marker_center": [candidate.x, candidate.y],
                    "relative_vertical_position": {
                        "A": "uppermost",
                        "B": "upper-middle",
                        "C": "middle",
                        "D": "lower-middle",
                        "E": "lowermost",
                    }[candidate.label],
                    "circle_center": [circle.cx, circle.cy],
                    "radius": circle.r,
                    "center_offset_pixels": center_offset,
                    "radius_offset_pixels": radius_offset,
                    "same_circle_residual_pixels": residual,
                }
            )
        return measurements

    def _reasoning_text(self) -> str:
        if self.left_circle is None:
            raise RuntimeError("arc_connect left circle is not initialized")
        width, height = self.canvas_dimensions
        mask_left = self.mask_x_center - self.mask_width / 2.0
        mask_right = self.mask_x_center + self.mask_width / 2.0
        correct_candidate = next(item for item in self.candidates if item.label == self.correct_label)
        right_crossings = self._get_y_at_x(self.left_circle, mask_right) or []
        if not right_crossings:
            raise RuntimeError("left circle no longer crosses the right mask boundary")
        right_y = min(right_crossings, key=lambda value: abs(value - correct_candidate.y))
        center_crossings = self._get_y_at_x(self.left_circle, self.mask_x_center) or []
        if not center_crossings:
            raise RuntimeError("left circle no longer crosses the mask center")
        center_y = min(center_crossings, key=lambda value: abs(value - right_y))
        left_crossings = self._get_y_at_x(self.left_circle, mask_left) or []
        if not left_crossings:
            raise RuntimeError("left circle no longer crosses the mask boundary")
        left_y = min(left_crossings, key=lambda value: abs(value - center_y))
        vertical_region = "upper" if left_y < height * 0.4 else "lower" if left_y > height * 0.6 else "middle"
        bend = "downward" if self.left_circle.cy > left_y else "upward"
        measurements = self._continuity_measurements()
        comparisons = []
        for item in measurements:
            residual = float(item["same_circle_residual_pixels"])
            if residual <= 1e-6:
                verdict = "same center and radius, so position, tangent, and curvature remain continuous"
            else:
                direction = self._direction_name(
                    float(item["circle_center"][0]) - self.left_circle.cx,
                    float(item["circle_center"][1]) - self.left_circle.cy,
                )
                verdict = (
                    f"its implied circle center is shifted {direction}, so the hidden continuation changes "
                    "position or tangent"
                )
            comparisons.append(
                f"{item['label']} is the {item['relative_vertical_position']} right fragment: {verdict}"
            )
        return (
            f"The visible left fragment reaches the mask in the {vertical_region} part of the image "
            f"and bends {bend}; continuing that circular bend fixes both its hidden centerline and tangent. "
            + "; ".join(comparisons)
            + f". Only {self.correct_label} preserves the same circle, so it is the unique smooth continuation."
            + f"\n\n{FIVE_CHOICE_CODEC.format(self.correct_label)}"
        )


    def _authored_observation(self) -> str:
        if self.left_circle is None:
            raise RuntimeError("arc_connect left circle is not initialized")
        height = float(self.canvas_dimensions[1])
        mask_left = self.mask_x_center - self.mask_width / 2.0
        mask_right = self.mask_x_center + self.mask_width / 2.0
        correct_candidate = next(item for item in self.candidates if item.label == self.correct_label)
        right_y = min(
            self._get_y_at_x(self.left_circle, mask_right) or [correct_candidate.y],
            key=lambda value: abs(value - correct_candidate.y),
        )
        center_y = min(
            self._get_y_at_x(self.left_circle, self.mask_x_center) or [right_y],
            key=lambda value: abs(value - right_y),
        )
        left_y = min(
            self._get_y_at_x(self.left_circle, mask_left) or [center_y],
            key=lambda value: abs(value - center_y),
        )
        vertical_region = "upper" if left_y < height * 0.4 else "lower" if left_y > height * 0.6 else "middle"
        bend = "downward" if self.left_circle.cy > left_y else "upward"
        return (
            f"The near-black fragment on the left meets the gray band in the {vertical_region} part of the "
            f"canvas, bending {bend} as it slips behind the occluder; each lettered marker only names its "
            f"adjacent right fragment, so the curves themselves carry the evidence. Continuing that same circular "
            f"bend across the band fixes the hidden centerline, tangent direction, and curvature the arc must "
            f"re-emerge with, and exactly one of the five right fragments lies on that same circle — the rest sit "
            f"on visibly shifted or differently curved circles."
        )

    def build_record_extra(self) -> Dict[str, Any]:
        if self.left_circle is None:
            raise RuntimeError("arc_connect left circle is not initialized")
        measurements = self._continuity_measurements()
        derived = min(measurements, key=lambda item: float(item["same_circle_residual_pixels"]))
        structured = {
            "left_circle": self.left_circle.to_dict(),
            "candidate_circles": [
                {"label": candidate.label, **circle.to_dict()}
                for candidate, circle in zip(sorted(self.candidates, key=lambda item: item.label), self.circles)
            ],
            "candidate_measurements": measurements,
            "mask": {
                "center_x": self.mask_x_center,
                "width": self.mask_width,
                "left_x": self.mask_x_center - self.mask_width / 2.0,
                "right_x": self.mask_x_center + self.mask_width / 2.0,
                "top_y": self.mask_top,
                "bottom_y": self.mask_bottom,
            },
            "arc_span_degrees": self.arc_span_deg,
            "correct_option": self.correct_label,
            "derived_correct_option": str(derived["label"]),
            "multiple_solution_policy": "generator_guarantees_one_and_only_one_candidate_circle_equals_the_left_circle",
            "state_trace": list(self.STATE_TRACE),
            "answer_schema": {
                "terminal_text_regex": FIVE_CHOICE_CODEC.full_pattern,
                "requires_sample_specific_reasoning_before_terminal_answer": True,
                "terminal_text_contains_exact_answer": True,
                "no_block_after_terminal_image": True,
            },
            "verifier_contract": {
                "profile": "arc_connect_multimodal_semantic_hard_gate_v1",
                "judged_artifacts": ["text", "final_image", "full_video"],
                "strict_binary_correctness": True,
                "performance_policy": "latency_and_throughput_are_diagnostics_only",
            },
            "render_geometry": {
                "canvas_dimensions": list(self.canvas_dimensions),
                "point_radius": self.point_radius,
                "line_width": max(3, int(round(min(self.canvas_dimensions) * 0.015))),
                "colors": {
                    "background": [255, 255, 255],
                    "mask_fill": list(self.MASK_FILL),
                    "mask_edge": list(self.MASK_EDGE),
                    "right_arcs": list(self.ARC_COLOR),
                    "left_arc": list(self.LEFT_ARC_COLOR),
                    "selected_outline": list(self.CANDIDATE_HIGHLIGHT_COLOR),
                    "selected_fill": list(self.CANDIDATE_HIGHLIGHT_FILL),
                },
            },
            "expected_video_frame_count": self.EXPECTED_VIDEO_FRAMES,
            "video_fps": 16,
        }
        return {
            "task_name": "arc_connect",
            "task_family": "geometry",
            "structured_answer": structured,
            "task_contract": {
                "logic": {
                    "match_rule": "same_circle_position_tangent_and_curvature_continuity",
                    "candidate_marker_semantics": "identifier_only",
                    "multiple_solution_policy": structured["multiple_solution_policy"],
                },
                "answer": {
                    "parser_profile": "sample_specific_reasoning_then_terminal_uppercase_option",
                },
                "vision": {
                    "mask_fate": "band_narrows_symmetrically_while_existing_arcs_are_revealed_from_both_sides",
                    "nonmatching_arc_fate": "each_right_continuation_is_revealed_only_where_the_shrinking_band_uncovers_it",
                    "unchanged_objects": "all_arc_geometry_candidate_centers_labels_background_scale_and_camera",
                    "gravity": "none",
                },
                "verifier": structured["verifier_contract"],
            },
        }

    def save_puzzle(self) -> PointTargetPuzzleRecord:
        return super().save_puzzle()

    @staticmethod
    def _blend_color(start: Sequence[int], end: Sequence[int], progress: float) -> Tuple[int, int, int]:
        progress = max(0.0, min(1.0, progress))
        return tuple(
            int(round(float(a) + (float(b) - float(a)) * progress))
            for a, b in zip(start, end)
        )

    def _draw_candidates_with_progress(self, draw: Any, progress: float) -> None:
        if progress <= 0.0:
            self.draw_candidates(draw, highlight_label=None)
            return
        if progress >= 1.0:
            self.draw_candidates(draw, highlight_label=self.correct_label)
            return
        ss = draw.factor
        raw_draw = draw.raw
        font = ImageFont.load_default(round(self.CANDIDATE_LABEL_FONT_SIZE * ss))
        point_radius = round(self.point_radius * ss)
        fill = self._blend_color(self.CANDIDATE_BASE_FILL, self.CANDIDATE_HIGHLIGHT_FILL, progress)
        outline = self._blend_color(self.CANDIDATE_OUTLINE_COLOR, self.CANDIDATE_HIGHLIGHT_COLOR, progress)
        for candidate in sorted(self.candidates, key=lambda item: item.label):
            cx = round(candidate.x * ss)
            cy = round(candidate.y * ss)
            bbox = (cx - point_radius, cy - point_radius, cx + point_radius, cy + point_radius)
            selected = candidate.label == self.correct_label
            raw_draw.ellipse(
                bbox,
                fill=fill if selected else self.CANDIDATE_BASE_FILL,
                outline=outline if selected else self.CANDIDATE_OUTLINE_COLOR,
                width=round(self.CANDIDATE_OUTLINE_WIDTH * ss),
            )
            text_bbox = font.getbbox(candidate.label)
            tx = cx - (text_bbox[0] + text_bbox[2]) // 2
            ty = cy - (text_bbox[1] + text_bbox[3]) // 2
            raw_draw.text((tx, ty), candidate.label, fill=self.CANDIDATE_TEXT_COLOR, font=font)

    def _render_state(
        self,
        *,
        mask_factor: float,
        highlight_progress: float,
    ) -> Image.Image:
        width, height = self.canvas_dimensions
        draw, base = self.get_draw_base()
        span_rad = math.radians(self.arc_span_deg)
        stroke = max(3, int(round(min(width, height) * 0.015)))
        mask_left = self.mask_x_center - self.mask_width / 2
        mask_right = self.mask_x_center + self.mask_width / 2
        mask_factor = max(0.0, min(1.0, float(mask_factor)))
        current_width = self.mask_width * mask_factor
        visible_left = self.mask_x_center - current_width / 2.0
        visible_right = self.mask_x_center + current_width / 2.0

        for index, circle in enumerate(self.circles):
            right_crossings = self._get_y_at_x(circle, mask_right) or []
            if not right_crossings:
                continue
            right_y = min(
                right_crossings,
                key=lambda value: abs(value - self.candidates[index].y),
            )
            right_angle = math.atan2(right_y - circle.cy, mask_right - circle.cx)
            test_x, _ = self._point_on_circle(circle, right_angle + 0.01)
            right_direction = 1.0 if test_x > mask_right else -1.0
            right_outer = right_angle + right_direction * span_rad

            if mask_factor > 0.01 and abs(visible_right - circle.cx) < circle.r:
                visible_crossings = self._get_y_at_x(circle, visible_right) or []
                if visible_crossings:
                    visible_y = min(
                        visible_crossings,
                        key=lambda value: abs(value - right_y),
                    )
                    right_start = math.atan2(
                        visible_y - circle.cy,
                        visible_right - circle.cx,
                    )
                else:
                    right_start = right_angle
            else:
                center_crossings = self._get_y_at_x(circle, self.mask_x_center) or []
                if not center_crossings:
                    continue
                center_y = min(center_crossings, key=lambda value: abs(value - right_y))
                right_start = math.atan2(
                    center_y - circle.cy,
                    self.mask_x_center - circle.cx,
                )
            self._draw_arc_safe(
                draw,
                circle,
                math.degrees(right_start),
                math.degrees(right_outer),
                self.ARC_COLOR,
                stroke,
            )

            if self.candidates[index].label != self.correct_label:
                continue
            center_crossings = self._get_y_at_x(circle, self.mask_x_center) or []
            left_crossings = self._get_y_at_x(circle, mask_left) or []
            if not center_crossings or not left_crossings:
                continue
            center_y = min(center_crossings, key=lambda value: abs(value - right_y))
            left_y = min(left_crossings, key=lambda value: abs(value - center_y))
            left_angle = math.atan2(left_y - circle.cy, mask_left - circle.cx)
            test_left_x, _ = self._point_on_circle(circle, left_angle - 0.01)
            left_direction = -1.0 if test_left_x < mask_left else 1.0
            left_outer = left_angle + left_direction * span_rad
            if mask_factor > 0.01 and abs(visible_left - circle.cx) < circle.r:
                visible_crossings = self._get_y_at_x(circle, visible_left) or []
                if visible_crossings:
                    visible_y = min(
                        visible_crossings,
                        key=lambda value: abs(value - center_y),
                    )
                    left_start = math.atan2(
                        visible_y - circle.cy,
                        visible_left - circle.cx,
                    )
                else:
                    left_start = left_angle
            else:
                left_start = math.atan2(
                    center_y - circle.cy,
                    self.mask_x_center - circle.cx,
                )
            self._draw_arc_safe(
                draw,
                circle,
                math.degrees(left_start),
                math.degrees(left_outer),
                self.LEFT_ARC_COLOR,
                stroke,
            )

        if mask_factor > 1e-6:
            mask_fill = self._blend_color((255, 255, 255), self.MASK_FILL, mask_factor)
            mask_edge = self._blend_color((255, 255, 255), self.MASK_EDGE, mask_factor)
            draw.rectangle(
                (visible_left, 0.0, visible_right, float(height)),
                fill=mask_fill,
            )
            edge_width = max(2, int(round(min(width, height) * 0.006)))
            draw.line([(visible_left, 0.0), (visible_left, float(height))], fill=mask_edge, width=edge_width)
            draw.line([(visible_right, 0.0), (visible_right, float(height))], fill=mask_edge, width=edge_width)

        self._draw_candidates_with_progress(draw, highlight_progress)
        return base

    def _render(self, highlight_label: Optional[str], mask_factor: Optional[float] = None) -> Image.Image:
        solved = bool(highlight_label)
        if mask_factor is None:
            mask_factor = 0.0 if solved else 1.0
        return self._render_state(
            mask_factor=max(0.0, min(1.0, float(mask_factor))),
            highlight_progress=1.0 if solved else 0.0,
        )

    def _render_recorded_solution_image(self) -> Image.Image:
        """Render the destructive mask removal from task semantics.

        The shared point-target command replay is additive by design, whereas
        this task's answer must remove the opaque band.  Re-rendering the
        solved scene preserves every already visible arc's original identity
        and color while exposing exactly the previously hidden portions.
        """

        return self._downscale(
            self._render_state(mask_factor=0.0, highlight_progress=1.0)
        )

    def _draw_arc_safe(self, draw: ImageDraw.ImageDraw, c: CircleSpec, d1: float, d2: float, color: Tuple[int, int, int], w: int) -> None:
        d1, d2 = d1 % 360, d2 % 360
        mn, mx = min(d1, d2), max(d1, d2)
        if mx - mn > 180:
            start, end = mx, mn
        else:
            start, end = mn, mx
        draw.arc(c.bbox(), start=start, end=end, fill=color, width=w)

    def save_video_solution(self, pid: str) -> Optional[int]:
        frames: List[Image.Image] = []

        def frame(
            mask_factor: float,
            highlight_progress: float = 0.0,
        ) -> Image.Image:
            return self._downscale(
                self._render_state(
                    mask_factor=mask_factor,
                    highlight_progress=highlight_progress,
                )
            )

        input_frame = frame(1.0)
        frames.extend(input_frame.copy() for _ in range(4))
        frames.extend(frame(1.0 - index / 8.0) for index in range(1, 9))
        revealed = frame(0.0)
        frames.extend(revealed.copy() for _ in range(4))
        frames.extend(frame(0.0, index / 4.0) for index in range(1, 5))
        final_frame = frame(0.0, 1.0)
        frames.extend(final_frame.copy() for _ in range(9))
        if len(frames) != self.EXPECTED_VIDEO_FRAMES or len(frames) % 4 != 1:
            raise RuntimeError(f"arc_connect produced invalid frame count {len(frames)}")
        video_path = self.solution_dir / f"{pid}_solution.webm"
        arrays = [np.asarray(item.convert("RGB"), dtype=np.uint8) for item in frames]
        if not encode_rgb_frames_to_webm(arrays, video_path, fps=16):
            raise RuntimeError("Failed to encode arc_connect semantic video")
        return len(frames)

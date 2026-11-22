"""Rule-complete circle-tangent-line point-target generator."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.common.prompts import soften_precise_video_timing
from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.typed_answers import FIVE_CHOICE_CODEC


class CircleTangentLineGenerator(PointTargetPuzzleGenerator):
    """Select the candidate center on the tangent ray at a marked circle point."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: The input is white image containing exactly one large unfilled circle with a medium-width "
        "black outline, one small hollow black-outlined contact marker whose center lies on that circumference, "
        "and five separate circular candidates labeled A, B, C, D, and E outside the circle near one tangent "
        "direction. Infer the circle center from the circular outline. The mathematical radius is the segment "
        "from that center to the center of the hollow contact marker. A tangent at the contact is the infinite "
        "straight line through the contact that is perpendicular to that radius. Of its two opposite rays, use "
        "the ray that starts at the contact and points toward the A-E candidate cluster; all five candidate "
        "centers lie in that forward half-plane. A candidate counts only by the center of its marker: the marker "
        "outline, fill, and letter glyph have no geometric extent. Exactly one candidate center lies on the "
        "selected tangent ray, and every wrong center has a visible nonzero perpendicular offset from it. Do not "
        "treat proximity to the circle, a candidate outline touching an imagined line, or the opposite tangent "
        "ray as a match. Rule: Select the unique labeled candidate whose center satisfies the stated geometric "
        "relation. Required Output: Return exactly one uppercase option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/circle_tangent_line"
    CONSTRUCTION_COLOR: Tuple[int, int, int] = (20, 151, 188)
    RIGHT_ANGLE_SIZE = 24.0
    MIN_RUNNER_UP_OFFSET = 18.0
    EXPECTED_VIDEO_FRAMES = 29
    STATE_TRACE = [
        "input_hold",
        "radius_growth",
        "tangent_growth",
        "right_angle_marker_growth",
        "correct_candidate_highlight",
        "final_hold",
    ]

    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Use the supplied input image as the fixed base and produce exactly final solution image. Preserve the "
        "white background, full original black circle outline, hollow contact marker, all five candidate centers, "
        "marker sizes, outlines, labels, spacing, scale, framing, and camera. Add a small cyan-blue center dot at "
        "the inferred center of the circle and a medium-width cyan-blue radius segment from that center to the "
        "contact-marker center. From the same contact center add one medium-width cyan-blue tangent segment along "
        "the perpendicular ray toward the candidate cluster until it reaches the center of the unique correct "
        "candidate; the contact and candidate markers are drawn above the segments, so their fills may cover the "
        "segment endpoints. Add one small cyan-blue square-corner mark between the inward radius direction and "
        "the tangent direction at the contact to encode the 90-degree angle. Change only the correct candidate to "
        "pale-red fill with a dark-red outline while keeping its black uppercase letter readable. Keep the other "
        "four candidates white with their original dark outlines and black letters. No source object moves, "
        "rotates, falls, compacts, refills, reorders, disappears, duplicates, changes identity, or changes size. "
        "Add no second tangent, extension beyond the selected candidate, extra center or contact marker, extra "
        "highlight, legend, formula, answer banner, or other text."
    )
    DEFAULT_TI2V_PROMPT = (
        "Input Image: The input is white image containing exactly one large unfilled circle with a medium-width "
        "black outline, one small hollow black-outlined contact marker whose center lies on that circumference, "
        "and five separate circular candidates labeled A, B, C, D, and E outside the circle near one tangent "
        "direction. Infer the circle center from the circular outline. The mathematical radius is the segment "
        "from that center to the center of the hollow contact marker. A tangent at the contact is the infinite "
        "straight line through the contact that is perpendicular to that radius. Of its two opposite rays, use "
        "the ray that starts at the contact and points toward the A-E candidate cluster; all five candidate "
        "centers lie in that forward half-plane. A candidate counts only by the center of its marker: the marker "
        "outline, fill, and letter glyph have no geometric extent. Exactly one candidate center lies on the "
        "selected tangent ray, and every wrong center has a visible nonzero perpendicular offset from it. Do not "
        "treat proximity to the circle, a candidate outline touching an imagined line, or the opposite tangent "
        "ray as a match. Rule: Select the unique labeled candidate whose center satisfies the stated geometric "
        "relation. Required Output: Preserve the input scene, then commit the solved construction and selected "
        "candidate. Use a fixed camera, preserve every unchanged object, and show each semantic construction or "
        "commit in rule order without teleportation, hard cuts, full-frame crossfades, flicker, disappearance, or "
        "identity changes."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        custom_ti2v_prompt = kwargs.get("ti2v_prompt")
        super().__init__(*args, **kwargs)
        self.solution_image_prompt = soften_precise_video_timing(self.DEFAULT_SOLUTION_IMAGE_PROMPT)
        self.ti2v_prompt = soften_precise_video_timing(
            str(custom_ti2v_prompt or self.DEFAULT_TI2V_PROMPT)
        )
        self._geometry_input_description = (
            "One circle, one hollow contact marker on its circumference, and five labeled candidate centers in "
            "the forward candidate cluster are visible."
        )
        self._geometry_rule_description = (
            "Infer the circle center and radius through the contact. Select the unique candidate center on the "
            "forward tangent ray perpendicular to that radius; marker outlines and the opposite ray do not count."
        )
        self._geometry_solution_description = (
            "Preserve the circle, contact, and candidates. In cyan, draw the radius to the contact, then the "
            "forward tangent to the unique candidate, add the right-angle mark, and mark only that candidate."
        )
        self._geometry_video_description = (
            "Hold the unchanged input, grow the cyan radius from the inferred center to the contact, grow the cyan "
            "tangent from the contact to the unique candidate, add the cyan right-angle mark, then smoothly mark "
            "the answer."
        )

    def _draw_right_angle(self, draw: Any, progress: float) -> None:
        if progress <= 0.0:
            return
        q1, q2, q3 = self._right_angle_points()
        if progress <= 0.5:
            ratio = progress / 0.5
            end = Point(q1.x + (q2.x - q1.x) * ratio, q1.y + (q2.y - q1.y) * ratio)
            self.draw_line(draw, [q1, end], width_factor=0.72, fill=self.CONSTRUCTION_COLOR)
            return
        self.draw_line(draw, [q1, q2], width_factor=0.72, fill=self.CONSTRUCTION_COLOR)
        ratio = (progress - 0.5) / 0.5
        end = Point(q2.x + (q3.x - q2.x) * ratio, q2.y + (q3.y - q2.y) * ratio)
        self.draw_line(draw, [q2, end], width_factor=0.72, fill=self.CONSTRUCTION_COLOR)

    def _draw_contact_marker(self, draw: Any) -> None:
        self.draw_anchor_marker(draw, self.point_on_circle, 7)

    def _render_state(
        self,
        *,
        radius_progress: float = 0.0,
        tangent_progress: float = 0.0,
        right_angle_progress: float = 0.0,
        highlight_progress: float = 0.0,
    ) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_circle(draw, self.circle_center, self.circle_radius)
        radius_progress = max(0.0, min(1.0, radius_progress))
        tangent_progress = max(0.0, min(1.0, tangent_progress))
        if radius_progress > 0.0:
            radius_end = Point(
                self.circle_center.x + (self.point_on_circle.x - self.circle_center.x) * radius_progress,
                self.circle_center.y + (self.point_on_circle.y - self.circle_center.y) * radius_progress,
            )
            self.draw_line(
                draw,
                [self.circle_center, radius_end],
                fill=self.CONSTRUCTION_COLOR,
                join_start=True,
            )
            ss = draw.factor
            dot_radius = max(1, round(4.2 * ss))
            cx = round(self.circle_center.x * ss)
            cy = round(self.circle_center.y * ss)
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.ellipse(
                (cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius),
                fill=(*self.CONSTRUCTION_COLOR, int(round(255 * radius_progress))),
            )
            base.paste(overlay, (0, 0), overlay)
        if tangent_progress > 0.0:
            tangent_end = Point(
                self.point_on_circle.x + (self.target_point.x - self.point_on_circle.x) * tangent_progress,
                self.point_on_circle.y + (self.target_point.y - self.point_on_circle.y) * tangent_progress,
            )
            self.draw_line(
                draw,
                [self.point_on_circle, tangent_end],
                fill=self.CONSTRUCTION_COLOR,
                join_start=True,
            )
        self._draw_right_angle(draw, max(0.0, min(1.0, right_angle_progress)))
        self._draw_contact_marker(draw)
        self._draw_candidates_with_highlight_progress(draw, highlight_progress)
        return base

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        solved = bool(highlight_label)
        return self._render_state(
            radius_progress=1.0 if solved else 0.0,
            tangent_progress=1.0 if solved else 0.0,
            right_angle_progress=1.0 if solved else 0.0,
            highlight_progress=1.0 if solved else 0.0,
        )

    def save_video_solution(self, pid: str) -> Optional[int]:
        frames: List[Image.Image] = []

        def frame(**kwargs: float) -> Image.Image:
            return self._downscale(self._render_state(**kwargs))

        input_frame = frame()
        frames.extend(input_frame.copy() for _ in range(4))
        frames.extend(frame(radius_progress=index / 6.0) for index in range(1, 7))
        frames.extend(
            frame(radius_progress=1.0, tangent_progress=index / 8.0)
            for index in range(1, 9)
        )
        frames.extend(
            frame(
                radius_progress=1.0,
                tangent_progress=1.0,
                right_angle_progress=index / 4.0,
            )
            for index in range(1, 5)
        )
        frames.extend(
            frame(
                radius_progress=1.0,
                tangent_progress=1.0,
                right_angle_progress=1.0,
                highlight_progress=index / 3.0,
            )
            for index in range(1, 4)
        )
        final_frame = frame(
            radius_progress=1.0,
            tangent_progress=1.0,
            right_angle_progress=1.0,
            highlight_progress=1.0,
        )
        frames.extend(final_frame.copy() for _ in range(4))
        if len(frames) != self.EXPECTED_VIDEO_FRAMES or len(frames) % 4 != 1:
            raise RuntimeError(f"circle_tangent_line produced invalid frame count {len(frames)}")
        output_path = self.solution_dir / f"{pid}_solution.webm"
        arrays = [np.asarray(item.convert("RGB"), dtype=np.uint8) for item in frames]
        if not encode_rgb_frames_to_webm(arrays, output_path, fps=16):
            raise RuntimeError("Failed to encode circle_tangent_line semantic video")
        return len(frames)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        anchor_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        for _ in range(5000):
            center = self.pick_target_point(0.42, padding=anchor_padding + self.canvas_short_side * 0.08)
            radius = self._rng.uniform(self.canvas_short_side * 0.22, self.canvas_short_side * 0.36)
            if not self.circle_fits(center, radius, extra_padding=max(self.line_width, anchor_padding * 0.2)):
                continue
            radius_angle = self._rng.uniform(0.0, math.tau)
            contact = Point(center.x + radius * math.cos(radius_angle), center.y + radius * math.sin(radius_angle))
            tangent_angle = radius_angle + math.pi / 2.0
            if self._rng.random() < 0.5:
                tangent_angle += math.pi
            try:
                target = self.sample_point_along_direction(contact, tangent_angle, min_distance=self.canvas_short_side * 0.26, max_distance=self.canvas_short_side * 0.38, padding=anchor_padding)
                if not self.point_can_host_candidate(target, extra_padding=self.canvas_short_side * 0.03):
                    continue
                strict_gap = max(self.point_radius * 3.0 + 6.0, self.minimum_candidate_spacing() * 1.2)
                self.place_candidates(target, min_separation=strict_gap, accept=lambda point: self.distance(center, point) >= radius + self.point_radius + 10.0)
            except RuntimeError:
                continue
            self.circle_center = center
            self.circle_radius = radius
            self.point_on_circle = contact
            self.target_point = target
            self._radius_angle = radius_angle % math.tau
            self._tangent_angle = tangent_angle % math.tau
            measurements = self._candidate_measurements()
            ordered = sorted(measurements, key=lambda item: float(item['tangent_ray_residual_pixels']))
            if ordered[0]['label'] != self.correct_label or float(ordered[0]['tangent_ray_residual_pixels']) > 1e-06 or float(ordered[1]['tangent_ray_residual_pixels']) < self.MIN_RUNNER_UP_OFFSET or (min((float(item['forward_distance_pixels']) for item in measurements)) <= 12.0):
                continue
            return self.save_puzzle()
        raise RuntimeError('Failed to find valid tangent-line geometry after 5000 tries')

    @staticmethod
    def _bearing_degrees(angle: float) -> float:
        return math.degrees(angle % math.tau)

    def _candidate_measurements(self) -> List[Dict[str, Any]]:
        ux = math.cos(self._radius_angle)
        uy = math.sin(self._radius_angle)
        tx = math.cos(self._tangent_angle)
        ty = math.sin(self._tangent_angle)
        measurements: List[Dict[str, Any]] = []
        for candidate in sorted(self.candidates, key=lambda item: item.label):
            dx = candidate.x - self.point_on_circle.x
            dy = candidate.y - self.point_on_circle.y
            distance = math.hypot(dx, dy)
            forward = dx * tx + dy * ty
            signed_offset = dx * ux + dy * uy
            angle_error = abs(math.degrees(math.acos(max(-1.0, min(1.0, signed_offset / max(distance, 1e-9))))) - 90.0)
            residual = math.hypot(abs(signed_offset), max(0.0, -forward))
            measurements.append(
                {
                    "label": candidate.label,
                    "center": [candidate.x, candidate.y],
                    "relative_direction_from_contact": self._direction_name(dx, dy),
                    "forward_distance_pixels": forward,
                    "signed_perpendicular_offset_pixels": signed_offset,
                    "perpendicular_offset_pixels": abs(signed_offset),
                    "radius_candidate_angle_degrees": math.degrees(
                        math.acos(max(-1.0, min(1.0, signed_offset / max(distance, 1e-9))))
                    ),
                    "right_angle_error_degrees": angle_error,
                    "tangent_ray_residual_pixels": residual,
                }
            )
        return measurements

    def _authored_observation(self) -> str:
        tangent_heading = self._heading_phrase(
            math.cos(self._tangent_angle),
            math.sin(self._tangent_angle),
        )
        return (
            f"The large circle sits {self._canvas_region(self.circle_center)}, and the hollow contact marker "
            f"rides its rim {self._relative_position_phrase(self.circle_center, self.point_on_circle)} the "
            f"inferred center. A tangent must cross the contact radius at a right angle, so from the contact the "
            f"tangent ray runs {tangent_heading}, into the candidate cluster "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> Dict[str, Any]:
        measurements = self._candidate_measurements()
        derived_index = next(index for index, item in enumerate(measurements) if item["label"] == self.correct_label)
        structured = {
            "circle_center": self.circle_center.to_list(),
            "circle_radius": self.circle_radius,
            "contact_point": self.point_on_circle.to_list(),
            "candidate_centers": [
                {"label": item["label"], "center": item["center"]}
                for item in measurements
            ],
            "radius_bearing_degrees": self._bearing_degrees(self._radius_angle),
            "tangent_ray_bearing_degrees": self._bearing_degrees(self._tangent_angle),
            "candidate_measurements": measurements,
            "correct_option": self.correct_label,
            "correct_index": derived_index,
            "runner_up_offset_pixels": sorted(
                float(item["tangent_ray_residual_pixels"]) for item in measurements
            )[1],
            "multiple_solution_policy": "generator_guarantees_one_forward_candidate_center_on_the_tangent_ray",
            "state_trace": list(self.STATE_TRACE),
            "answer_schema": {
                "terminal_text_regex": FIVE_CHOICE_CODEC.full_pattern,
                "requires_sample_specific_reasoning_before_terminal_answer": True,
                "terminal_text_contains_exact_answer": True,
                "no_block_after_terminal_image": True,
            },
            "verifier_contract": {
                "profile": "circle_tangent_line_multimodal_semantic_hard_gate_v1",
                "judged_artifacts": ["text", "final_image", "full_video"],
                "strict_binary_correctness": True,
                "performance_policy": "latency_and_throughput_are_diagnostics_only",
            },
            "render_geometry": {
                "canvas_dimensions": list(self.canvas_dimensions),
                "point_radius": self.point_radius,
                "line_width": self.line_width,
                "right_angle_marker_size": self.RIGHT_ANGLE_SIZE,
                "colors": {
                    "background": [255, 255, 255],
                    "source_circle": list(self.CANDIDATE_OUTLINE_COLOR),
                    "construction": list(self.CONSTRUCTION_COLOR),
                    "selected_outline": list(self.CANDIDATE_HIGHLIGHT_COLOR),
                    "selected_fill": list(self.CANDIDATE_HIGHLIGHT_FILL),
                },
            },
            "expected_video_frame_count": self.EXPECTED_VIDEO_FRAMES,
            "video_fps": 16,
        }
        return {
            "task_name": "circle_tangent_line",
            "task_family": "geometry",
            "circle_center": structured["circle_center"],
            "circle_radius": structured["circle_radius"],
            "contact_point": structured["contact_point"],
            "structured_answer": structured,
            "task_contract": {
                "logic": {
                    "tangent_definition": "line_through_contact_perpendicular_to_center_contact_radius",
                    "ray_choice": "the_forward_ray_toward_the_candidate_cluster",
                    "candidate_semantics": "circle_center_only",
                    "multiple_solution_policy": structured["multiple_solution_policy"],
                },
                "answer": {
                    "parser_profile": "strict_uppercase_option_letter_a_to_e",
                },
                "vision": {
                    "unchanged_objects": "source_circle_contact_all_candidate_centers_labels_background_and_camera",
                    "gravity": "none",
                    "forbidden_mutations": [
                        "falling",
                        "compaction",
                        "refill",
                        "deletion",
                        "duplication",
                        "reflow",
                        "camera_motion",
                    ],
                },
                "verifier": structured["verifier_contract"],
            },
        }

    def save_puzzle(self) -> PointTargetPuzzleRecord:
        return super().save_puzzle()

    @staticmethod
    def _blend_color(start: Sequence[int], end: Sequence[int], progress: float) -> Tuple[int, int, int]:
        return tuple(
            int(round(float(a) + (float(b) - float(a)) * progress))
            for a, b in zip(start, end)
        )

    def _draw_candidates_with_highlight_progress(self, draw: Any, progress: float) -> None:
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

    def _right_angle_points(self) -> Tuple[Point, Point, Point]:
        ux = math.cos(self._radius_angle)
        uy = math.sin(self._radius_angle)
        tx = math.cos(self._tangent_angle)
        ty = math.sin(self._tangent_angle)
        p = self.point_on_circle
        q1 = Point(p.x - ux * self.RIGHT_ANGLE_SIZE, p.y - uy * self.RIGHT_ANGLE_SIZE)
        q2 = Point(q1.x + tx * self.RIGHT_ANGLE_SIZE, q1.y + ty * self.RIGHT_ANGLE_SIZE)
        q3 = Point(p.x + tx * self.RIGHT_ANGLE_SIZE, p.y + ty * self.RIGHT_ANGLE_SIZE)
        return q1, q2, q3

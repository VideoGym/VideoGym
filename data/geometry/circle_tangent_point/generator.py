"""Rule-complete external-point circle tangency generator."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.common.prompts import soften_precise_video_timing
from data.geometry.core.point_target_base import Point, PointCandidate, PointTargetPuzzleGenerator, PointTargetPuzzleRecord
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.typed_answers import FIVE_CHOICE_CODEC


class CircleTangentPointGenerator(PointTargetPuzzleGenerator):
    """Select the displayed circumference point tangent to one external point."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: One circle, one hollow external point, and five labeled candidates on the circumference are "
        "shown. Rule: Select the displayed contact where the segment to the external point is perpendicular to "
        "the radius; the other mathematical tangent contact is not an option. Required Output: Return exactly one "
        "uppercase option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/circle_tangent_point"
    CONSTRUCTION_COLOR: Tuple[int, int, int] = (20, 151, 188)
    RIGHT_ANGLE_SIZE = 24.0
    MIN_RUNNER_UP_ERROR_DEGREES = 15.0
    EXPECTED_VIDEO_FRAMES = 29
    STATE_TRACE = [
        "input_hold",
        "external_tangent_growth",
        "center_and_radius_growth",
        "right_angle_marker_growth",
        "correct_candidate_highlight",
        "final_hold",
    ]

    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Use the supplied input image as the fixed base and produce exactly final solution image. Preserve the "
        "white background, complete original black circle outline, hollow external marker, all five candidate "
        "centers, marker sizes, outlines, labels, spacing, scale, framing, and camera. Add one small cyan-blue "
        "center dot at the inferred circle center. Add a medium-width cyan-blue radius from that center to the "
        "center of the unique correct candidate and one medium-width cyan-blue tangent segment from the "
        "external-marker center to the same candidate center. Draw the external and candidate markers above the "
        "segments so their white or red fills may cover the segment endpoints. Add one small cyan-blue "
        "square-corner mark inside the angle between the inward radius direction and the direction from the "
        "correct candidate toward the external marker. Change only the correct candidate to pale-red fill with a "
        "dark-red outline while keeping its black uppercase letter readable; keep the other four candidates white "
        "with their original dark outlines and letters. No source object moves, rotates, falls, compacts, "
        "refills, reorders, disappears, duplicates, changes identity, or changes size. Add no second tangent, "
        "line extension past an endpoint, extra center or external marker, extra highlight, legend, formula, "
        "answer banner, or other text."
    )
    DEFAULT_TI2V_PROMPT = (
        "Input Image: One circle, one hollow external point, and five labeled candidates on the circumference are "
        "shown. Rule: Select the displayed contact where the segment to the external point is perpendicular to "
        "the radius; the other mathematical tangent contact is not an option. Required Output: Hold the unchanged "
        "input. Grow the tangent from the external marker to the unique contact, then grow the radius from the "
        "circle center to that contact, add the right-angle mark, and finally mark the correct candidate. Preserve "
        "all completed construction. Use a fixed camera, preserve every unchanged object, and show each semantic "
        "construction or commit in rule order without teleportation, hard cuts, full-frame crossfades, flicker, "
        "disappearance, or identity changes."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        custom_ti2v_prompt = kwargs.get("ti2v_prompt")
        super().__init__(*args, **kwargs)
        self.solution_image_prompt = soften_precise_video_timing(self.DEFAULT_SOLUTION_IMAGE_PROMPT)
        self.ti2v_prompt = soften_precise_video_timing(
            str(custom_ti2v_prompt or self.DEFAULT_TI2V_PROMPT)
        )
        self._geometry_input_description = (
            "One circle, one hollow external point, and five labeled candidates on the circumference are shown."
        )
        self._geometry_rule_description = (
            "Select the displayed contact where the segment to the external point is perpendicular to the radius; "
            "the other mathematical tangent contact is not an option."
        )
        self._geometry_solution_description = (
            "Preserve the source circle, external marker, candidates, and camera. In cyan, draw the tangent from the "
            "external marker to the unique contact first; then connect the circle center to that contact and add a "
            "right-angle mark. Mark only the correct candidate in the terminal image."
        )
        self._geometry_video_description = (
            "Hold the unchanged input. Grow the cyan tangent from the external marker to the unique contact, then "
            "grow the cyan radius from the circle center to that contact, add the cyan right-angle mark, and finally "
            "mark the correct candidate. Preserve all completed construction."
        )

    def calculate_tangent_point(self, center: Point, radius: float, external: Point) -> Point:
        distance = self.distance(center, external)
        projection_distance = radius * radius / distance
        ux = (external.x - center.x) / distance
        uy = (external.y - center.y) / distance
        projection = Point(
            center.x + projection_distance * ux,
            center.y + projection_distance * uy,
        )
        height = math.sqrt(max(0.0, radius * radius - projection_distance * projection_distance))
        return Point(projection.x - height * uy, projection.y + height * ux)

    def place_candidates_on_circle(self, center: Point, radius: float, true_point: Point) -> bool:
        labels = list(self.option_labels)
        correct_index = self._rng.randrange(len(labels))
        self.correct_label = labels[correct_index]
        target_angle = math.atan2(true_point.y - center.y, true_point.x - center.x)
        angular_spread = math.radians(self._rng.uniform(96.0, 132.0))
        angular_step = angular_spread / (len(labels) - 1)
        start_angle = target_angle - correct_index * angular_step
        candidates: List[PointCandidate] = []
        for index, label in enumerate(labels):
            angle = start_angle + index * angular_step
            point = Point(center.x + radius * math.cos(angle), center.y + radius * math.sin(angle))
            if index == correct_index:
                point = true_point
            candidates.append(PointCandidate(x=point.x, y=point.y, label=label))
        if not self.validate_candidate_layout(
            candidates,
            min_distance=self.minimum_candidate_spacing(scale=0.9),
        ):
            return False
        self.candidates = candidates
        return True

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.04)
        for _ in range(5000):
            radius = self.canvas_short_side * self._rng.uniform(0.22, 0.36)
            center = self.pick_target_point(
                0.40,
                padding=candidate_padding + self.canvas_short_side * 0.08,
            )
            if not self.circle_fits(center, radius, extra_padding=candidate_padding):
                continue
            center_external_distance = self._rng.uniform(radius * 1.7, radius * 2.8)
            angle = self._rng.uniform(0.0, math.tau)
            external = Point(
                center.x + center_external_distance * math.cos(angle),
                center.y + center_external_distance * math.sin(angle),
            )
            target = self.calculate_tangent_point(center, radius, external)
            if not self.inside_canvas(external, padding=self.line_width):
                continue
            if not self.point_can_host_candidate(target):
                continue
            if self.distance(target, external) < self.canvas_short_side * 0.24:
                continue
            self.center = center
            self.r = radius
            self.external_point = external
            self.target_point = target
            if not self.place_candidates_on_circle(center, radius, target):
                continue
            measurements = self._candidate_measurements()
            ordered = sorted(measurements, key=lambda item: float(item["right_angle_error_degrees"]))
            if (
                ordered[0]["label"] != self.correct_label
                or float(ordered[0]["right_angle_error_degrees"]) > 1e-5
                or float(ordered[1]["right_angle_error_degrees"]) < self.MIN_RUNNER_UP_ERROR_DEGREES
            ):
                continue
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid circle tangent-point puzzle after 5000 tries")


    @staticmethod
    def _bearing_degrees(dx: float, dy: float) -> float:
        return math.degrees(math.atan2(dy, dx) % math.tau)

    def _candidate_measurements(self) -> List[Dict[str, Any]]:
        measurements: List[Dict[str, Any]] = []
        for candidate in sorted(self.candidates, key=lambda item: item.label):
            inward_x = self.center.x - candidate.x
            inward_y = self.center.y - candidate.y
            external_x = self.external_point.x - candidate.x
            external_y = self.external_point.y - candidate.y
            inward_length = math.hypot(inward_x, inward_y)
            external_length = math.hypot(external_x, external_y)
            cosine = (inward_x * external_x + inward_y * external_y) / max(
                1e-9,
                inward_length * external_length,
            )
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            residual = abs(inward_x * external_x + inward_y * external_y) / max(
                1e-9,
                inward_length,
            )
            measurements.append(
                {
                    "label": candidate.label,
                    "center": [candidate.x, candidate.y],
                    "relative_direction_from_circle_center": self._direction_name(
                        candidate.x - self.center.x,
                        candidate.y - self.center.y,
                    ),
                    "direction_from_candidate_to_external": self._direction_name(external_x, external_y),
                    "candidate_radius_bearing_degrees": self._bearing_degrees(
                        candidate.x - self.center.x,
                        candidate.y - self.center.y,
                    ),
                    "candidate_to_external_bearing_degrees": self._bearing_degrees(external_x, external_y),
                    "candidate_to_external_distance_pixels": external_length,
                    "radius_external_angle_degrees": angle,
                    "right_angle_error_degrees": abs(angle - 90.0),
                    "perpendicularity_residual_pixels": residual,
                }
            )
        return measurements

    def _reasoning_text(self) -> str:
        return (
            "A tangent meets the radius at its contact at a right angle. Comparing the displayed contact markers, "
            f"only {self.correct_label} has that perpendicular relation to the external segment; the other mathematical "
            "contact is not offered.\n\n"
            f"{FIVE_CHOICE_CODEC.format(self.correct_label)}"
        )

    def _authored_observation(self) -> str:
        return (
            f"The circle sits {self._canvas_region(self.center)}, and the hollow external marker stands clear of "
            f"it, {self._relative_position_phrase(self.center, self.external_point)} the circle. A segment from "
            f"that marker is tangent exactly where it meets the contact radius at a right angle, which can only "
            f"happen on the part of the rim facing the marker — so the true contact lies "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> Dict[str, Any]:
        measurements = self._candidate_measurements()
        derived_index = next(index for index, item in enumerate(measurements) if item["label"] == self.correct_label)
        runner_up_error = sorted(float(item["right_angle_error_degrees"]) for item in measurements)[1]
        structured = {
            "circle_center": self.center.to_list(),
            "circle_radius": self.r,
            "external_point": self.external_point.to_list(),
            "candidate_centers": [
                {"label": item["label"], "center": item["center"]}
                for item in measurements
            ],
            "candidate_measurements": measurements,
            "correct_option": self.correct_label,
            "correct_index": derived_index,
            "runner_up_right_angle_error_degrees": runner_up_error,
            "multiple_solution_policy": "the_other_mathematical_tangent_point_is_not_present_among_A_to_E",
            "state_trace": list(self.STATE_TRACE),
            "answer_schema": {
                "terminal_text_regex": FIVE_CHOICE_CODEC.full_pattern,
                "requires_sample_specific_reasoning_before_terminal_answer": True,
                "terminal_text_contains_exact_answer": True,
                "no_block_after_terminal_image": True,
            },
            "verifier_contract": {
                "profile": "circle_tangent_point_multimodal_semantic_hard_gate_v1",
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
            "task_name": "circle_tangent_point",
            "task_family": "geometry",
            "circle_center": structured["circle_center"],
            "circle_radius": structured["circle_radius"],
            "external_point": structured["external_point"],
            "structured_answer": structured,
            "task_contract": {
                "logic": {
                    "tangent_definition": "external_segment_perpendicular_to_inward_radius_at_displayed_candidate_center",
                    "candidate_semantics": "circle_center_only",
                    "multiple_solution_policy": structured["multiple_solution_policy"],
                },
                "answer": {
                    "parser_profile": "sample_specific_reasoning_then_strict_uppercase_option_letter_a_to_e",
                },
                "vision": {
                    "unchanged_objects": "source_circle_external_marker_all_candidate_centers_labels_background_and_camera",
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
        radius = round(self.point_radius * ss)
        fill = self._blend_color(self.CANDIDATE_BASE_FILL, self.CANDIDATE_HIGHLIGHT_FILL, progress)
        outline = self._blend_color(self.CANDIDATE_OUTLINE_COLOR, self.CANDIDATE_HIGHLIGHT_COLOR, progress)
        for candidate in sorted(self.candidates, key=lambda item: item.label):
            cx = round(candidate.x * ss)
            cy = round(candidate.y * ss)
            bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
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
        inward_x = (self.center.x - self.target_point.x) / self.r
        inward_y = (self.center.y - self.target_point.y) / self.r
        external_distance = self.distance(self.target_point, self.external_point)
        external_x = (self.external_point.x - self.target_point.x) / external_distance
        external_y = (self.external_point.y - self.target_point.y) / external_distance
        q1 = Point(
            self.target_point.x + inward_x * self.RIGHT_ANGLE_SIZE,
            self.target_point.y + inward_y * self.RIGHT_ANGLE_SIZE,
        )
        q2 = Point(
            q1.x + external_x * self.RIGHT_ANGLE_SIZE,
            q1.y + external_y * self.RIGHT_ANGLE_SIZE,
        )
        q3 = Point(
            self.target_point.x + external_x * self.RIGHT_ANGLE_SIZE,
            self.target_point.y + external_y * self.RIGHT_ANGLE_SIZE,
        )
        return q1, q2, q3

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

    def _draw_external_point(self, draw: Any) -> None:
        self.draw_anchor_marker(draw, self.external_point, 7)

    def _render_state(
        self,
        *,
        radius_progress: float = 0.0,
        tangent_progress: float = 0.0,
        right_angle_progress: float = 0.0,
        highlight_progress: float = 0.0,
    ) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_circle(draw, self.center, self.r)
        radius_progress = max(0.0, min(1.0, radius_progress))
        tangent_progress = max(0.0, min(1.0, tangent_progress))
        if radius_progress > 0.0:
            radius_end = Point(
                self.center.x + (self.target_point.x - self.center.x) * radius_progress,
                self.center.y + (self.target_point.y - self.center.y) * radius_progress,
            )
            self.draw_line(
                draw,
                [self.center, radius_end],
                fill=self.CONSTRUCTION_COLOR,
                join_start=True,
            )
            ss = draw.factor
            dot_radius = max(1, round(4.2 * ss))
            cx = round(self.center.x * ss)
            cy = round(self.center.y * ss)
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.ellipse(
                (cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius),
                fill=(*self.CONSTRUCTION_COLOR, int(round(255 * radius_progress))),
            )
            base.paste(overlay, (0, 0), overlay)
        if tangent_progress > 0.0:
            tangent_end = Point(
                self.external_point.x + (self.target_point.x - self.external_point.x) * tangent_progress,
                self.external_point.y + (self.target_point.y - self.external_point.y) * tangent_progress,
            )
            self.draw_line(
                draw,
                [self.external_point, tangent_end],
                fill=self.CONSTRUCTION_COLOR,
                join_start=True,
            )
        self._draw_right_angle(draw, max(0.0, min(1.0, right_angle_progress)))
        self._draw_external_point(draw)
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
        frames.extend(
            frame(tangent_progress=index / 8.0)
            for index in range(1, 9)
        )
        frames.extend(
            frame(tangent_progress=1.0, radius_progress=index / 6.0)
            for index in range(1, 7)
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
            raise RuntimeError(f"circle_tangent_point produced invalid frame count {len(frames)}")
        output_path = self.solution_dir / f"{pid}_solution.webm"
        arrays = [np.asarray(item.convert("RGB"), dtype=np.uint8) for item in frames]
        if not encode_rgb_frames_to_webm(arrays, output_path, fps=16):
            raise RuntimeError("Failed to encode circle_tangent_point semantic video")
        return len(frames)

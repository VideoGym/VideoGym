"""Rule-complete isosceles-trapezoid completion puzzle generator."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFont

from data.common.prompts import soften_precise_video_timing
from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord
from data.common.video_encoding import encode_rgb_frames_to_webm


def _cross(first: Point, second: Point, third: Point) -> float:
    return (
        (second.x - first.x) * (third.y - first.y)
        - (second.y - first.y) * (third.x - first.x)
    )


def _strictly_convex(points: Sequence[Point]) -> bool:
    if len(points) != 4:
        return False
    values = [
        _cross(points[index], points[(index + 1) % 4], points[(index + 2) % 4])
        for index in range(4)
    ]
    return all(value > 1e-5 for value in values) or all(value < -1e-5 for value in values)


class IsoscelesTrapezoidGenerator(PointTargetPuzzleGenerator):
    """Select the fourth vertex that closes one strict isosceles trapezoid."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: The input is a white image containing exactly one fixed open black polyline with two straight "
        "segments and three distinct visible vertices, plus five separate circular candidates labeled A, B, C, D, "
        "and E near the missing fourth corner. For the rule below, call the polyline's sole bend J, its farther "
        "open endpoint U, and its nearer open endpoint V, with distances measured from J. These local aliases are "
        "defined by the visible geometry and are not additional printed labels. The generator guarantees JU is "
        "strictly longer than JV, so the endpoint naming is unique. JU is the visible long base and JV is one "
        "visible leg. A candidate center X is correct only when the quadrilateral in the fixed order J-U-X-V is "
        "strictly convex, has no crossing edges, and satisfies all of the following: VX points in the same "
        "direction as JU and is parallel to JU, 0 < length(VX) < length(JU), and length(UX) = length(JV). Thus JU "
        "and VX are the unequal parallel bases and JV and UX are the equal non-parallel legs. A parallelogram or "
        "rectangle with equal base lengths is invalid; a zero-length or oppositely directed shorter base, a collinear "
        "corner, a self-crossing or concave order, a candidate on another vertex, or a merely "
        "near-parallel/near-equal result is also invalid. Candidate centers alone count: each circular outline, "
        "fill, and letter glyph has no geometric extent. If a marker visually covers a line, the mathematical "
        "line continues straight beneath it. Exactly one displayed candidate satisfies every condition. Rule: Identify the "
        "unique candidate with zero parallel offset, zero leg mismatch, strict convexity, and a positive shorter "
        "base. Required Output: Return exactly one uppercase option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/isosceles_trapezoid"
    EXPECTED_VIDEO_FRAMES = 25
    MIN_RUNNER_UP_DECISION_RESIDUAL_PIXELS = 18.0
    STATE_TRACE = [
        "input_hold",
        "shorter_base_growth",
        "equal_leg_growth",
        "correct_candidate_highlight",
        "final_hold",
    ]

    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Use the supplied input as the fixed base and produce exactly one final solution image. Preserve the white "
        "background, the complete original open black polyline, its stroke width and three vertices, all "
        "five candidate centers, marker sizes, outlines, fills, labels, spacing, scale, framing, and camera. Add "
        "exactly two medium-width cyan-blue straight segments: first V-X, the shorter base from the nearer endpoint V "
        "to the correct candidate center X, and then X-U, the equal closing leg from X to the farther endpoint U. Join "
        "them cleanly to the original endpoints so the final outline is one strict convex quadrilateral in order "
        "J-U-X-V with JU parallel to VX, VX shorter than JU, and JV equal to UX. Render the candidate markers "
        "above the added edges so their white fills cover only incidental line crossings and the correct vertex "
        "endpoint. Change only the correct candidate to pale-red fill with a dark-red outline while keeping its "
        "black uppercase letter readable; leave the other four candidates white with their original dark outlines "
        "and letters. No source object or unselected candidate moves, rotates, falls, compacts, refills, "
        "reorders, disappears, duplicates, changes identity or size, or reflows. Add no guide line, parallel "
        "arrow, tick mark, extra edge, extra highlight, vertex label, formula, legend, answer banner, or other "
        "text."
    )
    DEFAULT_TI2V_PROMPT = (
        "Input Image: The input is a white image containing exactly one fixed open black polyline with two straight "
        "segments and three distinct visible vertices, plus five separate circular candidates labeled A, B, C, D, "
        "and E near the missing fourth corner. For the rule below, call the polyline's sole bend J, its farther "
        "open endpoint U, and its nearer open endpoint V, with distances measured from J. These local aliases are "
        "defined by the visible geometry and are not additional printed labels. The generator guarantees JU is "
        "strictly longer than JV, so the endpoint naming is unique. JU is the visible long base and JV is one "
        "visible leg. A candidate center X is correct only when the quadrilateral in the fixed order J-U-X-V is "
        "strictly convex, has no crossing edges, and satisfies all of the following: VX points in the same "
        "direction as JU and is parallel to JU, 0 < length(VX) < length(JU), and length(UX) = length(JV). Thus JU "
        "and VX are the unequal parallel bases and JV and UX are the equal non-parallel legs. A parallelogram or "
        "rectangle with equal base lengths is invalid; a zero-length or oppositely directed shorter base, a collinear "
        "corner, a self-crossing or concave order, a candidate on another vertex, or a merely "
        "near-parallel/near-equal result is also invalid. Candidate centers alone count: each circular outline, "
        "fill, and letter glyph has no geometric extent. If a marker visually covers a line, the mathematical "
        "line continues straight beneath it. Exactly one displayed candidate satisfies every condition. Rule: Identify the "
        "unique candidate with zero parallel offset, zero leg mismatch, strict convexity, and a positive shorter "
        "base. Required Output: Preserve the input scene, then commit the solved construction and selected "
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
            "A fixed unlabeled open two-edge polyline and five labeled candidate centers for the missing fourth "
            "vertex X are visible. Call the bend J, farther endpoint U, and nearer endpoint V; JU is the longer "
            "base and JV is one leg."
        )
        self._geometry_rule_description = (
            "Select the unique X for which J-U-X-V is strictly convex, VX is a positive shorter base parallel to "
            "JU in the same direction, and UX equals JV. Reject crossings, concavity, and merely near-equal cases."
        )
        self._geometry_solution_description = (
            "Preserve the open polyline and candidates. In cyan, draw the shorter parallel base from V to X, then "
            "the equal leg from X to U, and mark only the candidate that closes the valid isosceles trapezoid."
        )
        self._geometry_video_description = (
            "Hold the unchanged input, grow the cyan shorter parallel base from V to X, then grow the cyan equal "
            "leg from X to U while preserving the base, smoothly mark the unique candidate, and hold the closed "
            "trapezoid."
        )

    @staticmethod
    def _distance_to_segment(point: Point, start: Point, end: Point) -> float:
        dx = end.x - start.x
        dy = end.y - start.y
        denominator = dx * dx + dy * dy
        if denominator <= 1e-9:
            return math.hypot(point.x - start.x, point.y - start.y)
        progress = max(
            0.0,
            min(
                1.0,
                ((point.x - start.x) * dx + (point.y - start.y) * dy) / denominator,
            ),
        )
        projection = Point(start.x + progress * dx, start.y + progress * dy)
        return math.hypot(point.x - projection.x, point.y - projection.y)

    def _candidate_layout_clear_of_edges(self) -> bool:
        junction, base_endpoint, leg_endpoint, target = self.trapezoid_points
        edges = (
            (junction, base_endpoint),
            (junction, leg_endpoint),
            (leg_endpoint, target),
            (target, base_endpoint),
        )
        clearance = self.point_radius * 1.65 + self.line_width
        for candidate in self.candidates:
            if candidate.label == self.correct_label:
                continue
            center = Point(candidate.x, candidate.y)
            if any(self._distance_to_segment(center, start, end) < clearance for start, end in edges):
                return False
        return True

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        anchor_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(1400):
            center = self.pick_target_point(
                0.45,
                padding=anchor_padding + self.canvas_short_side * 0.12,
            )
            base_angle = self._rng.uniform(0.0, math.tau)
            axis_angle = base_angle + math.pi / 2.0
            base_half = self.canvas_short_side * self._rng.uniform(0.20, 0.26)
            top_half = base_half * self._rng.uniform(0.52, 0.72)
            height = self.canvas_short_side * self._rng.uniform(0.20, 0.28)

            bottom_center = self.point_on_ray(center, axis_angle + math.pi, height * 0.5)
            top_center = self.point_on_ray(center, axis_angle, height * 0.5)
            junction = self.point_on_ray(bottom_center, base_angle + math.pi, base_half)
            base_endpoint = self.point_on_ray(bottom_center, base_angle, base_half)
            leg_endpoint = self.point_on_ray(top_center, base_angle + math.pi, top_half)
            target = self.point_on_ray(top_center, base_angle, top_half)
            points = (junction, base_endpoint, leg_endpoint, target)
            if not all(self.point_can_host_candidate(point) for point in points):
                continue
            long_base = math.dist(junction.to_list(), base_endpoint.to_list())
            visible_leg = math.dist(junction.to_list(), leg_endpoint.to_list())
            short_base = math.dist(leg_endpoint.to_list(), target.to_list())
            if (
                height < self.canvas_short_side * 0.20
                or short_base < self.canvas_short_side * 0.18
                or long_base - short_base < self.canvas_short_side * 0.12
                or long_base - visible_leg < self.canvas_short_side * 0.08
                or not _strictly_convex((junction, base_endpoint, target, leg_endpoint))
            ):
                continue
            self.trapezoid_points = points
            self.junction = junction
            self.base_endpoint = base_endpoint
            self.leg_endpoint = leg_endpoint
            self.target_point = target
            for _layout_attempt in range(80):
                try:
                    self.place_candidates(target)
                except RuntimeError:
                    continue
                if not self._candidate_layout_clear_of_edges():
                    continue
                measurements = self._candidate_measurements()
                ordered = sorted(
                    measurements,
                    key=lambda item: float(item["decision_residual_pixels"]),
                )
                if (
                    ordered[0]["label"] == self.correct_label
                    and bool(ordered[0]["eligible"])
                    and float(ordered[0]["decision_residual_pixels"]) <= 1e-5
                    and float(ordered[1]["decision_residual_pixels"])
                    >= self.MIN_RUNNER_UP_DECISION_RESIDUAL_PIXELS
                ):
                    return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid isosceles-trapezoid puzzle")


    def _candidate_measurements(self) -> List[Dict[str, Any]]:
        junction, base_endpoint, leg_endpoint, _target = self.trapezoid_points
        base_dx = base_endpoint.x - junction.x
        base_dy = base_endpoint.y - junction.y
        base_length = math.hypot(base_dx, base_dy)
        unit_x, unit_y = base_dx / base_length, base_dy / base_length
        known_leg_length = math.hypot(
            leg_endpoint.x - junction.x,
            leg_endpoint.y - junction.y,
        )
        measurements: List[Dict[str, Any]] = []
        for candidate in sorted(self.candidates, key=lambda item: item.label):
            point = Point(candidate.x, candidate.y)
            top_dx = point.x - leg_endpoint.x
            top_dy = point.y - leg_endpoint.y
            forward = top_dx * unit_x + top_dy * unit_y
            signed_offset = top_dx * unit_y - top_dy * unit_x
            right_leg = math.hypot(point.x - base_endpoint.x, point.y - base_endpoint.y)
            mismatch = abs(right_leg - known_leg_length)
            convex = _strictly_convex((junction, base_endpoint, point, leg_endpoint))
            shorter_positive = 1e-5 < forward < base_length - 1e-5
            eligible = bool(
                abs(signed_offset) <= 1e-5
                and mismatch <= 1e-5
                and convex
                and shorter_positive
            )
            residual = math.hypot(abs(signed_offset), mismatch)
            if not convex or not shorter_positive:
                residual += 40.0
            measurements.append(
                {
                    "label": candidate.label,
                    "center": [candidate.x, candidate.y],
                    "direction_from_V": self._direction_name(top_dx, top_dy),
                    "parallel_signed_offset_pixels": signed_offset,
                    "parallel_offset_pixels": abs(signed_offset),
                    "top_base_forward_length_pixels": forward,
                    "closing_leg_length_pixels": right_leg,
                    "known_visible_leg_length_pixels": known_leg_length,
                    "leg_length_mismatch_pixels": mismatch,
                    "strict_convex": convex,
                    "shorter_positive_top_base": shorter_positive,
                    "eligible": eligible,
                    "decision_residual_pixels": residual,
                }
            )
        return measurements

    def _authored_observation(self) -> str:
        junction, base_endpoint, leg_endpoint, target = self.trapezoid_points
        base_direction = self._heading_phrase(
            base_endpoint.x - junction.x, base_endpoint.y - junction.y,
        )
        return (
            f"The bend J sits {self._canvas_region(junction)}, its farther open endpoint U lies "
            f"{self._relative_position_phrase(junction, base_endpoint)} it, and the nearer endpoint V sits "
            f"{self._relative_position_phrase(junction, leg_endpoint)} it; the long base JU runs "
            f"{base_direction} and JV is the one visible leg. The missing base must leave V parallel to JU in "
            f"the same direction while staying shorter than JU, and the closing leg back to U must equal JV; "
            f"those constraints pin the fourth corner {self._canvas_region(target)}."
        )

    def build_record_extra(self) -> Dict[str, Any]:
        junction, base_endpoint, leg_endpoint, target = self.trapezoid_points
        measurements = self._candidate_measurements()
        derived_index = next(
            index for index, item in enumerate(measurements)
            if item["label"] == self.correct_label
        )
        runner_up = sorted(float(item["decision_residual_pixels"]) for item in measurements)[1]
        long_base = math.hypot(base_endpoint.x - junction.x, base_endpoint.y - junction.y)
        short_base = math.hypot(target.x - leg_endpoint.x, target.y - leg_endpoint.y)
        leg_length = math.hypot(leg_endpoint.x - junction.x, leg_endpoint.y - junction.y)
        visible_vertices = {
            "J": junction.to_list(),
            "U": base_endpoint.to_list(),
            "V": leg_endpoint.to_list(),
        }
        structured = {
            "visible_vertices": visible_vertices,
            "candidate_centers": [
                {"label": item["label"], "center": item["center"]}
                for item in measurements
            ],
            "candidate_measurements": measurements,
            "derived_missing_vertex": target.to_list(),
            "long_base_length_pixels": long_base,
            "short_base_length_pixels": short_base,
            "equal_leg_length_pixels": leg_length,
            "correct_option": self.correct_label,
            "correct_index": derived_index,
            "runner_up_decision_residual_pixels": runner_up,
            "multiple_solution_policy": "exactly_one_displayed_candidate_forms_a_strict_convex_shorter_top_base_isosceles_trapezoid",
            "state_trace": list(self.STATE_TRACE),
            "answer_schema": {
                "terminal_text_regex": FIVE_CHOICE_CODEC.full_pattern,
                "requires_sample_specific_reasoning_before_terminal_answer": True,
                "terminal_text_contains_exact_answer": True,
                "no_block_after_terminal_image": True,
            },
            "verifier_contract": {
                "profile": "isosceles_trapezoid_multimodal_semantic_hard_gate_v1",
                "judged_artifacts": ["text", "final_image", "full_video"],
                "strict_binary_correctness": True,
                "performance_policy": "latency_and_throughput_are_diagnostics_only",
            },
            "render_geometry": {
                "canvas_dimensions": list(self.canvas_dimensions),
                "point_radius": self.point_radius,
                "line_width": self.line_width,
                "colors": {
                    "background": [255, 255, 255],
                    "source_edges": list(self.CANDIDATE_OUTLINE_COLOR),
                    "terminal_completion_edges": list(self.CONSTRUCTION_COLOR),
                    "selected_outline": list(self.CANDIDATE_HIGHLIGHT_COLOR),
                    "selected_fill": list(self.CANDIDATE_HIGHLIGHT_FILL),
                },
            },
            "expected_video_frame_count": self.EXPECTED_VIDEO_FRAMES,
            "video_fps": 16,
        }
        return {
            "task_name": "isosceles_trapezoid",
            "task_family": "geometry",
            "visible_vertices": visible_vertices,
            "structured_answer": structured,
            "task_contract": {
                "logic": {
                    "quadrilateral_order": "J-U-X-V",
                    "parallel_bases": "JU_parallel_to_VX_with_0_less_than_VX_less_than_JU",
                    "equal_legs": "JV_equals_UX",
                    "candidate_semantics": "marker_center_only",
                    "boundary_policy": "strict_convex_non_crossing_and_equal_base_parallelogram_excluded",
                },
                "answer": {
                    "parser_profile": "sample_specific_constraint_table_then_strict_uppercase_option_letter_a_to_e",
                },
                "vision": {
                    "unchanged_objects": "source_polyline_all_candidate_centers_labels_background_and_camera",
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
    def _blend_color(
        start: Sequence[int],
        end: Sequence[int],
        progress: float,
    ) -> Tuple[int, int, int]:
        return tuple(
            int(round(float(first) + (float(second) - float(first)) * progress))
            for first, second in zip(start, end)
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

    @staticmethod
    def _interpolate(start: Point, end: Point, progress: float) -> Point:
        return Point(
            start.x + (end.x - start.x) * progress,
            start.y + (end.y - start.y) * progress,
        )

    def _render_state(
        self,
        *,
        top_base_progress: float = 0.0,
        right_leg_progress: float = 0.0,
        highlight_progress: float = 0.0,
    ) -> Image.Image:
        draw, base = self.get_draw_base()
        junction, base_endpoint, leg_endpoint, target = self.trapezoid_points
        self.draw_line(draw, [leg_endpoint, junction, base_endpoint])
        top_base_progress = max(0.0, min(1.0, top_base_progress))
        right_leg_progress = max(0.0, min(1.0, right_leg_progress))
        if top_base_progress > 0.0:
            top_end = self._interpolate(leg_endpoint, target, top_base_progress)
            self.draw_line(
                draw,
                [leg_endpoint, top_end],
                fill=self.CONSTRUCTION_COLOR,
                join_start=True,
            )
        if right_leg_progress > 0.0:
            leg_end = self._interpolate(target, base_endpoint, right_leg_progress)
            self.draw_line(
                draw,
                [target, leg_end],
                fill=self.CONSTRUCTION_COLOR,
                join_end=right_leg_progress >= 1.0,
            )
        self._draw_candidates_with_highlight_progress(draw, highlight_progress)
        return base

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        solved = bool(highlight_label)
        return self._render_state(
            top_base_progress=1.0 if solved else 0.0,
            right_leg_progress=1.0 if solved else 0.0,
            highlight_progress=1.0 if solved else 0.0,
        )

    def save_video_solution(self, pid: str) -> Optional[int]:
        frames: List[Image.Image] = []

        def frame(**kwargs: float) -> Image.Image:
            return self._downscale(self._render_state(**kwargs))

        input_frame = frame()
        frames.extend(input_frame.copy() for _ in range(4))
        frames.extend(
            frame(top_base_progress=index / 6.0)
            for index in range(1, 7)
        )
        frames.extend(
            frame(top_base_progress=1.0, right_leg_progress=index / 6.0)
            for index in range(1, 7)
        )
        frames.extend(
            frame(
                top_base_progress=1.0,
                right_leg_progress=1.0,
                highlight_progress=index / 4.0,
            )
            for index in range(1, 5)
        )
        final_frame = frame(
            top_base_progress=1.0,
            right_leg_progress=1.0,
            highlight_progress=1.0,
        )
        frames.extend(final_frame.copy() for _ in range(5))
        if len(frames) != self.EXPECTED_VIDEO_FRAMES or len(frames) % 4 != 1:
            raise RuntimeError(f"isosceles_trapezoid produced invalid frame count {len(frames)}")
        output_path = self.solution_dir / f"{pid}_solution.webm"
        arrays = [np.asarray(item.convert("RGB"), dtype=np.uint8) for item in frames]
        if not encode_rgb_frames_to_webm(arrays, output_path, fps=16):
            raise RuntimeError("Failed to encode isosceles_trapezoid semantic video")
        return len(frames)

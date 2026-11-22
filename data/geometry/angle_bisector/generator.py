"""Rule-complete angle-bisector point-target generator."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFont

from data.common.prompts import soften_precise_video_timing
from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord
from data.common.video_encoding import encode_rgb_frames_to_webm


class AngleBisectorGenerator(PointTargetPuzzleGenerator):
    """Generate a visible two-ray angle with one exact bisector candidate."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: The input is white image containing exactly two finite solid-black line segments that share "
        "one endpoint. Treat the shared endpoint as the vertex and treat each segment as showing the direction of "
        "an infinite ray; the visible outer endpoints do not limit how far the mathematical rays extend. The two "
        "ray directions define one smaller interior angle of less than 180 degrees and one reflex angle of more "
        "than 180 degrees. Use only the smaller interior angle. Five non-overlapping circular candidates labeled "
        "A, B, C, D, and E are shown near that smaller angle. A candidate counts by the center of its circle "
        "only: its outline, fill area, and letter glyph do not count as the point. Exactly one candidate center "
        "lies on the internal angle bisector, the ray starting at the shared vertex that remains inside the "
        "smaller angle and has equal angular distance from the two black ray directions. Ignore neither black "
        "ray, do not use the external/reflex-angle bisector, and do not extend a candidate's circle or label as "
        "geometry. Rule: Select the unique labeled candidate whose center satisfies the stated geometric "
        "relation. Required Output: Return exactly one uppercase option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/angle_bisector"
    BISECTOR_COLOR: Tuple[int, int, int] = (20, 151, 188)
    EXPECTED_VIDEO_FRAMES = 21
    STATE_TRACE = [
        "input_hold",
        "bisector_vertex_to_candidate_growth",
        "correct_candidate_highlight",
        "final_hold",
    ]

    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Use the supplied input image as the fixed base. Produce exactly final solution image. Preserve the white "
        "background, both original black ray segments, their shared vertex and finite endpoints, all five "
        "candidate circle centers, sizes, labels, and the camera exactly. From the shared vertex draw one "
        "medium-width cyan-blue segment along the internal bisector of the smaller angle until it reaches the "
        "center of the unique correct candidate. "
        "Change only the correct candidate to pale-red fill with a dark-red outline; keep its black uppercase "
        "letter readable. Keep every other candidate white with its original dark outline and black letter. No "
        "object moves, falls, compacts, refills, disappears, duplicates, or changes order, and no extra lines, "
        "arcs, highlights, legends, answer banners, or text may appear."
    )
    DEFAULT_TI2V_PROMPT = (
        "Input Image: The input is white image containing exactly two finite solid-black line segments that share "
        "one endpoint. Treat the shared endpoint as the vertex and treat each segment as showing the direction of "
        "an infinite ray; the visible outer endpoints do not limit how far the mathematical rays extend. The two "
        "ray directions define one smaller interior angle of less than 180 degrees and one reflex angle of more "
        "than 180 degrees. Use only the smaller interior angle. Five non-overlapping circular candidates labeled "
        "A, B, C, D, and E are shown near that smaller angle. A candidate counts by the center of its circle "
        "only: its outline, fill area, and letter glyph do not count as the point. Exactly one candidate center "
        "lies on the internal angle bisector, the ray starting at the shared vertex that remains inside the "
        "smaller angle and has equal angular distance from the two black ray directions. Ignore neither black "
        "ray, do not use the external/reflex-angle bisector, and do not extend a candidate's circle or label as "
        "geometry. Rule: Select the unique labeled candidate whose center satisfies the stated geometric "
        "relation. Required Output: Preserve the input scene, then continuously draw the bisector from the shared "
        "vertex to the selected center and finally highlight that candidate. Use a fixed camera, preserve "
        "every unchanged object, and show each semantic construction or commit in rule order without teleportation, "
        "hard cuts, full-frame crossfades, flicker, disappearance, or identity changes."
    )

    def __init__(self, *args: Any, opening_range=(54.0, 112.0), **kwargs: Any) -> None:
        if not 0 < opening_range[0] <= opening_range[1] < 180:
            raise ValueError("opening_range must lie between zero and 180 degrees")
        self.opening_range = opening_range
        custom_ti2v_prompt = kwargs.get("ti2v_prompt")
        super().__init__(*args, **kwargs)
        self.solution_image_prompt = soften_precise_video_timing(self.DEFAULT_SOLUTION_IMAGE_PROMPT)
        self.ti2v_prompt = soften_precise_video_timing(
            str(custom_ti2v_prompt or self.DEFAULT_TI2V_PROMPT)
        )
        self._geometry_input_description = (
            "Two black rays share one vertex and five labeled candidate centers lie near their smaller interior "
            "angle. The visible segment endpoints indicate ray direction but do not bound the rays."
        )
        self._geometry_rule_description = (
            "Use the smaller, non-reflex angle. Select the unique candidate center on its internal bisector, "
            "equally inclined to both ray directions; marker outlines and letters have no geometric extent."
        )
        self._geometry_solution_description = (
            "Preserve both rays and every candidate. Add only the cyan internal bisector from the shared vertex to "
            "the correct candidate center, then mark that candidate."
        )
        self._geometry_video_description = (
            "Hold the unchanged input. Draw the cyan internal bisector from the shared vertex to the selected "
            "center, then smoothly mark the unique candidate and hold the solved state."
        )

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin = 64
        anchor_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            vertex = self.pick_target_point(
                0.5,
                padding=anchor_padding + self.canvas_short_side * 0.16,
            )
            opening = math.radians(self._rng.uniform(*self.opening_range))
            bisector_angle = self._rng.uniform(0.0, math.tau)
            angle1 = bisector_angle - opening / 2.0
            angle2 = bisector_angle + opening / 2.0
            ray_min = self.canvas_short_side * 0.24
            ray_max = self.canvas_short_side * 0.36
            target_min = self.canvas_short_side * 0.30
            target_max = self.canvas_short_side * 0.40
            try:
                p1 = self.sample_point_along_direction(
                    vertex,
                    angle1,
                    min_distance=ray_min,
                    max_distance=ray_max,
                    padding=self.line_width,
                )
                p2 = self.sample_point_along_direction(
                    vertex,
                    angle2,
                    min_distance=ray_min,
                    max_distance=ray_max,
                    padding=self.line_width,
                )
                target = self.sample_point_along_direction(
                    vertex,
                    bisector_angle,
                    min_distance=target_min,
                    max_distance=target_max,
                    padding=anchor_padding,
                )
                self.place_candidates_line(
                    target,
                    bisector_angle + math.pi / 2 + self._rng.uniform(-0.06, 0.06),
                )
            except RuntimeError:
                continue
            if min(self.distance(vertex, p1), self.distance(vertex, p2)) < ray_min:
                continue
            self.points = (p1, vertex, p2)
            self.target_point = target
            self._ray_angles = (angle1 % math.tau, angle2 % math.tau)
            self._opening_radians = opening
            self._bisector_angle = bisector_angle % math.tau
            return self.save_puzzle()
        raise RuntimeError("Failed to find valid angle bisector geometry after 999 tries")

    @staticmethod
    def _angular_distance(angle1: float, angle2: float) -> float:
        return abs((angle2 - angle1 + math.pi) % math.tau - math.pi)

    @staticmethod
    def _bearing_degrees(angle: float) -> float:
        return math.degrees(angle % math.tau)


    def _candidate_measurements(self) -> List[Dict[str, Any]]:
        vertex = self.points[1]
        angle1, angle2 = self._ray_angles
        measurements: List[Dict[str, Any]] = []
        for candidate in sorted(self.candidates, key=lambda item: item.label):
            dx = candidate.x - vertex.x
            dy = candidate.y - vertex.y
            bearing = math.atan2(dy, dx) % math.tau
            distance1 = self._angular_distance(bearing, angle1)
            distance2 = self._angular_distance(bearing, angle2)
            measurements.append(
                {
                    "label": candidate.label,
                    "center": [candidate.x, candidate.y],
                    "relative_direction": self._direction_name(dx, dy),
                    "bearing_degrees": self._bearing_degrees(bearing),
                    "ray1_distance_degrees": math.degrees(distance1),
                    "ray2_distance_degrees": math.degrees(distance2),
                    "angular_imbalance_degrees": math.degrees(abs(distance1 - distance2)),
                }
            )
        return measurements

    def _authored_observation(self) -> str:
        vertex = self.points[1]
        ray_one = self._heading_phrase(math.cos(self._ray_angles[0]), math.sin(self._ray_angles[0]))
        ray_two = self._heading_phrase(math.cos(self._ray_angles[1]), math.sin(self._ray_angles[1]))
        size = self._angle_size_phrase(math.degrees(self._opening_radians))
        heading = self._heading_phrase(math.cos(self._bisector_angle), math.sin(self._bisector_angle))
        return (
            f"From the shared vertex {self._canvas_region(vertex)}, one ray heads {ray_one} and the other "
            f"{ray_two}, spanning {size} between them. The internal bisector leaves the vertex {heading}, "
            "staying equally inclined to both rays, so I follow that direction into the candidate cluster."
        )

    def build_record_extra(self) -> Dict[str, Any]:
        measurements = self._candidate_measurements()
        correct_index = next(
            index for index, candidate in enumerate(sorted(self.candidates, key=lambda item: item.label))
            if candidate.label == self.correct_label
        )
        structured = {
            "vertex": self.points[1].to_list(),
            "ray_endpoints": [self.points[0].to_list(), self.points[2].to_list()],
            "candidate_centers": [
                {"label": item["label"], "center": item["center"]} for item in measurements
            ],
            "ray_bearings_degrees": [
                self._bearing_degrees(self._ray_angles[0]),
                self._bearing_degrees(self._ray_angles[1]),
            ],
            "smaller_opening_degrees": math.degrees(self._opening_radians),
            "bisector_bearing_degrees": self._bearing_degrees(self._bisector_angle),
            "candidate_measurements": measurements,
            "correct_option": self.correct_label,
            "correct_index": correct_index,
            "multiple_solution_policy": "generator_guarantees_exactly_one_candidate_center_on_internal_bisector",
            "state_trace": list(self.STATE_TRACE),
            "answer_schema": {
                "terminal_text_regex": FIVE_CHOICE_CODEC.full_pattern,
                "requires_sample_specific_reasoning_before_terminal_answer": True,
                "terminal_text_contains_exact_answer": True,
                "no_block_after_terminal_image": True,
            },
            "verifier_contract": {
                "profile": "angle_bisector_multimodal_semantic_hard_gate_v1",
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
                    "source_rays": list(self.CANDIDATE_OUTLINE_COLOR),
                    "bisector": list(self.BISECTOR_COLOR),
                    "selected_outline": list(self.CANDIDATE_HIGHLIGHT_COLOR),
                    "selected_fill": list(self.CANDIDATE_HIGHLIGHT_FILL),
                },
            },
            "expected_video_frame_count": self.EXPECTED_VIDEO_FRAMES,
            "video_fps": 16,
        }
        return {
            "task_name": "angle_bisector",
            "task_family": "geometry",
            "vertex": structured["vertex"],
            "ray_endpoints": structured["ray_endpoints"],
            "angle_degrees": structured["smaller_opening_degrees"],
            "structured_answer": structured,
            "task_contract": {
                "logic": {
                    "angle_choice": "smaller_interior_angle_only",
                    "candidate_semantics": "circle_center_only",
                    "finite_segment_semantics": "endpoints_show_ray_direction_but_do_not_bound_the_ray",
                    "multiple_solution_policy": structured["multiple_solution_policy"],
                },
                "answer": {
                    "parser_profile": "strict_uppercase_option_letter_a_to_e",
                },
                "vision": {
                    "unchanged_objects": "both_source_rays_all_candidate_centers_labels_background_and_camera",
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
        from data.geometry.core.prompt_paraphrase import workbench_active, draw_survey_markers
        if workbench_active():
            draw_survey_markers(type(self), draw, self.candidates, self.point_radius, self.correct_label, progress)
            return
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
        bisector_progress: float = 0.0,
        highlight_progress: float = 0.0,
    ) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, self.points)
        vertex = self.points[1]
        if bisector_progress > 0.0:
            end = Point(
                vertex.x + (self.target_point.x - vertex.x) * min(1.0, bisector_progress),
                vertex.y + (self.target_point.y - vertex.y) * min(1.0, bisector_progress),
            )
            self.draw_line(
                draw,
                [vertex, end],
                fill=self.BISECTOR_COLOR,
                join_start=True,
            )
        self._draw_candidates_with_highlight_progress(draw, highlight_progress)
        return base

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        solved = bool(highlight_label)
        return self._render_state(
            bisector_progress=1.0 if solved else 0.0,
            highlight_progress=1.0 if solved else 0.0,
        )

    def save_video_solution(self, pid: str) -> Optional[int]:
        frames: List[Image.Image] = []

        def frame(**kwargs: float) -> Image.Image:
            return self._downscale(self._render_state(**kwargs))

        input_frame = frame()
        frames.extend(input_frame.copy() for _ in range(4))
        frames.extend(
            frame(
                bisector_progress=index / 8.0,
            )
            for index in range(1, 9)
        )
        frames.extend(
            frame(
                bisector_progress=1.0,
                highlight_progress=index / 4.0,
            )
            for index in range(1, 5)
        )
        final_frame = frame(
            bisector_progress=1.0,
            highlight_progress=1.0,
        )
        frames.extend(final_frame.copy() for _ in range(5))
        if len(frames) != self.EXPECTED_VIDEO_FRAMES or len(frames) % 4 != 1:
            raise RuntimeError(f"angle_bisector produced invalid frame count {len(frames)}")
        output_path = self.solution_dir / f"{pid}_solution.webm"
        arrays = [np.asarray(item.convert("RGB"), dtype=np.uint8) for item in frames]
        if not encode_rgb_frames_to_webm(arrays, output_path, fps=16):
            raise RuntimeError("Failed to encode angle_bisector semantic video")
        return len(frames)

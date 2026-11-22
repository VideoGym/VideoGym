"""Rule-complete midpoint selection and construction generator."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFont

from data.common.prompts import soften_precise_video_timing
from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointCandidate,
    PointTargetPuzzleGenerator,
)
from data.common.video_encoding import encode_rgb_frames_to_webm


@dataclass
class Segment:
    # Held as points, not as bare pairs, so the shared frame composition can
    # see the segment's ends and slide them with the rest of the sample.
    start: Point
    end: Point

    def to_dict(self) -> dict:
        return {"start": self.start.to_list(), "end": self.end.to_list()}


@dataclass
class MidpointPuzzleRecord:
    id: str
    ti2v_prompt: str
    canvas_dimensions: Tuple[int, int]
    margin: int
    midpoint: Tuple[float, float]
    segment: Segment
    candidates: List[PointCandidate]
    point_radius: int
    line_width: int
    correct_option: str
    image: str
    solution_image_path: str
    structured_answer: Dict[str, Any]
    task_contract: Dict[str, Any]
    solution_image_prompt: Optional[str] = None
    seed: Optional[int] = None
    reasoning_image: Optional[str] = None
    solution_video_path: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None
    construction_trace: Optional[List[str]] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ti2v_prompt": self.ti2v_prompt,
            "canvas_dimensions": list(self.canvas_dimensions),
            "margin": self.margin,
            "midpoint": list(self.midpoint),
            "segment": self.segment.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "point_radius": self.point_radius,
            "line_width": self.line_width,
            "correct_option": self.correct_option,
            "answer": FIVE_CHOICE_CODEC.format(self.correct_option),
            "image": self.image,
            "reasoning_image": self.reasoning_image,
            "solution_image_path": self.solution_image_path,
            "solution_video_path": self.solution_video_path,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
            "construction_trace": list(self.construction_trace or []),
            "structured_answer": self.structured_answer,
            "task_contract": self.task_contract,
            "task_name": "midpoint",
            "task_family": "geometry",
            "seed": self.seed,
            "type": "midpoint",
        }


CandidatePoint = PointCandidate


class MidpointGenerator(PointTargetPuzzleGenerator):
    """Select the unique candidate at the coordinate midpoint of two anchors."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A straight segment joins two hollow endpoint markers, with five labeled candidate markers "
        "nearby. Rule: Select the candidate on the segment that divides it into two equal halves. Required "
        "Output: Return exactly one uppercase option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/midpoint"
    EXPECTED_VIDEO_FRAMES = 17
    MIN_RUNNER_UP_MIDPOINT_ERROR_PIXELS = 32.0
    CONSTRUCTION_TRACE = (
        "input_hold",
        "correct_candidate_highlight",
        "final_hold",
    )

    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Input Image: A straight segment joins two hollow endpoint markers, with five labeled candidate markers "
        "nearby. Rule: Select the candidate on the segment that divides it into two equal halves. Required "
        "Output: Preserve the segment and every marker, and mark only the midpoint candidate in the terminal image."
    )
    DEFAULT_TI2V_PROMPT = (
        "Input Image: A straight segment joins two hollow endpoint markers, with five labeled candidate markers "
        "nearby. Rule: Select the candidate on the segment that divides it into two equal halves. Required "
        "Output: Hold the unchanged input, then mark only the midpoint candidate and hold the solved state. Use a "
        "fixed camera, preserve every unchanged object, and show each semantic construction or commit in rule "
        "order without teleportation, hard cuts, full-frame crossfades, flicker, disappearance, or identity "
        "changes."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        custom_ti2v_prompt = kwargs.get("ti2v_prompt")
        super().__init__(*args, **kwargs)
        self.solution_image_prompt = soften_precise_video_timing(self.DEFAULT_SOLUTION_IMAGE_PROMPT)
        self.ti2v_prompt = soften_precise_video_timing(
            str(custom_ti2v_prompt or self.DEFAULT_TI2V_PROMPT)
        )
        self._geometry_input_description = (
            "A straight segment joins two hollow endpoint markers, with five labeled candidate markers nearby."
        )
        self._geometry_rule_description = (
            "Select the candidate on the segment that divides it into two equal halves."
        )
        self._geometry_solution_description = (
            "Preserve the segment and all markers; mark only the midpoint candidate in the terminal image."
        )
        self._geometry_video_description = (
            "Hold the unchanged input, then mark only the midpoint candidate and hold the solved state."
        )

    @staticmethod
    def _ordered_endpoints(segment: Segment) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((segment.start.x, segment.start.y), (segment.end.x, segment.end.y))

    def _candidate_measurements(self) -> List[Dict[str, Any]]:
        p, q = self._ordered_endpoints(self._segment)
        mx, my = self.target_point.x, self.target_point.y
        vx, vy = q[0] - p[0], q[1] - p[1]
        length_sq = vx * vx + vy * vy
        segment_length = math.sqrt(length_sq)
        half = segment_length * 0.5
        measurements: List[Dict[str, Any]] = []
        for candidate in sorted(self.candidates, key=lambda item: item.label):
            dx, dy = candidate.x - mx, candidate.y - my
            distance_p = math.hypot(candidate.x - p[0], candidate.y - p[1])
            distance_q = math.hypot(candidate.x - q[0], candidate.y - q[1])
            along = (
                ((candidate.x - p[0]) * vx + (candidate.y - p[1]) * vy) / length_sq
                if length_sq > 1e-9 else float("nan")
            )
            signed_perpendicular = (
                ((candidate.x - p[0]) * vy - (candidate.y - p[1]) * vx) / segment_length
                if segment_length > 1e-9 else float("nan")
            )
            midpoint_error = math.hypot(dx, dy)
            on_segment = abs(signed_perpendicular) <= 1e-5 and -1e-5 <= along <= 1.0 + 1e-5
            strictly_between = on_segment and 1e-5 < along < 1.0 - 1e-5
            eligible = midpoint_error <= 1e-5 and strictly_between
            measurements.append(
                {
                    "label": candidate.label,
                    "center": [candidate.x, candidate.y],
                    "direction_from_M": self._direction_name(dx, dy),
                    "offset_from_M": [dx, dy],
                    "midpoint_error_pixels": midpoint_error,
                    "distance_to_P_pixels": distance_p,
                    "distance_to_Q_pixels": distance_q,
                    "endpoint_distance_difference_pixels": abs(distance_p - distance_q),
                    "half_length_pixels": half,
                    "along_segment_fraction": along,
                    "perpendicular_offset_pixels": abs(signed_perpendicular),
                    "on_segment": on_segment,
                    "strictly_between_endpoints": strictly_between,
                    "eligible": eligible,
                }
            )
        return measurements

    def _anchor_clearance(self) -> float:
        """Centre distance that keeps a marker clear of an endpoint anchor."""

        anchor_radius = max(self.point_radius + 6, int(round(min(self.canvas_dimensions) * 0.028)))
        return self.point_radius + anchor_radius + 8.0

    def _candidate_layout_clear_of_anchors(self) -> bool:
        p, q = self._ordered_endpoints(self._segment)
        minimum = self._anchor_clearance()
        return all(
            min(math.dist((candidate.x, candidate.y), p), math.dist((candidate.x, candidate.y), q))
            >= minimum
            for candidate in self.candidates
        )

    def _authored_observation(self) -> str:
        p, q = self._ordered_endpoints(self._segment)
        run = self._heading_phrase(q[0] - p[0], q[1] - p[1])
        start = Point(p[0], p[1])
        end = Point(q[0], q[1])
        return (
            f"The segment runs {run}, from one hollow endpoint {self._canvas_region(start)} to the "
            f"other {self._canvas_region(end)}. The midpoint must lie on the segment itself with equal distances "
            f"to those two endpoints, so I walk in from both ends and meet halfway, "
            f"{self._canvas_region(self.target_point)}; a marker that is merely near the middle but off the "
            "segment, or on it but closer to one end, does not qualify."
        )

    def _build_structured_answer(self) -> Dict[str, Any]:
        p, q = self._ordered_endpoints(self._segment)
        measurements = self._candidate_measurements()
        runner_up = sorted(float(item["midpoint_error_pixels"]) for item in measurements)[1]
        return {
            "visible_endpoints": {"P": list(p), "Q": list(q)},
            "candidate_centers": [
                {"label": item["label"], "center": item["center"]}
                for item in measurements
            ],
            "candidate_measurements": measurements,
            "derived_midpoint": self.target_point.to_list(),
            "segment_length_pixels": math.dist(p, q),
            "half_length_pixels": math.dist(p, q) * 0.5,
            "correct_option": self.correct_label,
            "correct_index": ord(self.correct_label) - ord("A"),
            "runner_up_midpoint_error_pixels": runner_up,
            "multiple_solution_policy": "exactly_one_displayed_candidate_center_equals_the_coordinate_midpoint",
            "state_trace": list(self.CONSTRUCTION_TRACE),
            "answer_schema": {
                "terminal_text_regex": FIVE_CHOICE_CODEC.full_pattern,
                "requires_sample_specific_reasoning_before_terminal_answer": True,
                "terminal_text_contains_exact_answer": True,
                "no_block_after_terminal_image": True,
            },
            "verifier_contract": {
                "profile": "midpoint_multimodal_semantic_hard_gate_v1",
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
                    "segment": list(self.CANDIDATE_OUTLINE_COLOR),
                    "selected_outline": list(self.CANDIDATE_HIGHLIGHT_COLOR),
                    "selected_fill": list(self.CANDIDATE_HIGHLIGHT_FILL),
                },
            },
            "expected_video_frame_count": self.EXPECTED_VIDEO_FRAMES,
            "video_fps": 16,
        }

    def create_puzzle(self, *, puzzle_id: Optional[str]=None) -> MidpointPuzzleRecord:
        self._refresh_geometry_prompts()
        midpoint = self.pick_target_point(0.48, padding=self.candidate_anchor_padding(extra=self.canvas_short_side * 0.12))
        segment = self._build_segment(midpoint.to_list())
        self.target_point = midpoint
        self._segment = segment
        p, q = self._ordered_endpoints(segment)
        start, end = (Point(*p), Point(*q))
        anchor_clearance = self._anchor_clearance()
        half_length = math.dist(p, q) / 2.0
        for _ in range(120):
            self.place_candidates_on_incidence(midpoint, [Incidence.segment(start, end)], min_offset=self.MIN_RUNNER_UP_MIDPOINT_ERROR_PIXELS, max_offset=half_length - anchor_clearance, reject=lambda point: min(math.dist((point.x, point.y), p), math.dist((point.x, point.y), q)) < anchor_clearance)
            errors = sorted((math.hypot(candidate.x - midpoint.x, candidate.y - midpoint.y) for candidate in self.candidates))
            if len(errors) == 5 and errors[1] >= self.MIN_RUNNER_UP_MIDPOINT_ERROR_PIXELS and self._candidate_layout_clear_of_anchors():
                break
        else:
            raise RuntimeError('Failed to place five separated midpoint candidates')
        self._compose_off_centre()
        pid = puzzle_id or str(uuid.uuid4())
        puzzle_img = self._downscale(self._render_state())
        solution_img = self._downscale(self._render_state(segment_progress=1.0, first_half_progress=1.0, second_half_progress=1.0, highlight_progress=1.0))
        puzzle_path = self.puzzle_dir / f'{pid}_puzzle.png'
        solution_path = self.solution_dir / f'{pid}_solution.png'
        puzzle_img.save(puzzle_path)
        solution_img.save(solution_path)
        self.puzzle_path = puzzle_path
        self.solution_path = solution_path
        video_rel_path: Optional[str] = None
        video_num_frames: Optional[int] = None
        if self.record_video:
            video_num_frames = self.save_video_solution(pid)
            video_abs = self.solution_dir / f'{pid}_solution.webm'
            if video_abs.exists():
                video_rel_path = self.relativize_path(video_abs)
        solution_rel = self.relativize_path(solution_path)
        structured = self._build_structured_answer()
        task_contract = {'logic': {'endpoint_naming': 'smaller_x_then_smaller_y_is_P', 'midpoint_rule': 'coordinate_average_and_strictly_between_equal_halves', 'candidate_semantics': 'marker_center_only', 'boundary_policy': 'equidistant_off_segment_endpoint_beyond_endpoint_and_approximate_matches_are_invalid'}, 'answer': {'parser_profile': 'sample_specific_candidate_measurement_table_then_strict_uppercase_option_letter_a_to_e'}, 'vision': {'unchanged_objects': 'both_anchors_all_candidate_centers_labels_background_and_camera', 'gravity': 'none', 'forbidden_mutations': ['falling', 'compaction', 'refill', 'deletion', 'duplication', 'reflow', 'camera_motion']}, 'verifier': structured['verifier_contract']}
        return MidpointPuzzleRecord(id=pid, ti2v_prompt=self.ti2v_prompt, canvas_dimensions=self.canvas_dimensions, margin=self.margin, midpoint=midpoint.to_list(), segment=segment, candidates=self.candidates, point_radius=self.point_radius, line_width=self.line_width, correct_option=self.correct_label, image=self.relativize_path(puzzle_path), solution_image_path=solution_rel, structured_answer=structured, task_contract=task_contract, seed=self.seed, reasoning_image=self.relativize_path(puzzle_path), solution_video_path=video_rel_path, video_fps=16 if video_rel_path else None, video_num_frames=video_num_frames if video_rel_path else None, construction_trace=list(self.CONSTRUCTION_TRACE))

    def create_random_puzzle(self) -> MidpointPuzzleRecord:
        return self.create_puzzle()

    def _build_segment(self, midpoint: Tuple[float, float]) -> Segment:
        midpoint_point = Point(*midpoint)
        for _ in range(200):
            angle = self.rng.uniform(0.0, math.tau)
            try:
                first, second = self.sample_symmetric_segment(
                    midpoint_point,
                    angle,
                    # Each half must host two collinear distractors that clear
                    # the runner-up margin and the endpoint anchor, so the
                    # segment can no longer be short enough to force markers
                    # off the ink.
                    # Each half has to hold all four distractors, not two, so
                    # the answer can be the first marker along the segment as
                    # readily as the third.  A segment sized for a row centred
                    # on the midpoint hands over "the middle marker"

                    min_half_length=self.canvas_short_side * 0.33,
                    max_half_length=self.canvas_short_side * 0.345,
                    padding=self.point_radius + 8,
                )
            except RuntimeError:
                continue
            endpoints = sorted((first.to_list(), second.to_list()), key=lambda item: (item[0], item[1]))
            if all(self._inside_bounds(point) for point in endpoints):
                return Segment(start=Point(*endpoints[0]), end=Point(*endpoints[1]))
        left, top, right, bottom = self.canvas_bounds()
        mx, my = midpoint
        half_length = min(0.24 * (right - left), 0.24 * (bottom - top))
        return Segment(
            start=Point(max(left + 16, mx - half_length), my),
            end=Point(min(right - 16, mx + half_length), my),
        )

    def _inside_bounds(self, point: Tuple[float, float]) -> bool:
        left, top, right, bottom = self.canvas_bounds()
        return left <= point[0] <= right and top <= point[1] <= bottom

    @staticmethod
    def _interpolate(start: Tuple[float, float], end: Tuple[float, float], progress: float) -> Tuple[float, float]:
        value = max(0.0, min(1.0, float(progress)))
        return (
            start[0] + (end[0] - start[0]) * value,
            start[1] + (end[1] - start[1]) * value,
        )

    @staticmethod
    def _blend_color(start: Sequence[int], end: Sequence[int], progress: float) -> Tuple[int, int, int]:
        value = max(0.0, min(1.0, float(progress)))
        return tuple(int(round(a + (b - a) * value)) for a, b in zip(start, end))

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
        radius = round(self.point_radius * ss)
        fill = self._blend_color(self.CANDIDATE_BASE_FILL, self.CANDIDATE_HIGHLIGHT_FILL, progress)
        outline = self._blend_color(self.CANDIDATE_OUTLINE_COLOR, self.CANDIDATE_HIGHLIGHT_COLOR, progress)
        for candidate in sorted(self.candidates, key=lambda item: item.label):
            cx, cy = round(candidate.x * ss), round(candidate.y * ss)
            selected = candidate.label == self.correct_label
            raw_draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=fill if selected else self.CANDIDATE_BASE_FILL,
                outline=outline if selected else self.CANDIDATE_OUTLINE_COLOR,
                width=round(self.CANDIDATE_OUTLINE_WIDTH * ss),
            )
            bbox = font.getbbox(candidate.label)
            raw_draw.text(
                (cx - (bbox[0] + bbox[2]) // 2, cy - (bbox[1] + bbox[3]) // 2 + round(self.CANDIDATE_LABEL_OFFSET_Y * ss)),
                candidate.label,
                fill=self.CANDIDATE_TEXT_COLOR,
                font=font,
            )

    def _draw_anchor(self, draw: Any, point: Tuple[float, float]) -> None:
        anchor_radius = max(self.point_radius + 6, int(round(min(self.canvas_dimensions) * 0.028)))
        x, y = int(round(point[0])), int(round(point[1]))
        from data.geometry.core.prompt_paraphrase import workbench_active
        if workbench_active():
            body = anchor_radius + 10
            bevel = body * 0.4
            draw.polygon([(x-body+bevel,y-body),(x+body-bevel,y-body),
                          (x+body,y-body+bevel),(x+body,y+body-bevel),
                          (x+body-bevel,y+body),(x-body+bevel,y+body),
                          (x-body,y+body-bevel),(x-body,y-body+bevel)],
                         fill=(190,194,175), outline=(59,69,60), width=3)
            for side in (-1,1):
                draw.line(((x+side*(body-5)-2,y),(x+side*(body-5)+2,y)),
                          fill=(59,69,60),width=2)
        draw.ellipse(
            [x - anchor_radius, y - anchor_radius, x + anchor_radius, y + anchor_radius],
            fill=(250, 250, 250),
            outline=(30, 30, 30),
            width=max(3, anchor_radius // 3),
        )

    def _render_state(
        self,
        *,
        segment_progress: float = 0.0,
        first_half_progress: float = 0.0,
        second_half_progress: float = 0.0,
        highlight_progress: float = 0.0,
    ) -> Image.Image:
        draw, base = self.get_draw_base()
        p, q = self._ordered_endpoints(self._segment)
        construction_width = max(2, int(round(min(self.canvas_dimensions) * 0.01)))
        from data.geometry.core.prompt_paraphrase import workbench_active, draw_survey_rail, given_colour
        if workbench_active():
            draw_survey_rail(draw, [p, q], self.CANDIDATE_OUTLINE_COLOR, construction_width)
        draw.line(
            [tuple(map(round, p)), tuple(map(round, q))],
            fill=given_colour(self.CANDIDATE_OUTLINE_COLOR),
            width=construction_width,
        )
        self._draw_anchor(draw, p)
        self._draw_anchor(draw, q)
        self._draw_candidates_with_highlight_progress(draw, highlight_progress)
        return base

    def _render(
        self,
        *,
        highlight_label: Optional[str],
        midpoint: Optional[Tuple[float, float]] = None,
        segment: Optional[Segment] = None,
    ) -> Image.Image:
        if midpoint is not None:
            self.target_point = Point(*midpoint)
        if segment is not None:
            self._segment = segment
        solved = bool(highlight_label)
        return self._render_state(
            segment_progress=1.0 if solved else 0.0,
            first_half_progress=1.0 if solved else 0.0,
            second_half_progress=1.0 if solved else 0.0,
            highlight_progress=1.0 if solved else 0.0,
        )


    def save_video_solution(self, pid: str) -> Optional[int]:
        frames: List[Image.Image] = []

        def frame(**kwargs: float) -> Image.Image:
            return self._downscale(self._render_state(**kwargs))

        input_frame = frame()
        frames.extend(input_frame.copy() for _ in range(4))
        frames.extend(
            frame(highlight_progress=index / 4.0)
            for index in range(1, 5)
        )
        final_frame = frame(highlight_progress=1.0)
        frames.extend(final_frame.copy() for _ in range(9))
        if len(frames) != self.EXPECTED_VIDEO_FRAMES or len(frames) % 4 != 1:
            raise RuntimeError(f"midpoint produced invalid frame count {len(frames)}")
        output_path = self.solution_dir / f"{pid}_solution.webm"
        arrays = [np.asarray(item.convert("RGB"), dtype=np.uint8) for item in frames]
        if not encode_rgb_frames_to_webm(arrays, output_path, fps=16):
            raise RuntimeError("Failed to encode midpoint semantic video")
        return len(frames)


__all__ = ["MidpointGenerator", "MidpointPuzzleRecord", "Segment", "CandidatePoint"]

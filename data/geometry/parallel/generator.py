"""Rule-complete parallel-through-point option puzzle generator."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from data.geometry.circumcenter.generator import CircumcenterGenerator
from data.geometry.core.point_target_base import (
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.typed_answers import FIVE_CHOICE_CODEC


def distance_to_segment(point: Point, first: Point, second: Point) -> float:
    """Return the distance from a point to the finite segment through two points."""

    dx = second.x - first.x
    dy = second.y - first.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return math.hypot(point.x - first.x, point.y - first.y)
    along = ((point.x - first.x) * dx + (point.y - first.y) * dy) / length_sq
    along = max(0.0, min(1.0, along))
    return math.hypot(point.x - (first.x + dx * along), point.y - (first.y + dy * along))


def distance_to_infinite_line(point: Point, first: Point, second: Point) -> float:
    """Return the perpendicular distance to the infinite line through two points."""

    dx = second.x - first.x
    dy = second.y - first.y
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return float("inf")
    return abs(dx * (point.y - first.y) - dy * (point.x - first.x)) / length


class ParallelGenerator(CircumcenterGenerator):
    """Select the unique candidate on the parallel through a marked point."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: The input is white image containing exactly one fixed medium-width black reference segment, "
        "one separate small hollow circular through-point marker, and five separate circular candidates labeled "
        "A, B, C, D, and E. The reference segment is finite only as a visible direction sample: geometrically it "
        "determines an undirected infinite-line direction, so either of its two opposite orientations is the same "
        "parallel direction. The required line is the unique infinite straight line through the center of the "
        "hollow marker with that same direction. It need not pass through, touch, or lie within the endpoint "
        "projection of the visible reference segment. Candidate centers alone count; marker outlines, fills, and "
        "letter glyphs have no geometric extent, and a line hidden beneath an opaque marker continues straight "
        "through its center. Exactly one candidate center has zero perpendicular offset from the required line. "
        "Every other candidate center has a visibly nonzero offset; a nearly parallel or nearly collinear center "
        "is invalid unless its exact perpendicular offset is zero. Call the reference endpoint with smaller x R1 "
        "and the other endpoint R2, breaking an exact x tie by smaller y. Rule: Identify the unique zero-offset center and explain that "
        "reference-endpoint extent and line orientation do not change the parallel rule. Required Output: Return "
        "exactly one uppercase option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/parallel"
    # Circumcenter turns the shared frame composition off because its own
    # bisectors are cut by the frame; nothing here is, so this task takes the
    # family default back.
    COMPOSITION_REACH_RATIO: float = PointTargetPuzzleGenerator.COMPOSITION_REACH_RATIO
    CONSTRUCTION_COLOR: Tuple[int, int, int] = (20, 151, 188)
    MIN_RUNNER_UP_OFFSET_PIXELS = 24.0
    GUIDE_HALF_LENGTH_PIXELS = 36.0
    EXPECTED_VIDEO_FRAMES = 21
    STATE_TRACE = [
        "input_hold",
        "parallel_line_single_end_growth",
        "correct_candidate_highlight",
        "final_hold",
    ]

    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Use the supplied input as the fixed base and produce exactly final solution image. Preserve the white "
        "background, the complete original black reference segment and both endpoints, the hollow through-point "
        "marker, all five candidate centers, marker sizes, outlines, labels, spacing, scale, framing, and camera. "
        "Add exactly one medium-thin cyan-blue straight construction segment on the required parallel line. It "
        "has a short the visible scale tail on the side of T opposite the winning candidate, passes geometrically "
        "through T, and continues on the winner side until the correct candidate center. The opaque T and winner "
        "markers are drawn above it, so the line is hidden inside those two marker disks and visibly stops at "
        "their outside edges. The new segment must have exactly the same undirected direction as the reference "
        "segment and must not bend, overshoot, branch, or extend through a distractor. Change only the correct "
        "candidate to pale-red fill with a dark-red outline while keeping its black letter readable. Keep the "
        "other four candidates white and unchanged. No source object moves, rotates, falls, compacts, refills, "
        "reorders, disappears, duplicates, changes size or identity, or reflows. Add no extension of the "
        "reference segment, arrowhead, perpendicular guide, angle mark, second line, second highlight, coordinate "
        "label, legend, formula, answer banner, or other text."
    )
    DEFAULT_TI2V_PROMPT = (
        "Input Image: The input is white image containing exactly one fixed medium-width black reference segment, "
        "one separate small hollow circular through-point marker, and five separate circular candidates labeled "
        "A, B, C, D, and E. The reference segment is finite only as a visible direction sample: geometrically it "
        "determines an undirected infinite-line direction, so either of its two opposite orientations is the same "
        "parallel direction. The required line is the unique infinite straight line through the center of the "
        "hollow marker with that same direction. It need not pass through, touch, or lie within the endpoint "
        "projection of the visible reference segment. Candidate centers alone count; marker outlines, fills, and "
        "letter glyphs have no geometric extent, and a line hidden beneath an opaque marker continues straight "
        "through its center. Exactly one candidate center has zero perpendicular offset from the required line. "
        "Every other candidate center has a visibly nonzero offset; a nearly parallel or nearly collinear center "
        "is invalid unless its exact perpendicular offset is zero. Call the reference endpoint with smaller x R1 "
        "and the other endpoint R2, breaking an exact x tie by smaller y. Rule: Identify the unique zero-offset center and explain that "
        "reference-endpoint extent and line orientation do not change the parallel rule. Required Output: "
        "Hold the unchanged input. Draw the complete parallel construction continuously from its short tail on "
        "one side of T, through T, to the unique candidate center on the other side. Use one advancing endpoint: "
        "do not grow from T in both directions and do not reveal a guide or the full line all at once. Then mark "
        "the selected candidate. Use a fixed camera, preserve every unchanged object, and show each semantic construction or commit in rule order "
        "without teleportation, hard cuts, full-frame crossfades, flicker, disappearance, or identity changes."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._geometry_input_description = (
            "A finite black reference segment, one separate hollow through-point, and five labeled candidate "
            "centers are visible. The segment supplies an undirected infinite-line direction."
        )
        self._geometry_rule_description = (
            "Construct the unique line through the hollow point parallel to the reference direction and select "
            "the only candidate center on it. Segment endpoint extent and marker outlines do not count."
        )
        self._geometry_solution_description = (
            "Preserve the reference segment, through-point, and candidates. Add one complete cyan parallel line "
            "from its short tail through the through-point to the unique candidate center, and mark only that "
            "candidate."
        )
        self._geometry_video_description = (
            "Hold the unchanged input, grow the complete cyan parallel line continuously from its short tail "
            "through the hollow point to the unique candidate with one advancing endpoint, then smoothly mark the "
            "answer and hold the solved state."
        )

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin = 72
        spread = self.minimum_candidate_spacing()
        candidate_padding = self.candidate_anchor_padding(extra=spread * 2.15)
        anchor_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.025)

        for _ in range(4000):
            line_angle = self.rng.uniform(0.0, math.tau)
            try:
                target = self.pick_target_point(0.58, padding=candidate_padding)
            except ValueError:
                continue
            through_direction = line_angle + (math.pi if self.rng.random() < 0.5 else 0.0)
            try:
                through_point = self.sample_point_along_direction(
                    target,
                    through_direction,
                    min_distance=self.canvas_short_side * 0.27,
                    max_distance=self.canvas_short_side * 0.35,
                    padding=anchor_padding,
                )
                reference_mid = self.sample_point_along_direction(
                    through_point,
                    line_angle + math.pi / 2.0 * self.rng.choice((-1, 1)),
                    min_distance=self.canvas_short_side * 0.19,
                    max_distance=self.canvas_short_side * 0.27,
                    padding=self.line_width * 2.0,
                )
                p1 = self.sample_point_along_direction(
                    reference_mid,
                    line_angle,
                    min_distance=self.canvas_short_side * 0.14,
                    max_distance=self.canvas_short_side * 0.21,
                    padding=self.line_width * 2.0,
                )
                p2 = self.sample_point_along_direction(
                    reference_mid,
                    line_angle + math.pi,
                    min_distance=self.canvas_short_side * 0.14,
                    max_distance=self.canvas_short_side * 0.21,
                    padding=self.line_width * 2.0,
                )
            except RuntimeError:
                continue
            if self.distance(p1, p2) < self.canvas_short_side * 0.28:
                continue
            self.points = (through_point, p1, p2)
            self.target_point = target

            # A distractor may sit anywhere that is visibly off the required
            # line, clear of the through-point and clear of the reference
            # segment.  The five markers used to be a row laid out square to
            # the reference direction, with the answer on the rung the
            # through-point projects onto: "the marker nearest the hollow
            # point" then named the answer every time, because the row met the
            # required line at right angles and only the answer had no
            # perpendicular offset from it.  Seating the arrangement first and
            # taking one of its slots for the answer leaves the row no
            # relation to the line direction at all.
            segment_clear = float(self.point_radius) * 2.2
            anchor_clear = float(self.point_radius) * 2.2 + self.line_width

            def usable(point: Point) -> bool:
                if distance_to_infinite_line(point, through_point, target) < self.MIN_RUNNER_UP_OFFSET_PIXELS:
                    return False
                if self.distance(point, through_point) < anchor_clear:
                    return False
                return distance_to_segment(point, p1, p2) >= segment_clear

            try:
                self.seat_candidates(target, accept=usable, min_separation=spread)
            except RuntimeError:
                continue
            if min(
                distance_to_infinite_line(Point(item.x, item.y), p1, p2)
                for item in self.candidates
            ) < self.point_radius * 2.2:
                continue
            measurements = self._candidate_measurements()
            ordered = sorted(measurements, key=lambda item: float(item["perpendicular_offset_pixels"]))
            if (
                ordered[0]["label"] != self.correct_label
                or float(ordered[0]["perpendicular_offset_pixels"]) > 1e-5
                or float(ordered[1]["perpendicular_offset_pixels"]) < self.MIN_RUNNER_UP_OFFSET_PIXELS
            ):
                continue
            return self.save_puzzle()
        raise RuntimeError("Failed to generate valid balanced parallel geometry after 4000 tries")

    def _ordered_reference_endpoints(self) -> Tuple[Point, Point]:
        first, second = self.points[1], self.points[2]
        return (first, second) if (first.x, first.y) <= (second.x, second.y) else (second, first)

    def _reference_unit(self) -> Tuple[float, float]:
        first, second = self._ordered_reference_endpoints()
        dx, dy = second.x - first.x, second.y - first.y
        length = math.hypot(dx, dy)
        return dx / length, dy / length

    def _candidate_measurements(self) -> List[Dict[str, Any]]:
        through_point = self.points[0]
        ux, uy = self._reference_unit()
        measurements: List[Dict[str, Any]] = []
        for candidate in sorted(self.candidates, key=lambda item: item.label):
            dx = candidate.x - through_point.x
            dy = candidate.y - through_point.y
            along = dx * ux + dy * uy
            signed_offset = ux * dy - uy * dx
            measurements.append(
                {
                    "label": candidate.label,
                    "center": [candidate.x, candidate.y],
                    "direction_from_T": self._direction_name(dx, dy),
                    "along_line_projection_pixels": along,
                    "signed_perpendicular_offset_pixels": signed_offset,
                    "perpendicular_offset_pixels": abs(signed_offset),
                    "eligible": abs(signed_offset) <= 1e-5,
                }
            )
        return measurements

    def _reference_reasoning_text(self) -> str:
        r1, r2 = self._ordered_reference_endpoints()
        vector_x, vector_y = r2.x - r1.x, r2.y - r1.y
        orientation = self._direction_name(vector_x, vector_y)
        text = (
            f"The reference segment runs roughly {orientation}. I carry that same undirected direction through the "
            "hollow point T; extending either finite segment does not change the parallel relation."
        )
        comparison = (
            f"I compare the marker centers with the transferred line. Only {self.correct_label} lies exactly on it; "
            "the other markers are visible near-misses above or below the line."
        )
        terminal = (
            f"The required infinite line through T reaches the center of {self.correct_label}, while every other "
            "center remains off the line. Marker outlines and letters do not count, and a near miss is invalid. "
            f"Therefore {self.correct_label} is the unique correct candidate center."
        )
        return (
            f"{text}\n{comparison}\n{terminal}\n\n"
            f"{FIVE_CHOICE_CODEC.format(self.correct_label)}"
        )

    def build_record_extra(self) -> Dict[str, Any]:
        r1, r2 = self._ordered_reference_endpoints()
        through_point = self.points[0]
        ux, uy = self._reference_unit()
        measurements = self._candidate_measurements()
        runner_up = sorted(float(item["perpendicular_offset_pixels"]) for item in measurements)[1]
        derived_index = next(index for index, item in enumerate(measurements) if item["label"] == self.correct_label)
        structured = {
            "reference_endpoints": {"R1": r1.to_list(), "R2": r2.to_list()},
            "reference_vector": [r2.x - r1.x, r2.y - r1.y],
            "reference_unit_vector": [ux, uy],
            "through_point": through_point.to_list(),
            "candidate_centers": [{"label": item["label"], "center": item["center"]} for item in measurements],
            "candidate_measurements": measurements,
            "correct_option": self.correct_label,
            "correct_index": derived_index,
            "runner_up_perpendicular_offset_pixels": runner_up,
            "multiple_solution_policy": "exactly_one_displayed_candidate_center_is_on_the_parallel_through_T",
            "state_trace": list(self.STATE_TRACE),
            "answer_schema": {
                "terminal_text_regex": FIVE_CHOICE_CODEC.full_pattern,
                "requires_sample_specific_reasoning_before_terminal_answer": True,
                "terminal_text_contains_exact_answer": True,
                "no_block_after_terminal_image": True,
            },
            "verifier_contract": {
                "profile": "parallel_multimodal_semantic_hard_gate_v1",
                "judged_artifacts": ["text", "final_image", "full_video"],
                "strict_binary_correctness": True,
                "performance_policy": "latency_and_throughput_are_diagnostics_only",
            },
            "render_geometry": {
                "canvas_dimensions": list(self.canvas_dimensions),
                "point_radius": self.point_radius,
                "through_point_radius": 7,
                "line_width": self.line_width,
                "colors": {
                    "background": [255, 255, 255],
                    "reference_segment": list(self.CANDIDATE_OUTLINE_COLOR),
                    "parallel_construction": list(self.CONSTRUCTION_COLOR),
                    "selected_outline": list(self.CANDIDATE_HIGHLIGHT_COLOR),
                    "selected_fill": list(self.CANDIDATE_HIGHLIGHT_FILL),
                },
            },
            "expected_video_frame_count": self.EXPECTED_VIDEO_FRAMES,
            "video_fps": 16,
        }
        return {
            "task_name": "parallel",
            "task_family": "geometry",
            "source_line_endpoints": [r1.to_list(), r2.to_list()],
            "through_point": through_point.to_list(),
            "candidate_measurements": measurements,
            "structured_answer": structured,
            "construction_trace": list(self.STATE_TRACE),
            "task_contract": {
                "logic": {
                    "parallel_definition": "zero_perpendicular_offset_from_the_infinite_line_through_T_with_reference_direction",
                    "candidate_semantics": "marker_center_only",
                    "boundary_policy": "finite_reference_extent_and_opposite_vector_orientation_do_not_limit_the_parallel_line",
                },
                "answer": {
                    "parser_profile": "sample_specific_reference_vector_candidate_offset_table_then_strict_uppercase_option",
                },
                "vision": {
                    "unchanged_objects": "reference_segment_T_all_candidate_centers_labels_background_scale_framing_and_camera",
                    "gravity": "none",
                    "forbidden_mutations": ["falling", "compaction", "refill", "deletion", "duplication", "reflow", "camera_motion"],
                },
                "verifier": structured["verifier_contract"],
            },
        }

    def _construction_points(self) -> Tuple[Point, Point, Point]:
        through_point = self.points[0]
        dx = self.target_point.x - through_point.x
        dy = self.target_point.y - through_point.y
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        tail = Point(
            through_point.x - ux * self.GUIDE_HALF_LENGTH_PIXELS,
            through_point.y - uy * self.GUIDE_HALF_LENGTH_PIXELS,
        )
        guide_front = Point(
            through_point.x + ux * self.GUIDE_HALF_LENGTH_PIXELS,
            through_point.y + uy * self.GUIDE_HALF_LENGTH_PIXELS,
        )
        final_end = Point(
            self.target_point.x - ux * float(self.point_radius + 2),
            self.target_point.y - uy * float(self.point_radius + 2),
        )
        return tail, guide_front, final_end

    def _render_state(
        self,
        *,
        line_progress: float = 0.0,
        highlight_progress: float = 0.0,
    ) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, list(self._ordered_reference_endpoints()))
        progress = max(0.0, min(1.0, float(line_progress)))
        tail, _guide_front, final_end = self._construction_points()
        if progress > 0.0:
            current_end = Point(
                tail.x + (final_end.x - tail.x) * progress,
                tail.y + (final_end.y - tail.y) * progress,
            )
            self.draw_line(draw, [tail, current_end], width_factor=0.72, fill=self.CONSTRUCTION_COLOR)
        self._draw_through_point(draw)
        self._draw_candidates_with_highlight_progress(draw, highlight_progress)
        return base

    def _draw_through_point(self, draw: Any) -> None:
        self.draw_anchor_marker(draw, self.points[0], 7)

    def _video_overlay_extras(self, draw: Any) -> None:
        self._draw_through_point(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        solved = bool(highlight_label)
        return self._render_state(
            line_progress=1.0 if solved else 0.0,
            highlight_progress=1.0 if solved else 0.0,
        )

    def save_video_solution(self, pid: str) -> Optional[int]:
        frames: List[Image.Image] = []

        def frame(
            line: float = 0.0,
            highlight: float = 0.0,
        ) -> Image.Image:
            return self._downscale(
                self._render_state(
                    line_progress=line,
                    highlight_progress=highlight,
                )
            )

        input_frame = frame()
        frames.extend(input_frame.copy() for _ in range(4))
        frames.extend(frame(line=step / 8.0) for step in range(1, 9))
        frames.extend(frame(line=1.0, highlight=step / 4.0) for step in range(1, 5))
        final_frame = frame(line=1.0, highlight=1.0)
        frames.extend(final_frame.copy() for _ in range(5))
        if len(frames) != self.EXPECTED_VIDEO_FRAMES or len(frames) % 4 != 1:
            raise RuntimeError(f"parallel produced invalid frame count {len(frames)}")
        output_path = self.solution_dir / f"{pid}_solution.webm"
        arrays = [np.asarray(item.convert("RGB"), dtype=np.uint8) for item in frames]
        if not encode_rgb_frames_to_webm(arrays, output_path, fps=16):
            raise RuntimeError("Failed to encode parallel semantic video")
        return len(frames)

"""Rule-complete orthocenter option puzzle generator."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.geometry.circumcenter.generator import CircumcenterGenerator
from data.geometry.core.point_target_base import Point, PointTargetPuzzleRecord
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.typed_answers import FIVE_CHOICE_CODEC


def calculate_orthocenter(
    p1: Point,
    p2: Point,
    p3: Point,
) -> Optional[Point]:
    """Return the unique intersection of two altitude lines."""

    a1 = p2.x - p3.x
    b1 = p2.y - p3.y
    d1 = a1 * p1.x + b1 * p1.y
    a2 = p1.x - p3.x
    b2 = p1.y - p3.y
    d2 = a2 * p2.x + b2 * p2.y
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-9:
        return None
    return Point(
        (d1 * b2 - d2 * b1) / determinant,
        (a1 * d2 - a2 * d1) / determinant,
    )


def project_to_line(point: Point, first: Point, second: Point) -> Tuple[Point, float]:
    """Project ``point`` onto the infinite line through ``first`` and ``second``."""

    dx = second.x - first.x
    dy = second.y - first.y
    denominator = dx * dx + dy * dy
    if denominator <= 1e-9:
        return first, 0.0
    t = ((point.x - first.x) * dx + (point.y - first.y) * dy) / denominator
    return Point(first.x + t * dx, first.y + t * dy), t


class OrthocenterGenerator(CircumcenterGenerator):
    """Select the unique displayed point lying on all three triangle altitudes."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A fixed triangle and five labeled candidate markers are shown. A candidate may lie inside "
        "or outside the triangle. Rule: For each vertex, use the line through that vertex perpendicular to the "
        "opposite side or its extension. Select the unique candidate where all three altitudes meet. Required "
        "Output: Return exactly one uppercase option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/orthocenter"
    EXTENSION_COLOR: Tuple[int, int, int] = (20, 151, 188)
    MIN_RUNNER_UP_ALTITUDE_OFFSET_PIXELS = 20.0
    EXPECTED_VIDEO_FRAMES = 25
    STATE_TRACE = [
        "input_hold",
        "required_opposite_side_extensions_sequential_single_end_growth",
        "first_altitude_single_end_growth",
        "second_altitude_single_end_growth",
        "third_altitude_single_end_growth",
        "correct_candidate_highlight",
        "final_hold",
    ]

    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Use the supplied input as the fixed base and produce exactly final solution image. Preserve the white "
        "background, all three original black triangle edges and vertices, all five candidate centers, marker "
        "sizes, outlines, labels, spacing, scale, framing, and camera. For V1, then V2, then V3, add one "
        "medium-thin cyan-blue altitude construction. Each construction is one continuous straight segment "
        "spanning its source vertex, the common orthocenter, and its perpendicular foot on the opposite-side "
        "line; use the two farthest of those three collinear points as the visible segment endpoints. If and only "
        "if a foot lies outside the finite opposite side, also add one thinner muted-violet straight side "
        "extension from the nearer original side endpoint to that foot. If a foot lies on the finite side, add no "
        "extension for it. Draw cyan/violet construction beneath the original black triangle and beneath all "
        "candidate markers, so black edges remain unchanged and markers cover only incidental crossings. Change "
        "only the correct candidate to pale-red fill with a dark-red outline while keeping its black letter "
        "readable; keep the other four candidates white and unchanged. No source object moves, rotates, falls, "
        "compacts, refills, reorders, disappears, duplicates, changes size or identity, or reflows. Add no "
        "right-angle square, perpendicular-bisector, angle-bisector, circle, second highlight, vertex/side label, "
        "legend, formula, answer banner, or other text."
    )
    DEFAULT_TI2V_PROMPT = (
        "Input Image: A fixed triangle and five labeled candidate markers are shown. A candidate may lie inside "
        "or outside the triangle. Rule: For each vertex, use the line through that vertex perpendicular to the "
        "opposite side or its extension. Select the unique candidate where all three altitudes meet. Required "
        "Output: Hold the unchanged input. Draw each required opposite-side extension separately from the original "
        "side endpoint to its perpendicular foot. Finish one extension before starting another. Then draw the three "
        "complete altitude segments one at a time, each continuously from one segment endpoint to the other; never "
        "grow an altitude in both directions from its vertex or reveal a whole line at once. Mark the answer only "
        "after all lines are complete. Use a fixed camera, preserve every unchanged object, and show each semantic construction or "
        "commit in rule order without teleportation, hard cuts, full-frame crossfades, flicker, disappearance, or "
        "identity changes."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._geometry_input_description = (
            "A fixed triangle and five labeled candidate markers are shown. A candidate may lie inside or outside the triangle."
        )
        self._geometry_rule_description = (
            "For each vertex, use the line through that vertex perpendicular to the opposite side or its extension. "
            "Select the unique candidate where all three altitudes meet."
        )
        self._geometry_solution_description = (
            "Preserve the triangle and candidates. In cyan, add required opposite-side extensions one complete "
            "line at a time when needed, then add the three complete altitudes one at a time. Mark only the common-"
            "intersection candidate in the terminal image."
        )
        self._geometry_video_description = (
            "Hold the unchanged input. In cyan, draw required opposite-side extensions one at a time from side "
            "endpoint to foot, then draw the three altitude segments one at a time from one endpoint to the other, "
            "and mark the answer only after every line is complete."
        )

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        requested_triangle_type = "obtuse" if self.rng.random() < 0.45 else "acute"

        def triangle_is_usable(a: Point, b: Point, c: Point) -> bool:
            maximum_angle = max(self.triangle_angles(a, b, c))
            if requested_triangle_type == "obtuse" and maximum_angle < 102.0:
                return False
            if requested_triangle_type == "acute" and maximum_angle > 76.0:
                return False
            center = calculate_orthocenter(a, b, c)
            if center is None or not self.point_can_host_candidate(
                center,
                extra_padding=self.canvas_short_side * 0.018,
            ):
                return False
            vertices = (a, b, c)
            if min(math.hypot(center.x - v.x, center.y - v.y) for v in vertices) < self.point_radius * 3.2:
                return False
            for index, vertex in enumerate(vertices):
                others = [vertices[j] for j in range(3) if j != index]
                foot, _ = project_to_line(vertex, others[0], others[1])
                if not self.inside_canvas(foot, padding=self.line_width * 2.0):
                    return False
            return True

        for _ in range(1000):
            p1, p2, p3 = self.sample_triangle_vertices(
                jitter_ratio=0.56,
                min_side_ratio=0.19,
                min_area_ratio=0.038,
                min_altitude_ratio=0.14,
                min_angle_deg=32.0 if requested_triangle_type == "obtuse" else 38.0,
                max_angle_deg=116.0 if requested_triangle_type == "obtuse" else 76.0,
                forbidden_angle_windows=((78.0, 102.0),),
                max_attempts=20000 if requested_triangle_type == "obtuse" else 4000,
                validator=triangle_is_usable,
            )
            center = calculate_orthocenter(p1, p2, p3)
            if center is None:
                continue
            self.triangle_points = (p1, p2, p3)
            self.center = center
            self.target_point = center
            try:
                self.place_candidates(center)
            except RuntimeError:
                continue
            if any(
                min(math.hypot(candidate.x - vertex.x, candidate.y - vertex.y) for vertex in self.triangle_points)
                < self.point_radius * 2.2
                for candidate in self.candidates
            ):
                continue
            ordered = sorted(
                self._candidate_measurements(),
                key=lambda item: float(item["maximum_altitude_offset_pixels"]),
            )
            if (
                ordered[0]["label"] != self.correct_label
                or float(ordered[0]["maximum_altitude_offset_pixels"]) > 1e-5
                or float(ordered[1]["maximum_altitude_offset_pixels"])
                < self.MIN_RUNNER_UP_ALTITUDE_OFFSET_PIXELS
            ):
                continue
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid orthocenter puzzle after 1000 tries")

    def _triangle_type(self) -> str:
        maximum = max(self.triangle_angles(*self.triangle_points))
        if maximum < 90.0 - 1e-6:
            return "acute"
        if maximum > 90.0 + 1e-6:
            return "obtuse"
        return "right"

    def _semantic_phase_roles_for_sample(self) -> tuple[str, ...]:
        if not hasattr(self, "triangle_points"):
            return super()._semantic_phase_roles_for_sample()
        altitude_data = self._altitude_data()
        names = self._visible_point_names(self._ordered_vertices())
        roles: List[str] = []
        for index, item in enumerate(altitude_data):
            if item["extension"] is None:
                continue
            endpoint_index = (
                (index + 1) % 3
                if float(item["side_parameter"]) < 0.0
                else (index + 2) % 3
            )
            roles.append(
                f"the required opposite-side extension beyond the {names[endpoint_index]} for the {names[index]} altitude, with earlier extensions retained"
            )
        roles.extend(
            f"the complete altitude through the {name}, with all earlier extensions and altitudes retained"
            for name in names
        )
        return tuple(roles)

    def _altitude_data(self) -> List[Dict[str, Any]]:
        vertices = self._ordered_vertices()
        data: List[Dict[str, Any]] = []
        for index, vertex in enumerate(vertices):
            side_first = vertices[(index + 1) % 3]
            side_second = vertices[(index + 2) % 3]
            foot, side_t = project_to_line(vertex, side_first, side_second)
            points = (vertex, self.center, foot)
            endpoint_a, endpoint_b = max(
                ((first, second) for first in points for second in points if first is not second),
                key=lambda pair: math.hypot(pair[1].x - pair[0].x, pair[1].y - pair[0].y),
            )
            extension: Optional[Tuple[Point, Point]] = None
            if side_t < 0.0:
                extension = (side_first, foot)
                foot_status = f"beyond V{(index + 1) % 3 + 1}"
            elif side_t > 1.0:
                extension = (side_second, foot)
                foot_status = f"beyond V{(index + 2) % 3 + 1}"
            else:
                foot_status = "on the finite opposite side"
            data.append(
                {
                    "vertex": vertex,
                    "side_first": side_first,
                    "side_second": side_second,
                    "foot": foot,
                    "side_parameter": side_t,
                    "foot_status": foot_status,
                    "segment_start": endpoint_a,
                    "segment_end": endpoint_b,
                    "extension": extension,
                }
            )
        return data

    @staticmethod
    def _altitude_offset(point: Point, vertex: Point, side_first: Point, side_second: Point) -> float:
        side_dx = side_second.x - side_first.x
        side_dy = side_second.y - side_first.y
        return abs(
            (point.x - vertex.x) * side_dx + (point.y - vertex.y) * side_dy
        ) / max(math.hypot(side_dx, side_dy), 1e-9)

    def _candidate_measurements(self) -> List[Dict[str, Any]]:
        altitude_data = self._altitude_data()
        vertices = self._ordered_vertices()
        centroid_x = sum(point.x for point in vertices) / 3.0
        centroid_y = sum(point.y for point in vertices) / 3.0
        measurements: List[Dict[str, Any]] = []
        for candidate in sorted(self.candidates, key=lambda item: item.label):
            point = Point(candidate.x, candidate.y)
            offsets = [
                self._altitude_offset(
                    point,
                    item["vertex"],
                    item["side_first"],
                    item["side_second"],
                )
                for item in altitude_data
            ]
            measurements.append(
                {
                    "label": candidate.label,
                    "center": [candidate.x, candidate.y],
                    "direction_from_triangle_centroid": self._direction_name(
                        candidate.x - centroid_x,
                        candidate.y - centroid_y,
                    ),
                    "altitude_line_offsets_pixels": offsets,
                    "maximum_altitude_offset_pixels": max(offsets),
                    "mean_altitude_offset_pixels": sum(offsets) / 3.0,
                }
            )
        return measurements

    def _reasoning_text(self) -> str:
        vertices = self._ordered_vertices()
        names = self._visible_point_names(vertices)
        measurements = self._candidate_measurements()
        positions = ", ".join(
            f"{item['label']} lies {item['direction_from_triangle_centroid']} of the triangle's middle"
            for item in measurements
        )
        altitude_data = self._altitude_data()
        extended: List[str] = []
        for index, item in enumerate(altitude_data):
            if item["extension"] is None:
                continue
            endpoint_index = (index + 1) % 3 if float(item["side_parameter"]) < 0.0 else (index + 2) % 3
            extended.append(
                f"the side opposite the {names[index]} beyond the {names[endpoint_index]}"
            )
        extension_text = (
            "The exterior perpendicular feet require extending " + ", and ".join(extended) + ". "
            if extended
            else "Every perpendicular foot lands on its finite opposite side, so no side extension is needed. "
        )
        distractors = ", ".join(
            item["label"] for item in measurements if item["label"] != self.correct_label
        )
        location = self._point_in_triangle(self.center, vertices)
        return (
            f"The triangle is {self._triangle_type()}, so I expect its orthocenter {location}. Visually, {positions}. "
            f"{extension_text}The three vertex-to-opposite-side perpendiculars all pass through "
            f"{self.correct_label}; {distractors} each miss at least one altitude. Thus {self.correct_label} is the "
            "unique common intersection."
            f"\n\n{FIVE_CHOICE_CODEC.format(self.correct_label)}"
        )

    def _authored_observation(self) -> str:
        vertices = self._ordered_vertices()
        names = self._visible_point_names(vertices)
        angles = self.triangle_angles(*vertices)
        widest_index = max(range(3), key=lambda index: angles[index])
        if angles[widest_index] > 90.0:
            shape_clause = (
                f"the corner at the {names[widest_index]} opens wider than a right angle, so the "
                "perpendicular feet from the other two corners land beyond it and both of those sides must "
                "be extended before their altitudes can reach their feet; the three altitude lines therefore "
                "cross outside the triangle"
            )
        else:
            shape_clause = (
                "every corner is sharper than a right angle, so each perpendicular foot lands on its finite "
                "opposite side and the three altitudes cross inside the triangle"
            )
        return (
            f"The triangle presents its {names[0]}, {names[1]}, and {names[2]}, and from each corner I run "
            f"the line perpendicular to the opposite side. Here {shape_clause}, meeting "
            f"{self._canvas_region(self.center)}."
        )

    def build_record_extra(self) -> Dict[str, Any]:
        vertices = self._ordered_vertices()
        altitude_data = self._altitude_data()
        measurements = self._candidate_measurements()
        derived_index = next(
            index for index, item in enumerate(measurements)
            if item["label"] == self.correct_label
        )
        runner_up = sorted(
            float(item["maximum_altitude_offset_pixels"])
            for item in measurements
        )[1]
        extension_count = sum(item["extension"] is not None for item in altitude_data)
        structured = {
            "triangle_vertices": [point.to_list() for point in vertices],
            "orthocenter": self.center.to_list(),
            "orthocenter_location_relative_to_triangle": self._point_in_triangle(self.center, vertices),
            "triangle_type": self._triangle_type(),
            "altitude_feet": [item["foot"].to_list() for item in altitude_data],
            "altitude_foot_side_parameters": [item["side_parameter"] for item in altitude_data],
            "altitude_foot_statuses": [item["foot_status"] for item in altitude_data],
            "altitude_segments": [
                [item["segment_start"].to_list(), item["segment_end"].to_list()]
                for item in altitude_data
            ],
            "required_side_extensions": [
                None
                if item["extension"] is None
                else [item["extension"][0].to_list(), item["extension"][1].to_list()]
                for item in altitude_data
            ],
            "candidate_centers": [
                {"label": item["label"], "center": item["center"]}
                for item in measurements
            ],
            "candidate_measurements": measurements,
            "correct_option": self.correct_label,
            "correct_index": derived_index,
            "runner_up_maximum_altitude_offset_pixels": runner_up,
            "multiple_solution_policy": "exactly_one_displayed_candidate_is_the_common_intersection_of_three_altitude_lines",
            "state_trace": list(self.STATE_TRACE),
            "answer_schema": {
                "terminal_text_regex": FIVE_CHOICE_CODEC.full_pattern,
                "requires_sample_specific_reasoning_before_terminal_answer": True,
                "terminal_text_contains_exact_answer": True,
                "no_block_after_terminal_image": True,
            },
            "verifier_contract": {
                "profile": "orthocenter_multimodal_semantic_hard_gate_v1",
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
                    "source_triangle": list(self.CANDIDATE_OUTLINE_COLOR),
                    "altitude_construction": list(self.CONSTRUCTION_COLOR),
                    "side_extension": list(self.EXTENSION_COLOR),
                    "selected_outline": list(self.CANDIDATE_HIGHLIGHT_COLOR),
                    "selected_fill": list(self.CANDIDATE_HIGHLIGHT_FILL),
                },
            },
            "expected_video_frame_count": self.EXPECTED_VIDEO_FRAMES + 4 * extension_count,
            "video_fps": 16,
        }
        return {
            "task_name": "orthocenter",
            "task_family": "geometry",
            "triangle_vertices": structured["triangle_vertices"],
            "orthocenter": structured["orthocenter"],
            "altitude_feet": structured["altitude_feet"],
            "structured_answer": structured,
            "task_contract": {
                "logic": {
                    "orthocenter_definition": "unique_common_intersection_of_three_vertex_to_opposite_line_altitudes",
                    "candidate_semantics": "marker_center_only",
                    "boundary_policy": "inside_for_acute_at_right_vertex_for_right_and_outside_for_obtuse_with_side_extensions_allowed",
                },
                "answer": {
                    "parser_profile": "sample_specific_altitude_offset_table_then_strict_uppercase_option_letter_a_to_e",
                },
                "vision": {
                    "unchanged_objects": "source_triangle_all_candidate_centers_labels_background_scale_framing_and_camera",
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

    def _draw_altitude_progress(
        self,
        draw: Any,
        item: Dict[str, Any],
        *,
        extension_progress: float,
        altitude_progress: float,
    ) -> None:
        extension_progress = max(0.0, min(1.0, extension_progress))
        altitude_progress = max(0.0, min(1.0, altitude_progress))
        start = item["segment_start"]
        end = Point(
            start.x + (item["segment_end"].x - start.x) * altitude_progress,
            start.y + (item["segment_end"].y - start.y) * altitude_progress,
        )
        extension = item["extension"]
        if extension is not None and extension_progress > 0.0:
            extension_end = Point(
                extension[0].x + (extension[1].x - extension[0].x) * extension_progress,
                extension[0].y + (extension[1].y - extension[0].y) * extension_progress,
            )
            self.draw_line(
                draw,
                [extension[0], extension_end],
                width_factor=0.50,
                fill=self.EXTENSION_COLOR,
            )
        if altitude_progress > 0.0:
            self.draw_line(
                draw,
                [start, end],
                width_factor=0.72,
                fill=self.CONSTRUCTION_COLOR,
            )

    def _render_state(
        self,
        *,
        extension_progresses: Sequence[float] = (0.0, 0.0, 0.0),
        altitude_progresses: Sequence[float] = (0.0, 0.0, 0.0),
        highlight_progress: float = 0.0,
    ) -> Image.Image:
        draw, base = self.get_draw_base()
        from data.geometry.core.prompt_paraphrase import workbench_active, draw_survey_rail, given_colour
        from data.common.prompt_paraphrase import use_prompt_paraphrase
        if workbench_active():
            draw_survey_rail(draw, [(p.x,p.y) for p in list(self.triangle_points)+[self.triangle_points[0]]], self.CANDIDATE_OUTLINE_COLOR, self.line_width)
        for item, extension, altitude in zip(
            self._altitude_data(),
            extension_progresses,
            altitude_progresses,
        ):
            self._draw_altitude_progress(
                draw,
                item,
                extension_progress=float(extension),
                altitude_progress=float(altitude),
            )
        # The embossed body sits beneath the construction. Only the thin
        # given centreline is reasserted above it, preserving visible arcs.
        source_colour=given_colour(self.CANDIDATE_OUTLINE_COLOR)
        with use_prompt_paraphrase(None):
            self.draw_line(draw, list(self.triangle_points) + [self.triangle_points[0]], fill=source_colour)
        self._draw_candidates_with_highlight_progress(draw, highlight_progress)
        return base

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        solved = bool(highlight_label)
        return self._render_state(
            extension_progresses=(1.0, 1.0, 1.0) if solved else (0.0, 0.0, 0.0),
            altitude_progresses=(1.0, 1.0, 1.0) if solved else (0.0, 0.0, 0.0),
            highlight_progress=1.0 if solved else 0.0,
        )

    def save_video_solution(self, pid: str) -> Optional[int]:
        frames: List[Image.Image] = []

        def frame(
            extensions: Sequence[float],
            altitudes: Sequence[float],
            highlight: float = 0.0,
        ) -> Image.Image:
            return self._downscale(
                self._render_state(
                    extension_progresses=extensions,
                    altitude_progresses=altitudes,
                    highlight_progress=highlight,
                )
            )

        input_frame = frame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        frames.extend(input_frame.copy() for _ in range(4))
        required = [item["extension"] is not None for item in self._altitude_data()]
        extension_progresses = [0.0, 0.0, 0.0]
        for extension_index, needed in enumerate(required):
            if not needed:
                continue
            for step in range(1, 5):
                current = list(extension_progresses)
                current[extension_index] = step / 4.0
                frames.append(frame(current, (0.0, 0.0, 0.0)))
            extension_progresses[extension_index] = 1.0
        altitude_progresses = [0.0, 0.0, 0.0]
        for altitude_index in range(3):
            for step in range(1, 5):
                current = list(altitude_progresses)
                current[altitude_index] = step / 4.0
                frames.append(frame(extension_progresses, current))
            altitude_progresses[altitude_index] = 1.0
        frames.extend(
            frame(extension_progresses, altitude_progresses, step / 4.0)
            for step in range(1, 5)
        )
        final_frame = frame(extension_progresses, altitude_progresses, 1.0)
        frames.extend(final_frame.copy() for _ in range(5))
        expected_frames = self.EXPECTED_VIDEO_FRAMES + 4 * sum(required)
        if len(frames) != expected_frames or len(frames) % 4 != 1:
            raise RuntimeError(f"orthocenter produced invalid frame count {len(frames)}")
        output_path = self.solution_dir / f"{pid}_solution.webm"
        arrays = [np.asarray(item.convert("RGB"), dtype=np.uint8) for item in frames]
        if not encode_rgb_frames_to_webm(arrays, output_path, fps=16):
            raise RuntimeError("Failed to encode orthocenter semantic video")
        return len(frames)

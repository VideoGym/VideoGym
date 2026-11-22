"""Rule-complete circumcenter option puzzle generator."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFont

from data.common.prompts import soften_precise_video_timing
from data.geometry.core.point_target_base import Point, PointTargetPuzzleGenerator, PointTargetPuzzleRecord
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.typed_answers import FIVE_CHOICE_CODEC


def calculate_circumcenter(
    p1: Point,
    p2: Point,
    p3: Point,
) -> Tuple[Optional[Point], Optional[float]]:
    """Return the unique circumcenter/radius, or ``(None, None)`` if collinear."""

    ax, ay = p1.x, p1.y
    bx, by = p2.x, p2.y
    cx, cy = p3.x, p3.y
    denominator = 2.0 * (
        ax * (by - cy)
        + bx * (cy - ay)
        + cx * (ay - by)
    )
    if abs(denominator) < 1e-9:
        return None, None
    square_a = ax * ax + ay * ay
    square_b = bx * bx + by * by
    square_c = cx * cx + cy * cy
    ux = (
        square_a * (by - cy)
        + square_b * (cy - ay)
        + square_c * (ay - by)
    ) / denominator
    uy = (
        square_a * (cx - bx)
        + square_b * (ax - cx)
        + square_c * (bx - ax)
    ) / denominator
    center = Point(ux, uy)
    return center, math.hypot(ax - ux, ay - uy)


class CircumcenterGenerator(PointTargetPuzzleGenerator):
    """Select the unique displayed point equidistant from three vertices."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A fixed non-collinear triangle and five labeled candidate markers are shown. Rule: Select "
        "the unique candidate at the common intersection of the three side perpendicular bisectors, equivalently "
        "the point equally distant from all three vertices. Required Output: Return exactly one uppercase option "
        "label in the form \"Answer: X.\""
    )

    # This task's three bisectors have no ends of their own: they run until
    # the frame cuts them, and its whole-video verifier rebuilds those cut
    # chords from the record and reads each one's stroke growth off them.
    # Sliding the triangle inside a fixed frame re-cuts the chords, which
    # changes how far each stroke travels per frame, and the reference video
    # then fails its own sequential-stroke gate on roughly one seed in twenty
    # -- twice the rate it already failed at.  Until that gate is loosened to
    # the extent it can actually observe, this task composes its frame the way
    # it always did.  Its answer therefore still reads 0.25 nearest the frame
    # centre against 0.200 chance (0.41 before the shared seating).
    COMPOSITION_REACH_RATIO: float = 0.0

    DEFAULT_OUTPUT_DIR = "outputs/geometry/circumcenter"
    CONSTRUCTION_COLOR: Tuple[int, int, int] = (20, 151, 188)
    MIN_RUNNER_UP_DISTANCE_SPREAD_PIXELS = 24.0
    EXPECTED_VIDEO_FRAMES = 33
    STATE_TRACE = [
        "input_hold",
        "first_perpendicular_bisector_single_end_growth",
        "second_perpendicular_bisector_single_end_growth",
        "third_perpendicular_bisector_single_end_growth",
        "circumcircle_single_end_growth",
        "correct_candidate_highlight",
        "final_hold",
    ]

    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Use the supplied input image as the fixed base and produce exactly final solution image. Preserve the "
        "white background, all three original black triangle edges and vertices, all five candidate centers, "
        "marker sizes, outlines, labels, spacing, scale, framing, and camera. Add the three medium-thin cyan-blue "
        "side perpendicular bisectors, each as one complete straight segment from one clipped canvas endpoint to "
        "the other. Add one medium-width cyan-blue circumcircle centered on their common intersection and passing "
        "exactly through all three vertices. Keep the original black triangle stroke visibly above the cyan "
        "construction, and draw all five candidate markers above it so their fills cover only incidental "
        "crossings. Change only the correct candidate to pale-red "
        "fill with a dark-red outline while keeping its black uppercase letter readable; keep the other four "
        "candidates white with their original dark outlines and letters. No source object moves, rotates, falls, "
        "compacts, refills, reorders, disappears, duplicates, changes identity, changes size, or reflows. Add no "
        "second circle, extra construction lines, extra highlight, vertex labels, legend, "
        "formula, answer banner, or other text."
    )
    DEFAULT_TI2V_PROMPT = (
        "Input Image: A fixed non-collinear triangle and five labeled candidate markers are shown. Rule: Select "
        "the unique candidate at the common intersection of the three side perpendicular bisectors, equivalently "
        "the point equally distant from all three vertices. Required Output: Hold the unchanged input. Draw the "
        "three side perpendicular bisectors one at a time. Grow each complete bisector continuously from one of its "
        "two endpoints to the other, keeping that starting end fixed for the whole stroke; finish it before starting "
        "another, and never reveal a line from its midpoint, from both ends, or all at once. Then draw the "
        "circumcircle continuously in one direction "
        "from one point around to that same point, and finally mark the correct candidate. Use a fixed camera, "
        "preserve every unchanged object, and show each semantic construction or commit in rule order without "
        "teleportation, hard cuts, full-frame crossfades, flicker, disappearance, or identity changes."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        custom_ti2v_prompt = kwargs.get("ti2v_prompt")
        super().__init__(*args, **kwargs)
        self.solution_image_prompt = soften_precise_video_timing(self.DEFAULT_SOLUTION_IMAGE_PROMPT)
        self.ti2v_prompt = soften_precise_video_timing(
            str(custom_ti2v_prompt or self.DEFAULT_TI2V_PROMPT)
        )
        if self._task_name_for_contract() == "circumcenter":
            self._geometry_input_description = (
                "A fixed non-collinear triangle and five labeled candidate markers are shown."
            )
            self._geometry_rule_description = (
                "Select the unique candidate at the common intersection of the three side perpendicular bisectors, "
                "equivalently the point equally distant from all three vertices."
            )
            self._geometry_solution_description = (
                "Preserve the triangle and candidates. Draw the three cyan side perpendicular bisectors "
                "cumulatively, one complete line at a time; after they meet at one candidate, draw the cyan "
                "circumcircle through the vertices and mark "
                "only that candidate in the terminal image."
            )
            self._geometry_video_description = (
                "Hold the unchanged input. Draw the three cyan side perpendicular bisectors one at a time, each "
                "from one of its endpoints to the other with that starting end held fixed. Then draw the cyan "
                "circumcircle in one "
                "continuous direction about their common intersection and finally mark the correct candidate."
            )

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        minimum_radius = self.canvas_short_side * 0.09

        def circle_is_usable(a: Point, b: Point, c: Point) -> bool:
            center, radius = calculate_circumcenter(a, b, c)
            if (
                center is None
                or radius is None
                or radius < minimum_radius
                or not self.point_can_host_candidate(center)
                or not self.circle_fits(center, radius, extra_padding=self.line_width)
            ):
                return False
            vertices = (a, b, c)
            for index, first in enumerate(vertices):
                second = vertices[(index + 1) % 3]
                edge_length = math.hypot(second.x - first.x, second.y - first.y)
                center_to_edge_line = abs(
                    (second.x - first.x) * (center.y - first.y)
                    - (second.y - first.y) * (center.x - first.x)
                ) / max(edge_length, 1e-9)
                if center_to_edge_line < self.point_radius * 1.8:
                    return False
            return True

        for _ in range(1000):
            p1, p2, p3 = self.sample_triangle_vertices(
                jitter_ratio=0.56,
                min_side_ratio=0.18,
                min_area_ratio=0.035,
                min_altitude_ratio=0.13,
                min_angle_deg=38.0,
                max_angle_deg=108.0,
                validator=circle_is_usable,
            )
            center, radius = calculate_circumcenter(p1, p2, p3)
            if center is None or radius is None:
                continue
            self.triangle_points = (p1, p2, p3)
            self.center = center
            self.target_point = center
            self.r = radius
            try:
                self.place_candidates(center)
            except RuntimeError:
                continue
            ordered = sorted(
                self._candidate_measurements(),
                key=lambda item: float(item["distance_spread_pixels"]),
            )
            if (
                ordered[0]["label"] != self.correct_label
                or float(ordered[0]["distance_spread_pixels"]) > 1e-5
                or float(ordered[1]["distance_spread_pixels"])
                < self.MIN_RUNNER_UP_DISTANCE_SPREAD_PIXELS
            ):
                continue
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid circumcenter puzzle after 1000 tries")

    def _semantic_phase_roles_for_sample(self) -> tuple[str, ...]:
        if not hasattr(self, "triangle_points"):
            return super()._semantic_phase_roles_for_sample()
        vertices = self._ordered_vertices()
        names = self._visible_point_names(vertices)
        roles = [
            (
                f"the perpendicular bisector of the side joining the {names[index]} and "
                f"the {names[(index + 1) % 3]}, with earlier bisectors retained"
            )
            for index in range(3)
        ]
        roles.append(
            "all three retained perpendicular bisectors plus the circumcircle through the visible vertices before candidate selection"
        )
        return tuple(roles)


    def _ordered_vertices(self) -> List[Point]:
        vertices = list(self.triangle_points)
        centroid_x = sum(point.x for point in vertices) / 3.0
        centroid_y = sum(point.y for point in vertices) / 3.0
        clockwise = sorted(
            vertices,
            key=lambda point: math.atan2(point.y - centroid_y, point.x - centroid_x),
        )
        start = min(range(3), key=lambda index: (clockwise[index].y, clockwise[index].x))
        return clockwise[start:] + clockwise[:start]

    @staticmethod
    def _point_in_triangle(point: Point, vertices: Sequence[Point]) -> str:
        signs: List[float] = []
        for index in range(3):
            a = vertices[index]
            b = vertices[(index + 1) % 3]
            signs.append(
                (b.x - a.x) * (point.y - a.y)
                - (b.y - a.y) * (point.x - a.x)
            )
        if any(abs(value) <= 1e-6 for value in signs) and not (
            any(value < -1e-6 for value in signs)
            and any(value > 1e-6 for value in signs)
        ):
            return "on the triangle boundary"
        if not (
            any(value < 0.0 for value in signs)
            and any(value > 0.0 for value in signs)
        ):
            return "inside the triangle"
        return "outside the triangle"

    def _candidate_measurements(self) -> List[Dict[str, Any]]:
        vertices = self._ordered_vertices()
        centroid_x = sum(point.x for point in vertices) / 3.0
        centroid_y = sum(point.y for point in vertices) / 3.0
        measurements: List[Dict[str, Any]] = []
        for candidate in sorted(self.candidates, key=lambda item: item.label):
            distances = [
                math.hypot(candidate.x - vertex.x, candidate.y - vertex.y)
                for vertex in vertices
            ]
            measurements.append(
                {
                    "label": candidate.label,
                    "center": [candidate.x, candidate.y],
                    "direction_from_triangle_centroid": self._direction_name(
                        candidate.x - centroid_x,
                        candidate.y - centroid_y,
                    ),
                    "vertex_distances_pixels": distances,
                    "distance_spread_pixels": max(distances) - min(distances),
                    "mean_vertex_distance_pixels": sum(distances) / 3.0,
                }
            )
        return measurements

    def _reasoning_text(self) -> str:
        return (
            "Each side's perpendicular bisector contains every point equidistant from that side's endpoints. The three "
            f"bisectors meet at candidate {self.correct_label}, so a circle centered there passes through all three "
            "vertices.\n\n"
            f"{FIVE_CHOICE_CODEC.format(self.correct_label)}"
        )

    def _authored_observation(self) -> str:
        vertices = self._ordered_vertices()
        names = self._visible_point_names(vertices)
        location = self._point_in_triangle(self.center, vertices)
        return (
            f"The triangle stretches between its {names[0]}, {names[1]}, and {names[2]}. Each side's "
            "perpendicular bisector holds every point equally distant from that side's two endpoints, so the "
            "one point equally distant from all three vertices is where the three bisectors cross; here that "
            f"crossing falls {location}, {self._canvas_region(self.center)}, and the circle centered there "
            "passes through all three corners."
        )

    def build_record_extra(self) -> Dict[str, Any]:
        vertices = self._ordered_vertices()
        measurements = self._candidate_measurements()
        derived_index = next(
            index for index, item in enumerate(measurements)
            if item["label"] == self.correct_label
        )
        runner_up = sorted(
            float(item["distance_spread_pixels"]) for item in measurements
        )[1]
        structured = {
            "triangle_vertices": [point.to_list() for point in vertices],
            "circumcenter": self.center.to_list(),
            "circumradius": self.r,
            "candidate_centers": [
                {"label": item["label"], "center": item["center"]}
                for item in measurements
            ],
            "candidate_measurements": measurements,
            "correct_option": self.correct_label,
            "correct_index": derived_index,
            "runner_up_distance_spread_pixels": runner_up,
            "circumcenter_location_relative_to_triangle": self._point_in_triangle(
                self.center,
                vertices,
            ),
            "multiple_solution_policy": "exactly_one_displayed_candidate_is_the_unique_non_collinear_circumcenter",
            "state_trace": list(self.STATE_TRACE),
            "answer_schema": {
                "terminal_text_regex": FIVE_CHOICE_CODEC.full_pattern,
                "requires_sample_specific_reasoning_before_terminal_answer": True,
                "terminal_text_contains_exact_answer": True,
                "no_block_after_terminal_image": True,
            },
            "verifier_contract": {
                "profile": "circumcenter_multimodal_semantic_hard_gate_v1",
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
                    "construction": list(self.CONSTRUCTION_COLOR),
                    "selected_outline": list(self.CANDIDATE_HIGHLIGHT_COLOR),
                    "selected_fill": list(self.CANDIDATE_HIGHLIGHT_FILL),
                },
            },
            "expected_video_frame_count": self.EXPECTED_VIDEO_FRAMES,
            "video_fps": 16,
        }
        return {
            "task_name": "circumcenter",
            "task_family": "geometry",
            "triangle_vertices": structured["triangle_vertices"],
            "circumcenter": structured["circumcenter"],
            "circumradius": structured["circumradius"],
            "structured_answer": structured,
            "task_contract": {
                "logic": {
                    "circumcenter_definition": "unique_point_equidistant_from_three_non_collinear_vertices",
                    "candidate_semantics": "marker_center_only",
                    "boundary_policy": "inside_for_acute_on_hypotenuse_midpoint_for_right_outside_for_obtuse",
                },
                "answer": {
                    "parser_profile": "sample_specific_vertex_distance_table_then_strict_uppercase_option_letter_a_to_e",
                },
                "vision": {
                    "unchanged_objects": "source_triangle_all_candidate_centers_labels_background_and_camera",
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
        radius = round(self.point_radius * ss)
        fill = self._blend_color(
            self.CANDIDATE_BASE_FILL,
            self.CANDIDATE_HIGHLIGHT_FILL,
            progress,
        )
        outline = self._blend_color(
            self.CANDIDATE_OUTLINE_COLOR,
            self.CANDIDATE_HIGHLIGHT_COLOR,
            progress,
        )
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
            raw_draw.text(
                (tx, ty),
                candidate.label,
                fill=self.CANDIDATE_TEXT_COLOR,
                font=font,
            )

    def _draw_circumcircle_progress(self, draw: Any, progress: float) -> None:
        progress = max(0.0, min(1.0, progress))
        if progress <= 0.0:
            return
        center = self.center
        radius = self.r
        bbox = (
            center.x - radius,
            center.y - radius,
            center.x + radius,
            center.y + radius,
        )
        if progress >= 1.0:
            draw.ellipse(
                bbox,
                outline=self.CONSTRUCTION_COLOR,
                width=self.line_width,
            )
            return
        start_vertex = min(
            self.triangle_points,
            key=lambda point: (point.y, point.x),
        )
        start_degrees = math.degrees(
            math.atan2(
                start_vertex.y - center.y,
                start_vertex.x - center.x,
            )
        )
        draw.arc(
            bbox,
            start=start_degrees,
            end=start_degrees + 360.0 * progress,
            fill=self.CONSTRUCTION_COLOR,
            width=self.line_width,
        )

    def _perpendicular_bisectors(self) -> List[Tuple[Point, Point, Point]]:
        # The reference authoring order is deterministic for reproducibility;
        # endpoint names, not invisible ordinal labels, describe each side.
        # Candidate verification accepts every cumulative permutation of the
        # same three perpendicular bisectors.
        vertices = list(self._ordered_vertices())
        bisectors: List[Tuple[Point, Point, Point]] = []
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % len(vertices)]
            midpoint = Point((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)
            side_angle = math.atan2(second.y - first.y, second.x - first.x)
            start, end = self.clip_line_to_canvas(
                midpoint,
                side_angle + math.pi / 2.0,
                padding=self.line_width + 5,
            )
            bisectors.append((midpoint, start, end))
        return bisectors

    def _draw_bisector_progress(
        self,
        draw: Any,
        bisector: Tuple[Point, Point, Point],
        progress: float,
    ) -> None:
        progress = max(0.0, min(1.0, progress))
        if progress <= 0.0:
            return
        _midpoint, start, end = bisector
        current_end = Point(
            start.x + (end.x - start.x) * progress,
            start.y + (end.y - start.y) * progress,
        )
        self.draw_line(
            draw,
            [start, current_end],
            width_factor=0.65,
            fill=self.CONSTRUCTION_COLOR,
        )

    def _render_state(
        self,
        *,
        bisector_progresses: Sequence[float] = (0.0, 0.0, 0.0),
        circle_progress: float = 0.0,
        highlight_progress: float = 0.0,
    ) -> Image.Image:
        draw, base = self.get_draw_base()
        from data.geometry.core.prompt_paraphrase import workbench_active, draw_survey_rail, given_colour
        from data.common.prompt_paraphrase import use_prompt_paraphrase
        if workbench_active():
            draw_survey_rail(draw, [(p.x,p.y) for p in list(self.triangle_points)+[self.triangle_points[0]]], self.CANDIDATE_OUTLINE_COLOR, self.line_width)
        for bisector, progress in zip(self._perpendicular_bisectors(), bisector_progresses):
            self._draw_bisector_progress(draw, bisector, float(progress))
        self._draw_circumcircle_progress(draw, circle_progress)
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
            bisector_progresses=(1.0, 1.0, 1.0) if solved else (0.0, 0.0, 0.0),
            circle_progress=1.0 if solved else 0.0,
            highlight_progress=1.0 if solved else 0.0,
        )

    def save_video_solution(self, pid: str) -> Optional[int]:
        frames: List[Image.Image] = []

        def frame(**kwargs: float) -> Image.Image:
            return self._downscale(self._render_state(**kwargs))

        input_frame = frame()
        frames.extend(input_frame.copy() for _ in range(4))
        bisector_progresses = [0.0, 0.0, 0.0]
        for bisector_index in range(3):
            for step in range(1, 5):
                current = list(bisector_progresses)
                current[bisector_index] = step / 4.0
                frames.append(frame(bisector_progresses=tuple(current)))
            bisector_progresses[bisector_index] = 1.0
        frames.extend(
            frame(bisector_progresses=(1.0, 1.0, 1.0), circle_progress=index / 8.0)
            for index in range(1, 9)
        )
        frames.extend(
            frame(
                bisector_progresses=(1.0, 1.0, 1.0),
                circle_progress=1.0,
                highlight_progress=index / 4.0,
            )
            for index in range(1, 5)
        )
        final_frame = frame(
            bisector_progresses=(1.0, 1.0, 1.0),
            circle_progress=1.0,
            highlight_progress=1.0,
        )
        frames.extend(final_frame.copy() for _ in range(5))
        if len(frames) != self.EXPECTED_VIDEO_FRAMES or len(frames) % 4 != 1:
            raise RuntimeError(f"circumcenter produced invalid frame count {len(frames)}")
        output_path = self.solution_dir / f"{pid}_solution.webm"
        arrays = [
            np.asarray(item.convert("RGB"), dtype=np.uint8)
            for item in frames
        ]
        if not encode_rgb_frames_to_webm(arrays, output_path, fps=16):
            raise RuntimeError("Failed to encode circumcenter semantic video")
        return len(frames)

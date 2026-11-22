"""Rule-complete incenter option puzzle generator."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.geometry.circumcenter.generator import CircumcenterGenerator
from data.geometry.core.point_target_base import Point, PointTargetPuzzleRecord


def distance(p1: Point, p2: Point) -> float:
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def calculate_incenter_and_inradius(
    p1: Point,
    p2: Point,
    p3: Point,
) -> Tuple[Optional[Point], Optional[float]]:
    """Return the unique interior angle-bisector intersection and inradius."""

    a = distance(p2, p3)
    b = distance(p1, p3)
    c = distance(p1, p2)
    perimeter = a + b + c
    if perimeter <= 1e-9 or a + b <= c + 1e-9 or a + c <= b + 1e-9 or b + c <= a + 1e-9:
        return None, None
    center = Point(
        (a * p1.x + b * p2.x + c * p3.x) / perimeter,
        (a * p1.y + b * p2.y + c * p3.y) / perimeter,
    )
    semiperimeter = perimeter / 2.0
    area_term = semiperimeter * (semiperimeter - a) * (semiperimeter - b) * (semiperimeter - c)
    if area_term <= 1e-9:
        return None, None
    return center, math.sqrt(area_term) / semiperimeter


class IncenterGenerator(CircumcenterGenerator):
    """Select the unique displayed interior point equidistant from three sides."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: The input is white image containing exactly one non-collinear triangle drawn as three fixed "
        "medium-width black straight edges with no fill, plus five separate circular candidates labeled A, B, C, "
        "D, and E. All five candidate centers lie strictly inside the triangle. Candidate centers alone count: "
        "marker outlines, fills, and letter glyphs have no geometric extent. If a marker overlaps any "
        "construction, the underlying geometry continues unchanged beneath the opaque marker. Call the topmost "
        "triangle vertex V1, then call the remaining vertices V2 and V3 in clockwise image order; break an exact "
        "vertical tie by smaller x. Define side S1 as V1-V2, S2 as V2-V3, and S3 as V3-V1. A candidate's distance "
        "to a side is the shortest perpendicular distance from its center to that finite black edge; for the "
        "unique winning interior point, all three perpendicular feet lie on their respective edge segments. "
        "Exactly one displayed center has equal distance to S1, S2, and S3. It is the unique intersection of the "
        "triangle's three internal angle bisectors and the center of the circle inside the triangle tangent to "
        "all three sides. A point on a side is invalid because its distance to that side is zero, and an exterior "
        "equal-distance point would be an excenter rather than the requested interior point. Use interior "
        "membership plus equal perpendicular side distances as the only decision rule. Rule: Identify the unique "
        "zero-spread interior candidate and the common inradius. Required Output: Return exactly one uppercase "
        "option label in the form \"Answer: X.\""
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/incenter"
    MIN_RUNNER_UP_DISTANCE_SPREAD_PIXELS = 18.0
    EXPECTED_VIDEO_FRAMES = 33
    STATE_TRACE = [
        "input_hold",
        "first_angle_bisector_single_end_growth",
        "second_angle_bisector_single_end_growth",
        "third_angle_bisector_single_end_growth",
        "incenter_marker",
        "incircle_single_end_growth",
        "correct_candidate_highlight",
        "final_hold",
    ]

    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Use the supplied input as the fixed base and produce exactly final solution image. Preserve the white "
        "background, all three original black triangle edges and vertices, all five candidate centers, marker "
        "sizes, outlines, labels, spacing, scale, framing, and camera. Using the correct candidate as the common "
        "center, add "
        "exactly three medium-thin cyan-blue complete internal angle-bisector segments, each running from one "
        "triangle vertex through the common intersection to the opposite side. Draw a small black hollow marker "
        "at that intersection, then add one "
        "medium-width cyan-blue incircle centered there with radius equal to the distance from the center to any "
        "triangle side. Do not draw perpendicular spokes. Draw the original black edges above the cyan construction at tangencies and "
        "draw all candidate markers above the construction. Change only the correct candidate to pale-red fill "
        "with a dark-red outline while keeping its black uppercase letter readable; keep the other four "
        "candidates white and unchanged. No source object moves, rotates, falls, compacts, refills, reorders, "
        "disappears, duplicates, changes size or identity, or reflows. Add no side extension, second circle, "
        "perpendicular spoke, tangent-square glyph, extra highlight, vertex/side label, legend, "
        "formula, answer banner, or other text."
    )
    DEFAULT_TI2V_PROMPT = (
        "Input Image: The input is white image containing exactly one non-collinear triangle drawn as three fixed "
        "medium-width black straight edges with no fill, plus five separate circular candidates labeled A, B, C, "
        "D, and E. All five candidate centers lie strictly inside the triangle. Candidate centers alone count: "
        "marker outlines, fills, and letter glyphs have no geometric extent. If a marker overlaps any "
        "construction, the underlying geometry continues unchanged beneath the opaque marker. Call the topmost "
        "triangle vertex V1, then call the remaining vertices V2 and V3 in clockwise image order; break an exact "
        "vertical tie by smaller x. Define side S1 as V1-V2, S2 as V2-V3, and S3 as V3-V1. A candidate's distance "
        "to a side is the shortest perpendicular distance from its center to that finite black edge; for the "
        "unique winning interior point, all three perpendicular feet lie on their respective edge segments. "
        "Exactly one displayed center has equal distance to S1, S2, and S3. It is the unique intersection of the "
        "triangle's three internal angle bisectors and the center of the circle inside the triangle tangent to "
        "all three sides. A point on a side is invalid because its distance to that side is zero, and an exterior "
        "equal-distance point would be an excenter rather than the requested interior point. Use interior "
        "membership plus equal perpendicular side distances as the only decision rule. Rule: Identify the unique "
        "zero-spread interior candidate and the common inradius. Required Output: Hold the unchanged input. Draw "
        "the three complete internal angle-bisector segments one at a time, each from a triangle vertex through "
        "their common intersection to the opposite side. Then place the hollow center marker. Do not draw "
        "perpendicular spokes. Next, draw the incircle continuously in one direction from one tangent point around "
        "to the same point, using the center-to-side distance as its radius, and "
        "mark the selected candidate. Use a fixed camera, preserve every unchanged "
        "object, and show each semantic construction or commit in rule order without teleportation, hard cuts, "
        "full-frame crossfades, flicker, disappearance, or identity changes."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._geometry_input_description = (
            "A fixed non-collinear triangle and five labeled candidate centers strictly inside it are visible."
        )
        self._geometry_rule_description = (
            "Select the unique interior candidate whose perpendicular distances to all three finite triangle "
            "sides are equal. It is the common internal angle-bisector intersection and incircle center."
        )
        self._geometry_solution_description = (
            "Preserve the triangle and candidates. Add the three complete cyan internal angle-bisector segments one "
            "at a time from their vertices through the common center to the opposite sides, place a hollow marker "
            "at that center, draw the cyan incircle using the center-to-side distance as its radius, and mark only "
            "that center."
        )
        self._geometry_video_description = (
            "Hold the unchanged input, grow the three complete cyan internal angle-bisector segments one at a time "
            "from their vertices through the common center to the opposite sides, place the hollow center marker, "
            "draw the cyan incircle continuously while retaining the bisectors, then smoothly mark the answer."
        )

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        minimum_radius = self.canvas_short_side * 0.105

        def triangle_is_usable(a: Point, b: Point, c: Point) -> bool:
            center, radius = calculate_incenter_and_inradius(a, b, c)
            return bool(
                center is not None
                and radius is not None
                and radius >= minimum_radius
                and self.point_can_host_candidate(center)
                and self.circle_fits(center, radius, extra_padding=self.line_width)
            )

        for _ in range(1000):
            try:
                p1, p2, p3 = self.sample_triangle_vertices(
                    jitter_ratio=0.56,
                    min_side_ratio=0.18,
                    min_area_ratio=0.035,
                    min_altitude_ratio=0.13,
                    min_angle_deg=38.0,
                    max_angle_deg=108.0,
                    validator=triangle_is_usable,
                )
            except RuntimeError:
                # A run of three vertices that never met the angle, area and
                # inradius window is one more rejection, not the end of the
                # sample: letting it out of the loop killed roughly one seed
                # in three hundred outright, and the CLI stops a whole run on
                # the first exception.
                continue
            center, radius = calculate_incenter_and_inradius(p1, p2, p3)
            if center is None or radius is None:
                continue
            self.triangle_points = (p1, p2, p3)
            self.center = center
            self.target_point = center
            self.r = radius
            if not self._place_candidates_inside_incircle():
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
        raise RuntimeError("Failed to generate a valid incenter puzzle after 1000 tries")

    def _place_candidates_inside_incircle(self) -> bool:
        minimum_spacing = self.point_radius * 2.6
        # Distractors stay inside the incircle, so none can be dismissed for
        # sitting near a side.  Ringing them around the centre put the answer
        # at the middle of its own cloud, which answered the sample without
        # any bisector; the cloud is laid out first and one of
        # its slots taken for the answer instead.
        p1, p2, p3 = self.triangle_points
        sides = ((p1, p2), (p2, p3), (p3, p1))
        # Stated as a disc around the incircle centre this would be a region
        # centred on the answer, which puts the answer at the middle of
        # whatever cloud fits inside it.  What the task actually needs is that
        # no distractor hugs a side -- a marker there is dismissible without
        # any bisector -- so the region is the inner triangle instead, and a
        # cloud can sit off-centre in it.
        keep_off = max(self.r * 0.22, self.point_radius + 6.0)
        if keep_off >= self.r - self.point_radius:
            return False
        def inside_with_clearance(point: Point) -> bool:
            # Distance to a side *line* is signless, so it alone lets a marker
            # sit outside the triangle as long as it is far from all three
            # edges.  Containment is checked first.
            for first, second in sides:
                cross = (second.x - first.x) * (point.y - first.y) - (
                    second.y - first.y
                ) * (point.x - first.x)
                if cross * orientation <= 0.0:
                    return False
            return all(
                self.distance_point_to_line(point, first, second) >= keep_off
                for first, second in sides
            )

        orientation = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
        try:
            self.seat_candidates(
                self.center,
                accept=inside_with_clearance,
                min_separation=minimum_spacing,
            )
        except RuntimeError:
            return False
        return True

    @staticmethod
    def _line_intersection(
        a: Point,
        b: Point,
        c: Point,
        d: Point,
    ) -> Optional[Point]:
        abx, aby = b.x - a.x, b.y - a.y
        cdx, cdy = d.x - c.x, d.y - c.y
        denominator = abx * cdy - aby * cdx
        if abs(denominator) <= 1e-9:
            return None
        acx, acy = c.x - a.x, c.y - a.y
        t = (acx * cdy - acy * cdx) / denominator
        return Point(a.x + t * abx, a.y + t * aby)

    def _angle_bisector_segments(self) -> List[Tuple[Point, Point]]:
        """Return the three vertex-to-opposite-side internal bisector segments."""
        vertices = self._ordered_vertices()
        segments: List[Tuple[Point, Point]] = []
        for index, vertex in enumerate(vertices):
            opposite_a = vertices[(index + 1) % 3]
            opposite_b = vertices[(index + 2) % 3]
            endpoint = self._line_intersection(
                vertex,
                self.center,
                opposite_a,
                opposite_b,
            )
            segments.append((vertex, endpoint or self.center))
        return segments

    def _semantic_phase_roles_for_sample(self) -> tuple[str, ...]:
        if not hasattr(self, "triangle_points"):
            return super()._semantic_phase_roles_for_sample()
        return (
            "the first complete internal angle-bisector segment from a vertex through the incenter to the opposite side",
            "the first two retained complete internal angle-bisector segments",
            "all three retained complete internal angle-bisector segments through their common incenter",
            "the hollow incenter marker followed by the incircle whose radius is the center-to-side distance",
        )

    @staticmethod
    def _distance_to_side(point: Point, a: Point, b: Point) -> float:
        edge_length = math.hypot(b.x - a.x, b.y - a.y)
        return abs(
            (b.x - a.x) * (point.y - a.y)
            - (b.y - a.y) * (point.x - a.x)
        ) / max(edge_length, 1e-9)

    @staticmethod
    def _projection_to_side(point: Point, a: Point, b: Point) -> Point:
        dx = b.x - a.x
        dy = b.y - a.y
        denominator = dx * dx + dy * dy
        t = ((point.x - a.x) * dx + (point.y - a.y) * dy) / max(denominator, 1e-9)
        return Point(a.x + t * dx, a.y + t * dy)

    def _tangent_feet(self) -> List[Point]:
        vertices = self._ordered_vertices()
        return [
            self._projection_to_side(self.center, vertices[index], vertices[(index + 1) % 3])
            for index in range(3)
        ]

    def _candidate_measurements(self) -> List[Dict[str, Any]]:
        vertices = self._ordered_vertices()
        centroid_x = sum(point.x for point in vertices) / 3.0
        centroid_y = sum(point.y for point in vertices) / 3.0
        measurements: List[Dict[str, Any]] = []
        for candidate in sorted(self.candidates, key=lambda item: item.label):
            point = Point(candidate.x, candidate.y)
            distances = [
                self._distance_to_side(point, vertices[index], vertices[(index + 1) % 3])
                for index in range(3)
            ]
            measurements.append(
                {
                    "label": candidate.label,
                    "center": [candidate.x, candidate.y],
                    "direction_from_triangle_centroid": self._direction_name(
                        candidate.x - centroid_x,
                        candidate.y - centroid_y,
                    ),
                    "side_distances_pixels": distances,
                    "distance_spread_pixels": max(distances) - min(distances),
                    "mean_side_distance_pixels": sum(distances) / 3.0,
                    "strictly_inside_triangle": True,
                }
            )
        return measurements

    def _reasoning_text(self) -> str:
        measurements = self._candidate_measurements()
        positions = ", ".join(
            f"{item['label']} is {item['direction_from_triangle_centroid']} of the triangle's middle"
            for item in measurements
        )
        distractors = ", ".join(item["label"] for item in measurements if item["label"] != self.correct_label)
        return (
            f"All candidates lie inside the triangle; visually, {positions}. The three internal angle bisectors "
            f"meet only at {self.correct_label}. The candidates "
            f"{distractors} are closer to at least one side than another, so their circles cannot touch all three "
            f"sides. Therefore {self.correct_label} is the internal angle-bisector intersection and incircle center. "
            f"I use the distance from that center to a triangle side as the incircle radius, without drawing a spoke."
            f"\n\n{FIVE_CHOICE_CODEC.format(self.correct_label)}"
        )

    def _authored_observation(self) -> str:
        vertices = self._ordered_vertices()
        names = self._visible_point_names(vertices)
        return (
            f"The triangle's corners are the {names[0]}, the {names[1]}, and the {names[2]}, and every "
            "labeled marker sits strictly inside the outline. The internal bisector of each corner angle "
            "keeps equal perpendicular distance to that corner's two sides, so all three bisectors funnel to "
            "a single interior point equally distant from the three sides; they converge "
            f"{self._canvas_region(self.center)}, and the circle grown from that common side distance touches "
            "every side."
        )

    def build_record_extra(self) -> Dict[str, Any]:
        vertices = self._ordered_vertices()
        feet = self._tangent_feet()
        measurements = self._candidate_measurements()
        derived_index = next(index for index, item in enumerate(measurements) if item["label"] == self.correct_label)
        runner_up = sorted(float(item["distance_spread_pixels"]) for item in measurements)[1]
        structured = {
            "triangle_vertices": [point.to_list() for point in vertices],
            "incenter": self.center.to_list(),
            "inradius": self.r,
            "tangent_feet": [point.to_list() for point in feet],
            "candidate_centers": [
                {"label": item["label"], "center": item["center"]}
                for item in measurements
            ],
            "candidate_measurements": measurements,
            "correct_option": self.correct_label,
            "correct_index": derived_index,
            "runner_up_distance_spread_pixels": runner_up,
            "multiple_solution_policy": "exactly_one_strictly_interior_displayed_candidate_is_equidistant_from_all_three_sides",
            "state_trace": list(self.STATE_TRACE),
            "answer_schema": {
                "terminal_text_regex": FIVE_CHOICE_CODEC.full_pattern,
                "requires_sample_specific_reasoning_before_terminal_answer": True,
                "terminal_text_contains_exact_answer": True,
                "no_block_after_terminal_image": True,
            },
            "verifier_contract": {
                "profile": "incenter_multimodal_semantic_hard_gate_v1",
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
            "task_name": "incenter",
            "task_family": "geometry",
            "triangle_vertices": structured["triangle_vertices"],
            "incenter": structured["incenter"],
            "inradius": structured["inradius"],
            "tangent_feet": structured["tangent_feet"],
            "structured_answer": structured,
            "task_contract": {
                "logic": {
                    "incenter_definition": "unique_strictly_interior_point_equidistant_from_three_finite_sides",
                    "candidate_semantics": "marker_center_only",
                    "boundary_policy": "on_side_or_exterior_equal_distance_points_are_invalid",
                },
                "answer": {
                    "parser_profile": "sample_specific_side_distance_table_then_strict_uppercase_option_letter_a_to_e",
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

    def _draw_incircle_progress(self, draw: Any, progress: float) -> None:
        progress = max(0.0, min(1.0, progress))
        if progress <= 0.0:
            return
        bbox = (
            self.center.x - self.r,
            self.center.y - self.r,
            self.center.x + self.r,
            self.center.y + self.r,
        )
        if progress >= 1.0:
            draw.ellipse(bbox, outline=self.CONSTRUCTION_COLOR, width=self.line_width)
            return
        start_foot = min(self._tangent_feet(), key=lambda point: (point.y, point.x))
        start_degrees = math.degrees(math.atan2(start_foot.y - self.center.y, start_foot.x - self.center.x))
        draw.arc(
            bbox,
            start=start_degrees,
            end=start_degrees + 360.0 * progress,
            fill=self.CONSTRUCTION_COLOR,
            width=self.line_width,
        )

    def _render_state(
        self,
        *,
        bisector_progresses: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        marker_progress: float = 0.0,
        circle_progress: float = 0.0,
        highlight_progress: float = 0.0,
    ) -> Image.Image:
        draw, base = self.get_draw_base()
        from data.geometry.core.prompt_paraphrase import workbench_active, draw_survey_rail, given_colour
        from data.common.prompt_paraphrase import use_prompt_paraphrase
        if workbench_active():
            draw_survey_rail(draw, [(p.x,p.y) for p in list(self.triangle_points)+[self.triangle_points[0]]], self.CANDIDATE_OUTLINE_COLOR, self.line_width)
        for (start, end), raw_progress in zip(
            self._angle_bisector_segments(),
            bisector_progresses,
        ):
            progress = max(0.0, min(1.0, float(raw_progress)))
            if progress <= 0.0:
                continue
            current_end = Point(
                start.x + (end.x - start.x) * progress,
                start.y + (end.y - start.y) * progress,
            )
            self.draw_line(
                draw,
                [start, current_end],
                width_factor=0.72,
                fill=self.CONSTRUCTION_COLOR,
                join_start=True,
            )
        if marker_progress > 0.0:
            self.draw_anchor_marker(draw, self.center, 7)
        self._draw_incircle_progress(draw, circle_progress)
        # The embossed body sits beneath the construction. Only the thin
        # given centreline is reasserted above it, preserving visible arcs.
        source_colour=given_colour(self.CANDIDATE_OUTLINE_COLOR)
        with use_prompt_paraphrase(None):
            self.draw_line(draw, list(self.triangle_points) + [self.triangle_points[0]], fill=source_colour)
        self._draw_candidates_with_highlight_progress(draw, highlight_progress)
        # The incenter candidate is centered on the construction point.
        # Redraw the hollow marker above the opaque candidate plate so it
        # remains visible in the cumulative and terminal states.
        if marker_progress > 0.0:
            self.draw_anchor_marker(draw, self.center, 7)
        return base

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        solved = bool(highlight_label)
        return self._render_state(
            bisector_progresses=(1.0, 1.0, 1.0) if solved else (0.0, 0.0, 0.0),
            marker_progress=1.0 if solved else 0.0,
            circle_progress=1.0 if solved else 0.0,
            highlight_progress=1.0 if solved else 0.0,
        )

    def save_video_solution(self, pid: str) -> Optional[int]:
        """Render the angle-bisector → center-marker → incircle visual program."""

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
        frames.append(
            frame(
                bisector_progresses=(1.0, 1.0, 1.0),
                marker_progress=1.0,
            )
        )
        frames.extend(
            frame(
                bisector_progresses=(1.0, 1.0, 1.0),
                marker_progress=1.0,
                circle_progress=step / 7.0,
            )
            for step in range(1, 8)
        )
        frames.extend(
            frame(
                bisector_progresses=(1.0, 1.0, 1.0),
                marker_progress=1.0,
                circle_progress=1.0,
                highlight_progress=step / 4.0,
            )
            for step in range(1, 5)
        )
        final_frame = frame(
            bisector_progresses=(1.0, 1.0, 1.0),
            marker_progress=1.0,
            circle_progress=1.0,
            highlight_progress=1.0,
        )
        frames.extend(final_frame.copy() for _ in range(5))
        if len(frames) != self.EXPECTED_VIDEO_FRAMES or len(frames) % 4 != 1:
            raise RuntimeError(f"incenter produced invalid frame count {len(frames)}")
        output_path = self.solution_dir / f"{pid}_solution.webm"
        arrays = [np.asarray(item.convert("RGB"), dtype=np.uint8) for item in frames]
        if not encode_rgb_frames_to_webm(arrays, output_path, fps=16):
            raise RuntimeError("Failed to encode incenter semantic video")
        return len(frames)

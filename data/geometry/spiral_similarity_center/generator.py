"""Spiral similarity center puzzle generator."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from PIL import Image

from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class SpiralSimilarityCenterGenerator(PointTargetPuzzleGenerator):
    """Find the fixed centre of a spiral similarity: one turn plus one shrink."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows one large black triangle, one smaller dark-gray triangle that is a "
        "turned and shrunk copy of it, and five labeled candidate circles A-E. "
        "Rule: One fixed centre carries the black triangle onto the gray one: every black vertex swings about that "
        "centre through the same angle, in the same direction, and lands on its matching gray vertex, whose "
        "distance to the centre is the same fraction of the black vertex's distance. Matching vertices are the ones "
        "with equal interior angles. No other point turns all three vertex pairs through one common angle at one "
        "common distance fraction. "
        "Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/spiral_similarity_center"
    DEFAULT_TI2V_PROMPT = (
        "Input Image: A white canvas shows one large black triangle, one smaller dark-gray triangle that is a "
        "turned and shrunk copy of it, and five labeled candidate circles A-E. "
        "Rule: One fixed centre carries the black triangle onto the gray one: every black vertex swings about that "
        "centre through the same angle, in the same direction, and lands on its matching gray vertex, whose "
        "distance to the centre is the same fraction of the black vertex's distance. Matching vertices are the ones "
        "with equal interior angles. No other point turns all three vertex pairs through one common angle at one "
        "common distance fraction. "
        "Required Output: The video first holds the two triangles and the candidate markers, then draws three cyan "
        "construction polylines one at a time, each running from one black vertex to the fixed centre and on to that "
        "vertex's matching gray vertex, keeping every completed polyline. In the final state, only the marker at the "
        "fixed centre changes to pale red fill with a dark red outline, while the other four markers remain white. "
        "Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    GRAY_COLOR = (100, 100, 100)

    # A spiral similarity degenerates into a homothety at a turn of 0 or 180
    # degrees and into a plain rotation at ratio 1.  Both families already
    # exist as their own tasks, so the sampled turn and ratio stay far from
    # every degenerate value and no instance can collapse back into one.
    MIN_TURN_DEGREES = 42.0
    MAX_TURN_DEGREES = 138.0
    MIN_RATIO = 0.34
    MAX_RATIO = 0.62

    # Perceptual margin.  A distractor is only refutable if
    # the three vertex pairs visibly disagree about it, so every wrong marker
    # must miss on the turn, on the distance fraction, or on both.
    MIN_TURN_SPREAD_DEGREES = 15.0
    MIN_RATIO_SPREAD = 1.60

    # The matching between black and gray vertices is read off equal interior
    # angles, so the triangle is constructed visibly scalene: consecutive
    # interior angles differ by at least MIN_INTERIOR_ANGLE_GAP_DEGREES, and
    # the upper gap keeps the smallest angle above 26 degrees.
    MIN_INTERIOR_ANGLE_GAP_DEGREES = 13.0
    MAX_INTERIOR_ANGLE_GAP_DEGREES = 32.0

    def __init__(self, *args, expanding=False, **kwargs):
        self.expanding = bool(expanding)
        if self.expanding:
            self.MIN_RATIO, self.MAX_RATIO = 1.6, 2.0
            for field in (
                "DEFAULT_RULE_PROMPT",
                "DEFAULT_TI2V_PROMPT",
                "DEFAULT_SOLUTION_IMAGE_PROMPT",
            ):
                setattr(
                    self,
                    field,
                    getattr(self, field)
                    .replace(
                        "one large black triangle, one smaller dark-gray triangle",
                        "one black triangle, one larger dark-gray triangle",
                    )
                    .replace("turned and shrunk", "turned and enlarged")
                    .replace("fraction", "multiple"),
                )
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self._build_configuration()
        return self.save_puzzle()

    def _build_configuration(self) -> None:
        """Sample one non-degenerate spiral similarity and its marker layout."""

        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            center = self.pick_target_point(
                0.36,
                padding=candidate_padding + self.canvas_short_side * 0.08,
            )
            ratio = self._rng.uniform(self.MIN_RATIO, self.MAX_RATIO)
            turn = math.radians(
                self._rng.uniform(self.MIN_TURN_DEGREES, self.MAX_TURN_DEGREES)
            ) * self._rng.choice((-1.0, 1.0))

            large_verts = self._scalene_triangle_around(center)
            if self.expanding:
                large_verts = [
                    Point(
                        center.x + (v.x - center.x) / ratio,
                        center.y + (v.y - center.y) / ratio,
                    )
                    for v in large_verts
                ]

            # Spiral similarity about the centre: turn by `turn`, then scale
            # every distance to the centre by `ratio`.
            cos_turn, sin_turn = math.cos(turn), math.sin(turn)
            small_verts = [
                Point(
                    center.x
                    + ratio
                    * ((v.x - center.x) * cos_turn - (v.y - center.y) * sin_turn),
                    center.y
                    + ratio
                    * ((v.x - center.x) * sin_turn + (v.y - center.y) * cos_turn),
                )
                for v in large_verts
            ]

            all_pts = large_verts + small_verts + [center]
            if not all(
                self.inside_canvas(p, padding=self.line_width + 4) for p in all_pts
            ):
                continue

            if self.triangle_area(*large_verts) < (self.canvas_short_side**2) * 0.018:
                continue

            # The two outlines legitimately cross, but a gray corner buried
            # inside a black stroke loses the corner shape that carries the
            # vertex matching, so keep each gray vertex off the black ink.
            black_outline = Incidence.polyline(list(large_verts) + [large_verts[0]])
            clearance = self.line_width + 3.0
            if any(black_outline.distance(sv) < clearance for sv in small_verts):
                continue
            if any(
                self.distance(center, sv) < self.canvas_short_side * 0.06
                for sv in small_verts
            ):
                continue

            if not self.point_can_host_candidate(center):
                continue
            try:
                self.place_candidates(center)
            except RuntimeError:
                continue

            # The centre lies on no drawn line, but a marker can still land on
            # a triangle outline.  When the correct marker is the only one
            # touching ink the sample answers itself exactly as a designed
            # incidence shortcut would, so redraw the scatter.
            if not self.candidate_incidence_is_answer_free(
                [
                    black_outline,
                    Incidence.polyline(list(small_verts) + [small_verts[0]]),
                ]
            ):
                continue

            self.large_triangle = tuple(large_verts)
            self.small_triangle = tuple(small_verts)
            self.target_point = center
            self.turn_degrees = math.degrees(turn)
            self.scale_ratio = ratio
            if not self._distractors_are_refutable():
                continue
            return
        raise RuntimeError("Failed to generate spiral similarity center puzzle.")

    def _scalene_triangle_around(self, center: Point) -> list[Point]:
        """Build a visibly scalene triangle whose interior contains ``center``.

        The shape is constructed from its own interior angles rather than
        sampled and re-sampled until it happens to be scalene: three sorted
        angles separated by real gaps are laid out on a circumcircle, then the
        whole triangle is translated so that ``center`` sits inside it.  The
        gaps are what makes matching black and gray vertices by equal interior
        angle a reliable reading of the picture.
        """

        first_gap = self._rng.uniform(
            self.MIN_INTERIOR_ANGLE_GAP_DEGREES, self.MAX_INTERIOR_ANGLE_GAP_DEGREES
        )
        second_gap = self._rng.uniform(
            self.MIN_INTERIOR_ANGLE_GAP_DEGREES, self.MAX_INTERIOR_ANGLE_GAP_DEGREES
        )
        smallest = (180.0 - 2.0 * first_gap - second_gap) / 3.0
        interior = [smallest, smallest + first_gap, smallest + first_gap + second_gap]
        self._rng.shuffle(interior)
        radius = self._rng.uniform(self.canvas_short_side * 0.20, self.canvas_short_side * 0.27)

        # A triangle inscribed in its circumcircle subtends twice each interior
        # angle on the opposite arc, which fixes the three vertex directions.
        start = self._rng.uniform(0.0, math.tau)
        directions = [
            start,
            start + 2.0 * math.radians(interior[2]),
            start + 2.0 * math.radians(interior[2]) + 2.0 * math.radians(interior[0]),
        ]
        vertices = [
            Point(radius * math.cos(angle), radius * math.sin(angle)) for angle in directions
        ]
        centroid = Point(
            sum(vertex.x for vertex in vertices) / 3.0,
            sum(vertex.y for vertex in vertices) / 3.0,
        )
        # Offsetting the centroid keeps the spiral centre well inside the
        # outline while still varying where inside it lands.
        inner_reach = min(
            self.distance_point_to_line(centroid, vertices[index], vertices[(index + 1) % 3])
            for index in range(3)
        )
        offset_angle = self._rng.uniform(0.0, math.tau)
        offset_reach = inner_reach * self._rng.uniform(0.0, 0.55)
        anchor = Point(
            center.x - centroid.x - math.cos(offset_angle) * offset_reach,
            center.y - centroid.y - math.sin(offset_angle) * offset_reach,
        )
        return [Point(vertex.x + anchor.x, vertex.y + anchor.y) for vertex in vertices]

    def _pair_transform(
        self, pivot: Point, black: Point, gray: Point
    ) -> Optional[Tuple[float, float]]:
        """Signed screen turn (degrees) and distance fraction about ``pivot``."""

        bx, by = black.x - pivot.x, black.y - pivot.y
        gx, gy = gray.x - pivot.x, gray.y - pivot.y
        black_reach = math.hypot(bx, by)
        gray_reach = math.hypot(gx, gy)
        if black_reach <= 1e-9 or gray_reach <= 1e-9:
            return None
        turn = math.degrees(math.atan2(bx * gy - by * gx, bx * gx + by * gy))
        return turn, gray_reach / black_reach

    def _pair_disagreement(self, pivot: Point) -> Tuple[float, float]:
        """Widest turn gap (degrees) and distance-fraction gap seen from ``pivot``."""

        readings = [
            self._pair_transform(pivot, black, gray)
            for black, gray in zip(self.large_triangle, self.small_triangle)
        ]
        if any(reading is None for reading in readings):
            return 360.0, float("inf")
        turns = [reading[0] for reading in readings if reading is not None]
        fractions = [reading[1] for reading in readings if reading is not None]
        turn_spread = max(
            abs((turns[i] - turns[j] + 180.0) % 360.0 - 180.0)
            for i in range(len(turns))
            for j in range(i + 1, len(turns))
        )
        return turn_spread, max(fractions) / max(min(fractions), 1e-9)

    def _distractors_are_refutable(self) -> bool:
        """Every wrong marker must visibly break the common turn or fraction."""

        for candidate in self.candidates:
            if candidate.label == self.correct_label:
                continue
            turn_spread, ratio_spread = self._pair_disagreement(
                Point(float(candidate.x), float(candidate.y))
            )
            if (
                turn_spread < self.MIN_TURN_SPREAD_DEGREES
                and ratio_spread < self.MIN_RATIO_SPREAD
            ):
                return False
        return True

    def _authored_observation(self) -> str:
        black = list(self.large_triangle)
        gray = list(self.small_triangle)
        names = self._visible_point_names(black)
        interior = self.triangle_angles(*self.large_triangle)
        widest = max(range(3), key=lambda index: interior[index])
        turn_direction = "clockwise" if self.turn_degrees > 0 else "counterclockwise"
        turn_size = int(round(abs(self.turn_degrees) / 5.0)) * 5
        fraction = self._fraction_phrase(self.scale_ratio)

        worst = max(
            (candidate for candidate in self.candidates if candidate.label != self.correct_label),
            key=lambda candidate: self._pair_disagreement(
                Point(float(candidate.x), float(candidate.y))
            )[0],
        )
        pivot = Point(float(worst.x), float(worst.y))
        readings = [self._pair_transform(pivot, b, g) for b, g in zip(black, gray)]
        turns = [0.0 if reading is None else reading[0] for reading in readings]
        first, second = max(
            ((i, j) for i in range(3) for j in range(i + 1, 3)),
            key=lambda pair: abs(
                (turns[pair[0]] - turns[pair[1]] + 180.0) % 360.0 - 180.0
            ),
        )
        return (
            f"The black triangle's widest corner is the {names[widest]}, and the gray triangle's widest corner sits "
            f"{self._relative_position_phrase(black[widest], gray[widest])} it, so those two are partners and the "
            f"other two pairs follow the same way round. Sighting from candidate {worst.label}, the pair at the "
            f"{names[first]} {self._turn_phrase(turns[first])} while the pair at the {names[second]} "
            f"{self._turn_phrase(turns[second])}, so no one turn about {worst.label} carries all three pairs. Only "
            f"one marker turns every black vertex {turn_direction} by about {turn_size} degrees onto its gray "
            f"partner, with each partner at {fraction} of its black vertex's distance from that marker."
        )

    @staticmethod
    def _turn_phrase(degrees: float) -> str:
        size = abs(float(degrees))
        if size < 12.0:
            return "barely turns at all"
        return f"swings about {int(round(size / 5.0)) * 5} degrees"

    @staticmethod
    def _fraction_phrase(ratio: float) -> str:
        if ratio < 0.40:
            return "roughly two fifths"
        if ratio < 0.46:
            return "a little under half"
        if ratio < 0.55:
            return "about half"
        return "a little over half"

    def build_record_extra(self) -> dict[str, object]:
        return {
            "large_triangle": [p.to_list() for p in self.large_triangle],
            "small_triangle": [p.to_list() for p in self.small_triangle],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, list(self.large_triangle) + [self.large_triangle[0]])
        # Small triangle drawn in gray with the same line_width so the two
        # shapes read as "same thickness, different tint."
        self.draw_line(
            draw,
            list(self.small_triangle) + [self.small_triangle[0]],
            fill=self.GRAY_COLOR,
        )
        if highlight_label:
            # One polyline per vertex pair, cornered at the fixed centre: the
            # three corners share the same opening angle and the same leg
            # ratio, which is exactly the relation that defines the centre.
            for black, gray in zip(self.large_triangle, self.small_triangle):
                self.draw_line(draw, [black, self.target_point, gray])
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

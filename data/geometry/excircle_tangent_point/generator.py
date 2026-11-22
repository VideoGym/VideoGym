"""Excircle tangent point puzzle generator.

Only the triangle and its marked vertex are shown.  The excircle opposite
the marked vertex is *not* drawn: it is the object the solver has to place
from the stated rule, and drawing it would reduce the task to reading off
where a visible circle meets a visible side.

The contact follows from the equal-tangent-length identity.  With A the
marked vertex and the excircle opposite A touching side BC at T,

    |B T| = s - c    and    |C T| = s - b

where a, b, c are the sides opposite A, B, C and s = (a + b + c) / 2.

The layout is normalized on the triangle, the candidate markers, and a
band reserved symmetrically around the whole target side.  Every term is a
function of the triangle alone, so the framing carries no information about
where along the side the contact falls.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PIL import Image

from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointCandidate,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


def _excircle_tangent_on_opposite_side(p_a: Point, p_b: Point, p_c: Point) -> Tuple[Point, Point, float]:
    """Compute the excircle opposite p_a's tangent point on side BC plus center and radius."""
    a = math.hypot(p_b.x - p_c.x, p_b.y - p_c.y)
    b = math.hypot(p_a.x - p_c.x, p_a.y - p_c.y)
    c = math.hypot(p_a.x - p_b.x, p_a.y - p_b.y)
    s = (a + b + c) / 2.0
    # Tangent point on BC at distance s - c from B (toward C).
    t_from_b = s - c
    if a < 1e-9:
        return p_b, p_b, 0.0
    ux = (p_c.x - p_b.x) / a
    uy = (p_c.y - p_b.y) / a
    tangent_point = Point(x=p_b.x + t_from_b * ux, y=p_b.y + t_from_b * uy)
    # Excircle center: I_A = (-a*A + b*B + c*C) / (-a + b + c)
    denom = -a + b + c
    if abs(denom) < 1e-9:
        return tangent_point, tangent_point, 0.0
    cx = (-a * p_a.x + b * p_b.x + c * p_c.x) / denom
    cy = (-a * p_a.y + b * p_b.y + c * p_c.y) / denom
    excircle_center = Point(x=cx, y=cy)
    area_sq = max(0.0, s * (s - a) * (s - b) * (s - c))
    area = math.sqrt(area_sq)
    if (s - a) <= 1e-9:
        return tangent_point, excircle_center, 0.0
    r_excircle = area / (s - a)
    return tangent_point, excircle_center, r_excircle


class ExcircleTangentPointGenerator(PointTargetPuzzleGenerator):
    """Find where the excircle opposite a marked vertex touches the opposite triangle side."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows a black triangle outline with one vertex marked by a filled dot, plus five "
        "labeled candidate markers A-E spread along the side opposite that marked vertex. "
        "Rule: The excircle opposite the marked vertex lies outside the triangle, on the far side of that opposite "
        "side, and touches that side as well as the extensions of the two sides meeting at the marked vertex. "
        "Identify the candidate sitting where that excircle touches the opposite side. "
        "Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/excircle_tangent_point"
    # Band reserved beyond each end of the target side and on its outer
    # face, as multiples of that side's length.  It keeps the solution arc
    # inside the canvas without letting the contact position influence the
    # framing of the input.
    SIDE_EXTENSION_RATIO = 0.08
    OUTER_CLEARANCE_RATIO = 0.20
    # Contact positions the triangle-shape guards can actually realize, and the
    # legibility floor for the marker ladder.
    MIN_CONTACT_FRACTION = 0.25
    MAX_CONTACT_FRACTION = 0.75
    MIN_TRIANGLE_ANGLE_DEG = 40.0
    MAX_TRIANGLE_ANGLE_DEG = 102.0
    # A blunt enough marked corner keeps the triangle from growing tall
    # relative to its target side, which is what sets the marker spacing.
    MIN_MARKED_ANGLE_DEG = 44.0
    # Apex height as a multiple of the target side.  Capping it keeps the
    # normalized side long enough to host five legible markers whatever the
    # contact fraction, so the ladder never has to be rejected and the
    # answer's rung stays uniformly distributed.
    MAX_APEX_HEIGHT_RATIO = 0.95
    MIN_MARKER_GAP = 28.0
    DEFAULT_TI2V_PROMPT = (
        "Input Image: A white canvas shows a black triangle outline of medium thickness with one vertex marked by a "
        "small filled black dot. Five candidate markers A-E lie along the side opposite that marked vertex; each is a "
        "small white circle with a thin dark gray outline and a black uppercase letter. "
        "Required Output: Hold the unchanged scene, then sweep in a single cyan arc that curves along the outer face "
        "of the marked vertex's opposite side and grazes it at one point. Finally recolor only the candidate at that "
        "contact to pale red fill with a dark red outline, leaving the other four markers white. Keep the triangle, "
        "the marked dot, the background, and the camera fixed. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Input Image: A white canvas shows a black triangle outline with one vertex marked by a filled dot, plus five "
        "labeled candidate markers A-E along the side opposite that marked vertex. "
        "Required Output: Preserve the triangle, the marked dot, and every marker. Add one cyan arc that curves along "
        "the outer face of the opposite side and grazes it at a single point, and recolor only the candidate at that "
        "contact to pale red fill with a dark red outline."
    )

    def _draw_marked_vertex(self, draw) -> None:
        cx, cy = round(self.marked_vertex.x), round(self.marked_vertex.y)
        r = max(3, self.point_radius // 2)
        if hasattr(draw, "raw"):
            raw = draw.raw
            ss = draw.factor
            raw.ellipse(
                (cx * ss - r * ss, cy * ss - r * ss, cx * ss + r * ss, cy * ss + r * ss),
                fill=(0, 0, 0),
            )
        else:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0))

    def _draw_reference(self, draw) -> None:
        """Source ink only: the triangle and the marked vertex."""

        va, vb, vc = self.triangle_points
        self.draw_line(draw, [va, vb, vc, va])
        self._draw_marked_vertex(draw)

    def _video_overlay_extras(self, draw) -> None:
        self._draw_reference(draw)

    def _side_frame(self) -> Tuple[Point, Point, float, Tuple[float, float], Tuple[float, float]]:
        """Return (B, C, |BC|, unit along B->C, unit normal away from the marked vertex)."""

        va, vb, vc = self.triangle_points
        length = max(1e-9, self.distance(vb, vc))
        tx, ty = (vc.x - vb.x) / length, (vc.y - vb.y) / length
        nx, ny = -ty, tx
        if (va.x - vb.x) * nx + (va.y - vb.y) * ny > 0.0:
            nx, ny = -nx, -ny
        return vb, vc, length, (tx, ty), (nx, ny)

    def _reserved_band(self) -> List[Point]:
        """Corners of the answer-independent band kept clear beyond the target side."""

        vb, vc, length, (tx, ty), (nx, ny) = self._side_frame()
        extension = self.SIDE_EXTENSION_RATIO * length
        clearance = self.OUTER_CLEARANCE_RATIO * length
        beyond_b = Point(vb.x - extension * tx, vb.y - extension * ty)
        beyond_c = Point(vc.x + extension * tx, vc.y + extension * ty)
        return [
            beyond_b,
            beyond_c,
            Point(beyond_b.x + clearance * nx, beyond_b.y + clearance * ny),
            Point(beyond_c.x + clearance * nx, beyond_c.y + clearance * ny),
        ]

    def _tangency_arc(self) -> Tuple[float, float]:
        """Screen-space (center bearing toward the contact, half sweep) of the drawn arc."""

        vb, vc, length, (tx, ty), (nx, ny) = self._side_frame()
        radius = self.excircle_radius
        along = (self.target_point.x - vb.x) * tx + (self.target_point.y - vb.y) * ty
        extension = self.SIDE_EXTENSION_RATIO * length
        clearance = self.OUTER_CLEARANCE_RATIO * length
        pad = float(self.line_width) + 4.0
        half_chord = max(0.0, min(along + extension, length - along + extension) - pad)
        rise = max(0.0, clearance - pad)
        by_chord = math.asin(max(0.0, min(1.0, half_chord / radius)))
        by_rise = math.acos(max(-1.0, min(1.0, 1.0 - rise / radius)))
        half_sweep = min(by_chord, by_rise)
        bearing = math.atan2(-ny, -nx)
        return bearing, half_sweep

    def _normalize_layout_to_canvas(self) -> Optional[Tuple[float, float, float]]:
        """Fit the answer-independent content to the canvas and return the transform.

        Only the triangle and the band reserved around the whole target side
        enter the bounding box.  The markers lie on that side and the arc lives
        inside the band, so nothing the contact position controls can shift the
        framing.
        """

        triangle_points = self.triangle_points
        points = list(triangle_points) + self._reserved_band()
        content_left = min(point.x for point in points)
        content_right = max(point.x for point in points)
        content_top = min(point.y for point in points)
        content_bottom = max(point.y for point in points)
        content_width = content_right - content_left
        content_height = content_bottom - content_top
        if content_width <= 1e-9 or content_height <= 1e-9:
            return None

        pad_x, pad_y = self._candidate_safe_padding(include_edge_buffer=True)
        pad_x = max(pad_x, float(self.line_width))
        pad_y = max(pad_y, float(self.line_width))
        # Markers and strokes are validated against the margin-inset drawable
        # rect, so the layout is fitted into exactly that rect.
        left, top, right, bottom = self.canvas_bounds()
        available_width = (right - left) - 2.0 * pad_x
        available_height = (bottom - top) - 2.0 * pad_y
        if available_width <= 1e-9 or available_height <= 1e-9:
            return None
        scale = min(available_width / content_width, available_height / content_height)
        if scale <= 0.0:
            return None
        tx = left + pad_x + (available_width - content_width * scale) / 2.0 - content_left * scale
        ty = top + pad_y + (available_height - content_height * scale) / 2.0 - content_top * scale

        def transform_point(point: Point) -> Point:
            return Point(x=point.x * scale + tx, y=point.y * scale + ty)

        self.triangle_points = (
            transform_point(triangle_points[0]),
            transform_point(triangle_points[1]),
            transform_point(triangle_points[2]),
        )
        self.marked_vertex = self.triangle_points[0]
        self.target_point = transform_point(self.target_point)
        self.excircle_center = transform_point(self.excircle_center)
        self.excircle_radius *= scale
        return scale, tx, ty

    def _place_side_candidates(self, side_b: Point, side_c: Point, target: Point) -> None:
        """Spread A-E evenly along the target side with the answer at a free rank.

        Every option lies on the side the contact lies on, so "which marker is
        on the figure?" cannot narrow the field.  The row is an
        even ladder, so its central marker is always the third one, and the
        answer's rank in that ladder is drawn uniformly from the ranks the side
        can host: "take the middle marker" therefore stays at chance.
        """

        count = len(self.option_labels)
        length = self.distance(side_b, side_c)
        along = self.distance(side_b, target)
        gap = self.ladder_gap_fraction * length
        if gap < self.MIN_MARKER_GAP:
            raise RuntimeError("normalized target side is too short for the marker row")
        rank = self.answer_rank
        tx = (side_c.x - side_b.x) / max(length, 1e-9)
        ty = (side_c.y - side_b.y) / max(length, 1e-9)
        labels = list(self.option_labels)
        self._rng.shuffle(labels)
        self.candidates = [
            PointCandidate(
                x=side_b.x + (along + (index - rank) * gap) * tx,
                y=side_b.y + (along + (index - rank) * gap) * ty,
                label=labels[index],
            )
            for index in range(count)
        ]
        self.correct_label = labels[rank]

    def _apex_for_contact(self, contact_fraction: float) -> Optional[Point]:
        """Apex of a unit-base triangle whose excircle contact sits at ``contact_fraction``.

        With B = (0, 0) and C = (1, 0), the contact at ``u`` from B forces
        ``|AC| - |AB| = 2u - 1``, so the apex lies on one branch of the
        hyperbola with foci B and C.  Sweeping that branch and keeping the
        admissible stretch gives a legal triangle for the requested contact by
        construction instead of by rejection.
        """

        difference = 2.0 * contact_fraction - 1.0
        semi_axis = abs(difference) / 2.0
        conjugate = math.sqrt(max(1e-12, 0.25 - semi_axis * semi_axis))
        branch = -1.0 if difference > 0.0 else 1.0
        admissible: List[float] = []
        for step in range(400):
            parameter = 0.25 + step * 0.006
            apex = Point(
                x=0.5 + branch * semi_axis * math.cosh(parameter),
                y=conjugate * math.sinh(parameter),
            )
            angles = self.triangle_angles(apex, Point(0.0, 0.0), Point(1.0, 0.0))
            if (
                min(angles) >= self.MIN_TRIANGLE_ANGLE_DEG
                and max(angles) <= self.MAX_TRIANGLE_ANGLE_DEG
                and angles[0] >= self.MIN_MARKED_ANGLE_DEG
                and abs(apex.y) <= self.MAX_APEX_HEIGHT_RATIO
            ):
                admissible.append(parameter)
        if not admissible:
            return None
        parameter = admissible[self._rng.randrange(len(admissible))] + self._rng.uniform(-0.003, 0.003)
        return Point(
            x=0.5 + branch * semi_axis * math.cosh(parameter),
            y=conjugate * math.sinh(parameter),
        )

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        # The marker row and the contact's share of the target side are chosen
        # first, and the triangle is then built to realize them.  Because the
        # row's placement and spacing never depend on which rung carries the
        # answer, no positional reading of the row -- its middle marker, the
        # marker nearest the side's midpoint, the marker nearest the canvas
        # centre -- beats chance.
        for _ in range(999):
            gap_fraction = self._rng.uniform(0.119, 0.125)
            low = self.MIN_CONTACT_FRACTION
            high = self.MAX_CONTACT_FRACTION - 4.0 * gap_fraction
            if high < low:
                continue
            first_rung = self._rng.uniform(low, high)
            rank = self._rng.randrange(len(self.option_labels))
            contact_fraction = first_rung + rank * gap_fraction

            apex = self._apex_for_contact(contact_fraction)
            if apex is None:
                continue
            scale = float(self.canvas_short_side)
            angle = self._rng.uniform(0.0, math.tau)
            flip = 1.0 if self._rng.random() < 0.5 else -1.0
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            def place(point: Point) -> Point:
                x, y = point.x * scale, point.y * scale * flip
                return Point(x=x * cos_a - y * sin_a, y=x * sin_a + y * cos_a)

            va = place(apex)
            vb = place(Point(0.0, 0.0))
            vc = place(Point(1.0, 0.0))

            tangent_point, excircle_center, r_excircle = _excircle_tangent_on_opposite_side(va, vb, vc)
            side_length = self.distance(vb, vc)
            if r_excircle <= 1e-6 or side_length <= 1e-6:
                continue
            if abs(self.distance(vb, tangent_point) / side_length - contact_fraction) > 1e-6:
                continue

            self.ladder_gap_fraction = gap_fraction
            self.answer_rank = rank
            self.triangle_points = (va, vb, vc)
            self.marked_vertex = va
            self.excircle_center = excircle_center
            self.excircle_radius = r_excircle
            self.target_point = tangent_point

            if self._normalize_layout_to_canvas() is None:
                continue
            _marked, side_b, side_c = self.triangle_points
            if not self.point_can_host_candidate(self.target_point):
                continue

            try:
                self._place_side_candidates(side_b, side_c, self.target_point)
            except RuntimeError:
                continue

            if not self._normalized_layout_is_valid():
                continue
            return self.save_puzzle()
        raise RuntimeError("Failed to lay out excircle_tangent_point puzzle")

    def _normalized_layout_is_valid(self) -> bool:
        if not all(
            self.inside_canvas(point, padding=self.line_width)
            for point in (*self.triangle_points, self.target_point)
        ):
            return False
        if not self.validate_candidate_layout(
            self.candidates, min_distance=self.minimum_candidate_spacing(scale=0.9),
        ):
            return False
        # Every marker sits on the target side, so incidence with the drawn
        # triangle cannot single the answer out.
        first, second, third = self.triangle_points
        if not self.candidate_incidence_is_answer_free(
            [
                Incidence.segment(first, second),
                Incidence.segment(second, third),
                Incidence.segment(third, first),
            ]
        ):
            return False

        tangent_point, center, radius = _excircle_tangent_on_opposite_side(*self.triangle_points)
        if self.distance(tangent_point, self.target_point) > 1.0:
            return False
        if self.distance(center, self.excircle_center) > 1.0:
            return False
        if abs(radius - self.excircle_radius) > 1.0:
            return False

        # The construction arc must read as a curve grazing the side, not as a
        # second stroke lying on top of it.
        _bearing, half_sweep = self._tangency_arc()
        if half_sweep < math.radians(9.0):
            return False
        if self.excircle_radius * (1.0 - math.cos(half_sweep)) < 9.0:
            return False
        return True

    def _authored_observation(self) -> str:
        marked, first, second = self.triangle_points
        names = [
            name.removesuffix(" visible vertex")
            for name in self._visible_point_names([marked, first, second])
        ]
        side_to_first = self.distance(marked, first)
        side_to_second = self.distance(marked, second)
        base = self.distance(first, second)
        share = self.distance(first, self.target_point) / max(base, 1e-9)
        percent = int(round(share * 20.0)) * 5
        if abs(side_to_first - side_to_second) <= 0.06 * max(side_to_first, side_to_second):
            leg_phrase = (
                "The two sides leaving the marked corner are about equally long, so the contact sits near the "
                "middle of the base"
            )
        else:
            nearer, farther = (
                (names[1], names[2]) if side_to_first < side_to_second else (names[2], names[1])
            )
            leg_phrase = (
                f"The side running to the {nearer} corner is the shorter of the two leaving the marked corner, "
                f"and a shorter side leaves a longer stretch of the base on its own end, so the contact is pulled "
                f"away from the {nearer} corner and toward the {farther} corner"
            )
        return (
            f"The marked corner is the {names[0]}, so the circle in play is the one that leans on the far face of "
            f"the base joining the {names[1]} and the {names[2]} corners while also touching the two sides of the "
            "marked corner once they are prolonged past those corners. Both tangent segments from the marked "
            "corner to that circle are equal, and together they cover the whole perimeter once, so each measures "
            "the half-perimeter; taking away the side to the "
            f"{names[1]} corner leaves the stretch from the {names[1]} corner to the contact. "
            f"{leg_phrase}, landing about {percent}% of the way along the base from the {names[1]} corner, "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "triangle_vertices": [p.to_list() for p in self.triangle_points],
            "marked_vertex_index": 0,
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        if highlight_label:
            bearing, half_sweep = self._tangency_arc()
            self.draw_arc(
                draw,
                self.excircle_center,
                self.excircle_radius,
                -(bearing + half_sweep),
                2.0 * half_sweep,
                fill=self.CONSTRUCTION_COLOR,
            )
        self._draw_reference(draw)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

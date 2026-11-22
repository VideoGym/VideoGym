"""Inscribed square vertex puzzle generator."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PIL import Image

from data.geometry.core.point_target_base import (
    Incidence,
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class InscribedSquareVertexGenerator(PointTargetPuzzleGenerator):
    """Find the raised corner of the square inscribed on a marked triangle side."""

    DEFAULT_RULE_PROMPT = (
        "Input Image: A white canvas shows a black triangle with one side colored blue, and five labeled candidate "
        "circles A-E spaced out along one of the two black sides. Rule: Inscribe the unique square that has one full "
        "side lying on the blue side and its two other corners on the two black sides. Select the square corner that "
        "lies on the marker-bearing side. Required Output: Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/inscribed_square_vertex"
    MARKED_SIDE_COLOR = (36, 119, 196)
    # The option row rides the triangle side the answer sits on, so that side is
    # kept long enough for four markers to fit past the answer on either side.
    HOST_SIDE_RANGE = (345.0, 370.0)
    MAX_SIDE = 372.0
    MIN_ANGLE_DEG = 30.0
    MAX_ANGLE_DEG = 110.0
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows a triangle drawn as a black outline of medium thickness with one of its sides "
        "colored blue at the same thickness. Five candidate markers A-E are spaced out along one of the two black "
        "sides; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the triangle and the candidate markers, then draws three cyan segments of medium thickness forming the inscribed square on the blue side, with its raised corner on the marker-bearing side at the correct candidate. "
        "In the final state, only that corner's marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    @staticmethod
    def _inscribed_square(pA: Point, pB: Point, pC: Point):
        """Inscribed square with bottom on AB, top touching AC and BC.

        Returns (bottom_left, bottom_right, top_right, top_left) or None.
        """
        # Base vector
        bx, by = pB.x - pA.x, pB.y - pA.y
        base_len = math.hypot(bx, by)
        if base_len < 1e-9:
            return None
        ux, uy = bx / base_len, by / base_len
        # Perpendicular inward (toward C)
        nx, ny = -uy, ux
        # Check C is on the correct side
        dot_c = (pC.x - pA.x) * nx + (pC.y - pA.y) * ny
        if dot_c < 1e-9:
            nx, ny = -nx, -ny
            dot_c = -dot_c

        # Height of C above AB
        h = dot_c
        # Square side = h * base_len / (h + base_len)
        s = h * base_len / (h + base_len)
        if s < 1e-6:
            return None

        # Project C onto AB to find the horizontal position
        # Bottom-left: at distance d from A along AB
        # The left side of the square must touch AC.
        # Parametric: left side goes from (pA + d*u) up by s*n
        # Intersection with AC: pA + t*(pC - pA) = pA + d*u + s_frac*s*n
        # For left edge touching AC:
        dAC_x, dAC_y = pC.x - pA.x, pC.y - pA.y
        dAC_dot_u = dAC_x * ux + dAC_y * uy
        dAC_dot_n = dAC_x * nx + dAC_y * ny
        if abs(dAC_dot_n) < 1e-9:
            return None
        # At height s along normal, the AC line is at: t such that t*dAC_dot_n = s
        t_left = s / dAC_dot_n
        d_left = t_left * dAC_dot_u

        bl = Point(pA.x + d_left * ux, pA.y + d_left * uy)
        br = Point(bl.x + s * ux, bl.y + s * uy)
        tl = Point(bl.x + s * nx, bl.y + s * ny)
        tr = Point(br.x + s * nx, br.y + s * ny)
        return bl, br, tr, tl

    def _shape_grid(self, host_length: float) -> List[Tuple[float, float]]:
        """(share, base-angle) pairs whose triangle is presentable and fits.

        The raised corner sits at ``base / (height + base)`` of the host side,
        so the share is a shape parameter of the triangle rather than a free
        placement choice.  Every pair that keeps all three interior angles
        inside the declared band and every side on the canvas is enumerated,
        and the share is then drawn uniformly across the shares that survive:
        pinning the corner near the middle of its own side would let "the
        marker nearest that side's midpoint" answer the sample without ever
        building the square.
        """

        pairs: List[Tuple[float, float]] = []
        for share_step in range(60, 171):
            share = share_step / 200.0
            for degrees in range(
                int(self.MIN_ANGLE_DEG), int(self.MAX_ANGLE_DEG) + 1, 2
            ):
                angle = math.radians(degrees)
                height = host_length * math.sin(angle)
                foot = host_length * math.cos(angle)
                base = height * share / (1.0 - share)
                if base > self.MAX_SIDE:
                    continue
                far = math.hypot(base - foot, height)
                if far > self.MAX_SIDE:
                    continue
                at_far = math.degrees(math.atan2(height, base - foot)) % 180.0
                apex = 180.0 - degrees - at_far
                if min(degrees, at_far, apex) < self.MIN_ANGLE_DEG:
                    continue
                if max(degrees, at_far, apex) > self.MAX_ANGLE_DEG:
                    continue
                pairs.append((share, angle))
        return pairs

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        for _ in range(999):
            host_length = self._rng.uniform(*self.HOST_SIDE_RANGE)
            pairs = self._shape_grid(host_length)
            if not pairs:
                continue
            shares = sorted({share for share, _angle in pairs})
            share = self._rng.choice(shares)
            angle = self._rng.choice([a for s, a in pairs if s == share])
            height = host_length * math.sin(angle)
            foot = host_length * math.cos(angle)
            base = height * share / (1.0 - share)
            placed = self._place_triangle(
                [(0.0, 0.0), (base, 0.0), (foot, height)]
            )
            if placed is None:
                continue
            pA, pB, pC = placed

            result = self._inscribed_square(pA, pB, pC)
            if result is None:
                continue
            bl, br, tr, tl = result
            target = tl
            if not self.point_can_host_candidate(target):
                continue

            try:
                self._place_square_candidates(target, pA, pC)
            except RuntimeError:
                continue

            self.triangle_points = (pA, pB, pC)
            self.square_pts = (bl, br, tr, tl)
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to generate inscribed square vertex puzzle.")

    def _place_triangle(
        self,
        local: List[Tuple[float, float]],
    ) -> Optional[Tuple[Point, Point, Point]]:
        """Lay the constructed triangle on the canvas at a free heading."""

        theta = self._rng.uniform(0.0, math.tau)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rotated = [
            (px * cos_t - py * sin_t, px * sin_t + py * cos_t) for px, py in local
        ]
        left, top, right, bottom = self.canvas_bounds()
        pad_x, pad_y = self._candidate_safe_padding(include_edge_buffer=True)
        low_x = left + pad_x - min(point[0] for point in rotated)
        high_x = right - pad_x - max(point[0] for point in rotated)
        low_y = top + pad_y - min(point[1] for point in rotated)
        high_y = bottom - pad_y - max(point[1] for point in rotated)
        if low_x > high_x or low_y > high_y:
            return None
        shift_x = self._rng.uniform(low_x, high_x)
        shift_y = self._rng.uniform(low_y, high_y)
        return tuple(  # type: ignore[return-value]
            Point(px + shift_x, py + shift_y) for px, py in rotated
        )

    def _place_square_candidates(self, target: Point, pA: Point, pC: Point) -> None:
        """Spread A-E along the triangle side the raised corner rides.

        Every option sits on that side, so neither "outside the triangle" nor
        "not on the triangle" can eliminate one.  A distractor
        is wrong because the square built on the blue side cannot reach the
        side there, not because it floats in the interior.  Both ends of the
        side keep a marker's clearance so no option collides with a corner.
        """

        marker_clear = max(self.point_radius * 2.4, 28.0)
        corner_clear = self.point_radius * 2.1
        self.place_candidates_on_incidence(
            target,
            [Incidence.segment(pA, pC)],
            min_offset=marker_clear,
            # A hair under the step, so a row laid out at exactly the step is
            # not pushed outward by rounding and the outermost ranks keep the
            # side length they need.
            min_separation=marker_clear * 0.98,
            reject=lambda point: (
                self.distance(point, pA) < corner_clear
                or self.distance(point, pC) < corner_clear
            ),
        )

    def _authored_observation(self) -> str:
        host_base, other_base, apex = self.triangle_points
        names = [
            name.removesuffix(" visible vertex")
            for name in self._visible_point_names([host_base, other_base, apex])
        ]
        base_run = self._heading_phrase(
            other_base.x - host_base.x, other_base.y - host_base.y
        )
        base_mid = Point(
            (host_base.x + other_base.x) / 2.0, (host_base.y + other_base.y) / 2.0
        )
        apex_relation = self._relative_position_phrase(base_mid, apex)
        return (
            f"The blue side runs {base_run} from the {names[0]} corner to the {names[1]} corner, and the "
            f"{names[2]} corner closes the triangle {apex_relation} it. The markers line the black side that "
            f"climbs from the {names[0]} corner to the {names[2]} corner, so that side carries the answer. The "
            "inscribed square lays one full side along the blue side and raises its two remaining corners until "
            "each touches one black side; the raised corner on the marker-bearing side sits "
            f"{self._canvas_region(self.target_point)}."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            # Ordered (marker-bearing base endpoint, other base endpoint,
            # apex): the blue side is the first pair and the answer is the
            # square corner on the side from the first vertex to the apex.
            "triangle_vertices": [p.to_list() for p in self.triangle_points],
            "square_vertices": [p.to_list() for p in self.square_pts],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, list(self.triangle_points) + [self.triangle_points[0]])
        host_base, other_base, _apex = self.triangle_points
        self.draw_line(draw, [host_base, other_base], fill=self.MARKED_SIDE_COLOR)
        if highlight_label:
            bl, br, tr, tl = self.square_pts
            # Draw as 3 separate segments (bottom is on the blue side, skip it)
            self.draw_line(draw, [bl, tl])
            self.draw_line(draw, [tl, tr])
            self.draw_line(draw, [tr, br])
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

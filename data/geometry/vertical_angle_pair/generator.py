"""Vertical angle pair puzzle generator.

Two straight black lines cross at a marked point P. One of the four
angle regions formed at P is highlighted by a thin black arc. Five
labeled candidate markers A-E are placed throughout the canvas; exactly
one candidate sits inside the angle region VERTICALLY OPPOSITE to the
highlighted region (the vertical-angle pair shares the same opening).
"""

from __future__ import annotations

import math
from typing import List, Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Point,
    PointCandidate,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


class VerticalAnglePairGenerator(PointTargetPuzzleGenerator):
    """Find which candidate sits inside the vertical-angle region."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows two black lines crossing at one point. A thin arc near the intersection highlights one of the "
        "four angle regions. Five labeled candidate markers A-E are placed around the canvas. Identify the candidate that "
        "sits inside the angle region vertically opposite to the highlighted one (the vertical-angle pair sharing the same "
        "opening). Answer with one option from A-E."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/vertical_angle_pair"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows two long black straight lines of medium thickness crossing at one shared point in the middle of the canvas, partitioning the plane into four angle regions; a thin black quarter-circle arc near the intersection highlights one of the four regions. "
        "Five small labeled candidate markers A-E are scattered around the canvas; each marker is a small white circle with a thin dark gray outline and a black uppercase letter, and exactly one marker sits inside the angle region directly opposite the highlighted region. "
        "The video first holds the two crossing lines, the highlighted region arc, and the candidate markers, then traces a thin black mirror-image arc inside the vertical region to anchor the equality between the highlighted angle and its vertical pair. "
        "In the final state, only the candidate inside the vertical region changes to pale red fill with a dark red outline, while the other four markers stay white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    MIN_MARKER_GAP_RATIO = 0.16
    MIN_LINE_CLEARANCE_RATIO = 0.05
    MIN_MARKER_REACH_RATIO = 0.16
    MAX_MARKER_REACH_RATIO = 0.34

    @property
    def MIN_MARKER_GAP(self) -> float:
        return self.canvas_short_side * self.MIN_MARKER_GAP_RATIO

    @property
    def MIN_LINE_CLEARANCE(self) -> float:
        return self.canvas_short_side * self.MIN_LINE_CLEARANCE_RATIO

    @property
    def MIN_MARKER_REACH(self) -> float:
        return self.canvas_short_side * self.MIN_MARKER_REACH_RATIO

    @property
    def MAX_MARKER_REACH(self) -> float:
        return self.canvas_short_side * self.MAX_MARKER_REACH_RATIO

    def _region_half_opening(self, region: int) -> float:
        return (self.alpha_rad if region % 2 == 0 else math.pi - self.alpha_rad) / 2.0

    def _marker_in_region(self, region: int) -> Optional[Point]:
        """One marker inside a region, seated the same way in every region.

        Both quantities a blind reading can pick up on are drawn from the same
        distribution whichever region the marker lands in: how far it sits from
        the crossing, and how far it sits from the nearer of the two lines.
        The second one is why it matters.  The vertical pair is the *narrow*
        pair of regions here, and a point inside a narrow wedge hugs the lines
        that bound it, so drawing an angle inside the wedge instead would put
        the answer nearer the ink than four distractors spread over the wide
        regions -- 0.35 of the time against 0.200 chance, on a reading that
        never looks at the highlighted region at all.  The
        clearance is therefore drawn first, capped by what the narrowest wedge
        could host, and the angle inside the wedge follows from it.
        """

        half = self._region_half_opening(region)
        reach = self._rng.uniform(self.MIN_MARKER_REACH, self.MAX_MARKER_REACH)
        span = reach * math.sin(self.alpha_rad / 2.0)
        if span <= self.MIN_LINE_CLEARANCE:
            return None
        clearance = self._rng.uniform(self.MIN_LINE_CLEARANCE, span)
        offset = half - math.asin(min(1.0, clearance / reach))
        angle = (
            self.bisector_angle
            + region * math.pi / 2.0
            + self._rng.choice((-1.0, 1.0)) * offset
        )
        point = self.point_on_ray(self.cross, angle, reach)
        if not self._candidate_fits(PointCandidate(x=point.x, y=point.y, label="A")):
            return None
        return point

    def _draw_arc_in_region(self, draw, region_index: int) -> None:
        # Render a thin arc of radius ~ self.canvas_short_side*0.07 around
        # the intersection inside the indicated region. The arc spans the
        # opening of that region (half_opening on each side of bisector).
        # NOTE: draw_arc expects math y-up convention; negate to match
        # the image y-down convention used elsewhere in this generator.
        bisector = self.bisector_angle + region_index * math.pi / 2
        opening = self.alpha_rad if region_index % 2 == 0 else (math.pi - self.alpha_rad)
        arc_radius = self.canvas_short_side * 0.07
        start_angle = bisector - opening / 2.0
        sweep = opening
        self.draw_arc(draw, self.cross, arc_radius, -start_angle, -sweep, width_factor=0.6)

    def _video_overlay_extras(self, draw) -> None:
        # Crossing lines and highlighted arc must persist across the video.
        self.draw_line(draw, [self.line1_p0, self.line1_p1])
        self.draw_line(draw, [self.line2_p0, self.line2_p1])
        self._draw_arc_in_region(draw, 0)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        cw, ch = self.canvas_dimensions
        for _ in range(999):
            cross = Point(
                x=cw * self._rng.uniform(0.40, 0.60),
                y=ch * self._rng.uniform(0.40, 0.60),
            )
            self.cross = cross
            # First line at angle theta1, second at theta1 + alpha (with alpha
            # in 40..85 deg so the four regions have visually distinct widths).
            # theta1 turns through the whole circle, not a half of it: the
            # vertical region's own heading is then uniform too, where before
            # it always pointed into the same half of the plane and "not the
            # bottom-most marker" ruled the answer out 0.99 of the time

            theta1 = self._rng.uniform(0.0, math.tau)
            alpha = math.radians(self._rng.uniform(40.0, 85.0))
            theta2 = theta1 + alpha
            self.theta1 = theta1
            self.alpha_rad = alpha
            # Region 0's bisector lies between theta1 and theta2.
            self.bisector_angle = theta1 + alpha / 2.0

            # Compute line endpoints clipped to canvas.
            try:
                self.line1_p0, self.line1_p1 = self.clip_line_to_canvas(cross, theta1, padding=self.line_width)
                self.line2_p0, self.line2_p1 = self.clip_line_to_canvas(cross, theta2, padding=self.line_width)
            except Exception:
                continue

            # One marker in the vertical region and one in each of the other
            # three, plus a fifth in a region drawn from the three the answer
            # does not occupy.  Every one of them is seated by the same
            # sampler, so which region a marker sits in is the only thing
            # that tells it apart from the answer.
            # The fifth marker doubles up in one of the two *wide* regions,
            # never in the narrow one opposite the answer.  Both narrow regions
            # then hold exactly one marker, so nothing about the arrangement
            # tells the vertical region from the highlighted one; doubling the
            # narrow region instead pulled the cloud's centroid straight away
            # from the answer and made it the marker furthest from that
            # centroid 0.38 of the time.
            spare = self._rng.choice((1, 3))
            regions = [2, 0, 1, 3, spare]
            # Seated in a shuffled order, not answer-first.  Five markers this
            # far apart barely fit the ring they are drawn from, so whichever
            # one goes down first is the one the rest end up crowding against
            # at exactly the minimum gap: seating the answer first made it the
            # least isolated marker 0.43 of the time against 0.200 chance

            self._rng.shuffle(regions)
            placed: List[Point] = []
            for region in regions:
                for _ in range(200):
                    point = self._marker_in_region(region)
                    if point is None:
                        continue
                    if any(
                        self.distance(point, other) < self.MIN_MARKER_GAP
                        for other in placed
                    ):
                        continue
                    placed.append(point)
                    break
                else:
                    break
            if len(placed) < len(regions):
                continue

            labels = list(self.option_labels)
            self._rng.shuffle(labels)
            candidates: List[PointCandidate] = []
            for i, point in enumerate(placed):
                candidates.append(PointCandidate(x=point.x, y=point.y, label=labels[i]))
            if not self.validate_candidate_layout(candidates, min_distance=self.MIN_MARKER_GAP * 0.95):
                continue
            answer = regions.index(2)
            self.candidates = candidates
            self.correct_label = labels[answer]
            self.target_point = placed[answer]
            return self.save_puzzle()
        raise RuntimeError("Failed to lay out vertical_angle_pair puzzle")

    def _authored_observation(self) -> str:
        highlighted_heading = self._heading_phrase(
            math.cos(self.bisector_angle), math.sin(self.bisector_angle)
        )
        opening_size = self._angle_size_phrase(math.degrees(self.alpha_rad))
        opposite_side = self._relative_position_phrase(self.cross, self.target_point)
        return (
            f"Two straight lines cross at the marked intersection {self._canvas_region(self.cross)}, carving four "
            f"angle regions; the thin arc highlights the region opening {highlighted_heading} from the crossing, "
            f"spanning {opening_size}. Its vertical partner keeps that same opening but points the opposite way "
            f"through the crossing, so the required marker sits {opposite_side} the intersection, wedged between "
            "the far continuations of the same two lines."
        )

    def build_record_extra(self) -> dict[str, object]:
        return {
            "intersection": self.cross.to_list(),
            "line1_endpoints": [self.line1_p0.to_list(), self.line1_p1.to_list()],
            "line2_endpoints": [self.line2_p0.to_list(), self.line2_p1.to_list()],
            "highlighted_region_index": 0,
            "vertical_region_index": 2,
            "alpha_deg": math.degrees(self.alpha_rad),
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        self.draw_line(draw, [self.line1_p0, self.line1_p1])
        self.draw_line(draw, [self.line2_p0, self.line2_p1])
        self._draw_arc_in_region(draw, 0)
        if highlight_label:
            # Mirror arc inside vertical region for emphasis.
            self._draw_arc_in_region(draw, 2)
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

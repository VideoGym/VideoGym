"""Ray reflection puzzle generator.
"""

from __future__ import annotations
from typing import Optional
from PIL import Image, ImageDraw
import math
from data.geometry.core.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class RayReflectGenerator(PointTargetPuzzleGenerator):
    """Trace a ray through finite mirror segments."""

    DEFAULT_RULE_PROMPT=(
        "A white canvas shows a black mirror line, a solid black source disc, an incoming ray segment with a small "
        "arrowhead pointing toward the mirror, and five labeled candidate circles A-E. Determine which candidate lies on "
        "the outgoing reflected ray after mentally reflecting the incoming ray off the mirror. Answer with one option from A-E."
    )
    DEFAULT_OUTPUT_DIR = "outputs/geometry/ray_reflect"
    DEFAULT_TI2V_PROMPT=(
        "A square white canvas shows one black mirror segment of medium thickness, a small solid black source disc away from the mirror, and a thick black incoming-ray segment running from the source toward the mirror at a non-perpendicular angle and tipped with a small triangular arrowhead near the mirror surface (the ray stops short of the mirror so the precise reflection point is implicit). "
        "Five candidate markers A-E are placed in a short row near where the reflected ray would land; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the mirror, the source disc, the incoming ray with arrowhead, and the candidate markers, then extends that incoming ray in cyan from the arrowhead to the reflection point on the mirror and continues it away from the mirror as a cyan reflected segment toward the correct candidate, without redrawing the already-visible incoming ray. "
        "In the terminal state, only the candidate on the reflected ray changes to pale red fill with a dark red outline, while the other markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "A white canvas shows a black mirror line, a solid black source disc, an arrowhead-tipped incoming ray segment, "
        "and five labeled candidate circles A-E. Extend the incoming ray from the arrowhead to the reflection point on "
        "the mirror, then continue it away from the mirror as a matching medium-thickness reflected segment toward the "
        "correct candidate. In the solved image, only the candidate on the reflected ray changes to pale red fill with a "
        "dark red outline, while the other markers remain white."
    )

    def _reflect_point(self, point_to_reflect: Point, line_p1: Point, line_p2: Point) -> Point:
        """Helper to reflect a point across a line defined by two other points."""
        dx = line_p2.x - line_p1.x
        dy = line_p2.y - line_p1.y

        line_len_sq = dx * dx + dy * dy
        # Handle degenerate case where the line is just a point
        if line_len_sq < 1e-9:
            return point_to_reflect

        # Project the vector (point_to_reflect - line_p1) onto the line vector (line_p2 - line_p1)
        # t is the projection factor
        t = ((point_to_reflect.x - line_p1.x) * dx + (point_to_reflect.y - line_p1.y) * dy) / line_len_sq

        # Find the closest point on the line to point_to_reflect
        closest_point = Point(x=line_p1.x + t * dx, y=line_p1.y + t * dy)

        # The reflected point is P' = P + 2*(Closest - P) = 2*Closest - P
        reflected_point = Point(
            x=2 * closest_point.x - point_to_reflect.x,
            y=2 * closest_point.y - point_to_reflect.y
        )
        return reflected_point

    def __init__(self, *args, reflections=1, **kwargs):
        self.reflection_count = int(reflections)
        if self.reflection_count != 1:
            self.DEFAULT_RULE_PROMPT = "Input Image: Finite black mirror segments, a filled light source, an incoming arrow, and five candidate markers A-E are shown. Rule: Continue the light ray, reflecting at every encountered mirror with equal incidence and reflection angles. The nearest forward mirror intersection is the next contact. Select the candidate on the outgoing ray after it has cleared all mirrors. Required Output: Answer with one option from A-E."
            self.DEFAULT_TI2V_PROMPT = "Input Image: Finite black mirror segments, a filled light source, an incoming arrow, and five candidate markers A-E are shown. Rule: At each encountered mirror, reflect the ray with equal incidence and reflection angles; always take the nearest forward contact and continue until the ray has cleared all mirrors. Required Output: Extend the incoming arrow in cyan through every reflection, in travel order, as one continuous polyline to the correct candidate; highlight only that marker. Preserve all mirrors, the source, and the original arrow."
            self.DEFAULT_SOLUTION_IMAGE_PROMPT = self.strip_video_instruction(self.DEFAULT_TI2V_PROMPT)
        super().__init__(*args, **kwargs)

    def _create_multiple(self):
        from data.geometry.core.ray_optics import trace_mirrors
        if self.reflection_count != 4:
            raise ValueError("multi-mirror layout requires four reflections")
        for _ in range(1000):
            coordinates=[(105,65),(365,115),(140,205),(365,295),(140,385),(300,440)]
            pts=[Point(x+self._rng.uniform(-15,15),y+self._rng.uniform(-10,10)) for x,y in coordinates]
            # Rigid orientations vary the scene without changing its ray law.
            angle=self._rng.choice((0,math.pi/2,math.pi,3*math.pi/2))
            pts=[Point(256+(p.x-256)*math.cos(angle)-(p.y-256)*math.sin(angle),256+(p.x-256)*math.sin(angle)+(p.y-256)*math.cos(angle)) for p in pts]
            mirrors=[]
            for a,b,c in zip(pts,pts[1:],pts[2:]):
                incoming=((b.x-a.x)/self.distance(a,b),(b.y-a.y)/self.distance(a,b))
                outgoing=((c.x-b.x)/self.distance(b,c),(c.y-b.y)/self.distance(b,c))
                tangent=(incoming[0]+outgoing[0],incoming[1]+outgoing[1]);norm=math.hypot(*tangent)
                half=self._rng.uniform(35,43)
                mirrors.append((Point(b.x-half*tangent[0]/norm,b.y-half*tangent[1]/norm),Point(b.x+half*tangent[0]/norm,b.y+half*tangent[1]/norm)))
            source,target=pts[0],pts[-1];tip=Point(source.x+0.75*(pts[1].x-source.x),source.y+0.75*(pts[1].y-source.y))
            try:contacts,direction,order=trace_mirrors(source.to_list(),tip.to_list(),[[a.to_list(),b.to_list()] for a,b in mirrors])
            except ValueError:continue
            if order!=list(range(4)):continue
            if any(self.distance(Point(*got),want)>1e-5 for got,want in zip(contacts,pts[1:-1])):continue
            if not all(self.inside_canvas(p,padding=24) for pair in mirrors for p in pair):continue
            try:
                self._place_ray_reflect_candidates(target, source, pts[-2])
            except RuntimeError:continue
            # Candidates and the incoming arrow must remain legible against
            # every mirror and the already traversed ray sections.
            if any(self._point_to_segment_dist(Point(c.x,c.y),a,b)<24 for c in self.candidates for a,b in mirrors):continue
            if any(self._point_to_segment_dist(Point(c.x,c.y),a,b)<24 for c in self.candidates for a,b in zip(pts[:-2],pts[1:-1])):continue
            self.points=(source,pts[1],*mirrors[0]);self.mirror_segments=mirrors;self.reflections=pts[1:-1];self.target_point=target
            return self.save_puzzle()
        raise RuntimeError("failed to construct a readable four-reflection path")

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        if self.reflection_count > 1:
            return self._create_multiple()
        self.margin = 50
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        tries = 0
        while tries < 999:
            tries += 1

            # 1. Define the mirror line segment
            mirror_mid = self.pick_target_point(0.52, padding=candidate_padding + self.canvas_short_side * 0.12)
            mirror_angle = self._rng.uniform(0.0, math.tau)
            try:
                m1, m2 = self.sample_symmetric_segment(
                    mirror_mid,
                    mirror_angle,
                    min_half_length=self.canvas_short_side * 0.2,
                    max_half_length=self.canvas_short_side * 0.3,
                    padding=self.line_width,
                )
            except RuntimeError:
                continue

            # 2. Define the light source
            source = self.pick_target_point(0.5, padding=self.canvas_short_side * 0.08)

            # 3. Ensure the source is not too close to the mirror line
            dx, dy = m2.x - m1.x, m2.y - m1.y
            dist_from_line = abs(dx * (source.y - m1.y) - dy * (source.x - m1.x)) / math.sqrt(dx*dx + dy*dy)
            if dist_from_line < self.canvas_short_side * 0.22:
                continue

            # 4. Pick a point on the mirror segment for the reflection to occur
            reflect_ratio = self._rng.uniform(0.24, 0.76)  # Avoid the very ends of the mirror
            p_mirror = Point(
                x=m1.x + reflect_ratio * (m2.x - m1.x),
                y=m1.y + reflect_ratio * (m2.y - m1.y),
            )

            # 5. Check the angle of incidence to avoid very shallow angles
            v_in = (p_mirror.x - source.x, p_mirror.y - source.y)
            v_mirror = (m2.x - m1.x, m2.y - m1.y)
            dot_product = v_in[0]*v_mirror[0] + v_in[1]*v_mirror[1]
            mag_in = math.sqrt(v_in[0]**2 + v_in[1]**2)
            mag_mirror = math.sqrt(v_mirror[0]**2 + v_mirror[1]**2)
            if mag_in < 1e-6 or mag_mirror < 1e-6: continue
            cos_angle = abs(dot_product / (mag_in * mag_mirror))
            if cos_angle > 0.88:  # Angle is < ~28 degrees, too shallow
                continue

            # 6. The key insight: the reflected ray appears to come from the reflection of the source.
            s_reflected = self._reflect_point(source, m1, m2)

            # 7. The reflected ray is the line from p_mirror through s_reflected. Find a target on it.
            ray_dir_x = s_reflected.x - p_mirror.x
            ray_dir_y = s_reflected.y - p_mirror.y
            ray_len = math.sqrt(ray_dir_x**2 + ray_dir_y**2)
            if ray_len < 1e-6: continue

            # Extend the ray to find a target point within the canvas
            ray_angle = math.atan2(-ray_dir_y, -ray_dir_x)
            try:
                target = self.sample_point_along_direction(
                    p_mirror,
                    ray_angle,
                    min_distance=self.canvas_short_side * 0.4,
                    max_distance=self.canvas_short_side * 0.62,
                    padding=candidate_padding,
                )
            except RuntimeError:
                continue

            # 8. Final validation
            if not self.point_can_host_candidate(target):
                continue
            # Make sure the reflected target sits well away from the source
            # disc and the incoming-ray segment so the candidate cluster
            # placed around target does not visually crash the source dot
            # or the arrow.
            min_target_clearance = self.canvas_short_side * 0.18
            if math.hypot(target.x - source.x, target.y - source.y) < min_target_clearance:
                continue

            try:
                self._place_ray_reflect_candidates(target, source, p_mirror)
            except RuntimeError:
                continue

            self.points = (source, p_mirror, m1, m2)
            self.target_point = target
            return self.save_puzzle()

        raise RuntimeError("Failed to find valid ray reflection geometry after 999 tries")

    def _place_ray_reflect_candidates(
        self,
        target: Point,
        source: Point,
        p_mirror: Point,
    ) -> None:
        """2D-scatter 5 candidates near target, with strict clearance from
        the source disc, the incoming-ray segment, AND the reflected-ray
        line itself (so no distractor accidentally sits on the same line
        as the answer — the answer must be the unique candidate on the
        outgoing ray).

        The outgoing ray is what the rule names, so the reflected-ray
        clearance is measured against the ray — forward-infinite from the
        contact — and not against the contact-to-target segment.  A segment
        distance clamps at the target, so it charges a distractor parked
        *past* the answer only its distance to the answer, and a marker the
        ray runs straight through passes the test: over seeds 90000-90299
        the ray crossed a distractor's own disc on 51 samples and came
        within the replay's uniqueness gate on 22, seed 90210 among them.
        Those samples had two candidates on the outgoing ray and no rule to
        separate them.

        This replaces the previous horizontal-row placement which was
        axis-sortable. The 2D scatter forces distractors to perpendicular
        offsets from the reflected ray, so eye-extrapolation of the
        reflection is the only winning strategy.
        """
        spread = self.minimum_candidate_spacing()
        marker_clear = max(self.point_radius * 2.6, spread * 0.95)
        source_clear = max(self.point_radius * 2.0, 28.0)
        incoming_ray_clear = max(self.point_radius * 1.6, 22.0)
        # Reflected-ray clearance: distractors MUST be off this ray so the
        # answer is the unique candidate on the outgoing ray.
        reflected_ray_clear = max(self.point_radius * 1.6, 22.0)

        def clear_of_rays(point: Point) -> bool:
            return (
                self.distance(point, source) >= source_clear
                and self._point_to_segment_dist(point, source, p_mirror) >= incoming_ray_clear
                and self._point_to_ray_dist(point, p_mirror, target) >= reflected_ray_clear
            )

        # The cloud is laid out before one of its slots is taken for the
        # answer, so the answer is not the marker its four neighbours enclose
        #; eye-extrapolation of the reflection stays the only
        # winning strategy.
        self.seat_candidates(
            target, accept=clear_of_rays, min_separation=marker_clear
        )

    @staticmethod
    def _point_to_ray_dist(P: Point, A: Point, B: Point) -> float:
        """Distance from P to the ray that leaves A through B and never stops.

        Behind the origin the ray is only the point A, so the distance there
        is the distance to A; ahead of it the ray is the whole half-line, so
        the distance is the perpendicular offset however far along P sits.
        """
        ABx, ABy = B.x - A.x, B.y - A.y
        length = math.hypot(ABx, ABy)
        if length < 1e-9:
            return math.hypot(P.x - A.x, P.y - A.y)
        ux, uy = ABx / length, ABy / length
        APx, APy = P.x - A.x, P.y - A.y
        if APx * ux + APy * uy <= 0.0:
            return math.hypot(APx, APy)
        return abs(APx * uy - APy * ux)

    @staticmethod
    def _point_to_segment_dist(P: Point, A: Point, B: Point) -> float:
        ABx, ABy = B.x - A.x, B.y - A.y
        APx, APy = P.x - A.x, P.y - A.y
        ab2 = ABx * ABx + ABy * ABy
        if ab2 < 1e-9:
            return math.hypot(APx, APy)
        t = max(0.0, min(1.0, (APx * ABx + APy * ABy) / ab2))
        cx = A.x + t * ABx
        cy = A.y + t * ABy
        return math.hypot(P.x - cx, P.y - cy)

    def _authored_observation(self) -> str:
        if self.reflection_count > 1:
            return "Extend the incoming arrow to the nearest mirror, reflect its direction across that mirror, and repeat at each next forward intersection. Four successive contacts determine the outgoing ray and its candidate."
        source, p_mirror, m1, m2 = self.points
        mirror_heading = self._heading_phrase(m2.x - m1.x, m2.y - m1.y)
        incoming_heading = self._heading_phrase(p_mirror.x - source.x, p_mirror.y - source.y)
        outgoing_heading = self._heading_phrase(
            self.target_point.x - p_mirror.x, self.target_point.y - p_mirror.y
        )
        return (
            f"The mirror segment runs {mirror_heading}, and the filled source disc sits "
            f"{self._canvas_region(source)} with its arrow-tipped ray heading {incoming_heading} toward the "
            f"mirror. Extended along its arrow, the ray meets the mirror {self._canvas_region(p_mirror)}; "
            "reflection keeps the incoming and outgoing inclinations to the mirror equal, so the outgoing ray "
            f"leaves that contact {outgoing_heading}, and I keep the marker whose center it passes through."
        )

    def build_record_extra(self) -> dict[str, object]:
        source, p_mirror, m1, m2 = self.points
        if self.reflection_count > 1:
            return {"light_source":source.to_list(),
                    "mirror_segments":[[a.to_list(),b.to_list()] for a,b in self.mirror_segments],
                    "incoming_ray_tip":[source.x+0.75*(p_mirror.x-source.x),source.y+0.75*(p_mirror.y-source.y)]}
        return {
            "light_source": source.to_list(),
            "mirror_line": [m1.to_list(), m2.to_list()],
            "reflection_point": p_mirror.to_list(),
        }

    def _draw_source_point(self, draw) -> None:
        source, _, _, _ = self.points
        # Solid black source disc — clearly the "starting point of the
        # ray" — so the puzzle reads from the image alone.
        cx, cy = round(source.x), round(source.y)
        r = 8
        draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=self.CANDIDATE_OUTLINE_COLOR,
            outline=self.CANDIDATE_OUTLINE_COLOR,
            width=1,
        )

    def _draw_arrowhead(self, draw, tip: Point, direction_angle: float, size: float) -> None:
        """Triangular arrowhead at `tip` pointing along `direction_angle`."""
        ux = math.cos(direction_angle)
        uy = math.sin(direction_angle)
        px = -uy
        py = ux
        base_back = size * 0.95
        half_w = size * 0.55
        bx = tip.x - ux * base_back
        by = tip.y - uy * base_back
        p_left = (round(bx + px * half_w), round(by + py * half_w))
        p_right = (round(bx - px * half_w), round(by - py * half_w))
        p_tip = (round(tip.x), round(tip.y))
        draw.polygon(
            [p_tip, p_left, p_right],
            fill=self.CANDIDATE_OUTLINE_COLOR,
            outline=self.CANDIDATE_OUTLINE_COLOR,
            width=1,
        )

    def _video_overlay_extras(self, draw: ImageDraw.ImageDraw) -> None:
        self._draw_source_point(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()

        source, p_mirror, m1, m2 = self.points

        # Draw the mirror line — same width as every other geometry line.
        for first,last in (self.mirror_segments if self.reflection_count > 1 else [(m1,m2)]):
            self.draw_line(draw, [first,last])

        # Incoming ray runs from the source most of the way to the
        # mirror, but stops short so the reflection point isn't
        # given away. We choose 75% so the angle of incidence is
        # visually unambiguous yet the precise mirror-hit location is
        # left implicit.
        ray_tip_t = 0.75
        ray_tip = Point(
            x=source.x + ray_tip_t * (p_mirror.x - source.x),
            y=source.y + ray_tip_t * (p_mirror.y - source.y),
        )
        self.draw_line(draw, [source, ray_tip])
        # Arrowhead at the ray tip, pointing along the incoming
        # direction (source -> mirror). Sized proportional to line_width
        # so it scales with the rendering contract.
        incoming_angle = math.atan2(p_mirror.y - source.y, p_mirror.x - source.x)
        self._draw_arrowhead(
            draw, ray_tip, incoming_angle,
            size=max(self.line_width * 2.6, 12.0),
        )

        # Solution-only additions: extend the ray from the stub tip to
        # the reflection point, then continue it as the reflected
        # segment to the answer candidate. ONE polyline so the
        # animation flows smoothly through the mirror.
        if highlight_label:
            self.draw_line(draw, [ray_tip, *(self.reflections if self.reflection_count > 1 else [p_mirror]), self.target_point])

        self._draw_source_point(draw)

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

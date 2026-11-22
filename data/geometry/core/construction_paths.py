"""Terminal-image construction geometry for point-target tasks.

``coordinate_target`` replays an ordered *stroke* grammar for every task whose
whole-video contract contains a serial construction, and holds the terminal
image to the same geometry.  Other point-target tasks draw a construction too -- an
orbit, a secant, a pair of tangents, the complement of an arc -- and nothing
required it: the terminal gates could see that the input scene survived and
that the right marker turned red, so a delivery that simply omitted the
construction, or omitted one member of it, was accepted.  That is the
aggregate-comparison failure: the
present members pay for the missing one because nothing counts them.

This module supplies the construction geometry those tasks declare,
recomputed from the record's own semantic fields plus the caller's
independently replayed answer -- never from a saved ``correct_option``, a
reference image or a solver trace.  :func:`terminal_construction_paths`
retains the unordered member-set contract used by terminal-image verification.
:func:`serial_construction_path_variants` separately groups those members into
complete pen commands and enumerates only the order variants allowed by the
visible rule.  Keeping the two views separate prevents a terminal obligation
from silently imposing an arbitrary video order.

Returning ``None`` means this module owns no geometry for the task and the
terminal construction dimension stays masked; returning an empty list means it
owns the task but the supplied semantic input was incomplete, which the caller
treats as a fail-closed condition.
"""

from __future__ import annotations

import math
from itertools import permutations
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data.geometry.core.point_target_base import PointTargetPuzzleGenerator

Point = Tuple[float, float]

# Positions sampled along a declared circular sweep.  The caller resamples
# whatever it needs from the polyline, so this only has to be fine enough that
# the chords stay inside the stroke: at the family's largest usable radius a
# 96-segment full turn deviates from the circle by well under a pixel.
_ARC_STEPS = 96


def _pt(value: Any) -> Optional[Point]:
    if isinstance(value, Mapping):
        value = (value.get("x"), value.get("y"))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _pts(value: Any, count: int) -> Optional[List[Point]]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        return None
    parsed = [_pt(item) for item in value]
    return None if any(item is None for item in parsed) else [item for item in parsed if item]


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _candidate(record: Mapping[str, Any], label: str) -> Optional[Point]:
    for item in record.get("candidates") or []:
        if isinstance(item, Mapping) and str(item.get("label") or "").strip().upper() == label:
            return _pt(item)
    return None


def _rotate(point: Point, pivot: Point, degrees: float) -> Point:
    """Rotate in the generators' convention: positive is counter-clockwise.

    Canvas y grows downward, so both sine terms flip relative to the usual
    Cartesian matrix.  Same convention as the answer replay.
    """

    angle = math.radians(degrees)
    dx, dy = point[0] - pivot[0], point[1] - pivot[1]
    return (
        pivot[0] + dx * math.cos(angle) + dy * math.sin(angle),
        pivot[1] - dx * math.sin(angle) + dy * math.cos(angle),
    )


def _reflect(point: Point, first: Point, second: Point) -> Optional[Point]:
    dx, dy = second[0] - first[0], second[1] - first[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-9:
        return None
    ratio = ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / denominator
    foot = (first[0] + ratio * dx, first[1] + ratio * dy)
    return (2.0 * foot[0] - point[0], 2.0 * foot[1] - point[1])


def _path(points: Sequence[Point]) -> np.ndarray:
    return np.asarray(points, dtype=np.float32)


def _reverse_path(path: np.ndarray) -> np.ndarray:
    """Return an independent copy of a centreline in the opposite direction."""

    return np.asarray(path[::-1], dtype=np.float32).copy()


def _trimmed_segment(
    start: Point,
    end: Point,
    *,
    start_offset: float = 0.0,
    end_offset: float = 0.0,
) -> Optional[np.ndarray]:
    """Match ``PointTargetPuzzleGenerator.trim_segment`` in pure geometry."""

    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9 or start_offset + end_offset >= length:
        return None
    ux, uy = dx / length, dy / length
    return _path(
        [
            (start[0] + ux * start_offset, start[1] + uy * start_offset),
            (end[0] - ux * end_offset, end[1] - uy * end_offset),
        ]
    )


def _merge_paths(paths: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Join path members that meet at one endpoint into one pen command."""

    if not paths:
        return None
    merged = np.asarray(paths[0], dtype=np.float32)
    if merged.ndim != 2 or merged.shape[0] < 2 or merged.shape[1] != 2:
        return None
    for member in paths[1:]:
        member_array = np.asarray(member, dtype=np.float32)
        if (
            member_array.ndim != 2
            or member_array.shape[0] < 2
            or member_array.shape[1] != 2
            or not np.allclose(merged[-1], member_array[0], atol=1e-3, rtol=0.0)
        ):
            return None
        merged = np.concatenate((merged, member_array[1:]), axis=0)
    return merged.copy()


def _point_radius(record: Mapping[str, Any]) -> Optional[float]:
    radius = _number(record.get("point_radius"))
    if radius is None:
        radius = float(PointTargetPuzzleGenerator.POINT_RADIUS)
    return radius if radius >= 0.0 else None


def _margin(record: Mapping[str, Any], *, default: Optional[float] = None) -> float:
    """Working margin, with an optional task-owned canonical default."""

    declared = _number(record.get("margin"))
    if declared is not None and declared >= 0.0:
        return declared
    if default is not None:
        return float(default)
    return float(max(18, round(0.10 * _canvas_short_side(record))))


def _turn(point: Point, pivot: Point, degrees: float) -> np.ndarray:
    """The circular sweep ``point`` traces turning ``degrees`` about ``pivot``."""

    return _path(
        [_rotate(point, pivot, degrees * index / _ARC_STEPS) for index in range(_ARC_STEPS + 1)]
    )


def _screen_arc(centre: Point, radius: float, start: float, sweep: float) -> np.ndarray:
    """Arc sampled in screen angles, where the angle grows clockwise on screen."""

    return _path(
        [
            (
                centre[0] + radius * math.cos(start + sweep * index / _ARC_STEPS),
                centre[1] + radius * math.sin(start + sweep * index / _ARC_STEPS),
            )
            for index in range(_ARC_STEPS + 1)
        ]
    )


def _canvas_short_side(record: Mapping[str, Any]) -> float:
    dimensions = record.get("canvas_dimensions") or [512, 512]
    try:
        return float(min(float(dimensions[0]), float(dimensions[1])))
    except (TypeError, ValueError, IndexError):
        return 512.0


def _clipped_full_line(
    record: Mapping[str, Any],
    point: Point,
    direction: Point,
    *,
    padding: float,
) -> Optional[np.ndarray]:
    """The full line through ``point`` clipped to the padded canvas."""

    dimensions = record.get("canvas_dimensions") or [512, 512]
    try:
        width, height = float(dimensions[0]), float(dimensions[1])
    except (TypeError, ValueError, IndexError):
        width = height = 512.0
    dx, dy = direction
    if math.hypot(dx, dy) <= 1e-9:
        return None
    crossings: List[Tuple[float, Point]] = []
    if abs(dx) > 1e-9:
        for x in (padding, width - padding):
            ratio = (x - point[0]) / dx
            y = point[1] + ratio * dy
            if padding - 1e-6 <= y <= height - padding + 1e-6:
                crossings.append((ratio, (x, y)))
    if abs(dy) > 1e-9:
        for y in (padding, height - padding):
            ratio = (y - point[1]) / dy
            x = point[0] + ratio * dx
            if padding - 1e-6 <= x <= width - padding + 1e-6:
                crossings.append((ratio, (x, y)))
    if len(crossings) < 2:
        return None
    crossings.sort(key=lambda item: item[0])
    return _path([crossings[0][1], crossings[-1][1]])


def _circumcentre(a: Point, b: Point, c: Point) -> Optional[Point]:
    area = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(area) <= 1e-9:
        return None
    sa = a[0] * a[0] + a[1] * a[1]
    sb = b[0] * b[0] + b[1] * b[1]
    sc = c[0] * c[0] + c[1] * c[1]
    return (
        (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / area,
        (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / area,
    )


def _tangent_contacts(centre: Point, radius: float, external: Point) -> Optional[List[Point]]:
    dx, dy = external[0] - centre[0], external[1] - centre[1]
    distance = math.hypot(dx, dy)
    if distance <= radius + 1e-9:
        return None
    along = radius * radius / distance
    height = radius * math.sqrt(max(0.0, distance * distance - radius * radius)) / distance
    ux, uy = dx / distance, dy / distance
    base = (centre[0] + along * ux, centre[1] + along * uy)
    return [
        (base[0] - height * uy, base[1] + height * ux),
        (base[0] + height * uy, base[1] - height * ux),
    ]


TERMINAL_CONSTRUCTION_TASKS = frozenset(
    {
        "angle_trisection",
        "arc_center",
        "circle_inversion",
        "closest_pair",
        "diameter_opposite",
        "double_reflection",
        "double_rotation",
        "equilateral_triangle_vertex",
        "excircle_tangent_point",
        "external_angle_bisector",
        "external_ratio_point",
        "external_tangent_point",
        "glide_reflection",
        "inverse_rotation",
        "midline_of_triangle",
        "orthogonal_projection",
        "parabola_vertex",
        "parallelogram",
        "perpendicular",
        "perpendicular_bisector",
        "point_rotation",
        "ray_intersection",
        "reflection",
        "rhombus_vertex",
        "right_triangle",
        "spiral_continuation",
        "symmetric_completion",
        "tangent_line_pair_angle",
        "translation_composition",
        "trapezoid_median",
        "vector_translation",
        "vertical_angle_pair",
    }
)

# Tasks whose terminal delivery adds nothing but the marker highlight: the
# candidates already sit on the arc, segment or pairing the rule ranks, so the
# solution has no construction to draw.  Declaring that emptiness is what lets
# the added-member gate treat *any* changed ink outside the candidate discs as
# unexplained, instead of having nothing to compare it against.  Membership is
# measured, not assumed: across three seeds each, every legal terminal image of
# these tasks changes no pixels outside the candidate discs.
NO_CONSTRUCTION_TASKS = frozenset(
    {
        "angle_supplement",
        "arc_bisector",
        "segment_trisection",
        "tangent_circle_touch",
    }
)


def terminal_construction_paths(
    task_name: str,
    record: Mapping[str, Any],
    expected: str,
    *,
    line_width: float,
) -> Optional[List[np.ndarray]]:
    """Construction the terminal image must carry, from the record's own fields."""

    name = str(task_name)
    if name not in TERMINAL_CONSTRUCTION_TASKS:
        return None
    target = _candidate(record, str(expected).strip().upper())

    if name == "double_rotation":
        source = _pt(record.get("source_point"))
        pivot1, pivot2 = _pt(record.get("pivot1")), _pt(record.get("pivot2"))
        first, second = _number(record.get("rotation1_degrees")), _number(record.get("rotation2_degrees"))
        if None in (source, pivot1, pivot2, first, second):
            return []
        intermediate = _rotate(source, pivot1, first)
        # The solution redraws each turn at the radius the point actually
        # travels on, over the small arrowed indicator the input already shows.
        return [_turn(source, pivot1, first), _turn(intermediate, pivot2, second)]

    if name == "inverse_rotation":
        image_point, pivot = _pt(record.get("image_point")), _pt(record.get("pivot"))
        degrees = _number(record.get("rotation_degrees"))
        if None in (image_point, pivot, degrees):
            return []
        # The question is where the point came *from*, so the drawn orbit runs
        # backwards through the declared turn.
        return [_turn(image_point, pivot, -degrees)]

    if name == "external_ratio_point":
        start = _pt(record.get("start_anchor"))
        if start is None or target is None:
            return []
        return [_path([start, target])]

    if name == "double_reflection":
        source = _pt(record.get("source_point"))
        axis1 = _pts(record.get("reflection_axis_1"), 2)
        axis2 = _pts(record.get("reflection_axis_2"), 2)
        if source is None or axis1 is None or axis2 is None:
            return []
        intermediate = _reflect(source, axis1[0], axis1[1])
        if intermediate is None or target is None:
            return []
        return [_path([source, intermediate]), _path([intermediate, target])]

    if name == "ray_intersection":
        rays = record.get("rays")
        if not isinstance(rays, (list, tuple)) or not rays or target is None:
            return []
        starts = [_pt(ray.get("start")) if isinstance(ray, Mapping) else None for ray in rays]
        if any(start is None for start in starts):
            return []
        return [_path([start, target]) for start in starts if start is not None]

    if name == "right_triangle":
        if target is None:
            return []
        peers = [
            point
            for point in (
                _pt(item) for item in record.get("candidates") or [] if isinstance(item, Mapping)
            )
            if point is not None and math.hypot(point[0] - target[0], point[1] - target[1]) > 1e-6
        ]
        if len(peers) < 2:
            return []

        def squareness(pair: Tuple[Point, Point]) -> float:
            first = (pair[0][0] - target[0], pair[0][1] - target[1])
            second = (pair[1][0] - target[0], pair[1][1] - target[1])
            scale = math.hypot(*first) * math.hypot(*second)
            if scale <= 1e-9:
                return 1.0
            return abs((first[0] * second[0] + first[1] * second[1]) / scale)

        pairs = [
            (peers[i], peers[j]) for i in range(len(peers)) for j in range(i + 1, len(peers))
        ]
        arm_one, arm_two = min(pairs, key=squareness)
        # The marker on the square corner runs one leg of a small square along
        # each arm, at the fraction of the canvas the renderer uses.
        size = 0.045 * _canvas_short_side(record)
        first = math.atan2(arm_one[1] - target[1], arm_one[0] - target[0])
        second = math.atan2(arm_two[1] - target[1], arm_two[0] - target[0])
        leg_one = (target[0] + math.cos(first) * size, target[1] + math.sin(first) * size)
        leg_two = (target[0] + math.cos(second) * size, target[1] + math.sin(second) * size)
        corner = (
            target[0] + (math.cos(first) + math.cos(second)) * size,
            target[1] + (math.sin(first) + math.sin(second)) * size,
        )
        return [
            _path([arm_one, target]),
            _path([target, arm_two]),
            _path([arm_two, arm_one]),
            _path([leg_one, corner, leg_two]),
        ]

    if name == "arc_center":
        samples = _pts(record.get("source_arc_points"), 3)
        if samples is None:
            return []
        centre = _circumcentre(*samples)
        if centre is None:
            return []
        radius = math.hypot(samples[0][0] - centre[0], samples[0][1] - centre[1])
        # The record states the three samples in the generator's mathematical
        # frame while the arc is rasterised in screen coordinates, so the two
        # differ by the y flip about the centre.  The circle -- and therefore
        # the replayed answer -- is the same either way; which half of it the
        # complement covers is not, so the samples are read back into the frame
        # the arc was drawn in before the sweep is taken.
        samples = [(point[0], 2.0 * centre[1] - point[1]) for point in samples]
        bearings = [math.atan2(point[1] - centre[1], point[0] - centre[0]) for point in samples]
        # The three samples run along the given arc, so the complement is the
        # sweep from the last back to the first that does not pass the middle.
        sweep = (bearings[0] - bearings[2]) % (2.0 * math.pi)
        if ((bearings[1] - bearings[2]) % (2.0 * math.pi)) < sweep:
            sweep -= 2.0 * math.pi
        return [_screen_arc(centre, radius, bearings[2], sweep)]

    if name == "tangent_line_pair_angle":
        centre = _pt(record.get("circle_center"))
        external = _pt(record.get("external_point"))
        radius = _number(record.get("radius"))
        if centre is None or external is None or radius is None:
            return []
        contacts = _tangent_contacts(centre, radius, external)
        if contacts is None:
            return []
        # The opening the two tangents make at the anchor is marked with an
        # arc, so the declared set stays complete: nothing the solution draws
        # is left for the added-member gate to read as unexplained ink.
        first = math.atan2(contacts[0][1] - external[1], contacts[0][0] - external[0])
        second = math.atan2(contacts[1][1] - external[1], contacts[1][0] - external[0])
        opening = (second - first + math.pi) % (2.0 * math.pi) - math.pi
        mark_radius = max(
            18.0,
            min(
                0.45 * math.hypot(contacts[0][0] - external[0], contacts[0][1] - external[1]),
                0.14 * _canvas_short_side(record),
            ),
        )
        return [_path([external, contact]) for contact in contacts] + [
            _screen_arc(external, mark_radius, first, opening)
        ]

    if name == "excircle_tangent_point":
        vertices = _pts(record.get("triangle_vertices"), 3)
        marked = _number(record.get("marked_vertex_index"))
        if vertices is None or marked is None:
            return []
        index = int(marked) % 3
        apex = vertices[index]
        left, right = vertices[(index + 1) % 3], vertices[(index + 2) % 3]
        side = math.hypot(right[0] - left[0], right[1] - left[1])
        opposite = math.hypot(apex[0] - right[0], apex[1] - right[1])
        adjacent = math.hypot(apex[0] - left[0], apex[1] - left[1])
        semi = (side + opposite + adjacent) / 2.0
        denominator = -side + opposite + adjacent
        if side <= 1e-9 or abs(denominator) <= 1e-9 or (semi - side) <= 1e-9:
            return []
        centre = (
            (-side * apex[0] + opposite * left[0] + adjacent * right[0]) / denominator,
            (-side * apex[1] + opposite * left[1] + adjacent * right[1]) / denominator,
        )
        area = math.sqrt(max(0.0, semi * (semi - side) * (semi - opposite) * (semi - adjacent)))
        radius = area / (semi - side)
        if radius <= 1e-9:
            return []
        unit = ((right[0] - left[0]) / side, (right[1] - left[1]) / side)
        normal = (-unit[1], unit[0])
        if (apex[0] - left[0]) * normal[0] + (apex[1] - left[1]) * normal[1] > 0.0:
            normal = (-normal[0], -normal[1])
        contact = (left[0] + (semi - adjacent) * unit[0], left[1] + (semi - adjacent) * unit[1])
        # The drawn arc is the part of the excircle that fits between the band
        # the layout reserves beyond the target side; both limits come from the
        # side's own length, exactly as the renderer computes them.
        along = (contact[0] - left[0]) * unit[0] + (contact[1] - left[1]) * unit[1]
        extension = 0.08 * side
        clearance = 0.20 * side
        pad = float(line_width) + 4.0
        half_chord = max(0.0, min(along + extension, side - along + extension) - pad)
        rise = max(0.0, clearance - pad)
        half_sweep = min(
            math.asin(max(0.0, min(1.0, half_chord / radius))),
            math.acos(max(-1.0, min(1.0, 1.0 - rise / radius))),
        )
        if half_sweep <= 1e-6:
            return []
        bearing = math.atan2(-normal[1], -normal[0])
        return [_screen_arc(centre, radius, bearing - half_sweep, 2.0 * half_sweep)]

    if name == "closest_pair":
        pairs = [
            point
            for point in (
                _pt(item) for item in record.get("candidates") or [] if isinstance(item, Mapping)
            )
            if point is not None
        ]
        if len(pairs) < 2:
            return []
        best = min(
            (
                (pairs[i], pairs[j])
                for i in range(len(pairs))
                for j in range(i + 1, len(pairs))
            ),
            key=lambda pair: math.hypot(pair[0][0] - pair[1][0], pair[0][1] - pair[1][1]),
        )
        return [_path(best)]

    if name == "vector_translation":
        source = _pt(record.get("source_point"))
        if source is None or target is None:
            return []
        return [_path([source, target])]

    if name in {"orthogonal_projection", "reflection", "diameter_opposite"}:
        source = _pt(record.get("source_point"))
        if source is None or target is None:
            return []
        return [_path([source, target])]

    if name == "perpendicular":
        through = _pt(record.get("through_point"))
        if through is None or target is None:
            return []
        return [_path([through, target])]

    if name == "midline_of_triangle":
        marked = _pt(record.get("marked_midpoint"))
        if marked is None or target is None:
            return []
        return [_path([marked, target])]

    if name == "angle_trisection":
        vertex = _pt(record.get("vertex"))
        if vertex is None or target is None:
            return []
        return [_path([vertex, target])]

    if name == "trapezoid_median":
        midpoints = _pts(record.get("leg_midpoints"), 2)
        if midpoints is None:
            return []
        return [_path(midpoints)]

    if name == "spiral_continuation":
        points = record.get("spiral_points")
        tip = _pt(points[-1]) if isinstance(points, (list, tuple)) and points else None
        if tip is None or target is None:
            return []
        return [_path([tip, target])]

    if name == "perpendicular_bisector":
        endpoints = _pts(record.get("segment_endpoints"), 2)
        if endpoints is None:
            return []
        # This task sets its working margin to 72 before sampling, rather
        # than using the family's proportional default. Canonical semantic
        # input omits renderer metadata, so replay that task-owned default.
        margin = _margin(record, default=72.0)
        middle = ((endpoints[0][0] + endpoints[1][0]) / 2.0, (endpoints[0][1] + endpoints[1][1]) / 2.0)
        direction = (-(endpoints[1][1] - endpoints[0][1]), endpoints[1][0] - endpoints[0][0])
        # The renderer clips the bisector to its working bounds -- the canvas
        # inset by the family margin -- and then by the stroke width.
        line = _clipped_full_line(record, middle, direction, padding=margin + float(line_width))
        return [] if line is None else [line]

    if name == "equilateral_triangle_vertex":
        base = _pts(record.get("base"), 2)
        if base is None or target is None:
            return []
        return [_path([base[0], target]), _path([target, base[1]])]

    if name == "glide_reflection":
        source = _pt(record.get("source_point"))
        axis = _pts(record.get("axis"), 2)
        if source is None or axis is None or target is None:
            return []
        mirrored = _reflect(source, axis[0], axis[1])
        if mirrored is None:
            return []
        return [_path([source, mirrored]), _path([mirrored, target])]

    if name == "translation_composition":
        source = _pt(record.get("source_point"))
        first = _pts(record.get("vector1"), 2)
        second = _pts(record.get("vector2"), 2)
        if source is None or first is None or second is None:
            return []
        step = (first[1][0] - first[0][0], first[1][1] - first[0][1])
        intermediate = (source[0] + step[0], source[1] + step[1])
        if target is None:
            return []
        return [_path([source, intermediate]), _path([intermediate, target])]

    if name == "circle_inversion":
        source = _pt(record.get("source_point"))
        centre = _pt(record.get("inversion_center"))
        if target is None or source is None or centre is None:
            return []
        far = max((source, target), key=lambda point: math.dist(centre, point))
        return [_path([centre, far])]

    if name in {"rhombus_vertex", "parallelogram"}:
        shown = _pts(record.get("shown_vertices" if name == "rhombus_vertex" else "known_vertices"), 3)
        if shown is None or target is None:
            return []
        # The missing vertex closes the parallelogram against one apex; the
        # apex is whichever shown vertex the other two predict the answer
        # from, and the two closing sides run from those others to the answer.
        def closure_error(apex_index: int) -> float:
            others = [shown[i] for i in range(3) if i != apex_index]
            return math.hypot(
                others[0][0] + others[1][0] - shown[apex_index][0] - target[0],
                others[0][1] + others[1][1] - shown[apex_index][1] - target[1],
            )

        apex = min(range(3), key=closure_error)
        first, second = [shown[i] for i in range(3) if i != apex]
        return [_path([first, target]), _path([target, second])]

    if name == "symmetric_completion":
        edges = record.get("visible_edges")
        if not isinstance(edges, (list, tuple)) or target is None:
            return []
        counts: dict = {}
        for edge in edges:
            pair = _pts(edge, 2)
            if pair is None:
                return []
            for point in pair:
                key = (round(point[0], 3), round(point[1], 3))
                counts[key] = counts.get(key, 0) + 1
        loose = [key for key, count in counts.items() if count == 1]
        if len(loose) != 2:
            return []
        return [
            _path([(loose[0][0], loose[0][1]), target]),
            _path([target, (loose[1][0], loose[1][1])]),
        ]

    if name == "external_tangent_point":
        first_raw, second_raw = record.get("circle_1"), record.get("circle_2")
        if not isinstance(first_raw, Mapping) or not isinstance(second_raw, Mapping):
            return []
        centre1, centre2 = _pt(first_raw.get("center")), _pt(second_raw.get("center"))
        radius1, radius2 = _number(first_raw.get("radius")), _number(second_raw.get("radius"))
        if None in (centre1, centre2, radius1, radius2) or target is None:
            return []
        distance = math.hypot(centre2[0] - centre1[0], centre2[1] - centre1[1])
        if distance <= abs(radius1 - radius2) + 1e-9:
            return []
        heading = math.atan2(centre2[1] - centre1[1], centre2[0] - centre1[0])
        offset = math.acos(max(-1.0, min(1.0, (radius1 - radius2) / distance)))
        branches = []
        for sign in (1.0, -1.0):
            angle = heading + sign * offset
            contact1 = (centre1[0] + radius1 * math.cos(angle), centre1[1] + radius1 * math.sin(angle))
            contact2 = (centre2[0] + radius2 * math.cos(angle), centre2[1] + radius2 * math.sin(angle))
            branches.append((contact1, contact2))
        contact1, contact2 = min(
            branches,
            key=lambda pair: math.hypot(pair[1][0] - target[0], pair[1][1] - target[1]),
        )
        return [_path([contact1, contact2])]

    if name == "parabola_vertex":
        axis_degrees = _number(record.get("axis_angle_deg"))
        if axis_degrees is None or target is None:
            return []
        # The renderer marks the symmetry axis with one short segment through
        # the vertex: a fixed fraction back against the opening and a longer
        # one into it, both read off the canvas' short side.
        angle = math.radians(axis_degrees)
        open_x, open_y = -math.sin(angle), math.cos(angle)
        short_side = _canvas_short_side(record)
        back, forward = 0.05 * short_side, 0.16 * short_side
        return [
            _path(
                [
                    (target[0] - open_x * back, target[1] - open_y * back),
                    (target[0] + open_x * forward, target[1] + open_y * forward),
                ]
            )
        ]

    if name == "point_rotation":
        source = _pt(record.get("source_point"))
        pivot = _pt(record.get("pivot"))
        degrees = _number(record.get("rotation_degrees"))
        if None in (source, pivot, degrees):
            return []
        return [_turn(source, pivot, degrees)]

    if name == "external_angle_bisector":
        vertex = _pt(record.get("vertex"))
        if vertex is None or target is None:
            return []
        # The obligation is the drawn external bisector out to the answer; the
        # remainder of that line and the internal bisector it is built from
        # are attribution corridors, since their drawn lengths belong to the
        # renderer rather than the record.
        return [_path([vertex, target])]

    if name == "vertical_angle_pair":
        cross = _pt(record.get("intersection"))
        line1 = _pts(record.get("line1_endpoints"), 2)
        line2 = _pts(record.get("line2_endpoints"), 2)
        if cross is None or line1 is None or line2 is None or target is None:
            return []
        # The solution mirrors the marked-angle arc into the region that
        # holds the answer.  The region's boundary directions are the two
        # half-line bearings bracketing the answer's own bearing.
        bearings = sorted(
            math.atan2(point[1] - cross[1], point[0] - cross[0]) % (2.0 * math.pi)
            for line in (line1, line2)
            for point in line
        )
        aim = math.atan2(target[1] - cross[1], target[0] - cross[0]) % (2.0 * math.pi)
        radius = 0.07 * _canvas_short_side(record)
        for index, low in enumerate(bearings):
            high = bearings[(index + 1) % 4]
            sweep = (high - low) % (2.0 * math.pi)
            if (aim - low) % (2.0 * math.pi) < sweep:
                return [_screen_arc(cross, radius, low, sweep)]
        return []

    return []


def serial_construction_path_variants(
    task_name: str,
    record: Mapping[str, Any],
    expected: str,
    *,
    line_width: float,
) -> Optional[List[List[np.ndarray]]]:
    """Return every legal complete serial centreline programme for a task.

    The outer list enumerates legal variants.  Each variant is an ordered list
    of pen-command centrelines, and each centreline is an ``N x 2`` float
    array in canvas coordinates.  Composite polylines remain one command;
    partial arcs and smooth turns remain ordered sampled centrelines.  A
    marker-only commit/hold is intentionally absent because it contributes no
    construction path.

    ``None`` means the task is outside :data:`TERMINAL_CONSTRUCTION_TASKS`.
    An empty list means the task is owned here but the supplied raw semantic
    input or independently replayed answer is incomplete.
    """

    name = str(task_name)
    if name not in TERMINAL_CONSTRUCTION_TASKS:
        return None

    members = terminal_construction_paths(
        name,
        record,
        expected,
        line_width=line_width,
    )
    if not members:
        return []
    target = _candidate(record, str(expected).strip().upper())

    # These solution strokes stop at the candidate-disc rim.  The terminal
    # member contract extends beneath the later marker for mask robustness;
    # the serial contract must instead reproduce the recorder's true endpoint.
    end_trim_tasks = {
        "angle_trisection",
        "diameter_opposite",
        "orthogonal_projection",
        "perpendicular",
        "reflection",
        "spiral_continuation",
        "vector_translation",
    }
    if name in end_trim_tasks:
        radius = _point_radius(record)
        if radius is None or target is None:
            return []
        trimmed = _trimmed_segment(
            (float(members[0][0, 0]), float(members[0][0, 1])),
            target,
            end_offset=radius,
        )
        return [] if trimmed is None else [[trimmed]]

    if name == "circle_inversion":
        if record.get("_circle_source_started"):
            source = _pt(record.get("source_point"))
            if source is None or target is None:
                return []
            trimmed = _trimmed_segment(source, target, end_offset=_point_radius(record))
            variants = [] if trimmed is None else [[trimmed]]
            center = _pt(record.get("inversion_center"))
            radius = _number(record.get("inversion_radius"))
            if center is not None and radius is not None and 0 < radius < math.dist(source, center):
                # Frozen source-started wording also asks to pass through O,
                # although the inverse is between O and the outside source.
                # Retain the direct connector and its complete radial reading.
                variants.append([_path([source, center])])
            return variants
        return [[members[0].copy()]]

    if name == "closest_pair":
        radius = _point_radius(record)
        if radius is None or members[0].shape[0] < 2:
            return []
        start = (float(members[0][0, 0]), float(members[0][0, 1]))
        end = (float(members[0][-1, 0]), float(members[0][-1, 1]))
        trimmed = _trimmed_segment(start, end, start_offset=radius, end_offset=radius)
        return [] if trimmed is None else [[trimmed], [_reverse_path(trimmed)]]

    if name in {"double_reflection", "glide_reflection"}:
        radius = _point_radius(record)
        if radius is None or target is None or len(members) != 2:
            return []
        second_start = (float(members[1][0, 0]), float(members[1][0, 1]))
        second = _trimmed_segment(second_start, target, end_offset=radius)
        return [] if second is None else [[members[0].copy(), second]]

    if name == "translation_composition":
        source = _pt(record.get("source_point"))
        first = _pts(record.get("vector1"), 2)
        second = _pts(record.get("vector2"), 2)
        radius = _point_radius(record)
        if source is None or first is None or second is None or target is None or radius is None:
            return []
        vectors = (
            (first[1][0] - first[0][0], first[1][1] - first[0][1]),
            (second[1][0] - second[0][0], second[1][1] - second[0][1]),
        )
        variants: List[List[np.ndarray]] = []
        for first_index in (0, 1):
            dx, dy = vectors[first_index]
            intermediate = (source[0] + dx, source[1] + dy)
            final_leg = _trimmed_segment(intermediate, target, end_offset=radius)
            if final_leg is None:
                return []
            variants.append([_path([source, intermediate]), final_leg])
        return variants

    if name == "external_angle_bisector":
        vertex = _pt(record.get("vertex"))
        first = _pt(record.get("ray1_end"))
        second = _pt(record.get("ray2_end"))
        margin = _margin(record)
        width = _number(line_width)
        if (
            vertex is None
            or first is None
            or second is None
            or target is None
            or width is None
        ):
            return []
        first_length = math.hypot(first[0] - vertex[0], first[1] - vertex[1])
        second_length = math.hypot(second[0] - vertex[0], second[1] - vertex[1])
        if first_length <= 1e-9 or second_length <= 1e-9:
            return []
        internal = (
            (first[0] - vertex[0]) / first_length + (second[0] - vertex[0]) / second_length,
            (first[1] - vertex[1]) / first_length + (second[1] - vertex[1]) / second_length,
        )
        internal_norm = math.hypot(*internal)
        if internal_norm <= 1e-9:
            return []
        internal_unit = (internal[0] / internal_norm, internal[1] / internal_norm)
        ray_length = 0.5 * (first_length + second_length)
        internal_end = (
            vertex[0] + internal_unit[0] * ray_length * 0.72,
            vertex[1] + internal_unit[1] * ray_length * 0.72,
        )
        external_unit = (-internal_unit[1], internal_unit[0])
        clipped = _clipped_full_line(
            record,
            vertex,
            external_unit,
            padding=margin + width + 8.0,
        )
        if clipped is None:
            return []
        forward_distance = max(
            float(
                (point[0] - vertex[0]) * external_unit[0]
                + (point[1] - vertex[1]) * external_unit[1]
            )
            for point in clipped
        )
        target_distance = math.hypot(target[0] - vertex[0], target[1] - vertex[1])
        travel = max(target_distance * 1.2, forward_distance * 0.98)
        external_end = (
            vertex[0] + external_unit[0] * travel,
            vertex[1] + external_unit[1] * travel,
        )
        return [[_path([vertex, internal_end]), _path([vertex, external_end])]]

    if name == "ray_intersection":
        if len(members) != 3:
            return []
        return [
            [members[index].copy() for index in order]
            for order in permutations(range(3))
        ]

    if name == "tangent_line_pair_angle":
        if len(members) != 3:
            return []
        variants = []
        for order in ((0, 1), (1, 0)):
            for reverse_arc in (False, True):
                arc = _reverse_path(members[2]) if reverse_arc else members[2].copy()
                variants.append([members[order[0]].copy(), members[order[1]].copy(), arc])
        return variants

    if name == "right_triangle":
        if len(members) != 4:
            return []
        arm_one = (float(members[0][0, 0]), float(members[0][0, 1]))
        right = (float(members[0][-1, 0]), float(members[0][-1, 1]))
        arm_two = (float(members[1][-1, 0]), float(members[1][-1, 1]))
        vertices = (arm_one, right, arm_two)
        marker = members[3]
        variants = []
        for direction in (1, -1):
            for start_index in range(3):
                order = [
                    vertices[(start_index + direction * offset) % 3]
                    for offset in range(3)
                ]
                outline = _path(order + [order[0]])
                variants.append([outline, marker.copy()])
                variants.append([outline.copy(), _reverse_path(marker)])
        return variants

    if name in {
        "equilateral_triangle_vertex",
        "parallelogram",
        "rhombus_vertex",
        "symmetric_completion",
    }:
        merged = _merge_paths(members)
        if merged is None:
            return []
        if name == "parallelogram":
            # The renderer starts at the second open endpoint; both directions
            # are nevertheless equivalent under the prompt's "one ... other".
            reversed_merged = _reverse_path(merged)
            return [[reversed_merged], [merged]]
        return [[merged], [_reverse_path(merged)]]

    if name == "arc_center":
        # terminal_construction_paths follows the complementary locus from the
        # opposite endpoint; the recorder starts at the given arc's tail.
        return [[_reverse_path(members[0])], [members[0].copy()]]

    if name in {
        "excircle_tangent_point",
        "vertical_angle_pair",
        "parabola_vertex",
        "perpendicular_bisector",
        "trapezoid_median",
    }:
        return [[members[0].copy()], [_reverse_path(members[0])]]

    if name == "external_tangent_point":
        # The solution's hollow contact marker is a later zero-progress hold,
        # not a construction path.
        return [[members[0].copy()], [_reverse_path(members[0])]]

    # Remaining tasks have one causally directed command, or a fixed sequence
    # of directed smooth turns.  Copy the arrays so callers cannot mutate the
    # terminal member view through shared storage.
    return [[member.copy() for member in members]]


def terminal_attribution_paths(
    task_name: str,
    record: Mapping[str, Any],
    expected: str,
    *,
    line_width: float,
) -> List[np.ndarray]:
    """Extra source-derived corridors a legal delivery may put ink in.

    Separate from :func:`terminal_construction_paths` because these primitives
    are *not* obligations, only ink the delivery may legally carry: renderer
    restatements whose drawn extent belongs to the renderer rather than the
    record. Overshooting an attribution corridor never rejects a legal
    delivery; requiring one from the delivery could.
    """

    name = str(task_name)

    if name == "circumcenter":
        vertices = _pts(record.get("triangle_vertices"), 3)
        if vertices is None:
            return []
        result = []
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % 3]
            middle = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
            direction = (first[1] - second[1], second[0] - first[0])
            line = _clipped_full_line(record, middle, direction, padding=0.0)
            if line is not None:
                result.append(line)
        return result

    if name == "excircle_center":
        vertices = _pts(record.get("triangle_vertices"), 3)
        marked = _number(record.get("marked_vertex_index"))
        if vertices is None or marked is None:
            return []
        apex = vertices[int(marked) % 3]
        result = []
        for endpoint in (vertices[(int(marked) + 1) % 3], vertices[(int(marked) + 2) % 3]):
            direction = (endpoint[0] - apex[0], endpoint[1] - apex[1])
            line = _clipped_full_line(record, endpoint, direction, padding=0.0)
            if line is not None:
                far = max(line, key=lambda point: (point[0] - endpoint[0]) * direction[0]
                          + (point[1] - endpoint[1]) * direction[1])
                result.append(_path([endpoint, far]))
        return result

    if name == "excircle_tangent_point":
        from data.geometry.core.coordinate_replay import _triangle_excenter

        vertices = _pts(record.get("triangle_vertices"), 3)
        marked = _number(record.get("marked_vertex_index"))
        if vertices is None or marked is None:
            return []
        index = int(marked) % 3
        centre = _triangle_excenter(vertices, index)
        first, second = vertices[(index + 1) % 3], vertices[(index + 2) % 3]
        dx, dy = second[0] - first[0], second[1] - first[1]
        radius = abs(dx * (centre[1] - first[1]) - dy * (centre[0] - first[0])) / math.hypot(dx, dy)
        return [_screen_arc(centre, radius, 0.0, 2.0 * math.pi)]

    if name == "double_rotation":
        source = _pt(record.get("source_point"))
        pivot1, pivot2 = _pt(record.get("pivot1")), _pt(record.get("pivot2"))
        first = _number(record.get("rotation1_degrees"))
        second = _number(record.get("rotation2_degrees"))
        if None in (source, pivot1, pivot2, first, second):
            return []
        # The solution restates each turn indicator at the indicator radius
        # while it draws the full orbit, which leaves changed ink around the
        # pivots that the orbit corridors do not cover; the indicators exist
        # in the input already, so they cannot be obligations.
        indicator = 0.05 * _canvas_short_side(record)
        intermediate = _rotate(source, pivot1, first)

        def seat(point: Point, pivot: Point) -> Optional[Point]:
            length = math.hypot(point[0] - pivot[0], point[1] - pivot[1])
            if length <= 1e-9:
                return None
            scale = indicator / length
            return (
                pivot[0] + (point[0] - pivot[0]) * scale,
                pivot[1] + (point[1] - pivot[1]) * scale,
            )

        seats = [(seat(source, pivot1), pivot1, first), (seat(intermediate, pivot2), pivot2, second)]
        return [_turn(start, pivot, degrees) for start, pivot, degrees in seats if start is not None]

    if name == "point_rotation":
        source = _pt(record.get("source_point"))
        pivot = _pt(record.get("pivot"))
        degrees = _number(record.get("rotation_degrees"))
        if None in (source, pivot, degrees):
            return []
        # The angle indicator at 0.45 of the orbit radius is input ink the
        # solution restates; its re-rasterised fringe can cross the change
        # threshold without carrying any state.
        length = math.hypot(source[0] - pivot[0], source[1] - pivot[1])
        if length <= 1e-9:
            return []
        scale = 0.45
        seat = (
            pivot[0] + (source[0] - pivot[0]) * scale,
            pivot[1] + (source[1] - pivot[1]) * scale,
        )
        return [_turn(seat, pivot, degrees)]

    if name == "external_angle_bisector":
        vertex = _pt(record.get("vertex"))
        first = _pt(record.get("ray1_end"))
        second = _pt(record.get("ray2_end"))
        if vertex is None or first is None or second is None:
            return []
        # The drawn internal bisector's length and the external line's
        # clipped extent are the renderer's own; the whole of both lines is
        # excused rather than measured.
        length1 = math.hypot(first[0] - vertex[0], first[1] - vertex[1])
        length2 = math.hypot(second[0] - vertex[0], second[1] - vertex[1])
        if length1 <= 1e-9 or length2 <= 1e-9:
            return []
        internal = (
            (first[0] - vertex[0]) / length1 + (second[0] - vertex[0]) / length2,
            (first[1] - vertex[1]) / length1 + (second[1] - vertex[1]) / length2,
        )
        if math.hypot(*internal) <= 1e-9:
            return []
        external = (-internal[1], internal[0])
        corridors = []
        for direction in (internal, external):
            line = _clipped_full_line(record, vertex, direction, padding=float(line_width) + 8.0)
            if line is not None:
                corridors.append(line)
        return corridors

    if name == "spiral_continuation":
        points = record.get("spiral_points")
        if isinstance(points, (list, tuple)) and len(points) >= 2:
            parsed = _pts(points, len(points))
            if parsed is not None:
                return [_path(parsed)]
        return []

    return []


__all__ = [
    "TERMINAL_CONSTRUCTION_TASKS",
    "serial_construction_path_variants",
    "terminal_attribution_paths",
    "terminal_construction_paths",
]

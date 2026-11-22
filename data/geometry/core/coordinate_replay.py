"""Independent semantic replay for geometry point/option tasks.

The point-target generators historically persisted ``correct_option`` next to
their candidates.  That field is useful while authoring references, but it is
not a legitimate verifier oracle.  This module derives the accepted label(s)
from the canonical source geometry and then matches the derived construction
against candidate centers.  It deliberately never reads ``correct_option``,
``ground_truth``, ``structured_answer`` or authored solution traces.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


Point = tuple[float, float]


class CoordinateReplayError(ValueError):
    """Raised when canonical source geometry is insufficient or ambiguous."""


def _point(value: object) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CoordinateReplayError(f"expected finite point, got {value!r}")
    try:
        point = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise CoordinateReplayError(f"expected numeric point, got {value!r}") from exc
    if not all(math.isfinite(component) for component in point):
        raise CoordinateReplayError(f"expected finite point, got {value!r}")
    return point


def _points(value: object, *, count: int | None = None) -> list[Point]:
    if not isinstance(value, (list, tuple)):
        raise CoordinateReplayError("expected point sequence")
    result = [_point(item) for item in value]
    if count is not None and len(result) != count:
        raise CoordinateReplayError(f"expected {count} points, got {len(result)}")
    return result


def _candidate_points(record: Mapping[str, Any]) -> list[tuple[str, Point]]:
    raw = record.get("candidates")
    if not isinstance(raw, list) or not raw:
        raise CoordinateReplayError("missing candidate geometry")
    result: list[tuple[str, Point]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise CoordinateReplayError("invalid candidate entry")
        label = str(item.get("label") or "").strip().upper()
        if len(label) != 1 or not label.isalpha():
            raise CoordinateReplayError("invalid candidate label")
        result.append((label, _point((item.get("x"), item.get("y")))))
    if len({label for label, _ in result}) != len(result):
        raise CoordinateReplayError("duplicate candidate labels")
    return result


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _add(first: Point, second: Point) -> Point:
    return first[0] + second[0], first[1] + second[1]


def _sub(first: Point, second: Point) -> Point:
    return first[0] - second[0], first[1] - second[1]


def _scale(point: Point, factor: float) -> Point:
    return point[0] * factor, point[1] * factor


def _midpoint(first: Point, second: Point) -> Point:
    return (first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0


def _unit(vector: Point) -> Point:
    length = math.hypot(*vector)
    if length <= 1e-9:
        raise CoordinateReplayError("degenerate vector")
    return vector[0] / length, vector[1] / length


def _rotate(point: Point, pivot: Point, degrees: float) -> Point:
    """Rotate in the generators' mathematical convention (positive is CCW).

    Canvas y grows downward, so the screen-coordinate matrix differs from the
    usual Cartesian matrix in both sine terms.
    """
    angle = math.radians(degrees)
    dx, dy = point[0] - pivot[0], point[1] - pivot[1]
    return (
        pivot[0] + dx * math.cos(angle) + dy * math.sin(angle),
        pivot[1] - dx * math.sin(angle) + dy * math.cos(angle),
    )


def _project(point: Point, first: Point, second: Point) -> Point:
    vector = _sub(second, first)
    denom = vector[0] ** 2 + vector[1] ** 2
    if denom <= 1e-9:
        raise CoordinateReplayError("degenerate projection line")
    t = ((point[0] - first[0]) * vector[0] + (point[1] - first[1]) * vector[1]) / denom
    return first[0] + t * vector[0], first[1] + t * vector[1]


def _reflect(point: Point, first: Point, second: Point) -> Point:
    projection = _project(point, first, second)
    return 2.0 * projection[0] - point[0], 2.0 * projection[1] - point[1]


def _line_intersection(a: Point, b: Point, c: Point, d: Point) -> Point:
    ab = _sub(b, a)
    cd = _sub(d, c)
    denominator = ab[0] * cd[1] - ab[1] * cd[0]
    if abs(denominator) <= 1e-9:
        raise CoordinateReplayError("parallel construction lines")
    ac = _sub(c, a)
    t = (ac[0] * cd[1] - ac[1] * cd[0]) / denominator
    return a[0] + t * ab[0], a[1] + t * ab[1]


def _spiral_similarity_center(source: Sequence[Point], image: Sequence[Point]) -> Point:
    """Fixed point of the one turn-and-scale map carrying ``source`` to ``image``.

    Two corresponding pairs already determine the map, so the third pair is a
    genuine consistency check: if the two visible triangles are not related by
    a single spiral similarity in the stored vertex order, the recomputation
    fails closed instead of returning a plausible-looking point.
    """

    if len(source) != 3 or len(image) != 3:
        raise CoordinateReplayError("spiral similarity needs three corresponding vertex pairs")
    black = [complex(point[0], point[1]) for point in source]
    gray = [complex(point[0], point[1]) for point in image]
    span = black[1] - black[0]
    if abs(span) <= 1e-9:
        raise CoordinateReplayError("degenerate source triangle")
    factor = (gray[1] - gray[0]) / span
    if abs(factor - 1.0) <= 1e-9:
        raise CoordinateReplayError("the two triangles differ by a translation, not a spiral similarity")
    center = (gray[0] - factor * black[0]) / (1.0 - factor)
    scale = max(abs(vertex - center) for vertex in black)
    for vertex, partner in zip(black, gray):
        residual = abs(center + factor * (vertex - center) - partner)
        if residual > max(1e-6, scale * 1e-6):
            raise CoordinateReplayError("visible vertex pairs do not share one spiral similarity")
    return center.real, center.imag


def _circumcenter(a: Point, b: Point, c: Point) -> Point:
    denominator = 2.0 * (
        a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])
    )
    if abs(denominator) <= 1e-9:
        raise CoordinateReplayError("collinear circle samples")
    a2 = a[0] ** 2 + a[1] ** 2
    b2 = b[0] ** 2 + b[1] ** 2
    c2 = c[0] ** 2 + c[1] ** 2
    return (
        (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / denominator,
        (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / denominator,
    )


def _closest_candidate_to_targets(
    record: Mapping[str, Any],
    targets: Iterable[Point],
    *,
    tolerance: float = 1e-3,
) -> tuple[str, dict[str, Any]]:
    candidates = _candidate_points(record)
    target_list = list(targets)
    if not target_list:
        raise CoordinateReplayError("no independently derived target")
    ranked = sorted(
        (min(_distance(center, target) for target in target_list), label)
        for label, center in candidates
    )
    if ranked[0][0] > tolerance:
        raise CoordinateReplayError(
            f"no candidate matches derived target (best residual {ranked[0][0]:.6f})"
        )
    if len(ranked) > 1 and abs(ranked[1][0] - ranked[0][0]) <= 1e-7:
        raise CoordinateReplayError("multiple candidates match derived target")
    return ranked[0][1], {
        "derived_targets": [[float(x), float(y)] for x, y in target_list],
        "candidate_residuals": {label: float(residual) for residual, label in ranked},
        "unique_solution": True,
    }


def _choose_by_scores(
    record: Mapping[str, Any],
    scores: Mapping[str, float],
    *,
    maximum: float,
    minimum_gap: float = 1e-6,
) -> tuple[str, dict[str, Any]]:
    labels = {label for label, _ in _candidate_points(record)}
    if set(scores) != labels:
        raise CoordinateReplayError("score labels do not match candidate labels")
    ranked = sorted((float(score), label) for label, score in scores.items())
    if not ranked or ranked[0][0] > maximum:
        raise CoordinateReplayError(f"best geometric residual {ranked[0][0] if ranked else math.inf:.6f} exceeds tolerance")
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < minimum_gap:
        raise CoordinateReplayError("geometric replay is not unique")
    return ranked[0][1], {
        "candidate_residuals": {label: score for score, label in ranked},
        "unique_solution": True,
    }


def _angle_delta(first: float, second: float) -> float:
    return abs((second - first + math.pi) % (2.0 * math.pi) - math.pi)


def _triangle_excenter(vertices: Sequence[Point], marked: int) -> Point:
    a, b, c = vertices
    side_weights = [_distance(b, c), _distance(a, c), _distance(a, b)]
    weights = list(side_weights)
    weights[marked] *= -1.0
    denominator = sum(weights)
    if abs(denominator) <= 1e-9:
        raise CoordinateReplayError("degenerate excenter")
    return (
        sum(weight * point[0] for weight, point in zip(weights, vertices)) / denominator,
        sum(weight * point[1] for weight, point in zip(weights, vertices)) / denominator,
    )


def _fermat_point(vertices: Sequence[Point]) -> Point:
    # The generator rejects triangles with a >=120-degree angle.  Weiszfeld's
    # iteration therefore converges to the interior Fermat/geometric-median
    # point without a vertex special case.
    current = (
        sum(point[0] for point in vertices) / 3.0,
        sum(point[1] for point in vertices) / 3.0,
    )
    for _ in range(500):
        distances = [max(_distance(current, point), 1e-12) for point in vertices]
        denominator = sum(1.0 / distance for distance in distances)
        updated = (
            sum(point[0] / distance for point, distance in zip(vertices, distances)) / denominator,
            sum(point[1] / distance for point, distance in zip(vertices, distances)) / denominator,
        )
        if _distance(current, updated) <= 1e-10:
            return updated
        current = updated
    return current


def _inscribed_square(vertices: Sequence[Point]) -> Point:
    # The square rests on the marked side, which the scene draws in blue and
    # the semantic input lists first; the answer is the raised corner on the
    # side that runs from the first of those two vertices to the apex.
    first, second, opposite = vertices[0], vertices[1], vertices[2]
    base = _sub(second, first)
    length = math.hypot(*base)
    unit = _unit(base)
    normal = (-unit[1], unit[0])
    height = (opposite[0] - first[0]) * normal[0] + (opposite[1] - first[1]) * normal[1]
    if height < 0:
        normal = (-normal[0], -normal[1])
        height = -height
    side = height * length / (height + length)
    opposite_projection = (opposite[0] - first[0]) * unit[0] + (opposite[1] - first[1]) * unit[1]
    left_distance = side * opposite_projection / height
    bottom_left = _add(first, _scale(unit, left_distance))
    return _add(bottom_left, _scale(normal, side))


def _fit_parabola_vertex(record: Mapping[str, Any]) -> Point:
    samples = _points(record.get("source_curve_points"))
    if len(samples) < 5:
        raise CoordinateReplayError("parabola source samples are missing")
    angle = math.radians(float(record["axis_angle_deg"]))
    tangent = (math.cos(angle), math.sin(angle))
    normal = (-tangent[1], tangent[0])
    origin = samples[len(samples) // 2]
    local_x = np.asarray([_sub(point, origin)[0] * tangent[0] + _sub(point, origin)[1] * tangent[1] for point in samples])
    local_y = np.asarray([_sub(point, origin)[0] * normal[0] + _sub(point, origin)[1] * normal[1] for point in samples])
    quadratic, linear, intercept = np.polyfit(local_x, local_y, 2)
    if abs(quadratic) <= 1e-12:
        raise CoordinateReplayError("source curve is not parabolic")
    vx = -linear / (2.0 * quadratic)
    vy = quadratic * vx * vx + linear * vx + intercept
    return _add(origin, _add(_scale(tangent, float(vx)), _scale(normal, float(vy))))


def _right_triangle_answer(record: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates = _candidate_points(record)
    scores = {label: math.inf for label, _ in candidates}
    for (label_a, a), (label_b, b), (label_c, c) in itertools.combinations(candidates, 3):
        for label, vertex, first, second in (
            (label_a, a, b, c), (label_b, b, a, c), (label_c, c, a, b)
        ):
            va, vb = _sub(first, vertex), _sub(second, vertex)
            denom = math.hypot(*va) * math.hypot(*vb)
            if denom > 1e-9:
                scores[label] = min(scores[label], abs((va[0] * vb[0] + va[1] * vb[1]) / denom))
    return _choose_by_scores(record, scores, maximum=1e-6, minimum_gap=0.05)


def _closest_pair_answer(record: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates = _candidate_points(record)
    ranked = sorted(
        (_distance(first[1], second[1]), "".join(sorted((first[0], second[0]))))
        for first, second in itertools.combinations(candidates, 2)
    )
    if len(ranked) < 2 or ranked[0][0] <= 1e-9 or ranked[1][0] / ranked[0][0] < 1.05:
        raise CoordinateReplayError("closest pair is not unique")
    return ranked[0][1], {
        "pair_distances": {label: float(distance) for distance, label in ranked},
        "unique_solution": True,
    }


def derive_coordinate_answer(task_name: str, record: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the independently derived typed payload and replay diagnostics."""

    name = str(task_name)
    targets: list[Point] = []

    if name == "closest_pair":
        answer, metrics = _closest_pair_answer(record)
    elif name == "right_triangle":
        answer, metrics = _right_triangle_answer(record)
    elif name in {"angle_supplement", "inscribed_angle_arc", "tangent_line_pair_angle"}:
        angles = [float(value) for value in record["candidate_angles_deg"]]
        candidates = _candidate_points(record)
        if len(angles) != len(candidates):
            raise CoordinateReplayError("candidate-angle mapping mismatch")
        if name == "angle_supplement":
            target_angle = 180.0 - float(record["source_angle_deg"])
        elif name == "inscribed_angle_arc":
            center = _point(record["circle_center"])
            endpoints = _points(record["arc_endpoints"], count=2)
            bearings = [math.atan2(point[1] - center[1], point[0] - center[0]) for point in endpoints]
            target_angle = math.degrees(_angle_delta(bearings[0], bearings[1]))
        else:
            center = _point(record["circle_center"])
            external = _point(record["external_point"])
            radius = float(record["radius"])
            target_angle = 2.0 * math.degrees(math.asin(radius / _distance(center, external)))
        scores = {label: abs(angle - target_angle) for (label, _), angle in zip(candidates, angles)}
        answer, metrics = _choose_by_scores(record, scores, maximum=1e-5, minimum_gap=1.0)
        metrics["derived_target_angle_degrees"] = target_angle
    elif name == "angle_trisection":
        vertex = _point(record["vertex"])
        first = _point(record["reference_ray_endpoint"])
        second = _point(record["other_ray_endpoint"])
        angle1 = math.atan2(first[1] - vertex[1], first[0] - vertex[0])
        angle2 = math.atan2(second[1] - vertex[1], second[0] - vertex[0])
        delta = (angle2 - angle1) % (2.0 * math.pi)
        if delta > math.pi:
            delta -= 2.0 * math.pi
        target_angle = angle1 + delta / 3.0
        scores = {
            label: _angle_delta(target_angle, math.atan2(center[1] - vertex[1], center[0] - vertex[0]))
            for label, center in _candidate_points(record)
        }
        answer, metrics = _choose_by_scores(record, scores, maximum=1e-6, minimum_gap=0.02)
    elif name == "arc_bisector":
        center = _point(record["arc_center"])
        radius = float(record["radius"])
        angle = math.radians(float(record["arc_start_deg"]) + float(record["arc_sweep_deg"]) / 2.0)
        targets = [(center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "arc_center":
        samples = _points(record.get("source_arc_points"), count=3)
        targets = [_circumcenter(*samples)]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "circle_inversion":
        center, source = _point(record["inversion_center"]), _point(record["source_point"])
        vector = _sub(source, center)
        factor = float(record["inversion_radius"]) ** 2 / (vector[0] ** 2 + vector[1] ** 2)
        targets = [_add(center, _scale(vector, factor))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "circle_tangent_intersection":
        center = _point(record["center"])
        first, second = _points(record["circumference_points"], count=2)
        radial1, radial2 = _sub(first, center), _sub(second, center)
        targets = [_line_intersection(first, _add(first, (-radial1[1], radial1[0])), second, _add(second, (-radial2[1], radial2[0])))]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "diagonal_intersection":
        quad = _points(record["quadrilateral"], count=4)
        targets = [_line_intersection(quad[0], quad[2], quad[1], quad[3])]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "diameter_opposite":
        center, source = _point(record["circle_center"]), _point(record["source_point"])
        targets = [(2.0 * center[0] - source[0], 2.0 * center[1] - source[1])]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "dilation_to_circle":
        first, second = record["circle_1"], record["circle_2"]
        c1, c2, r1, r2 = _point(first["center"]), _point(second["center"]), float(first["radius"]), float(second["radius"])
        if abs(r1 - r2) <= 1e-9:
            raise CoordinateReplayError("equal circles have no finite external center")
        targets = [((r1 * c2[0] - r2 * c1[0]) / (r1 - r2), (r1 * c2[1] - r2 * c1[1]) / (r1 - r2))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "double_reflection":
        source = _point(record["source_point"])
        axis1, axis2 = _points(record["reflection_axis_1"], count=2), _points(record["reflection_axis_2"], count=2)
        targets = [_reflect(_reflect(source, *axis1), *axis2)]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "double_rotation":
        source = _point(record["source_point"])
        targets = [_rotate(_rotate(source, _point(record["pivot1"]), float(record["rotation1_degrees"])), _point(record["pivot2"]), float(record["rotation2_degrees"]))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "equilateral_triangle_vertex":
        first, second = _points(record["base"], count=2)
        midpoint, vector = _midpoint(first, second), _sub(second, first)
        offset = (-vector[1] * math.sqrt(3.0) / 2.0, vector[0] * math.sqrt(3.0) / 2.0)
        targets = [_add(midpoint, offset), _sub(midpoint, offset)]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "excircle_center":
        vertices = _points(record["triangle_vertices"], count=3)
        targets = [_triangle_excenter(vertices, int(record["marked_vertex_index"]))]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "excircle_tangent_point":
        vertices = _points(record["triangle_vertices"], count=3)
        marked = int(record["marked_vertex_index"])
        a, b, c = vertices[marked], vertices[(marked + 1) % 3], vertices[(marked + 2) % 3]
        side_a, side_b, side_c = _distance(b, c), _distance(a, c), _distance(a, b)
        distance_from_b = (side_a + side_b + side_c) / 2.0 - side_c
        targets = [_add(b, _scale(_unit(_sub(c, b)), distance_from_b))]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "external_angle_bisector":
        vertex = _point(record["vertex"])
        u1, u2 = _unit(_sub(_point(record["ray1_end"]), vertex)), _unit(_sub(_point(record["ray2_end"]), vertex))
        direction = _unit(_sub(u1, u2))
        scores = {}
        for label, center in _candidate_points(record):
            vector = _sub(center, vertex)
            scores[label] = abs(vector[0] * direction[1] - vector[1] * direction[0]) / max(math.hypot(*vector), 1e-9)
        answer, metrics = _choose_by_scores(record, scores, maximum=1e-6, minimum_gap=0.02)
    elif name == "external_ratio_point":
        start, through, ratio = _point(record["start_anchor"]), _point(record["through_anchor"]), float(record["ratio"])
        targets = [((ratio * through[0] - start[0]) / (ratio - 1.0), (ratio * through[1] - start[1]) / (ratio - 1.0))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name in {"external_tangent_point", "internal_tangent_point"}:
        first, second = record["circle_1"], record["circle_2"]
        c1, c2, r1, r2 = _point(first["center"]), _point(second["center"]), float(first["radius"]), float(second["radius"])
        # Which side of the tangent the far centre lies on is the whole rule.
        # An external tangent leaves both centres together, so the far centre
        # sits a radius *behind* the contact normal; an internal tangent
        # separates them, so it sits a radius in front.  Scoring the unsigned
        # distance collapsed all four common tangents onto one relation, and a
        # distractor parked on an internal contact then recomputed as a second
        # exact answer for the external task (and vice versa).
        far_offset = r2 if name == "external_tangent_point" else -r2
        scores = {}
        for label, point in _candidate_points(record):
            radial = _sub(point, c1)
            radial_error = abs(math.hypot(*radial) - r1)
            far_centre_offset = (_sub(c2, point)[0] * radial[0] + _sub(c2, point)[1] * radial[1]) / max(r1, 1e-9)
            scores[label] = radial_error + abs(far_centre_offset + far_offset)
        answer, metrics = _choose_by_scores(record, scores, maximum=1e-5, minimum_gap=1.0)
    elif name == "fermat_point":
        targets = [_fermat_point(_points(record["triangle_vertices"], count=3))]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "glide_reflection":
        source, axis = _point(record["source_point"]), _points(record["axis"], count=2)
        marker1, marker2 = _points(record["translation_markers"], count=2)
        targets = [_add(_reflect(source, *axis), _sub(marker2, marker1))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "homothety_center":
        large, small = _points(record["large_triangle"], count=3), _points(record["small_triangle"], count=3)
        targets = [_line_intersection(large[0], small[0], large[1], small[1])]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "spiral_similarity_center":
        large, small = _points(record["large_triangle"], count=3), _points(record["small_triangle"], count=3)
        targets = [_spiral_similarity_center(large, small)]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "incircle_tangent_point":
        side = _points(record["side_vertices"], count=2)
        targets = [_project(_point(record["incenter"]), *side)]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "inscribed_square_vertex":
        targets = [_inscribed_square(_points(record["triangle_vertices"], count=3))]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "inverse_rotation":
        targets = [_rotate(_point(record["image_point"]), _point(record["pivot"]), -float(record["rotation_degrees"]))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "kite_vertex":
        top, wing, bottom = _points(record["shown_vertices"], count=3)
        targets = [_reflect(wing, top, bottom)]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "midline_of_triangle":
        vertices = _points(record["triangle_vertices"], count=3)
        marked = _point(record["marked_midpoint"])
        mids = [_midpoint(vertices[0], vertices[1]), _midpoint(vertices[1], vertices[2]), _midpoint(vertices[2], vertices[0])]
        targets = [point for point in mids if _distance(point, marked) > 1e-3]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "napoleon_vertex":
        vertices = _points(record["triangle_vertices"], count=3)
        constructed = _points(record["constructed_equilateral_vertices"], count=3)
        side_lengths = [
            _distance(constructed[0], constructed[1]),
            _distance(constructed[1], constructed[2]),
            _distance(constructed[2], constructed[0]),
        ]
        if min(side_lengths) <= 1e-6 or max(side_lengths) - min(side_lengths) > max(side_lengths) * 1e-6:
            raise CoordinateReplayError("visible blue construction is not an equilateral triangle")

        shared_pairs: list[tuple[int, int]] = []
        for construction_index, point in enumerate(constructed):
            matches = [
                triangle_index
                for triangle_index, vertex in enumerate(vertices)
                if _distance(point, vertex) <= 1e-5
            ]
            if len(matches) == 1:
                shared_pairs.append((construction_index, matches[0]))
        if len(shared_pairs) != 2 or len({triangle_index for _, triangle_index in shared_pairs}) != 2:
            raise CoordinateReplayError("visible blue construction must share exactly one black-triangle side")

        shared_construction_indices = {construction_index for construction_index, _ in shared_pairs}
        apex_index = next(index for index in range(3) if index not in shared_construction_indices)
        opposite_index = next(
            index
            for index in range(3)
            if index not in {triangle_index for _, triangle_index in shared_pairs}
        )
        first, second = (constructed[index] for index in sorted(shared_construction_indices))
        apex, opposite = constructed[apex_index], vertices[opposite_index]
        edge = _sub(second, first)
        apex_side = edge[0] * (apex[1] - first[1]) - edge[1] * (apex[0] - first[0])
        opposite_side = edge[0] * (opposite[1] - first[1]) - edge[1] * (opposite[0] - first[0])
        if abs(apex_side) <= 1e-6 or abs(opposite_side) <= 1e-6 or apex_side * opposite_side >= 0.0:
            raise CoordinateReplayError("visible blue equilateral triangle is not constructed outward")

        targets = [
            (
                sum(point[0] for point in constructed) / 3.0,
                sum(point[1] for point in constructed) / 3.0,
            )
        ]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "nine_point_center":
        vertices = _points(record["triangle_vertices"], count=3)
        circum = _circumcenter(*vertices)
        orth = (sum(p[0] for p in vertices) - 2.0 * circum[0], sum(p[1] for p in vertices) - 2.0 * circum[1])
        targets = [_midpoint(circum, orth)]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "orthogonal_projection":
        targets = [_project(_point(record["source_point"]), *_points(record["line_points"], count=2))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "parabola_vertex":
        targets = [_fit_parabola_vertex(record)]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "parallelogram":
        first, second, opposite = _points(record["known_vertices"], count=3)
        targets = [(first[0] + second[0] - opposite[0], first[1] + second[1] - opposite[1])]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "perpendicular":
        through = _point(record["through_point"])
        first, second = _points(record["source_line_endpoints"], count=2)
        direction = _unit(_sub(second, first))
        scores = {}
        for label, center in _candidate_points(record):
            vector = _sub(center, through)
            scores[label] = abs(vector[0] * direction[0] + vector[1] * direction[1]) / max(math.hypot(*vector), 1e-9)
        answer, metrics = _choose_by_scores(record, scores, maximum=1e-6, minimum_gap=0.02)
    elif name == "perpendicular_bisector":
        first, second = _points(record["segment_endpoints"], count=2)
        scores = {label: abs(_distance(center, first) - _distance(center, second)) for label, center in _candidate_points(record)}
        answer, metrics = _choose_by_scores(record, scores, maximum=1e-5, minimum_gap=2.0)
    elif name == "point_rotation":
        targets = [_rotate(_point(record["source_point"]), _point(record["pivot"]), float(record["rotation_degrees"]))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "power_of_point":
        external = _point(record["external_point"])
        near, far = _points(record["full_secant_chord"], count=2)
        partial_near = _point(record["partial_secant_near_chord"])
        product = _distance(external, near) * _distance(external, far)
        far_distance = product / _distance(external, partial_near)
        targets = [_add(external, _scale(_unit(_sub(partial_near, external)), far_distance))]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "ray_intersection":
        rays = record["rays"]
        if not isinstance(rays, list) or len(rays) < 2:
            raise CoordinateReplayError("missing ray fragments")
        first = [_point(rays[0]["start"]), _point(rays[0]["end"])]
        second = [_point(rays[1]["start"]), _point(rays[1]["end"])]
        targets = [_line_intersection(*first, *second)]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "ray_reflect":
        if "mirror_segments" in record:
            from data.geometry.core.ray_optics import trace_mirrors
            contacts,direction,_ = trace_mirrors(record["light_source"],record["incoming_ray_tip"],record["mirror_segments"])
            contact=contacts[-1]
        else:
            source, mirror = _point(record["light_source"]), _points(record["mirror_line"], count=2)
            contact = _point(record["reflection_point"])
            virtual_source = _reflect(source, *mirror)
            direction = _unit(_sub(contact, virtual_source))
        scores = {}
        for label, center in _candidate_points(record):
            offset = abs(_sub(center, contact)[0] * direction[1] - _sub(center, contact)[1] * direction[0])
            forward = _sub(center, contact)[0] * direction[0] + _sub(center, contact)[1] * direction[1]
            scores[label] = offset + (1e6 if forward <= 0 else 0.0)
        answer, metrics = _choose_by_scores(record, scores, maximum=1e-5, minimum_gap=5.0)
    elif name == "reflection":
        targets = [_reflect(_point(record["source_point"]), *_points(record["reflection_axis"], count=2))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "regular_octagon_vertex":
        shown = _points(record["shown_vertices"])
        if len(shown) < 3:
            raise CoordinateReplayError("too few consecutive octagon vertices")
        previous = _sub(shown[-1], shown[-2])
        # The visible vertices are ordered around the screen-space octagon.
        # Continue the equal edge with the same 45-degree screen rotation;
        # no saved full polygon or missing vertex is consulted.
        angle = math.pi / 4.0
        next_edge = (
            previous[0] * math.cos(angle) - previous[1] * math.sin(angle),
            previous[0] * math.sin(angle) + previous[1] * math.cos(angle),
        )
        targets = [_add(shown[-1], next_edge)]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "rhombus_vertex":
        shown = _points(record["shown_vertices"], count=3)
        # The apex is read off the drawn shape — the one vertex whose two
        # sides are equal — rather than trusted from the stored order, and a
        # figure without exactly one such vertex fails closed.
        apexes = [
            index
            for index in range(3)
            if abs(
                _distance(shown[index], shown[(index + 1) % 3])
                - _distance(shown[index], shown[(index + 2) % 3])
            )
            <= 1e-6 * max(_distance(shown[index], shown[(index + 1) % 3]), 1.0)
        ]
        if len(apexes) != 1:
            raise CoordinateReplayError("visible triangle does not have one isosceles apex")
        apex = apexes[0]
        targets = [_reflect(shown[apex], shown[(apex + 1) % 3], shown[(apex + 2) % 3])]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "secant_intersection":
        first, second = _points(record["chord1_endpoints"], count=2), _points(record["chord2_endpoints"], count=2)
        targets = [_line_intersection(*first, *second)]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "segment_trisection":
        # The rule measures the third from the endpoint the scene draws as a
        # solid disc, so the other trisection point is not an answer.
        start, end = _point(record["marked_endpoint"]), _point(record["open_endpoint"])
        targets = [_add(start, _scale(_sub(end, start), 1.0 / 3.0))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "spiral_continuation":
        points = _points(record["spiral_points"])
        vectors = [_sub(points[index + 1], points[index]) for index in range(len(points) - 1)]
        lengths = [math.hypot(*vector) for vector in vectors]
        increment = float(np.median(np.diff(lengths)))
        previous = vectors[-1]
        direction = _unit((-previous[1], previous[0]))
        targets = [_add(points[-1], _scale(direction, lengths[-1] + increment))]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "square_vertex_from_diagonal":
        first, second = _points(record["diagonal_endpoints"], count=2)
        midpoint, half = _midpoint(first, second), _scale(_sub(second, first), 0.5)
        offset = (-half[1], half[0])
        targets = [_add(midpoint, offset), _sub(midpoint, offset)]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "symmetric_completion":
        axis = _points(record["axis"], count=2)
        endpoints = []
        for edge in record["visible_edges"]:
            endpoints.extend(_points(edge, count=2))
        unique: list[Point] = []
        for point in endpoints:
            if all(_distance(point, other) > 1e-4 for other in unique):
                unique.append(point)
        targets = []
        for point in unique:
            reflected = _reflect(point, *axis)
            if all(_distance(reflected, other) > 1e-3 for other in unique):
                targets.append(reflected)
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "tangent_circle_touch":
        first, second = record["circle_1"], record["circle_2"]
        c1, c2, r1 = _point(first["center"]), _point(second["center"]), float(first["radius"])
        targets = [_add(c1, _scale(_unit(_sub(c2, c1)), r1))]
        answer, metrics = _closest_candidate_to_targets(record, targets, tolerance=1e-2)
    elif name == "translation_composition":
        source = _point(record["source_point"])
        vector1, vector2 = _points(record["vector1"], count=2), _points(record["vector2"], count=2)
        targets = [_add(source, _add(_sub(vector1[1], vector1[0]), _sub(vector2[1], vector2[0])))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "trapezoid_median":
        first, second = _points(record["leg_midpoints"], count=2)
        targets = [_midpoint(first, second)]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "triangle_center":
        vertices = _points(record["triangle_vertices"], count=3)
        targets = [(sum(p[0] for p in vertices) / 3.0, sum(p[1] for p in vertices) / 3.0)]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "triangle_from_midpoints":
        first, second, third = _points(record["midpoints"], count=3)
        targets = [
            _sub(_add(third, second), first),
            _sub(_add(third, first), second),
            _sub(_add(second, first), third),
        ]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "vector_translation":
        source, start, end = _point(record["source_point"]), _point(record["vector_start"]), _point(record["vector_end"])
        targets = [_add(source, _sub(end, start))]
        answer, metrics = _closest_candidate_to_targets(record, targets)
    elif name == "vertical_angle_pair":
        intersection = _point(record["intersection"])
        line1, line2 = _points(record["line1_endpoints"], count=2), _points(record["line2_endpoints"], count=2)
        first = math.atan2(line1[1][1] - line1[0][1], line1[1][0] - line1[0][0])
        second = math.atan2(line2[1][1] - line2[0][1], line2[1][0] - line2[0][0])
        delta = (second - first) % (2.0 * math.pi)
        if delta > math.pi:
            delta -= math.pi
        target_angle = first + delta / 2.0 + math.pi
        scores = {
            label: _angle_delta(target_angle, math.atan2(center[1] - intersection[1], center[0] - intersection[0]))
            for label, center in _candidate_points(record)
        }
        answer, metrics = _choose_by_scores(record, scores, maximum=math.radians(35.0), minimum_gap=math.radians(5.0))
    else:
        raise CoordinateReplayError(f"no independent coordinate replay for {name}")

    return answer, {
        "coordinate_source_replay_valid": True,
        "coordinate_source_replay_reason": "ok",
        "coordinate_derived_answer": answer,
        "coordinate_replay_task": name,
        **metrics,
    }


__all__ = ["CoordinateReplayError", "derive_coordinate_answer"]

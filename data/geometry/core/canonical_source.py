"""Canonical, source-only semantic inputs for the Geometry family.

Geometry generators keep solved constructions beside visible primitives so
they can author reference images and videos.  Those solved fields are useful
inside the generator, but they are not verifier inputs.  This module is the
single family boundary that selects only the geometry visible in the puzzle
and the rule parameters stated to the model.

The allowlists are intentionally task-specific.  Generic key-name scrubbing
cannot distinguish, for example, a visible external point from a saved target
point, or a requested rotation from a renderer's solved target pose.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


_POINT_SOURCE_FIELDS: dict[str, tuple[str, ...]] = {
    "angle_bisector": ("vertex", "ray_endpoints"),
    "angle_supplement": ("source_angle_deg", "candidate_angles_deg"),
    "angle_trisection": ("vertex", "reference_ray_endpoint", "other_ray_endpoint"),
    "arc_bisector": ("arc_center", "radius", "arc_start_deg", "arc_sweep_deg"),
    "arc_center": ("source_arc_points",),
    "arc_connect": ("left_circle", "candidate_circles", "mask", "arc_span_degrees"),
    "circle_inversion": ("inversion_center", "inversion_radius", "source_point"),
    "circle_tangent_intersection": ("center", "circumference_points"),
    "circle_tangent_line": ("circle_center", "circle_radius", "contact_point"),
    "circle_tangent_point": ("circle_center", "circle_radius", "external_point"),
    "circumcenter": ("triangle_vertices",),
    "closest_pair": (),
    "diagonal_intersection": ("quadrilateral",),
    "diameter_opposite": ("circle_center", "source_point"),
    "dilation_to_circle": ("circle_1", "circle_2"),
    "double_reflection": ("source_point", "reflection_axis_1", "reflection_axis_2"),
    "double_rotation": (
        "source_point",
        "pivot1",
        "pivot2",
        "rotation1_degrees",
        "rotation2_degrees",
    ),
    "equilateral_triangle_vertex": ("base",),
    "excircle_center": ("triangle_vertices", "marked_vertex_index"),
    "excircle_tangent_point": ("triangle_vertices", "marked_vertex_index"),
    "external_angle_bisector": ("vertex", "ray1_end", "ray2_end"),
    "external_ratio_point": ("start_anchor", "through_anchor", "ratio"),
    "external_tangent_point": ("circle_1", "circle_2"),
    "fermat_point": ("triangle_vertices",),
    "glide_reflection": ("source_point", "axis", "translation_markers"),
    "homothety_center": ("large_triangle", "small_triangle"),
    "incenter": ("triangle_vertices",),
    "incircle_tangent_point": ("triangle_vertices", "side_vertices", "incenter"),
    "inscribed_angle_arc": ("circle_center", "arc_endpoints", "candidate_angles_deg"),
    "inscribed_square_vertex": ("triangle_vertices",),
    "internal_tangent_point": ("circle_1", "circle_2"),
    "inverse_rotation": ("image_point", "pivot", "rotation_degrees"),
    "isosceles_trapezoid": ("visible_vertices",),
    "kite_vertex": ("shown_vertices",),
    "midline_of_triangle": ("triangle_vertices", "marked_midpoint"),
    "midpoint": ("segment",),
    "napoleon_vertex": ("triangle_vertices", "constructed_equilateral_vertices"),
    "nine_point_center": ("triangle_vertices",),
    "orthocenter": ("triangle_vertices",),
    "orthogonal_projection": ("source_point", "line_points"),
    "parabola_vertex": ("source_curve_points", "axis_angle_deg"),
    "parallel": ("source_line_endpoints", "through_point"),
    "parallelogram": ("known_vertices",),
    "perpendicular": ("source_line_endpoints", "through_point"),
    "perpendicular_bisector": ("segment_endpoints",),
    "point_rotation": ("source_point", "pivot", "rotation_degrees"),
    "power_of_point": ("external_point", "full_secant_chord", "partial_secant_near_chord"),
    "ray_intersection": ("rays",),
    "ray_reflect": ("light_source", "mirror_line", "reflection_point"),
    "reflection": ("source_point", "reflection_axis"),
    "regular_octagon_vertex": ("shown_vertices",),
    "rhombus_vertex": ("shown_vertices",),
    "right_triangle": (),
    "secant_intersection": ("chord1_endpoints", "chord2_endpoints"),
    "segment_trisection": ("marked_endpoint", "open_endpoint"),
    "spiral_continuation": ("spiral_points",),
    "spiral_similarity_center": ("large_triangle", "small_triangle"),
    "square_vertex_from_diagonal": ("diagonal_endpoints",),
    "symmetric_completion": ("axis", "visible_edges"),
    "tangent_circle_touch": ("circle_1", "circle_2"),
    "tangent_line_pair_angle": (
        "circle_center",
        "external_point",
        "radius",
        "candidate_angles_deg",
    ),
    "translation_composition": ("source_point", "vector1", "vector2"),
    "trapezoid_median": ("leg_midpoints",),
    "triangle_center": ("triangle_vertices",),
    "triangle_from_midpoints": ("midpoints",),
    "vector_translation": ("source_point", "vector_start", "vector_end"),
    "vertical_angle_pair": ("intersection", "line1_endpoints", "line2_endpoints"),
}


_RENDERED_SOURCE_FIELDS: dict[str, tuple[str, ...]] = {
    "convex_hull_count": ("points", "options"),
    "count_intersection_points": ("n", "lines", "options"),
    "lattice_points_strict_inside": ("verts", "options"),
    "mirror_image_select": (
        "source_vertices",
        "candidate_vertices",
        "path_color",
        "mirror_axis",
        "start_marker",
        "end_marker",
        "multiple_solution_policy",
    ),
    "region_count": ("circles", "options"),
    "right_triangle_selection": ("triangles", "candidates"),
    "symmetry_completion": ("grid_size", "axis", "source_cells", "candidates"),
    "triangle_inequality_check": ("triples",),
}


_CANDIDATE_SAFE_KEYS = {
    "label",
    "x",
    "y",
    "cells",
    "vertices",
    "highlighted_triangle_indices",
    "relative_angles_deg",
    "distance_sq",
    "sweep_deg",
}


def _copy_fields(source: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    missing = [field for field in fields if field not in source]
    if missing:
        raise ValueError(f"Geometry source is missing required fields: {', '.join(missing)}")
    return {field: deepcopy(source[field]) for field in fields}


def _clean_candidates(value: Any) -> Any:
    if not isinstance(value, list):
        return deepcopy(value)
    cleaned: list[Any] = []
    for item in value:
        if not isinstance(item, Mapping):
            cleaned.append(deepcopy(item))
            continue
        candidate = {
            str(key): deepcopy(raw)
            for key, raw in item.items()
            if str(key) in _CANDIDATE_SAFE_KEYS
        }
        cleaned.append(candidate)
    return cleaned


def _right_triangle_selection_source(structured: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only primitives that are visible on the right-triangle card."""

    raw_triangles = structured.get("triangles")
    raw_candidates = structured.get("candidates")
    if not isinstance(raw_triangles, list) or not raw_triangles:
        raise ValueError("right_triangle_selection is missing visible triangles")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 4:
        raise ValueError("right_triangle_selection is missing its four visible candidates")

    triangle_fields = ("vertices", "label", "color", "cell_index")
    triangles: list[dict[str, Any]] = []
    for triangle in raw_triangles:
        if not isinstance(triangle, Mapping):
            raise ValueError("right_triangle_selection triangle must be an object")
        triangles.append(_copy_fields(triangle, triangle_fields))

    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, Mapping) or "highlighted_triangle_indices" not in candidate:
            raise ValueError("right_triangle_selection candidate is missing its visible label set")
        candidates.append(
            {
                # The option row labels are fixed visible A-D identities.  The
                # authoring envelope historically omitted this redundant field,
                # so reconstruct it from the visible row order.
                "label": chr(ord("A") + index),
                "highlighted_triangle_indices": deepcopy(
                    candidate["highlighted_triangle_indices"]
                ),
            }
        )
    return {"triangles": triangles, "candidates": candidates}


def _candidate_centers(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return []
    centers: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        centers.append(
            {
                "label": str(item.get("label") or ""),
                "center": [float(item["x"]), float(item["y"])],
            }
        )
    return centers


def _canonical_point_source(
    task_name: str,
    record: Mapping[str, Any],
    structured: Mapping[str, Any],
) -> dict[str, Any]:
    fields = _POINT_SOURCE_FIELDS[task_name]
    if task_name == "ray_reflect" and "mirror_segments" in record:
        fields = ("light_source", "mirror_segments", "incoming_ray_tip")
    # A few task-local wrappers moved visible primitives into their structured
    # authoring envelope.  Prefer the top-level raw source, then fall back to
    # that envelope for the same explicitly allowlisted field.
    source: dict[str, Any] = {}
    for field in fields:
        if field in record:
            source[field] = deepcopy(record[field])
        elif field in structured:
            source[field] = deepcopy(structured[field])
        else:
            raise ValueError(f"Geometry task {task_name!r} is missing visible field {field!r}")
    raw_candidates = record.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"Geometry point task {task_name!r} must expose visible candidates")
    source["candidates"] = _clean_candidates(raw_candidates)

    # Specialized whole-video parsers consume the same source twice: once at
    # the record root and once through ``structured_answer``.  The canonical
    # state contains only a duplicate view of raw visible primitives; derived
    # centers, measurements, traces, targets, and render schedules are absent.
    if task_name in {
        "angle_bisector",
        "arc_connect",
        "circle_tangent_line",
        "circle_tangent_point",
        "circumcenter",
        "incenter",
        "isosceles_trapezoid",
        "midpoint",
        "orthocenter",
        "parallel",
    }:
        state = {key: deepcopy(value) for key, value in source.items() if key != "candidates"}
        state["candidate_centers"] = _candidate_centers(source["candidates"])
        if task_name == "midpoint":
            segment = source.get("segment")
            if isinstance(segment, Mapping):
                state["visible_endpoints"] = {
                    "P": deepcopy(segment.get("start")),
                    "Q": deepcopy(segment.get("end")),
                }
        source["semantic_state"] = state
    return source


def _canonical_rendered_source(
    task_name: str,
    structured: Mapping[str, Any],
) -> dict[str, Any]:
    if task_name == "rotation":
        # These are the stated operation and the four visible rigid poses, not
        # a saved winner.  Answer index, target pose metadata, and traces stay
        # in reference authoring only.
        state = {
            "requested_rotation": deepcopy(structured.get("target_transform")),
            "visible_candidate_poses": deepcopy(structured.get("candidate_transforms")),
        }
    elif task_name == "right_triangle_selection":
        # The structured authoring envelope also contains independently
        # derivable angles, dot products, right-angle flags, centroids and
        # selection radii.  None is a verifier source: vertices determine the
        # geometry and the visible option rows determine the candidate sets.
        state = _right_triangle_selection_source(structured)
    else:
        fields = _RENDERED_SOURCE_FIELDS.get(task_name)
        if fields is None:
            raise ValueError(f"Geometry task {task_name!r} has no canonical source allowlist")
        state = _copy_fields(structured, fields)

    if "candidates" in state:
        state["candidates"] = _clean_candidates(state["candidates"])
    return {"semantic_state": state}


def canonical_geometry_semantic_input(
    task_name: str,
    record: Mapping[str, Any],
    structured: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the sole canonical verifier input for one Geometry sample."""

    if task_name in _POINT_SOURCE_FIELDS:
        return _canonical_point_source(task_name, record, structured)
    return _canonical_rendered_source(task_name, structured)


__all__ = ["canonical_geometry_semantic_input"]

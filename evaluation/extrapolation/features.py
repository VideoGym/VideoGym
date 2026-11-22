"""Task-level exposure axes recomputed from public inputs, never saved answers."""

import math
import numpy as np

GEOMETRY_MECHANISMS = frozenset(
    {
        "circle_inversion",
        "homothety_center",
        "spiral_similarity_center",
        "double_reflection",
        "external_tangent_point",
        "tangent_circle_touch",
        "count_intersection_points",
        "region_count",
        "spiral_continuation",
        "arc_center",
        "diagonal_intersection",
    }
)
SEQUENCE_MECHANISMS = frozenset(
    {"growing_polygon_step", "increasing_size_sequence", "linear_sequence_next"}
)
STATE_MECHANISMS = frozenset(
    {
        "rule_table_1d_rollout",
        "lifelike_rule_rollout",
        "cyclic_threshold_wave",
        "sandpile_avalanche",
        "elevator_dispatch_tick",
        "protocol_handshake_tick",
    }
)


def _vector(a, b):
    return np.asarray(b, dtype=float) - np.asarray(a, dtype=float)


def mechanism_features(task, public):
    s = {**public, **public.get("semantic_state", {})}
    if task == "circle_inversion":
        ratio = math.dist(s["source_point"], s["inversion_center"]) / float(s["inversion_radius"])
        return {"source_region": "inside" if ratio < 1 else "outside" if ratio > 1 else "boundary"}
    if task in ("homothety_center", "spiral_similarity_center"):
        u = _vector(*s["large_triangle"][:2])
        v = _vector(*s["small_triangle"][:2])
        return (
            {"scale_sign": "positive" if np.dot(u, v) > 0 else "negative"}
            if task == "homothety_center"
            else {"similarity_ratio": float(np.linalg.norm(v) / np.linalg.norm(u))}
        )
    if task == "double_reflection":
        u = _vector(*s["reflection_axis_1"])
        v = _vector(*s["reflection_axis_2"])
        cross = abs(u[0] * v[1] - u[1] * v[0]) / (np.linalg.norm(u) * np.linalg.norm(v))
        return {"axis_relation": "parallel" if cross < 1e-8 else "intersecting"}
    if task in ("external_tangent_point", "tangent_circle_touch"):
        a, b = s["circle_1"], s["circle_2"]
        r1, r2 = float(a["radius"]), float(b["radius"])
        distance = math.dist(a["center"], b["center"])
        if task == "external_tangent_point":
            return {
                "circle_relation": "intersecting"
                if abs(r1 - r2) < distance < r1 + r2
                else "separate"
            }
        return {
            "tangency": "internal"
            if abs(distance - abs(r1 - r2)) < 1e-5
            else "external"
        }
    if task == "count_intersection_points":
        from data.geometry.count_intersection_points.generator import _raw_pair_intersections, _cluster_intersections
        groups = _cluster_intersections(_raw_pair_intersections(s["lines"]))
        return {"maximum_concurrency": max(len({i for pair in g["pairs"] for i in pair}) for g in groups)}
    if task == "region_count":
        from data.geometry.region_count.generator import _analyze_arrangement
        return {"boundary_components": _analyze_arrangement(s["circles"])["component_count"]}
    if task == "diagonal_intersection":
        a,b,c,d = map(np.asarray, s["quadrilateral"])
        t,u = np.linalg.solve(np.column_stack((c-a, b-d)), b-a)
        return {"diagonal_crossing_location": "interior" if 0 < t < 1 and 0 < u < 1 else "exterior"}


    if task == "spiral_continuation":
        points = s["spiral_points"]
        lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
        return {
            "length_step_sign": "positive" if lengths[1] > lengths[0] else "negative"
        }
    if task == "arc_center":
        a, m, b = map(np.asarray, s["source_arc_points"])
        # The angle at the visible midpoint subtends the complementary arc.
        u, v = a - m, b - m
        cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return {"arc_regime": "major" if cosine > 0 else "minor"}

    if task in ("growing_polygon_step", "increasing_size_sequence"):
        values = s[
            "observed_counts" if task == "growing_polygon_step" else "observed_heights"
        ]
        step = values[1] - values[0]
        if values[2] - values[1] != step or step == 0:
            raise ValueError("source is not a nonconstant arithmetic progression")
        return {"step_direction": "increasing" if step > 0 else "decreasing"}
    if task == "linear_sequence_next":
        from data.in_context_learning.linear_sequence_next.generator import (
            _infer_repeated_rule,
        )

        return {"spatial_rule": _infer_repeated_rule(s["observed_frames"])}
    if task in STATE_MECHANISMS:
        params, initial = s["params"], s["initial_state"]
        if task == "rule_table_1d_rollout":
            return {"rule_ones": sum(params["rule"])}
        if task == "lifelike_rule_rollout":
            return {"zero_birth": 0 in params["birth"]}
        if task == "cyclic_threshold_wave":
            return {"threshold": params["threshold"]}
        if task == "sandpile_avalanche":
            r, c = params["add"]
            grid = initial["grid"]
            return {
                "addition_region": "boundary"
                if r in (0, len(grid) - 1) or c in (0, len(grid[0]) - 1)
                else "interior"
            }
        if task == "elevator_dispatch_tick":
            return {
                "full_car": ",".join(
                    c["id"] for c in initial["cars"] if c["load"] == c["capacity"]
                )
            }
        return {"initial_phase": initial["client"] + ":" + initial["server"]}
    raise ValueError(task)

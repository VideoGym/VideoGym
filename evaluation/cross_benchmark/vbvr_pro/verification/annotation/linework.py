"""Discrete cost routes and line connectivity/order, parsed without neural judges."""

from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import cv2
import numpy as np
from evaluation.cross_benchmark.vbvr_pro.verification.navigation.grid import Spec, _agent, _scene_preserved
from evaluation.cross_benchmark.vbvr_pro.verification.annotation.selection import _mark_mask

SUPPORTED_IDS = {"G-41", "G-54", "O-34"}


@lru_cache(maxsize=128)
def _maximum_cost(costs: tuple[tuple[int, ...], ...], start: tuple[int, int], goal: tuple[int, int]) -> int:
    rows, cols = len(costs), len(costs[0])
    values = tuple(v for row in costs for v in row)
    neighbors = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        neighbors.append(
            tuple(
                rr * cols + cc
                for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
                if 0 <= rr < rows and 0 <= cc < cols
            )
        )
    target = goal[0] * cols + goal[1]

    @lru_cache(maxsize=None)
    def solve(node, mask):
        if node == target:
            return values[node]
        best = -(10**12)
        for nxt in neighbors[node]:
            if not mask & (1 << nxt):
                best = max(best, values[node] + solve(nxt, mask | (1 << nxt)))
        return best

    origin = start[0] * cols + start[1]
    return solve(origin, 1 << origin)


def _highest_cost(frames, first, final, metadata):
    s = metadata.get("semantic_ground_truth") or {}
    render = metadata.get("generic_declarative_render") or {}
    if not s or not render:
        return None
    costs = tuple(tuple(int(v) for v in row) for row in s["grid_costs"])
    rows, cols = len(costs), len(costs[0])
    if rows * cols > 16:
        raise ValueError("Maximum simple-path solver supports at most 16 cells")
    layers = render["first_frame"]["layers"]
    tiles = [v for v in layers if v.get("type") == "labeled_tile"]
    if len(tiles) != rows * cols:
        raise ValueError("Reference does not render every numeric grid cell")
    for tile in tiles:
        if int(tile["label"]) != costs[tile["row"]][tile["col"]]:
            raise ValueError("Rendered numeric label contradicts symbolic cost grid")
    body = next(v for v in layers if v.get("kind") == "ellipse" and v.get("fill_rgb") == [255, 255, 0])
    scale = first.shape[1] / render["canvas"]["width"]
    tile = tiles[0]
    spec = Spec(
        rows,
        cols,
        (tile.get("origin_x_px", 0) * scale, tile.get("origin_y_px", 0) * scale),
        tile["cell_size_px"] * scale,
        tuple(s["start_cell"]),
        tuple(s["goal_cell"]),
        set(),
        [],
        False,
        np.array([0, 255, 255.0]),
        body["rx_px"] * scale,
        [],
    )
    optimum = _maximum_cost(costs, spec.start, spec.goal)
    if optimum != s["path_total_cost"]:
        raise ValueError("Recorded maximum path cost contradicts independent simple-path solver")
    route = []
    positions = []
    invalid = []
    for i, frame in enumerate(frames):
        agents, _ = _agent(frame, spec)
        if len(agents) != 1 or not np.all(np.isfinite(agents[0])):
            invalid.append(i)
            positions.append(None)
            continue
        point = agents[0]
        positions.append(point)
        cell = spec.cell(point)
        if not route or cell != route[-1]:
            route.append(cell)
    adjacent = all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(route, route[1:]))
    in_grid = all(0 <= r < rows and 0 <= c < cols for r, c in route)
    cost = sum(costs[r][c] for r, c in route) if in_grid else None
    scene_ok, bad_scene = _scene_preserved(frames, first, spec, positions, final, unobserved_source=True)
    checks = {
        "one_agent_every_frame": not invalid,
        "starts_at_source": bool(route) and route[0] == spec.start,
        "ends_at_goal": bool(route) and route[-1] == spec.goal,
        "all_cells_in_grid": in_grid,
        "orthogonal_adjacent_steps": adjacent,
        "no_cell_revisited": len(route) == len(set(route)),
        "maximum_total_cost": cost == optimum,
        "numeric_grid_preserved_throughout": scene_ok,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "evidence": {
            "family": "maximum_simple_cost_route",
            "route": [list(c) for c in route],
            "observed_cost": cost,
            "optimal_cost": optimum,
            "invalid_agent_frames": invalid,
            "corrupted_scene_frames": bad_scene,
        },
    }


def _thin(mask):
    """Zhang-Suen topology-preserving thinning for explicit stroke graphs."""
    image = np.pad((mask > 0).astype(np.uint8), 1)
    while True:
        removed = 0
        for phase in (0, 1):
            p = image[1:-1, 1:-1]
            ns = [
                image[:-2, 1:-1],
                image[:-2, 2:],
                image[1:-1, 2:],
                image[2:, 2:],
                image[2:, 1:-1],
                image[2:, :-2],
                image[1:-1, :-2],
                image[:-2, :-2],
            ]
            count = sum(ns)
            transitions = sum((a == 0) & (b == 1) for a, b in zip(ns, ns[1:] + ns[:1]))
            if phase == 0:
                condition = (ns[0] * ns[2] * ns[4] == 0) & (ns[2] * ns[4] * ns[6] == 0)
            else:
                condition = (ns[0] * ns[2] * ns[6] == 0) & (ns[0] * ns[4] * ns[6] == 0)
            delete = (p == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & condition
            removed += np.count_nonzero(delete)
            p[delete] = 0
        if removed == 0:
            break
    return image[1:-1, 1:-1]


def _stroke_topology(mask, spur_limit=2):
    # Codec pinholes would otherwise create tiny cycles in a single solid
    # stroke. Closing at the two-pixel edge scale leaves visible loops intact.
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    skeleton = _thin(cleaned)
    points = set(map(tuple, np.argwhere(skeleton > 0)))
    graph = {}
    for r, c in points:
        neighbors = set()
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            nxt = (r + dr, c + dc)
            if nxt not in points:
                continue
            # Do not turn the three pixels of a raster corner into a triangle.
            if dr and dc and ((r + dr, c) in points or (r, c + dc) in points):
                continue
            neighbors.add(nxt)
        graph[(r, c)] = neighbors
    for _ in range(3):
        deleted = set()
        for start in [point for point, neighbors in graph.items() if len(neighbors) == 1]:
            path = [start]
            previous = None
            current = start
            while len(path) <= spur_limit + 1:
                neighbors = graph[current] - ({previous} if previous is not None else set())
                if len(neighbors) > 1:
                    if len(path) - 1 <= spur_limit:
                        deleted.update(path[:-1])
                    break
                if not neighbors:
                    break
                previous, current = current, next(iter(neighbors))
                path.append(current)
        if not deleted:
            break
        for point in deleted:
            for neighbor in graph[point]:
                if neighbor not in deleted:
                    graph[neighbor].discard(point)
            del graph[point]
    endpoints = np.asarray([point for point, neighbors in graph.items() if len(neighbors) == 1])
    branches = np.asarray([point for point, neighbors in graph.items() if len(neighbors) >= 3])
    return endpoints, branches


def _classified(image, colors):
    distances = np.stack([np.max(np.abs(image.astype(np.float32) - c), axis=2) for c in colors])
    nearest = np.argmin(distances, axis=0)
    distance = np.min(distances, axis=0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = []
    for i, color in enumerate(colors):
        hue = int(cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
        dh = np.abs(hsv[:, :, 0].astype(int) - hue)
        dh = np.minimum(dh, 180 - dh)
        masks.append(((nearest == i) & (distance < 55) & (dh <= 8)).astype(np.uint8))
    return masks


def _color_connections(frames, first, metadata):
    s = metadata.get("semantic_ground_truth") or {}
    render = metadata.get("generic_declarative_render") or {}
    shapes = [v for v in s.get("objects", []) if v.get("role") == "colored_shape"]
    if not shapes or not render:
        return None
    # Work in a fixed pixel domain; endpoints and line widths share that scale.
    h, w = first.shape[:2]
    scale = min(1.0, 384 / max(h, w))
    size = (round(w * scale), round(h * scale))
    a = cv2.resize(first, size, interpolation=cv2.INTER_AREA)
    seq = [cv2.resize(v, size, interpolation=cv2.INTER_AREA) for v in frames]
    canvas = render["canvas"]
    sx, sy = size[0] / canvas["width"], size[1] / canvas["height"]
    groups = defaultdict(list)
    for obj in shapes:
        groups[tuple(obj["rgb"][::-1])].append((obj["center"][0] * sx, obj["center"][1] * sy))
    colors = [np.asarray(v, float) for v in groups]
    source_masks = _classified(a, colors)
    end_masks = _classified(seq[-1], colors)
    line_width = (metadata.get("parameters") or {}).get("line_width", 10) * sx
    group_details = []
    checks = {
        "all_color_groups_connected": True,
        "no_extra_components_or_branches": True,
        "no_cross_color_connections": True,
        "all_original_shapes_preserved": True,
        "edits_use_only_allowed_colors": True,
        "curves_grow_without_erasure": True,
    }
    source_union = np.any(np.stack(source_masks), axis=0)
    # Codec edge phase around an existing colored object is not a newly drawn
    # curve. Use the same one-pixel object halo for temporal growth and final
    # stroke topology; otherwise static edges can appear to grow and erase.
    source_dilated = cv2.dilate(source_union.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    previous = [np.zeros(a.shape[:2], np.uint8) for _ in colors]
    curve_sizes = [[] for _ in colors]
    bad_frames = []
    for fi, frame in enumerate(seq):
        masks = _classified(frame, colors)
        permitted = np.any(np.stack(masks), axis=0) | source_union
        permitted = cv2.dilate(permitted.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        unknown = (np.max(np.abs(frame.astype(np.int16) - a.astype(np.int16)), axis=2) > 55) & ~permitted
        if np.count_nonzero(unknown) > max(12, a.shape[0] * a.shape[1] * 0.0003):
            bad_frames.append(fi)
        for i, mask in enumerate(masks):
            source = source_masks[i] > 0
            if np.count_nonzero(source & (mask == 0)) / max(np.count_nonzero(source), 1) > 0.04:
                checks["all_original_shapes_preserved"] = False
            curve = (mask > 0) & ~source_dilated
            curve_sizes[i].append(int(np.count_nonzero(curve)))
            old = previous[i] > 0
            tolerated = cv2.dilate(curve.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if np.count_nonzero(old & ~tolerated) > max(8, np.count_nonzero(old) * 0.03):
                checks["curves_grow_without_erasure"] = False
            previous[i] = curve.astype(np.uint8)
    checks["edits_use_only_allowed_colors"] = not bad_frames
    checks["curve_drawing_process_observed"] = all(
        counts[-1] > 0 and any(counts[-1] * 0.10 < value < counts[-1] * 0.90 for value in counts)
        for counts in curve_sizes
    )
    for i, (color, centers) in enumerate(groups.items()):
        mask = end_masks[i]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        center_labels = [
            int(labels[min(size[1] - 1, max(0, round(y))), min(size[0] - 1, max(0, round(x)))]) for x, y in centers
        ]
        connected = all(v > 0 and v == center_labels[0] for v in center_labels)
        significant = {j for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] >= max(8, line_width**2 * 0.3)}
        extras = significant - set(center_labels)
        curve = ((mask > 0) & ~source_dilated).astype(np.uint8)
        components, curve_labels, curve_stats, _ = cv2.connectedComponentsWithStats(curve, 8)
        fragments = []
        topology_ok = True
        for label in range(1, components):
            if curve_stats[label, cv2.CC_STAT_AREA] < max(8, line_width**2 * 0.3):
                continue
            endpoints, branches = _stroke_topology(
                (curve_labels == label).astype(np.uint8), max(2, round(line_width * 0.6))
            )
            fragments.append({"endpoints": len(endpoints), "branch_pixels": len(branches)})
            if len(endpoints) != 2 or len(branches) > 0:
                topology_ok = False
        if len(fragments) != len(centers) - 1:
            topology_ok = False
        foreign = np.any(np.stack([v > 0 for j, v in enumerate(source_masks) if j != i]), axis=0)
        new_pixels = (mask > 0) & (source_masks[i] == 0)
        crosses = np.count_nonzero(new_pixels & foreign) > max(3, line_width)
        checks["all_color_groups_connected"] &= connected
        checks["no_extra_components_or_branches"] &= not extras and topology_ok
        checks["no_cross_color_connections"] &= not crosses
        group_details.append(
            {
                "bgr": list(color),
                "endpoint_labels": center_labels,
                "connected": connected,
                "extra_components": len(extras),
                "fragments": fragments,
                "crosses_foreign_shape": bool(crosses),
            }
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "evidence": {"family": "color_connection_graph", "groups": group_details, "unknown_edit_frames": bad_frames},
    }


def _dot_connections(frames, first, metadata):
    s = metadata.get("semantic_ground_truth") or {}
    p = metadata.get("parameters") or {}
    render = metadata.get("generic_declarative_render") or {}
    if not s or not render:
        return None
    dots = sorted([v for v in s["objects"] if v["symbol"] == "dot"], key=lambda v: v["index"])
    order = s["connection_order"]
    if order != [v["index"] for v in sorted(dots, key=lambda v: v["number"])]:
        raise ValueError("Dot connection order contradicts numeric labels")
    h, w = first.shape[:2]
    scale = min(1.0, 512 / max(h, w))
    size = (round(w * scale), round(h * scale))
    a = cv2.resize(first, size, interpolation=cv2.INTER_AREA)
    seq = [cv2.resize(v, size, interpolation=cv2.INTER_AREA) for v in frames]
    sx, sy = size[0] / render["canvas"]["width"], size[1] / render["canvas"]["height"]
    centers = [np.asarray([v["center"][0] * sx, v["center"][1] * sy]) for v in dots]
    radii = [v["radius"] * sx for v in dots]
    color = np.asarray(s["line_color_rgb"][::-1], float)
    width = max(1, int(round(p["line_width"] * sx)))
    dots_mask = np.zeros(a.shape[:2], np.uint8)
    for center, radius in zip(centers, radii):
        cv2.circle(dots_mask, tuple(np.round(center).astype(int)), round(radius + 2), 255, -1)
    allowed = np.zeros(a.shape[:2], np.uint8)
    samples = []
    segments = []
    for u, v in zip(order, order[1:]):
        aa, bb = centers[u], centers[v]
        segments.append((aa, bb))
        cv2.line(allowed, tuple(np.round(aa).astype(int)), tuple(np.round(bb).astype(int)), 255, max(3, width + 6))
        length = np.linalg.norm(bb - aa)
        steps = max(4, int(length / 2))
        line = aa + (bb - aa) * np.linspace(0, 1, steps)[:, None]
        line = line[np.array([all(np.linalg.norm(pt - c) > r + 3 for c, r in zip(centers, radii)) for pt in line])]
        if len(line) < 3:
            raise ValueError("Dot segment has insufficient visible evidence outside original dots")
        samples.append(np.round(line).astype(int))
    # At a crossing or overlap, an earlier legal stroke is also evidence for
    # pixels on a later segment. Only the later segment's exclusive visible
    # portions can establish its drawing time.
    interference = max(2, width * 0.6) + max(3, width + 6) / 2
    exclusive = []
    for index, points in enumerate(samples):
        visible = np.ones(len(points), bool)
        for other, (aa, bb) in enumerate(segments):
            if other == index:
                continue
            direction = bb - aa
            progress = np.clip(np.sum((points - aa) * direction, axis=1) / np.dot(direction, direction), 0, 1)
            distance = np.linalg.norm(points - (aa + progress[:, None] * direction), axis=1)
            visible &= distance > interference
        exclusive.append(points[visible])
    observable = [i for i, points in enumerate(exclusive) if len(points) >= 3]
    all_coverage = []
    all_order_coverage = []
    extra_frames = []
    damaged_frames = []
    prior = np.zeros(len(samples))
    order_prior = np.zeros(len(observable))
    erasures = []
    order_violations = []
    for fi, frame in enumerate(seq):
        # A legitimate new segment may fade from the white canvas to its red
        # ink. Observe that color contribution, otherwise a partially drawn
        # shaft is treated as an unexplained background change.
        mask = (_mark_mask(frame, a, color, allow_fade=True) > 0).astype(np.uint8)
        mask[dots_mask > 0] = 0
        distance = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
        coverage = np.asarray(
            [float(np.mean(distance[pts[:, 1], pts[:, 0]] <= max(2, width * 0.6))) for pts in samples]
        )
        all_coverage.append(coverage.tolist())
        order_coverage = np.asarray([
            float(np.mean(distance[exclusive[i][:, 1], exclusive[i][:, 0]] <= max(2, width * 0.6)))
            for i in observable
        ])
        all_order_coverage.append(order_coverage.tolist())
        # Observing all edges first become complete in one frame does not
        # establish their strict numerical drawing order.
        newly_complete = np.count_nonzero((order_coverage >= 0.94) & (order_prior < 0.94))
        if newly_complete > 1:
            order_violations.append(fi)
        if np.any(coverage + 0.06 < prior):
            erasures.append(fi)
        prior = np.maximum(prior, coverage)
        first_incomplete = next((i for i, v in enumerate(order_coverage) if v < 0.94), len(order_coverage))
        if np.any(order_coverage[first_incomplete + 1 :] > 0.12):
            order_violations.append(fi)
        order_prior = np.maximum(order_prior, order_coverage)
        extra = (mask > 0) & (allowed == 0)
        if np.count_nonzero(extra) > max(8, np.count_nonzero(mask) * 0.02):
            extra_frames.append(fi)
        # Exclude only line pixels; every original dot and its label stays intact.
        smooth = cv2.GaussianBlur(frame, (3, 3), 0)
        base = cv2.GaussianBlur(a, (3, 3), 0)
        changed = np.max(np.abs(smooth.astype(np.int16) - base.astype(np.int16)), axis=2) > 55
        damage = False
        for center, radius in zip(centers, radii):
            zone = np.zeros(a.shape[:2], np.uint8)
            cv2.circle(zone, tuple(np.round(center).astype(int)), max(3, round(radius * 0.65)), 255, -1)
            if np.count_nonzero(changed & (zone > 0)) / max(np.count_nonzero(zone), 1) > 0.12:
                damage = True
        line_edge = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        unknown = changed & (line_edge == 0) & (dots_mask == 0)
        if damage or np.count_nonzero(unknown) > max(12, a.shape[0] * a.shape[1] * 0.0003):
            damaged_frames.append(fi)
    checks = {
        "all_required_straight_segments_complete": all(v >= 0.94 for v in all_coverage[-1]),
        "strict_numeric_connection_order": not order_violations,
        "no_extra_segments": not extra_frames,
        "no_line_erasure": not erasures,
        "dots_numbers_and_background_preserved": not damaged_frames,
        "progressive_connection_evidence": len({tuple(round(v, 1) for v in row) for row in all_coverage})
        >= min(4, len(samples) + 1),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "evidence": {
            "family": "ordered_dot_connection_segments",
            "connection_order": order,
            "final_segment_coverage": all_coverage[-1],
            "exclusive_segment_sample_counts": [len(points) for points in exclusive],
            "observable_order_segment_indices": observable,
            "final_exclusive_segment_coverage": all_order_coverage[-1],
            "order_violation_frames": order_violations,
            "extra_segment_frames": extra_frames,
            "erasure_frames": erasures,
            "damaged_frames": damaged_frames,
        },
    }


def verify(task, frames, first, final, metadata, reference):
    task_id = task.split("_", 1)[0]
    if task_id not in SUPPORTED_IDS:
        return None
    if not frames:
        return {"passed": False, "checks": {"frames_present": False}, "evidence": {}}
    if task_id == "G-41":
        return _highest_cost(frames, first, final, metadata)
    if task_id == "G-54":
        return _color_connections(frames, first, metadata)
    return _dot_connections(frames, first, metadata)

"""Dynamic topology, layout, and anti-shortcut labelling for unified graph tasks.

Every task in the unified Graph Theory suite samples its own small graph from
this module instead of replaying a fixed hard-coded template.  Topology
sampling is seeded and deterministic; layout acceptance is strict (no node
overlap, no edge crossing, no edge passing through a third node) with a bounded
rejection budget, and numeric labels are permuted so neither alphabetical nor
numeric order can act as a solution shortcut.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Callable, Dict, Iterable, Optional, Sequence

CANVAS = 512
NODE_RADIUS = 16.0
LAYOUT_MARGIN = 46.0
MIN_NODE_GAP = 4.0 * NODE_RADIUS
MIN_EDGE_NODE_GAP = NODE_RADIUS + 10.0
MAX_TOPOLOGY_ATTEMPTS = 300
MAX_LABEL_ATTEMPTS = 240

IntEdge = tuple[int, int]


def int_edge(left: int, right: int) -> IntEdge:
    return (left, right) if left < right else (right, left)


def adjacency_list(node_count: int, edges: Iterable[IntEdge]) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    return [sorted(set(values)) for values in adjacency]


def bfs_distances(node_count: int, edges: Iterable[IntEdge], source: int) -> list[int]:
    adjacency = adjacency_list(node_count, edges)
    distance = [-1] * node_count
    distance[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if distance[neighbor] < 0:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def all_pairs_distances(node_count: int, edges: Iterable[IntEdge]) -> list[list[int]]:
    return [bfs_distances(node_count, edges, node) for node in range(node_count)]


def connected_components(node_count: int, edges: Iterable[IntEdge]) -> list[list[int]]:
    adjacency = adjacency_list(node_count, edges)
    remaining = set(range(node_count))
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        seen = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        remaining -= seen
        components.append(sorted(seen))
    return sorted(components, key=lambda values: values[0])


def simple_paths(node_count: int, edges: Iterable[IntEdge], start: int, goal: int, *, limit: int = 4000) -> list[list[int]]:
    adjacency = adjacency_list(node_count, edges)
    paths: list[list[int]] = []

    def visit(node: int, path: list[int]) -> None:
        if len(paths) >= limit:
            return
        if node == goal:
            paths.append(list(path))
            return
        for neighbor in adjacency[node]:
            if neighbor not in path:
                path.append(neighbor)
                visit(neighbor, path)
                path.pop()

    visit(start, [start])
    return paths


def sample_tree_edges(rng: random.Random, node_count: int, mode: str | None = None) -> list[IntEdge]:
    """Sample one random spanning tree over ``0..node_count-1``.

    ``recursive`` attaches each new node to a uniformly chosen earlier node;
    ``preferential`` attaches degree-proportionally, giving a scale-free hub
    profile.  The mode itself is chosen by the caller's seeded rng.
    """

    if node_count < 2:
        raise ValueError("a tree needs at least two nodes")
    if mode is None:
        mode = rng.choice(("recursive", "preferential"))
    if mode not in {"recursive", "preferential"}:
        raise ValueError(f"unknown tree sampling mode: {mode!r}")
    edges: list[IntEdge] = []
    degree = [0] * node_count
    for node in range(1, node_count):
        if mode == "recursive":
            parent = rng.randrange(node)
        else:
            pool = list(range(node))
            weights = [degree[value] + 1 for value in pool]
            total = sum(weights)
            pick = rng.uniform(0.0, float(total))
            parent = pool[-1]
            running = 0.0
            for value, weight in zip(pool, weights):
                running += weight
                if pick <= running:
                    parent = value
                    break
        edges.append(int_edge(parent, node))
        degree[parent] += 1
        degree[node] += 1
    return edges


def add_chords(
    rng: random.Random,
    node_count: int,
    edges: Sequence[IntEdge],
    count: int,
    predicate: Optional[Callable[[int, int], bool]] = None,
) -> list[IntEdge]:
    """Return up to ``count`` new non-tree edges satisfying ``predicate``."""

    existing = {int_edge(*edge) for edge in edges}
    candidates = [
        (left, right)
        for left in range(node_count)
        for right in range(left + 1, node_count)
        if (left, right) not in existing and (predicate is None or predicate(left, right))
    ]
    rng.shuffle(candidates)
    return candidates[: max(0, count)]


def _point_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    px, py = float(point[0]), float(point[1])
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length_sq))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def _segments_strictly_cross(a1: Sequence[float], a2: Sequence[float], b1: Sequence[float], b2: Sequence[float]) -> bool:
    def orient(p: Sequence[float], q: Sequence[float], r: Sequence[float]) -> float:
        return (float(q[0]) - float(p[0])) * (float(r[1]) - float(p[1])) - (float(q[1]) - float(p[1])) * (float(r[0]) - float(p[0]))

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    return (o1 * o2 < -1e-9) and (o3 * o4 < -1e-9)


def layout_acceptable(
    positions: Sequence[Sequence[float]],
    edges: Sequence[IntEdge],
    *,
    minimum_node_gap: float = MIN_NODE_GAP,
) -> bool:
    """Strict visual acceptance: spacing, margins, clearance, no crossings."""

    node_count = len(positions)
    for x, y in positions:
        if not (LAYOUT_MARGIN <= x <= CANVAS - LAYOUT_MARGIN and LAYOUT_MARGIN <= y <= CANVAS - LAYOUT_MARGIN):
            return False
    for left in range(node_count):
        for right in range(left + 1, node_count):
            if math.dist(positions[left], positions[right]) < minimum_node_gap:
                return False
    for left, right in edges:
        for node in range(node_count):
            if node in (left, right):
                continue
            if _point_segment_distance(positions[node], positions[left], positions[right]) < MIN_EDGE_NODE_GAP:
                return False
    edge_list = list(edges)
    for index, first in enumerate(edge_list):
        for second in edge_list[index + 1 :]:
            if len({first[0], first[1], second[0], second[1]}) < 4:
                continue
            if _segments_strictly_cross(positions[first[0]], positions[first[1]], positions[second[0]], positions[second[1]]):
                return False
    return True


def _force_positions(
    rng: random.Random,
    node_count: int,
    edges: Sequence[IntEdge],
    *,
    center_x: float,
    center_y: float,
    half_width: float,
    half_height: float,
) -> list[tuple[float, float]]:
    """One seeded force simulation inside an axis-aligned box."""

    spread_radius = min(half_width, half_height) * 0.85
    positions: list[list[float]] = []
    for node in range(node_count):
        angle = 2.0 * math.pi * node / max(1, node_count) + rng.uniform(-0.35, 0.35)
        spread = spread_radius * (0.55 + 0.45 * rng.random())
        positions.append([center_x + spread * math.cos(angle), center_y + spread * math.sin(angle)])
    area = (2.0 * half_width) * (2.0 * half_height)
    ideal = math.sqrt(area / max(1, node_count)) * 0.95
    for step in range(240):
        cooling = 1.0 - 0.6 * (step / 240.0)
        displacement = [[0.0, 0.0] for _ in range(node_count)]
        for left in range(node_count):
            for right in range(left + 1, node_count):
                dx = positions[left][0] - positions[right][0]
                dy = positions[left][1] - positions[right][1]
                distance = math.hypot(dx, dy) or 1e-6
                force = (ideal * ideal / distance) * 0.018 * cooling
                ux, uy = dx / distance, dy / distance
                displacement[left][0] += ux * force
                displacement[left][1] += uy * force
                displacement[right][0] -= ux * force
                displacement[right][1] -= uy * force
        for left, right in edges:
            dx = positions[right][0] - positions[left][0]
            dy = positions[right][1] - positions[left][1]
            distance = math.hypot(dx, dy) or 1e-6
            force = ((distance - ideal * 0.85) * distance / ideal) * 0.02 * cooling
            ux, uy = dx / distance, dy / distance
            displacement[left][0] += ux * force
            displacement[left][1] += uy * force
            displacement[right][0] -= ux * force
            displacement[right][1] -= uy * force
        for node in range(node_count):
            displacement[node][0] += (center_x - positions[node][0]) * 0.0025
            displacement[node][1] += (center_y - positions[node][1]) * 0.0025
        for node in range(node_count):
            positions[node][0] = min(center_x + half_width, max(center_x - half_width, positions[node][0] + displacement[node][0]))
            positions[node][1] = min(center_y + half_height, max(center_y - half_height, positions[node][1] + displacement[node][1]))
    return [(x, y) for x, y in positions]


def _fit_to_canvas(positions: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Recenter one layout and scale it up (never down) to fill the canvas."""

    xs = [float(point[0]) for point in positions]
    ys = [float(point[1]) for point in positions]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    available = CANVAS - 2.0 * LAYOUT_MARGIN
    scale = min(
        1.35,
        available / span_x if span_x > 1.0 else 1.35,
        available / span_y if span_y > 1.0 else 1.35,
    )
    scale = max(1.0, scale)
    mid_x = (max(xs) + min(xs)) / 2.0
    mid_y = (max(ys) + min(ys)) / 2.0
    center = CANVAS / 2.0
    return [
        (
            round(min(CANVAS - LAYOUT_MARGIN, max(LAYOUT_MARGIN, center + (x - mid_x) * scale)), 1),
            round(min(CANVAS - LAYOUT_MARGIN, max(LAYOUT_MARGIN, center + (y - mid_y) * scale)), 1),
        )
        for x, y in zip(xs, ys)
    ]


def _layout_quality(positions: Sequence[Sequence[float]], edges: Sequence[IntEdge]) -> float:
    """Score an accepted layout: angular resolution, even edges, coverage."""

    adjacency: Dict[int, list[int]] = {}
    for left, right in edges:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    min_gap = math.pi
    for node, neighbors in adjacency.items():
        if len(neighbors) < 2:
            continue
        angles = sorted(
            math.atan2(positions[other][1] - positions[node][1], positions[other][0] - positions[node][0])
            for other in neighbors
        )
        gaps = [angles[index + 1] - angles[index] for index in range(len(angles) - 1)]
        gaps.append(2.0 * math.pi - (angles[-1] - angles[0]))
        min_gap = min(min_gap, min(gaps))
    lengths = [math.dist(positions[left], positions[right]) for left, right in edges]
    if lengths:
        mean_length = sum(lengths) / len(lengths)
        variation = math.sqrt(sum((value - mean_length) ** 2 for value in lengths) / len(lengths)) / max(1.0, mean_length)
    else:
        variation = 0.0
    xs = [float(point[0]) for point in positions]
    ys = [float(point[1]) for point in positions]
    coverage = min(max(xs) - min(xs), max(ys) - min(ys)) / (CANVAS - 2.0 * LAYOUT_MARGIN)
    return min_gap * 1.5 - variation * 0.8 + coverage * 0.6


def layout_force_directed(rng: random.Random, node_count: int, edges: Sequence[IntEdge]) -> Optional[list[tuple[float, float]]]:
    """Best acceptable seeded force layout out of several scored attempts."""

    center = CANVAS / 2.0
    half = (CANVAS - 2.0 * LAYOUT_MARGIN) / 2.0
    best: Optional[list[tuple[float, float]]] = None
    best_score = -math.inf
    for _ in range(5):
        candidate = _fit_to_canvas(
            _force_positions(
                rng, node_count, edges,
                center_x=center, center_y=center, half_width=half, half_height=half,
            )
        )
        if not layout_acceptable(candidate, edges):
            continue
        score = _layout_quality(candidate, edges)
        if score > best_score:
            best, best_score = candidate, score
    return best


def layout_components(
    rng: random.Random,
    node_count: int,
    edges: Sequence[IntEdge],
) -> Optional[list[tuple[float, float]]]:
    """Compact per-component layout for a deliberately disconnected graph.

    A plain force simulation pushes disconnected islands onto the canvas
    boundary, leaving a hollow unreadable ring.  Here every component gets
    its own sub-region sized by membership, keeping each island a compact
    visual cluster.
    """

    components = connected_components(node_count, edges)
    count = len(components)
    if count < 2 or count > 4:
        return None
    patterns = {
        2: ((0.32, 0.38), (0.68, 0.62)),
        3: ((0.30, 0.30), (0.70, 0.36), (0.48, 0.72)),
        4: ((0.30, 0.30), (0.70, 0.32), (0.30, 0.72), (0.70, 0.70)),
    }[count]
    anchors = [
        (
            LAYOUT_MARGIN + (CANVAS - 2.0 * LAYOUT_MARGIN) * (px + rng.uniform(-0.04, 0.04)),
            LAYOUT_MARGIN + (CANVAS - 2.0 * LAYOUT_MARGIN) * (py + rng.uniform(-0.04, 0.04)),
        )
        for px, py in patterns
    ]
    order = sorted(range(count), key=lambda index: -len(components[index]))
    anchor_slots = list(range(count))
    rng.shuffle(anchor_slots)
    positions: list[Optional[tuple[float, float]]] = [None] * node_count
    usable = CANVAS - 2.0 * LAYOUT_MARGIN
    for rank, component_index in enumerate(order):
        members = components[component_index]
        center_x, center_y = anchors[anchor_slots[rank]]
        share = math.sqrt(len(members) / node_count)
        half = max(MIN_NODE_GAP * 0.75, usable * 0.5 * share * 0.92)
        center_x = min(CANVAS - LAYOUT_MARGIN - half * 0.4, max(LAYOUT_MARGIN + half * 0.4, center_x))
        center_y = min(CANVAS - LAYOUT_MARGIN - half * 0.4, max(LAYOUT_MARGIN + half * 0.4, center_y))
        if len(members) == 1:
            positions[members[0]] = (round(center_x, 1), round(center_y, 1))
            continue
        local_index = {node: index for index, node in enumerate(members)}
        local_edges = [
            (local_index[left], local_index[right])
            for left, right in edges
            if left in local_index and right in local_index
        ]
        local = _force_positions(
            rng, len(members), local_edges,
            center_x=center_x, center_y=center_y, half_width=half, half_height=half,
        )
        for node, point in zip(members, local):
            positions[node] = (round(point[0], 1), round(point[1], 1))
    result = [position for position in positions if position is not None]
    if len(result) != node_count:
        return None
    final = [
        (
            min(CANVAS - LAYOUT_MARGIN, max(LAYOUT_MARGIN, x)),
            min(CANVAS - LAYOUT_MARGIN, max(LAYOUT_MARGIN, y)),
        )
        for x, y in result
    ]
    return final if layout_acceptable(final, edges) else None


def layout_layered_tree(
    node_count: int, edges: Sequence[IntEdge], root: int
) -> Optional[list[tuple[float, float]]]:
    """Deterministic tidy layered layout for one rooted tree."""

    adjacency = adjacency_list(node_count, edges)
    parent: Dict[int, Optional[int]] = {root: None}
    depth: Dict[int, int] = {root: 0}
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in parent:
                parent[neighbor] = node
                depth[neighbor] = depth[node] + 1
                queue.append(neighbor)
    if len(parent) != node_count:
        return None
    children: Dict[int, list[int]] = {node: sorted(child for child in range(node_count) if parent.get(child) == node) for node in range(node_count)}

    weight: Dict[int, int] = {}

    def measure(node: int) -> int:
        total = sum(measure(child) for child in children[node])
        weight[node] = max(1, total)
        return weight[node]

    measure(root)
    max_depth = max(depth.values())
    if max_depth < 1:
        return None
    top, bottom = 74.0, float(CANVAS) - 74.0
    if node_count <= 10 and max_depth > 1 and (bottom - top) / max_depth < MIN_NODE_GAP:
        return None
    inset = 18.0 if node_count <= 10 else 0.0
    left_bound, right_bound = LAYOUT_MARGIN + inset, float(CANVAS) - LAYOUT_MARGIN - inset
    if (
        node_count <= 10
        and weight[root] > 1
        and (right_bound - left_bound) / weight[root] < MIN_NODE_GAP
    ):
        return None
    # Larger trees may stagger leaves across depths. The final Euclidean
    # node/edge clearance gate remains unchanged; per-axis bounds alone
    # would reject such layouts even when no glyphs are close.
    xs: Dict[int, float] = {}

    def place(node: int, low: float, high: float) -> None:
        xs[node] = (low + high) / 2.0
        total = sum(weight[child] for child in children[node])
        cursor = low
        for child in children[node]:
            span = (high - low) * (weight[child] / total) if total else 0.0
            place(child, cursor, cursor + span)
            cursor += span

    place(root, left_bound, right_bound)
    result = [
        (round(xs[node], 1), round(top + (bottom - top) * depth[node] / max_depth, 1))
        for node in range(node_count)
    ]
    return (
        result
        if layout_acceptable(
            result,
            edges,
            minimum_node_gap=3.25 * NODE_RADIUS if node_count > 10 else MIN_NODE_GAP,
        )
        else None
    )


def labelling_acceptable(
    permutation: Sequence[int],
    edges: Sequence[IntEdge],
    answer_paths: Sequence[Sequence[int]] = (),
) -> bool:
    """Return whether one numeric labelling defeats order-based shortcuts.

    ``permutation[node]`` is the visible number of one topology node.  The
    labelling leaks when more than one graph edge joins consecutive numbers,
    or when any ordered answer path of at least three nodes reads as a
    monotone or unit-step numeric sequence.
    """

    consecutive = sum(1 for left, right in edges if abs(permutation[left] - permutation[right]) == 1)
    if consecutive > 1:
        return False
    for path in answer_paths:
        if len(path) < 3:
            continue
        sequence = [permutation[node] for node in path]
        steps = [right - left for left, right in zip(sequence, sequence[1:])]
        if all(step > 0 for step in steps) or all(step < 0 for step in steps):
            return False
        if all(abs(step) == 1 for step in steps):
            return False
    return True


__all__ = [
    "CANVAS",
    "LAYOUT_MARGIN",
    "MAX_TOPOLOGY_ATTEMPTS",
    "MIN_EDGE_NODE_GAP",
    "MIN_NODE_GAP",
    "NODE_RADIUS",
    "IntEdge",
    "add_chords",
    "adjacency_list",
    "all_pairs_distances",
    "bfs_distances",
    "connected_components",
    "int_edge",
    "labelling_acceptable",
    "layout_acceptable",
    "layout_components",
    "layout_force_directed",
    "layout_layered_tree",
    "sample_tree_edges",
    "simple_paths",
]

"""Source-only semantics for the shared labelled-graph task engine (v2).

Every unified Graph Theory task samples a fresh small graph per sample:
random recursive/preferential trees plus chords for general graphs, tidy
layered layouts for rooted trees, and force-directed layouts elsewhere.  Node
labels are the numbers 1..N assigned by a seeded random permutation that is
rejected whenever numeric order could act as a solution shortcut.  Nothing in
this module trusts a saved solution: generation proves the formal answer
(unique, or multi-certificate for ``trace_all_edges_once``) before any media
is authored, and the verifier replays the same solver independently.
"""

from __future__ import annotations

import random
import re
from collections import deque
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from data.graph_theory.core.graph_sampling import (
    MAX_LABEL_ATTEMPTS,
    MAX_TOPOLOGY_ATTEMPTS,
    add_chords,
    adjacency_list,
    all_pairs_distances,
    connected_components,
    int_edge,
    labelling_acceptable,
    layout_components,
    layout_force_directed,
    layout_layered_tree,
    sample_tree_edges,
    simple_paths,
)
from data.graph_theory.core.task_specs import NEW_GRAPH_TASK_NAMES, get_unified_graph_spec
from data.common.scalar_lists import (
    format_nonnegative_integer_answer,
    format_string_list_answer,
)
from data.common.typed_answers import format_answer_line


Edge = tuple[str, str]
SCHEMA_VERSION = "unified_labelled_graph_source_v2"
_LABEL_PATTERN = re.compile(r"(?:[1-9]|1[0-8])")


def edge_key(left: Any, right: Any) -> Edge:
    a, b = str(left).strip(), str(right).strip()
    return (a, b) if _label_sort_key(a) < _label_sort_key(b) else (b, a)


def _label_sort_key(label: str) -> tuple[int, str]:
    return (0, f"{int(label):03d}") if label.isdigit() else (1, label)


def _difficulty_size(difficulty: str | None, rng: random.Random) -> int:
    if difficulty == "min":
        return 6
    if difficulty == "max":
        return rng.choice((9, 10))
    if difficulty == "typical":
        return rng.choice((7, 8))
    return rng.choice((6, 7, 8, 9, 10))


def _chord_count(rng: random.Random, difficulty: str | None) -> int:
    if difficulty == "min":
        return 1
    if difficulty == "max":
        return rng.choice((2, 3))
    if difficulty == "typical":
        return rng.choice((1, 2))
    return rng.choice((1, 2, 3))


# ---------------------------------------------------------------------------
# Topology attempts (integer node namespace; layout and labelling come later)
# ---------------------------------------------------------------------------

_Topology = tuple[int, list[tuple[int, int]], Dict[str, Any], Dict[int, Mapping[str, Any]], str]


def _attempt_route(task_name: str, rng: random.Random, node_count: int, difficulty: str | None) -> Optional[_Topology]:
    edges = sample_tree_edges(rng, node_count)
    edges = edges + add_chords(rng, node_count, edges, _chord_count(rng, difficulty))
    distances = all_pairs_distances(node_count, edges)
    separation = 4 if difficulty == "max" else 3
    endpoints = [
        (start, goal)
        for start in range(node_count)
        for goal in range(start + 1, node_count)
        if distances[start][goal] >= separation
    ]
    if not endpoints:
        return None
    start, goal = rng.choice(endpoints)
    paths = simple_paths(node_count, edges, start, goal)
    if not paths:
        return None
    roles: Dict[str, Any] = {"start": start, "goal": goal}
    base = min(len(path) - 1 for path in paths)
    if task_name == "shortest_path_choice":
        winners = [path for path in paths if len(path) - 1 == base]
        if len(winners) != 1:
            return None
    elif task_name == "route_via_checkpoint":
        options: list[int] = []
        for candidate in range(node_count):
            if candidate in (start, goal):
                continue
            via = [path for path in paths if candidate in path]
            if not via:
                continue
            best = min(len(path) - 1 for path in via)
            winners = [path for path in via if len(path) - 1 == best]
            # The checkpoint is load-bearing: the legal optimum must be
            # strictly longer than the unconstrained optimum, and unique.
            if len(winners) == 1 and best > base:
                options.append(candidate)
        if not options:
            return None
        roles["checkpoint"] = rng.choice(options)
    elif task_name == "avoid_blocked_vertex":
        shortest = [path for path in paths if len(path) - 1 == base]
        interior = sorted({node for path in shortest for node in path[1:-1]})
        options = []
        for candidate in interior:
            legal = [path for path in paths if candidate not in path]
            if not legal:
                continue
            best = min(len(path) - 1 for path in legal)
            winners = [path for path in legal if len(path) - 1 == best]
            # Blocking is load-bearing: every unconstrained optimum uses the
            # blocked node, and the legal detour optimum is unique.
            if len(winners) == 1 and best > base:
                options.append(candidate)
        if not options:
            return None
        roles["blocked"] = rng.choice(options)
    else:
        raise ValueError(task_name)
    return node_count, edges, roles, {}, "force"


def _attempt_trace(rng: random.Random, node_count: int, difficulty: str | None) -> Optional[_Topology]:
    """Construct one branched semi-Eulerian graph (base trail plus ears).

    The anti-degeneration contract is structural: at least one branch vertex
    of degree >= 3 (an interior attachment has degree >= 4), and at least one
    ear whose branch is >= 2 edges deep, so the trail can never collapse into
    a branch-free chain.  Any branched semi-Eulerian graph admits more than
    one Euler trail, so this task accepts every valid trail.  Attachment
    degree is capped so no hub sprouts an unreadable fan of near-parallel
    edges.
    """

    if difficulty == "min":
        base_length = 3
    elif difficulty == "max":
        base_length = 5
    elif difficulty == "typical":
        base_length = 4
    else:
        base_length = rng.choice((3, 4, 5))
    if node_count - (base_length + 1) < 2:
        return None
    path = list(range(base_length + 1))
    edges = [int_edge(path[index], path[index + 1]) for index in range(base_length)]
    degree = [0] * node_count
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    remaining = [node for node in range(node_count) if node not in set(path)]
    rng.shuffle(remaining)
    ear_plan = [2]
    leftover = len(remaining) - 2
    while leftover > 0:
        chunk = min(leftover, rng.choice((1, 2)))
        ear_plan.append(chunk)
        leftover -= chunk
    for size in ear_plan:
        open_attachments = [node for node in path[1:-1] if degree[node] <= 3]
        if not open_attachments:
            return None
        attachment = rng.choice(open_attachments)
        internal = [remaining.pop() for _ in range(size)]
        chain = [attachment, *internal, attachment]
        for left, right in zip(chain, chain[1:]):
            edges.append(int_edge(left, right))
            degree[left] += 1
            degree[right] += 1
    roles = {"start": path[0], "goal": path[-1]}
    return node_count, edges, roles, {}, "force"


def _attempt_euler_endpoints(rng: random.Random, node_count: int, difficulty: str | None) -> Optional[_Topology]:
    edges = sample_tree_edges(rng, node_count)
    edges = edges + add_chords(rng, node_count, edges, _chord_count(rng, difficulty))
    degree = [0] * node_count
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    odd = [node for node in range(node_count) if degree[node] % 2 == 1]
    # Exactly two odd vertices, and the graph must not degenerate into one
    # chain whose endpoints are visible without any degree counting.
    if len(odd) != 2 or max(degree) < 3:
        return None
    return node_count, edges, {}, {}, "force"


def _attempt_bridge(rng: random.Random, node_count: int, difficulty: str | None) -> Optional[_Topology]:
    """Two cycle-based blobs joined by exactly one connectivity-critical edge."""

    if difficulty == "max" and node_count >= 8:
        left_size = rng.randint(4, node_count - 4)
    else:
        left_size = rng.randint(3, node_count - 3)
    left_nodes = list(range(left_size))
    right_nodes = list(range(left_size, node_count))

    def cycle_edges(nodes: list[int]) -> list[tuple[int, int]]:
        return [int_edge(nodes[index], nodes[(index + 1) % len(nodes)]) for index in range(len(nodes))]

    edges = cycle_edges(left_nodes) + cycle_edges(right_nodes)
    # Extra chords must be sampled in each blob's local index space.  Sampling
    # over the whole graph would create a second cross-blob connection and
    # destroy the unique bridge certificate.
    for blob in (left_nodes, right_nodes):
        if len(blob) < 4:
            continue
        local_edges = cycle_edges(list(range(len(blob))))
        local_chords = add_chords(rng, len(blob), local_edges, 1)
        edges.extend(int_edge(blob[left], blob[right]) for left, right in local_chords)
    bridge = int_edge(rng.choice(left_nodes), rng.choice(right_nodes))
    edges.append(bridge)
    initial = len(connected_components(node_count, edges))
    bridges = [
        edge for edge in edges
        if len(connected_components(node_count, [other for other in edges if other != edge])) > initial
    ]
    if bridges != [bridge]:
        return None
    return node_count, edges, {}, {}, "force"


def _attempt_components(rng: random.Random, node_count: int, difficulty: str | None) -> Optional[_Topology]:
    if difficulty == "min":
        component_target = 2
    elif difficulty == "max":
        component_target = rng.choice((3, 4))
    elif difficulty == "typical":
        component_target = rng.choice((2, 3))
    else:
        component_target = rng.choice((2, 3, 4))
    component_target = min(component_target, node_count - 2)
    sizes = [1] * component_target
    for _ in range(node_count - component_target):
        sizes[rng.randrange(component_target)] += 1
    rng.shuffle(sizes)
    edges: list[tuple[int, int]] = []
    cursor = 0
    for size in sizes:
        members = list(range(cursor, cursor + size))
        cursor += size
        if size >= 2:
            local = sample_tree_edges(rng, size)
            edges += [(members[a], members[b]) for a, b in local]
            if size >= 4 and rng.random() < 0.5:
                chords = add_chords(rng, size, local, 1)
                edges += [(members[a], members[b]) for a, b in chords]
    return node_count, edges, {}, {}, "components"


def _attempt_degree(rng: random.Random, node_count: int, difficulty: str | None) -> Optional[_Topology]:
    edges = sample_tree_edges(rng, node_count)
    edges = edges + add_chords(rng, node_count, edges, _chord_count(rng, difficulty))
    degree = [0] * node_count
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    minimum = 3 if difficulty == "max" else 2
    candidates = [node for node in range(node_count) if degree[node] >= minimum]
    if not candidates:
        return None
    query = rng.choice(candidates)
    return node_count, edges, {"query": query}, {query: {"degree_badge": "?"}}, "force"


def _attempt_two_hop(rng: random.Random, node_count: int, difficulty: str | None) -> Optional[_Topology]:
    edges = sample_tree_edges(rng, node_count)
    edges = edges + add_chords(rng, node_count, edges, _chord_count(rng, difficulty))
    distances = all_pairs_distances(node_count, edges)
    adjacency = adjacency_list(node_count, edges)
    require_merge = difficulty != "min"
    candidates = []
    for node in range(node_count):
        second_ring = [other for other in range(node_count) if distances[node][other] == 2]
        if len(second_ring) < 2:
            continue
        if require_merge:
            # At least one two-hop vertex must be reachable through two
            # different one-hop neighbors, so naive double counting is a real
            # distractor rather than a decorative rule.
            merged = any(
                sum(1 for middle in adjacency[other] if distances[node][middle] == 1) >= 2
                for other in second_ring
            )
            if not merged:
                continue
        candidates.append(node)
    if not candidates:
        return None
    return node_count, edges, {"query": rng.choice(candidates)}, {}, "force"


def _attempt_bipartite(rng: random.Random, node_count: int, difficulty: str | None) -> Optional[_Topology]:
    edges = sample_tree_edges(rng, node_count)
    depth = all_pairs_distances(node_count, edges)[0]
    side = [value % 2 for value in depth]
    chords = add_chords(rng, node_count, edges, _chord_count(rng, difficulty), lambda a, b: side[a] != side[b])
    edges = edges + chords
    degree = [0] * node_count
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    candidates = [node for node in range(node_count) if degree[node] >= 2]
    if not candidates:
        return None
    query = rng.choice(candidates)
    flip = rng.randrange(2)
    node_attrs = {
        node: {"bipartite_color": None if node == query else ("RED" if (side[node] ^ flip) == 0 else "BLUE")}
        for node in range(node_count)
    }
    return node_count, edges, {"query": query}, node_attrs, "force"


def _tree_relations(node_count: int, edges: Sequence[tuple[int, int]], root: int) -> tuple[Dict[int, Optional[int]], Dict[int, int], Dict[int, list[int]]]:
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
    children = {node: sorted(child for child in range(node_count) if parent.get(child) == node) for node in range(node_count)}
    return parent, depth, children


def _lca(parent: Mapping[int, Optional[int]], depth: Mapping[int, int], first: int, second: int) -> int:
    a, b = first, second
    while depth[a] > depth[b]:
        a = parent[a]  # type: ignore[assignment]
    while depth[b] > depth[a]:
        b = parent[b]  # type: ignore[assignment]
    while a != b:
        a = parent[a]  # type: ignore[assignment]
        b = parent[b]  # type: ignore[assignment]
    return a


def _attempt_tree(task_name: str, rng: random.Random, node_count: int, difficulty: str | None) -> Optional[_Topology]:
    edges = sample_tree_edges(rng, node_count)
    root = 0
    parent, depth, children = _tree_relations(node_count, edges, root)
    roles: Dict[str, Any] = {"root": root}
    if task_name == "leaf_count":
        leaves = [node for node in range(node_count) if node != root and not children[node]]
        if len(leaves) < 3:
            return None
    elif task_name == "depth_lookup":
        minimum = 3 if difficulty == "max" else 2
        candidates = [node for node in range(node_count) if depth[node] >= minimum]
        if not candidates:
            return None
        roles["query"] = rng.choice(candidates)
    elif task_name == "path_between_nodes":
        minimum = 3 if difficulty == "max" else 2
        if rng.random() < 0.4:
            # Absorbed ``tree_navigation`` mechanic: navigate from the green
            # root to one marked deep node (a leaf when one is deep enough).
            deep = [node for node in range(node_count) if depth[node] >= minimum]
            if not deep:
                return None
            deep_leaves = [node for node in deep if not children[node]]
            roles["query_start"] = root
            roles["query_goal"] = rng.choice(deep_leaves or deep)
        else:
            pairs = []
            for first in range(node_count):
                for second in range(first + 1, node_count):
                    node, other = first, second
                    steps = 0
                    while node != other:
                        if depth[node] >= depth[other]:
                            node = parent[node]  # type: ignore[assignment]
                        else:
                            other = parent[other]  # type: ignore[assignment]
                        steps += 1
                    if steps >= minimum:
                        pairs.append((first, second))
            if not pairs:
                return None
            first, second = rng.choice(pairs)
            roles["query_start"] = first
            roles["query_goal"] = second
    elif task_name == "closest_common_ancestor":
        minimum_lca_depth = 1 if difficulty == "max" else 0
        pairs = [
            (first, second)
            for first in range(node_count)
            for second in range(first + 1, node_count)
            if _lca(parent, depth, first, second) not in (first, second)
            and depth[_lca(parent, depth, first, second)] >= minimum_lca_depth
        ]
        if not pairs:
            return None
        roles["queries"] = list(rng.choice(pairs))
    elif task_name == "subtree_size":
        minimum = 2 if difficulty == "min" else (4 if difficulty == "max" else 3)
        # ``_is_int_ancestor(node, node)`` is true, so this closure count
        # already includes the subtree root itself.
        descendants = {node: sum(1 for other in range(node_count) if _is_int_ancestor(node, other, parent)) for node in range(node_count)}
        candidates = [node for node in range(node_count) if node != root and descendants[node] >= minimum]
        if not candidates:
            return None
        roles["query"] = rng.choice(candidates)
    elif task_name == "balance_comparison":
        if len(children[root]) != 2:
            return None
        sides = [
            sum(1 for node in range(node_count) if _is_int_ancestor(side, node, parent))
            for side in children[root]
        ]
        # Both branches must need genuine counting: no branch is a bare stub
        # and the two counts stay close enough that one glance cannot settle
        # the comparison.
        if min(sides) < 2 or abs(sides[0] - sides[1]) > 2:
            return None
    elif task_name == "diameter_endpoints":
        distances = all_pairs_distances(node_count, edges)
        best = max(distances[first][second] for first in range(node_count) for second in range(first + 1, node_count))
        winners = [
            (first, second)
            for first in range(node_count)
            for second in range(first + 1, node_count)
            if distances[first][second] == best
        ]
        if len(winners) != 1 or best < (4 if difficulty == "max" else 3):
            return None
        # The diameter is an unrooted predicate: no root role is exposed.
        return node_count, edges, {}, {}, "force"
    else:
        raise ValueError(task_name)
    return node_count, edges, roles, {}, "layered"


def _is_int_ancestor(ancestor: int, node: int, parent: Mapping[int, Optional[int]]) -> bool:
    current: Optional[int] = node
    while current is not None:
        if current == ancestor:
            return True
        current = parent[current]
    return False


def _attempt_task(task_name: str, rng: random.Random, difficulty: str | None, node_count: int | None = None) -> Optional[_Topology]:
    node_count = _difficulty_size(difficulty, rng) if node_count is None else node_count
    if task_name in {"shortest_path_choice", "route_via_checkpoint", "avoid_blocked_vertex"}:
        return _attempt_route(task_name, rng, node_count, difficulty)
    if task_name == "trace_all_edges_once":
        return _attempt_trace(rng, node_count, difficulty)
    if task_name == "euler_trail_endpoints":
        return _attempt_euler_endpoints(rng, node_count, difficulty)
    if task_name == "bridge_edge_choice":
        return _attempt_bridge(rng, node_count, difficulty)
    if task_name == "connected_component_count":
        return _attempt_components(rng, node_count, difficulty)
    if task_name == "degree_missing_value":
        return _attempt_degree(rng, node_count, difficulty)
    if task_name == "two_hop_neighbor_count":
        return _attempt_two_hop(rng, node_count, difficulty)
    if task_name == "bipartite_color_completion":
        return _attempt_bipartite(rng, node_count, difficulty)
    return _attempt_tree(task_name, rng, node_count, difficulty)


# ---------------------------------------------------------------------------
# Finalization: layout, numeric anti-shortcut labelling, and solution proof
# ---------------------------------------------------------------------------


def _build_source(
    task_name: str,
    node_count: int,
    edges: Sequence[tuple[int, int]],
    roles: Mapping[str, Any],
    node_attrs: Mapping[int, Mapping[str, Any]],
    positions: Sequence[Sequence[float]],
    labels: Sequence[str],
) -> Dict[str, Any]:
    nodes = []
    for index in range(node_count):
        node = {"label": labels[index], "position": [float(positions[index][0]), float(positions[index][1])]}
        if index in node_attrs:
            # Node attributes such as the hidden color or degree badge are
            # scalar observations, not references into the integer namespace.
            node.update(deepcopy(dict(node_attrs[index])))
        nodes.append(node)

    def convert(value: Any) -> Any:
        if isinstance(value, int) and 0 <= value < node_count:
            return labels[value]
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return {
        "schema_version": SCHEMA_VERSION,
        "task_name": task_name,
        "directed": False,
        "nodes": nodes,
        "edges": [{"u": labels[left], "v": labels[right]} for left, right in edges],
        "roles": {str(key): convert(value) for key, value in roles.items()},
        "params": {},
    }


def _solution_publishable(task_name: str, solved: Mapping[str, Any]) -> bool:
    if not solved.get("valid"):
        return False
    if task_name == "trace_all_edges_once":
        # Multi-certificate task: generation must prove a branched instance
        # with at least two distinct legal trails, never a degenerate chain.
        return len(solved.get("solutions") or []) >= 2
    return bool(solved.get("unique"))


def _finalize(task_name: str, rng: random.Random, built: _Topology) -> Optional[Dict[str, Any]]:
    node_count, edges, roles, node_attrs, layout_kind = built
    positions = None
    for _ in range(8):
        if layout_kind == "layered":
            positions = layout_layered_tree(node_count, edges, int(roles.get("root", 0)))
            if positions is not None and rng.random() < 0.5:
                # The tidy layout walks children in ascending index order, and
                # low-index nodes of a random recursive tree carry larger
                # subtrees, so unmirrored trees would lean left.  A seeded
                # mirror keeps left/right outcomes symmetric.
                positions = [(round(512.0 - x, 1), y) for x, y in positions]
        elif layout_kind == "components":
            positions = layout_components(rng, node_count, edges)
        else:
            positions = layout_force_directed(rng, node_count, edges)
        if positions is not None:
            break
    if positions is None:
        return None
    for _ in range(MAX_LABEL_ATTEMPTS):
        permutation = list(range(1, node_count + 1))
        rng.shuffle(permutation)
        labels = [str(value) for value in permutation]
        source = _build_source(task_name, node_count, edges, roles, node_attrs, positions, labels)
        solved = solve_graph_source(task_name, source)
        if not _solution_publishable(task_name, solved):
            # Uniqueness and trail multiplicity are topology properties, so a
            # relabelling cannot repair them; resample topology instead.
            return None
        index_of = {labels[node]: node for node in range(node_count)}
        answer_paths = [
            [index_of[str(value)] for value in path]
            for path in (solved.get("paths") or [])
        ]
        if labelling_acceptable(permutation, edges, answer_paths):
            return source
    return None


def generate_graph_source(
    task_name: str,
    rng: random.Random,
    difficulty: str | None = None,
    *,
    node_count: int | None = None,
    regime: str | None = None,
    minimum_answer: int | None = None,
) -> Dict[str, Any]:
    if task_name not in NEW_GRAPH_TASK_NAMES:
        raise ValueError(task_name)
    if node_count is not None and not 6 <= int(node_count) <= 18:
        raise ValueError("node_count must lie in 6..18")
    # Larger graphs retain the same topology distribution and rejection rules.
    # Their valid planar / two-odd-degree cases are rarer, so increase only
    # the explicit extrapolation rejection budget.
    attempts = MAX_TOPOLOGY_ATTEMPTS * (10 if node_count is not None and node_count > 10 else 1)
    for _ in range(attempts):
        if regime == "unequal_branches" and task_name == "balance_comparison":
            n = 9
            # Same rooted-tree objective, a 2-versus-6 split outside the
            # training absolute imbalance <= 2, with both branches nontrivial.
            tail = list(range(3, 9))
            rng.shuffle(tail)
            edges = [(0, 1), (1, 2), (0, tail[0])]
            edges += [(tail[rng.randrange(i)], tail[i]) for i in range(1, len(tail))]
            built = (n, edges, {"root": 0}, {}, "layered")
        else:
            built = _attempt_task(task_name, rng, difficulty, node_count)
        if (
            built is not None
            and regime == "ancestor_query"
            and task_name == "closest_common_ancestor"
        ):
            n, edges, roles, attrs, layout = built
            parent, depth, _ = _tree_relations(n, edges, roles["root"])
            descendant = rng.choice([v for v in range(n) if depth[v] >= 2])
            roles["queries"] = [parent[descendant], descendant]
        if built is None:
            continue
        if minimum_answer is not None:
            n, edges, roles, attrs, layout = built
            parent, depth, children = _tree_relations(n, edges, roles["root"])
            if task_name == "leaf_count":
                if (
                    sum(not children[v] for v in range(n) if v != roles["root"])
                    < minimum_answer
                ):
                    continue
            elif task_name == "path_between_nodes":
                distances = all_pairs_distances(n, edges)
                pairs = [
                    (a, b)
                    for a in range(n)
                    for b in range(a + 1, n)
                    if distances[a][b] >= minimum_answer
                ]
                if not pairs:
                    continue
                roles["query_start"], roles["query_goal"] = rng.choice(pairs)
            else:
                raise ValueError(
                    "minimum_answer is defined only for leaf count and tree path"
                )
        source = _finalize(task_name, rng, built)
        if source is not None:
            return source
    raise ValueError(f"{task_name}: bounded generation attempts exhausted")


# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------


def parse_graph_source(task_name: str, raw: Mapping[str, Any]) -> Dict[str, Any]:
    if task_name not in NEW_GRAPH_TASK_NAMES or not isinstance(raw, Mapping):
        raise ValueError("invalid task/source")
    state = raw.get("semantic_state") if isinstance(raw.get("semantic_state"), Mapping) else raw
    if not isinstance(state, Mapping) or state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("missing unified graph source v2 schema")
    nodes_raw = state.get("nodes")
    edges_raw = state.get("edges")
    if (
        not isinstance(nodes_raw, list)
        or not (2 <= len(nodes_raw) <= 18)
        or not isinstance(edges_raw, list)
    ):
        raise ValueError("invalid graph containers")
    nodes: list[dict[str, Any]] = []
    labels: set[str] = set()
    for raw_node in nodes_raw:
        if not isinstance(raw_node, Mapping):
            raise ValueError("invalid node")
        label = str(raw_node.get("label") or "").strip()
        position = raw_node.get("position")
        if not _LABEL_PATTERN.fullmatch(label) or label in labels or not isinstance(position, (list, tuple)) or len(position) != 2:
            raise ValueError("invalid node label/position")
        x, y = float(position[0]), float(position[1])
        if not (30.0 <= x <= 482.0 and 30.0 <= y <= 482.0):
            raise ValueError("node outside canvas")
        labels.add(label)
        node = {"label": label, "position": [x, y]}
        for key in ("bipartite_color", "degree_badge"):
            if key in raw_node:
                node[key] = raw_node[key]
        nodes.append(node)

    edges: list[dict[str, Any]] = []
    seen: set[Edge] = set()
    for item in edges_raw:
        if not isinstance(item, Mapping):
            raise ValueError("invalid edge")
        u, v = str(item.get("u") or ""), str(item.get("v") or "")
        key = edge_key(u, v)
        if u not in labels or v not in labels or u == v or key in seen:
            raise ValueError("invalid/duplicate edge")
        seen.add(key)
        edges.append({"u": u, "v": v})

    roles = deepcopy(dict(state.get("roles") or {}))
    params = deepcopy(dict(state.get("params") or {}))

    def labels_in(value: Any) -> Iterable[str]:
        if isinstance(value, str) and _LABEL_PATTERN.fullmatch(value):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from labels_in(item)
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from labels_in(item)

    if any(label not in labels for label in labels_in(roles)):
        raise ValueError("role references unknown node")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_name": task_name,
        "directed": False,
        "nodes": nodes,
        "edges": edges,
        "roles": roles,
        "params": params,
    }


# ---------------------------------------------------------------------------
# Independent solver
# ---------------------------------------------------------------------------


def _labels(source: Mapping[str, Any]) -> list[str]:
    return [str(node["label"]) for node in source["nodes"]]


def _edge_map(source: Mapping[str, Any]) -> Dict[Edge, dict[str, Any]]:
    return {edge_key(edge["u"], edge["v"]): dict(edge) for edge in source["edges"]}


def _adjacency(labels: Sequence[str], edges: Iterable[Edge]) -> Dict[str, list[str]]:
    adjacency = {label: [] for label in labels}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for label in adjacency:
        adjacency[label] = sorted(set(adjacency[label]), key=_label_sort_key)
    return adjacency


def _connected_components(labels: Sequence[str], edges: Iterable[Edge]) -> list[list[str]]:
    adjacency = _adjacency(labels, edges)
    remaining = set(labels)
    components: list[list[str]] = []
    while remaining:
        start = min(remaining, key=_label_sort_key)
        seen = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        remaining -= seen
        components.append(sorted(seen, key=_label_sort_key))
    return sorted(components, key=lambda values: _label_sort_key(values[0]))


def _simple_paths(labels: Sequence[str], edges: Iterable[Edge], start: str, goal: str) -> list[list[str]]:
    adjacency = _adjacency(labels, edges)
    paths: list[list[str]] = []

    def visit(node: str, path: list[str]) -> None:
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


def _path_between(labels: Sequence[str], edges: Iterable[Edge], start: str, goal: str) -> Optional[list[str]]:
    paths = _simple_paths(labels, edges, start, goal)
    if not paths:
        return None
    return min(paths, key=lambda path: (len(path), [_label_sort_key(value) for value in path]))


def _euler_trails(labels: Sequence[str], edges: Sequence[Edge], start: str, goal: str, limit: int = 4) -> list[list[str]]:
    adjacency: Dict[str, list[tuple[str, int]]] = {label: [] for label in labels}
    for index, (left, right) in enumerate(edges):
        adjacency[left].append((right, index))
        adjacency[right].append((left, index))
    trails: list[list[str]] = []
    used: set[int] = set()

    def visit(node: str, path: list[str]) -> None:
        if len(trails) >= limit:
            return
        if len(used) == len(edges):
            if node == goal:
                trails.append(list(path))
            return
        for neighbor, index in sorted(adjacency[node], key=lambda item: (_label_sort_key(item[0]), item[1])):
            if index in used:
                continue
            used.add(index)
            path.append(neighbor)
            visit(neighbor, path)
            path.pop()
            used.remove(index)

    visit(start, [start])
    return trails


def _tree_parent_depth(source: Mapping[str, Any]) -> tuple[str, Dict[str, Optional[str]], Dict[str, int]]:
    labels = _labels(source)
    edges = list(_edge_map(source))
    root = str(source["roles"].get("root") or "")
    if root not in labels or len(edges) != len(labels) - 1 or len(_connected_components(labels, edges)) != 1:
        raise ValueError("source is not one rooted tree")
    adjacency = _adjacency(labels, edges)
    parent: Dict[str, Optional[str]] = {root: None}
    depth = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in parent:
                parent[neighbor] = node
                depth[neighbor] = depth[node] + 1
                queue.append(neighbor)
    return root, parent, depth


def _unique_solution(answer: Any, **overlays: Any) -> Dict[str, Any]:
    return {"valid": True, "unique": True, "answer": answer, **overlays}


def solve_graph_source(task_name: str, raw_source: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        source = parse_graph_source(task_name, raw_source)
        labels = _labels(source)
        edge_map = _edge_map(source)
        edges = list(edge_map)
        roles = source["roles"]

        if task_name in {"shortest_path_choice", "route_via_checkpoint", "avoid_blocked_vertex"}:
            start, goal = str(roles["start"]), str(roles["goal"])
            all_paths = _simple_paths(labels, edges, start, goal)
            if not all_paths:
                return {"valid": False, "unique": False}
            unconstrained_edges = min(len(path) for path in all_paths) - 1
            paths = all_paths
            if task_name == "route_via_checkpoint":
                paths = [path for path in all_paths if str(roles["checkpoint"]) in path]
                if not paths:
                    return {"valid": False, "unique": False}
            elif task_name == "avoid_blocked_vertex":
                paths = [path for path in all_paths if str(roles["blocked"]) not in path]
                if not paths:
                    return {"valid": False, "unique": False}
            best_value = min(map(len, paths))
            winners = [path for path in paths if len(path) == best_value]
            if len(winners) != 1:
                return {"valid": True, "unique": False, "solutions": winners}
            if task_name == "route_via_checkpoint":
                return _unique_solution(
                    winners[0],
                    paths=[winners[0]],
                    nodes=winners[0],
                    support={
                        "checkpoint": str(roles["checkpoint"]),
                        "unconstrained_edges": unconstrained_edges,
                    },
                )
            if task_name == "avoid_blocked_vertex":
                shortcut = min(
                    (path for path in all_paths if len(path) - 1 == unconstrained_edges),
                    key=lambda path: [_label_sort_key(value) for value in path],
                )
                return _unique_solution(
                    winners[0],
                    paths=[winners[0]],
                    nodes=winners[0],
                    support={"blocked_shortcut": shortcut, "unconstrained_edges": unconstrained_edges},
                )
            runner_up = sorted({len(path) - 1 for path in all_paths} - {best_value - 1})
            return _unique_solution(
                winners[0],
                paths=[winners[0]],
                nodes=winners[0],
                support={"runner_up_edges": runner_up[0] if runner_up else None},
            )

        if task_name == "trace_all_edges_once":
            start, goal = str(roles["start"]), str(roles["goal"])
            trails = _euler_trails(labels, edges, start, goal, limit=4)
            if not trails:
                return {"valid": False, "unique": False}
            canonical = min(trails, key=lambda trail: [_label_sort_key(value) for value in trail])
            return {
                "valid": True,
                "unique": len(trails) == 1,
                "multi_certificate": True,
                "answer": canonical,
                "solutions": trails,
                "paths": [canonical],
                "nodes": canonical,
            }

        if task_name == "euler_trail_endpoints":
            degree = {label: 0 for label in labels}
            for left, right in edges:
                degree[left] += 1
                degree[right] += 1
            odd = sorted((label for label, value in degree.items() if value % 2 == 1), key=_label_sort_key)
            if len(_connected_components(labels, edges)) != 1 or len(odd) != 2:
                return {"valid": False, "unique": False}
            return _unique_solution(odd, nodes=odd, support={"degrees": degree})

        if task_name == "connected_component_count":
            components = _connected_components(labels, edges)
            return _unique_solution(len(components), components=components, nodes=labels)

        if task_name == "bridge_edge_choice":
            initial_components = len(_connected_components(labels, edges))
            bridges = [
                edge for edge in edges
                if len(_connected_components(labels, [other for other in edges if other != edge])) > initial_components
            ]
            if len(bridges) != 1:
                return {"valid": True, "unique": False, "solutions": bridges}
            sides = _connected_components(labels, [other for other in edges if other != bridges[0]])
            return _unique_solution(
                list(bridges[0]),
                edges=bridges,
                nodes=list(bridges[0]),
                support={"sides": sides},
            )

        if task_name == "diameter_endpoints":
            # Diameter is an unrooted-tree property, so it stays independent
            # of the rooted-tree helpers by design.
            distances: list[tuple[int, list[str]]] = []
            for index, left in enumerate(labels):
                for right in labels[index + 1 :]:
                    path = _path_between(labels, edges, left, right)
                    if path:
                        distances.append((len(path) - 1, path))
            best = max(distance for distance, _ in distances)
            winners = [path for distance, path in distances if distance == best]
            pairs = sorted({edge_key(path[0], path[-1]) for path in winners}, key=lambda pair: (_label_sort_key(pair[0]), _label_sort_key(pair[1])))
            if len(pairs) != 1:
                return {"valid": True, "unique": False, "solutions": [list(pair) for pair in pairs]}
            path = next(
                path
                for distance, path in distances
                if distance == best and edge_key(path[0], path[-1]) == pairs[0]
            )
            runner_up = max(
                ((distance, other) for distance, other in distances if edge_key(other[0], other[-1]) != pairs[0]),
                key=lambda item: (item[0], [_label_sort_key(value) for value in item[1]]),
                default=None,
            )
            return _unique_solution(
                list(pairs[0]),
                paths=[path],
                nodes=list(pairs[0]),
                support={
                    "diameter_path": path,
                    "runner_up_edges": runner_up[0] if runner_up else None,
                    "runner_up_pair": list(edge_key(runner_up[1][0], runner_up[1][-1])) if runner_up else None,
                },
            )

        if task_name in {
            "leaf_count", "depth_lookup", "path_between_nodes", "closest_common_ancestor", "subtree_size",
            "balance_comparison",
        }:
            root, parent, depth = _tree_parent_depth(source)
            adjacency = _adjacency(labels, edges)
            children = {label: sorted((neighbor for neighbor in adjacency[label] if parent.get(neighbor) == label), key=_label_sort_key) for label in labels}
            if task_name == "leaf_count":
                leaves = sorted((label for label in labels if label != root and not children[label]), key=_label_sort_key)
                return _unique_solution(len(leaves), nodes=leaves, support={"leaves": leaves})
            if task_name == "depth_lookup":
                query = str(roles["query"])
                path: list[str] = []
                node: Optional[str] = query
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return _unique_solution(depth[query], paths=[path], nodes=path, support={"path": path})
            if task_name == "path_between_nodes":
                path = _path_between(labels, edges, str(roles["query_start"]), str(roles["query_goal"]))
                if path is None:
                    return {"valid": False, "unique": False}
                apex = min(path, key=lambda label: depth[label])
                return _unique_solution(
                    path,
                    paths=[path],
                    nodes=path,
                    support={"apex": apex, "apex_depth": depth[apex]},
                )
            if task_name == "closest_common_ancestor":
                first, second = [str(value) for value in roles["queries"]]
                ancestors_first: list[str] = []
                node = first
                while node is not None:
                    ancestors_first.append(node)
                    node = parent[node]
                ancestors_second: list[str] = []
                node = second
                while node is not None:
                    ancestors_second.append(node)
                    node = parent[node]
                common = set(ancestors_first).intersection(ancestors_second)
                answer = max(common, key=lambda label: depth[label])
                path1 = list(reversed(ancestors_first))
                path2 = list(reversed(ancestors_second))
                return _unique_solution(
                    answer,
                    paths=[path1, path2],
                    nodes=[answer],
                    support={
                        "ancestor_paths": [path1, path2],
                        "shared": sorted(common, key=_label_sort_key),
                        "answer_depth": depth[answer],
                    },
                )
            if task_name == "subtree_size":
                query = str(roles["query"])
                descendants = sorted((label for label in labels if label == query or _is_ancestor(query, label, parent)), key=_label_sort_key)
                levels: list[list[str]] = []
                frontier = [query]
                while frontier:
                    next_frontier = sorted(
                        (child for label in frontier for child in children[label]),
                        key=_label_sort_key,
                    )
                    if not next_frontier:
                        break
                    levels.append(next_frontier)
                    frontier = next_frontier
                return _unique_solution(
                    len(descendants),
                    nodes=descendants,
                    support={"subtree": descendants, "levels": levels},
                )
            if task_name == "balance_comparison":
                node_positions = {node["label"]: node["position"] for node in source["nodes"]}
                root_children = children[root]
                if len(root_children) != 2:
                    return {"valid": False, "unique": False}
                left, right = sorted(root_children, key=lambda label: node_positions[label][0])
                left_nodes = sorted((label for label in labels if label == left or _is_ancestor(left, label, parent)), key=_label_sort_key)
                right_nodes = sorted((label for label in labels if label == right or _is_ancestor(right, label, parent)), key=_label_sort_key)
                answer = "LEFT" if len(left_nodes) > len(right_nodes) else ("RIGHT" if len(right_nodes) > len(left_nodes) else "EQUAL")
                return _unique_solution(
                    answer,
                    components=[left_nodes, right_nodes],
                    nodes=left_nodes + right_nodes,
                    support={
                        "left": left_nodes,
                        "right": right_nodes,
                        "left_child": left,
                        "right_child": right,
                        "root": root,
                    },
                )

        if task_name == "bipartite_color_completion":
            query = str(roles["query"])
            known = {str(node["label"]): node.get("bipartite_color") for node in source["nodes"]}
            adjacency = _adjacency(labels, edges)
            colors: Dict[str, str] = {}
            seeds = [(label, str(color)) for label, color in known.items() if color in {"RED", "BLUE"}]
            if not seeds:
                return {"valid": False, "unique": False}
            for seed, color in seeds:
                if seed in colors and colors[seed] != color:
                    return {"valid": False, "unique": False}
                colors[seed] = color
            queue = deque(colors)
            while queue:
                node = queue.popleft()
                expected = "BLUE" if colors[node] == "RED" else "RED"
                for neighbor in adjacency[node]:
                    if neighbor in colors and colors[neighbor] != expected:
                        return {"valid": False, "unique": False}
                    if neighbor not in colors:
                        colors[neighbor] = expected
                        queue.append(neighbor)
            return _unique_solution(colors[query], nodes=[query], node_color={query: colors[query]}, support={"neighbor_colors": {n: colors[n] for n in adjacency[query]}})

        if task_name == "degree_missing_value":
            query = str(roles["query"])
            incident = sorted(edge for edge in edges if query in edge)
            return _unique_solution(len(incident), edges=incident, nodes=[query], support={"incident_edges": incident})

        if task_name == "two_hop_neighbor_count":
            query = str(roles["query"])
            adjacency = _adjacency(labels, edges)
            first = set(adjacency[query])
            second = {neighbor for node in first for neighbor in adjacency[node]} - first - {query}
            predecessors = {
                node: sorted((middle for middle in first if node in adjacency[middle]), key=_label_sort_key)
                for node in sorted(second, key=_label_sort_key)
            }
            return _unique_solution(
                len(second),
                nodes=sorted(second, key=_label_sort_key),
                components=[sorted(first, key=_label_sort_key), sorted(second, key=_label_sort_key)],
                support={
                    "one_hop": sorted(first, key=_label_sort_key),
                    "two_hop": sorted(second, key=_label_sort_key),
                    "two_hop_predecessors": predecessors,
                },
            )

        raise ValueError(f"unsupported task {task_name}")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {"valid": False, "unique": False, "error": str(exc)}


def _is_ancestor(ancestor: str, node: str, parent: Mapping[str, Optional[str]]) -> bool:
    current: Optional[str] = node
    while current is not None:
        if current == ancestor:
            return True
        current = parent[current]
    return False


# ---------------------------------------------------------------------------
# Typed final answers
# ---------------------------------------------------------------------------


def format_graph_answer(task_name: str, answer: Any) -> str:
    answer_type = get_unified_graph_spec(task_name).answer_type
    if answer_type in {"path", "edge"}:
        return format_string_list_answer(answer)
    if answer_type == "integer":
        return format_nonnegative_integer_answer(answer)
    return format_answer_line(str(answer).upper())


__all__ = [
    "SCHEMA_VERSION",
    "edge_key",
    "format_graph_answer",
    "generate_graph_source",
    "parse_graph_source",
    "solve_graph_source",
]

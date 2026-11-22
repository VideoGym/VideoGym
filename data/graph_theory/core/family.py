from __future__ import annotations

from data.task_spec import TaskFamily, TaskRoute
from data.common.verifiers.routes import VerifierRoute

_UNIFIED_VERIFIER = VerifierRoute('data.graph_theory.core.verifier', 'verify_unified_graph')


def _route(name: str, class_name: str, *, answer_kind: str) -> TaskRoute:
    return TaskRoute(
        name,
        f'data.graph_theory.{name}.generator',
        class_name,
        verifier=_UNIFIED_VERIFIER,
        answer_kind=answer_kind,
    )


FAMILY = TaskFamily(
    name='graph_theory',
    description='Trace connectivity, traversal, degree, legal paths, and hierarchy in dynamically generated numeric-labeled node-edge graphs or rooted trees.',
    tasks=(
        _route('shortest_path_choice', 'ShortestPathChoiceGenerator', answer_kind='string_list'),
        _route('route_via_checkpoint', 'RouteViaCheckpointGenerator', answer_kind='string_list'),
        _route('avoid_blocked_vertex', 'AvoidBlockedVertexGenerator', answer_kind='string_list'),
        _route('trace_all_edges_once', 'TraceAllEdgesOnceGenerator', answer_kind='string_list'),
        _route('euler_trail_endpoints', 'EulerTrailEndpointsGenerator', answer_kind='string_list'),
        _route('connected_component_count', 'ConnectedComponentCountGenerator', answer_kind='nonnegative_integer'),
        _route('degree_missing_value', 'DegreeMissingValueGenerator', answer_kind='nonnegative_integer'),
        _route('two_hop_neighbor_count', 'TwoHopNeighborCountGenerator', answer_kind='nonnegative_integer'),
        _route('bipartite_color_completion', 'BipartiteColorCompletionGenerator', answer_kind='typed_text'),
        _route('bridge_edge_choice', 'BridgeEdgeChoiceGenerator', answer_kind='string_list'),
        _route('leaf_count', 'LeafCountGenerator', answer_kind='nonnegative_integer'),
        _route('depth_lookup', 'DepthLookupGenerator', answer_kind='nonnegative_integer'),
        _route('path_between_nodes', 'PathBetweenNodesGenerator', answer_kind='string_list'),
        _route('closest_common_ancestor', 'ClosestCommonAncestorGenerator', answer_kind='typed_text'),
        _route('subtree_size', 'SubtreeSizeGenerator', answer_kind='nonnegative_integer'),
        _route('diameter_endpoints', 'DiameterEndpointsGenerator', answer_kind='string_list'),
        _route('balance_comparison', 'BalanceComparisonGenerator', answer_kind='typed_text'),
    ),
)

__all__ = ["FAMILY"]

"""Task-level contracts for the shared labelled-graph task engine.

The 16 tasks below share one visual language and one dynamic source schema.
Each specification defines the mathematical predicate, prompt prose, and
output-effect vocabulary while sharing the renderer and source-replay boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping



@dataclass(frozen=True)
class UnifiedGraphSpec:
    name: str
    class_name: str
    answer_type: str
    kind: str  # "graph" or "tree" — selects legend wording and layout family
    question: str
    input_description: str
    rules: tuple[str, ...]
    delivery: str
    terminal: str
    process: str
    moving_nouns: str


def _spec(
    name: str,
    class_name: str,
    *,
    answer_type: str,
    kind: str,
    question: str,
    input_description: str,
    rules: tuple[str, ...],
    delivery: str,
    terminal: str,
    process: str,
    moving_nouns: str,
) -> UnifiedGraphSpec:
    return UnifiedGraphSpec(
        name=name,
        class_name=class_name,
        answer_type=answer_type,
        kind=kind,
        question=question,
        input_description=input_description,
        rules=rules,
        delivery=delivery,
        terminal=terminal,
        process=process,
        moving_nouns=moving_nouns,
    )


_ROUTE_INPUT = (
    "The image shows one undirected graph with numbered nodes. The node with the green ring is the start "
    "and the node with the red ring is the goal."
)
_TREE_INPUT = (
    "The image shows one rooted tree. The green-ringed node at the top is the root, and every edge descends "
    "from a parent to a child."
)
_ROUTE_PROCESS = (
    "Draw the route as one orange line that starts at the green node and advances along one edge at a time, "
    "each new edge beginning only after the previous one is complete, until the line reaches the red node."
)


SPECS: Mapping[str, UnifiedGraphSpec] = {
    "shortest_path_choice": _spec(
        "shortest_path_choice", "ShortestPathChoiceGenerator", answer_type="path", kind="graph",
        question="What is the unique fewest-edge route from the green start to the red goal?",
        input_description=_ROUTE_INPUT,
        rules=(
            "A route walks from the start to the goal along drawn edges without visiting any node twice.",
            "Deliver the route with the fewest edges; this graph has exactly one such route.",
        ),
        delivery="State the shortest route as the ordered list of node labels from start to goal.",
        terminal="Show the input graph with the chosen route highlighted as one orange line running from the green node to the red node.",
        process=_ROUTE_PROCESS,
        moving_nouns="the growing orange route",
    ),
    "route_via_checkpoint": _spec(
        "route_via_checkpoint", "RouteViaCheckpointGenerator", answer_type="path", kind="graph",
        question="What is the unique fewest-edge start-to-goal route that visits the gold checkpoint?",
        input_description=_ROUTE_INPUT + " One further node carries a gold ring: the checkpoint.",
        rules=(
            "A route walks from the green start to the red goal along drawn edges without visiting any node twice, and it must pass through the gold-ringed checkpoint on the way.",
            "Among all such routes deliver the one with the fewest edges; it is unique in this graph.",
        ),
        delivery="State the full route as the ordered list of node labels from start to goal.",
        terminal="Show the input graph with the full route highlighted as one orange line from the green node through the gold checkpoint to the red node.",
        process=(
            "Draw the route as one orange line that starts at the green node and advances along one edge at a time, "
            "passing visibly through the gold checkpoint, until it reaches the red node."
        ),
        moving_nouns="the growing orange route",
    ),
    "avoid_blocked_vertex": _spec(
        "avoid_blocked_vertex", "AvoidBlockedVertexGenerator", answer_type="path", kind="graph",
        question="What is the unique fewest-edge route that never enters the gray blocked node?",
        input_description=_ROUTE_INPUT + " The one node filled solid gray is blocked.",
        rules=(
            "A route walks from the green start to the red goal along drawn edges without visiting any node twice, and it may never enter the gray blocked node.",
            "Among the legal routes deliver the one with the fewest edges; it is unique in this graph.",
        ),
        delivery="State the legal route as the ordered list of node labels from start to goal.",
        terminal="Show the input graph with the legal route highlighted as one orange line from the green node to the red node, leaving the gray node untouched.",
        process=_ROUTE_PROCESS,
        moving_nouns="the growing orange route",
    ),
    "trace_all_edges_once": _spec(
        "trace_all_edges_once", "TraceAllEdgesOnceGenerator", answer_type="path", kind="graph",
        question="Which node sequence walks every edge exactly once from the green node to the red node?",
        input_description=_ROUTE_INPUT,
        rules=(
            "Walk along drawn edges from node to node; every edge must be used exactly once, while nodes may be revisited.",
            "The sequence must begin at the green node and end at the red node; any sequence satisfying these conditions is accepted.",
        ),
        delivery="State one complete walk as the ordered list of node labels it visits.",
        terminal="Show the input graph with every edge highlighted orange, the finished trail resting at the red node.",
        process=(
            "Trace the walk as one orange line that starts at the green node and covers one edge at a time in your chosen "
            "order, never lifting or recoloring an edge twice, until every edge is orange and the line rests at the red node."
        ),
        moving_nouns="the growing orange trail",
    ),
    "euler_trail_endpoints": _spec(
        "euler_trail_endpoints", "EulerTrailEndpointsGenerator", answer_type="edge", kind="graph",
        question="Which two nodes must be the endpoints of a walk that uses every edge exactly once?",
        input_description="The image shows one connected undirected graph with numbered nodes.",
        rules=(
            "A walk that uses every edge exactly once and starts and ends at two different nodes must start and end at nodes touched by an odd number of edges.",
            "Exactly two nodes here have an odd edge count; report that pair.",
        ),
        delivery="State the two endpoint labels as a pair, smaller number first.",
        terminal="Show the input graph with one orange ring around each of the two endpoint nodes.",
        process="Fade in one orange ring around each of the two endpoint nodes, both complete rings appearing together.",
        moving_nouns="the appearing orange rings",
    ),
    "connected_component_count": _spec(
        "connected_component_count", "ConnectedComponentCountGenerator", answer_type="integer", kind="graph",
        question="How many separate connected groups does the graph contain?",
        input_description="The image shows one undirected graph with numbered nodes; parts of it may be separate islands with no edge between them.",
        rules=(
            "Two nodes belong to the same group exactly when some chain of drawn edges joins them.",
            "Count the groups; a node with no edges counts as a group of its own.",
        ),
        delivery="State how many groups there are.",
        terminal="Show the input graph with every node's interior tinted in its group's shared pastel color, a different color per group.",
        process="Tint the interior of every node with its group's shared pastel color — a different color per group — with all tints fading in together.",
        moving_nouns="the appearing group tints",
    ),
    "degree_missing_value": _spec(
        "degree_missing_value", "DegreeMissingValueGenerator", answer_type="integer", kind="graph",
        question="How many edges touch the purple-ringed node?",
        input_description=(
            "The image shows one undirected graph with numbered nodes. The purple-ringed node carries a small "
            "yellow badge holding a question mark: its edge count is the unknown."
        ),
        rules=(
            "The degree of a node is the number of drawn edges touching it; count each edge once.",
            "Determine the degree of the purple-ringed node.",
        ),
        delivery="State the degree of the purple-ringed node.",
        terminal="Show the input graph with every edge touching the purple-ringed node highlighted orange and the badge's question mark replaced by the degree digit.",
        process=(
            "Highlight every edge touching the purple-ringed node in orange, all fading in together; in the badge, "
            "the question mark fades away and then the degree digit fades in at the same spot."
        ),
        moving_nouns="the appearing orange edge highlights and the badge digit",
    ),
    "two_hop_neighbor_count": _spec(
        "two_hop_neighbor_count", "TwoHopNeighborCountGenerator", answer_type="integer", kind="graph",
        question="How many distinct nodes sit exactly two hops from the purple-ringed node?",
        input_description="The image shows one undirected graph with numbered nodes and one purple-ringed query node.",
        rules=(
            "One hop crosses one drawn edge; a node's hop distance is the smallest number of hops that reaches it from the purple-ringed node.",
            "Count the nodes whose hop distance is exactly two, counting each node once no matter how many ways it can be reached.",
        ),
        delivery="State how many nodes sit exactly two hops away.",
        terminal="Show the input graph with every direct neighbor of the purple-ringed node tinted pale teal and one orange ring around every node exactly two hops away.",
        process=(
            "First tint the interior of every direct neighbor of the purple-ringed node pale teal, all together; "
            "keep those tints, then fade in one orange ring around every node exactly two hops away, all rings together."
        ),
        moving_nouns="the appearing teal tints and orange rings",
    ),
    "bipartite_color_completion": _spec(
        "bipartite_color_completion", "BipartiteColorCompletionGenerator", answer_type="color", kind="graph",
        question="Which color must the uncolored node take: RED or BLUE?",
        input_description="The image shows one connected graph whose nodes are filled red or blue, except a single white node whose color is missing.",
        rules=(
            "Every edge joins a red node with a blue node.",
            "The visible colors are consistent and force exactly one color for the white node.",
        ),
        delivery="State the forced color, RED or BLUE.",
        terminal="Show the input graph with the white node now filled in its forced color; every other node keeps its color.",
        process="Fade the white node's interior into its forced fill color while every other node and edge stays untouched.",
        moving_nouns="the white node's appearing fill color",
    ),
    "bridge_edge_choice": _spec(
        "bridge_edge_choice", "BridgeEdgeChoiceGenerator", answer_type="edge", kind="graph",
        question="Which single edge would disconnect the graph if removed?",
        input_description="The image shows one connected undirected graph with numbered nodes.",
        rules=(
            "An edge is a bridge when removing it splits the graph into two disconnected parts.",
            "Exactly one drawn edge here is a bridge; name it by its two endpoint labels.",
        ),
        delivery="State the bridge as its two endpoint labels, smaller number first.",
        terminal="Show the input graph with the bridge edge highlighted by one thicker orange stroke along its full length.",
        process="Fade one thicker orange stroke in over the bridge edge's full length while everything else stays untouched.",
        moving_nouns="the appearing orange edge highlight",
    ),
    "leaf_count": _spec(
        "leaf_count", "LeafCountGenerator", answer_type="integer", kind="tree",
        question="How many leaves does the rooted tree have?",
        input_description=_TREE_INPUT,
        rules=(
            "A leaf is a non-root node with no child below it.",
            "Count every leaf in the tree.",
        ),
        delivery="State how many leaves the tree has.",
        terminal="Show the input tree with one orange ring around every leaf.",
        process="Fade in one orange ring around every leaf, all complete rings appearing together.",
        moving_nouns="the appearing orange rings",
    ),
    "depth_lookup": _spec(
        "depth_lookup", "DepthLookupGenerator", answer_type="integer", kind="tree",
        question="At what depth does the purple-ringed node sit when the root has depth 0?",
        input_description=_TREE_INPUT + " The purple-ringed node is the query.",
        rules=(
            "The root sits at depth 0, and each step from a parent down to its child adds 1.",
            "Give the depth of the purple-ringed node.",
        ),
        delivery="State the depth of the purple-ringed node.",
        terminal="Show the input tree with the chain from the root down to the purple-ringed node highlighted as one orange line.",
        process=(
            "Draw one orange line downward from the root along the chain of parent-child edges, one edge at a time, "
            "stopping at the purple-ringed node."
        ),
        moving_nouns="the growing orange chain",
    ),
    "path_between_nodes": _spec(
        "path_between_nodes", "PathBetweenNodesGenerator", answer_type="path", kind="tree",
        question="What is the unique route from the blue-ringed node to the red-ringed node?",
        input_description=(
            "The image shows one tree whose edges descend from parents to children. "
            "The blue-ringed node and the red-ringed node are the two endpoints to connect."
        ),
        rules=(
            "Consecutive nodes of the route must share a drawn edge; edges may be walked upward or downward.",
            "A tree holds exactly one route between the two marked nodes; deliver it from the blue-ringed node to the red-ringed node.",
        ),
        delivery="State the route as the ordered list of node labels, starting at the blue-ringed node and ending at the red-ringed one.",
        terminal="Show the input tree with the connecting route highlighted as one orange line between the blue-ringed and red-ringed nodes.",
        process=(
            "Draw the route as one orange line that starts at the blue-ringed node and advances along one edge at a time "
            "until it reaches the red-ringed node."
        ),
        moving_nouns="the growing orange route",
    ),
    "closest_common_ancestor": _spec(
        "closest_common_ancestor", "ClosestCommonAncestorGenerator", answer_type="label", kind="tree",
        question="Which node is the deepest common ancestor of the two purple-ringed nodes?",
        input_description=_TREE_INPUT + " Two nodes carry purple rings: the queries.",
        rules=(
            "The ancestors of a node are the node itself plus every node on its upward chain to the root.",
            "Deliver the node that is an ancestor of both purple-ringed queries and sits deepest, farthest from the root.",
        ),
        delivery="State the label of the deepest common ancestor.",
        terminal="Show the input tree with both queries' upward chains to the root highlighted as teal lines and one orange ring around the deepest node the chains share.",
        process=(
            "Fade both purple-ringed nodes' complete upward chains to the root in together as teal lines; "
            "then fade in one orange ring around the deepest node the two chains share."
        ),
        moving_nouns="the appearing teal chains and orange ring",
    ),
    "subtree_size": _spec(
        "subtree_size", "SubtreeSizeGenerator", answer_type="integer", kind="tree",
        question="How many nodes does the subtree of the purple-ringed node contain, that node included?",
        input_description=_TREE_INPUT + " The purple-ringed node is the root of the subtree in question.",
        rules=(
            "The subtree of the purple-ringed node contains that node plus every node reachable from it by moving only downward.",
            "Count the nodes of that subtree, the purple-ringed node included.",
        ),
        delivery="State how many nodes the subtree contains.",
        terminal="Show the input tree with one orange ring around every subtree node — the purple-ringed node and all its descendants.",
        process="Fade in one orange ring around every node of the subtree — the purple-ringed node and all its descendants — all rings together.",
        moving_nouns="the appearing orange rings",
    ),
    "diameter_endpoints": _spec(
        "diameter_endpoints", "DiameterEndpointsGenerator", answer_type="edge", kind="graph",
        question="Which two nodes are joined by the longest route in the tree?",
        input_description="The image shows one unrooted tree: all numbered nodes are connected and no cycle exists.",
        rules=(
            "The length of a route is its number of edges, and a tree holds exactly one route between any two nodes.",
            "Find the two nodes whose connecting route is the longest in the whole tree; this tree has one unique such pair.",
        ),
        delivery="State that pair of node labels, smaller number first.",
        terminal="Show the input tree with the longest route highlighted as one teal line and one orange ring around each of its two end nodes.",
        process=(
            "Fade the complete longest route in as one teal line; then fade in one orange ring around each of its two "
            "end nodes, both rings together."
        ),
        moving_nouns="the appearing teal route and orange rings",
    ),
    "balance_comparison": _spec(
        "balance_comparison", "BalanceComparisonGenerator", answer_type="token", kind="tree",
        question="Which side of the root holds more nodes: LEFT, RIGHT, or EQUAL?",
        input_description=(
            "The image shows one rooted tree whose green-ringed root has exactly two child branches, "
            "one descending to the left and one to the right."
        ),
        rules=(
            "Every node below the root belongs to the branch of the root child it hangs from.",
            "Compare how many nodes the left and right branches contain, the two root children included: report LEFT if the left branch has more, RIGHT if the right branch has more, or EQUAL for a tie.",
        ),
        delivery="State LEFT, RIGHT, or EQUAL.",
        terminal=(
            "Show the input tree with the left branch's node interiors tinted pale blue, the right branch's tinted "
            "pale orange, and one orange ring around every node of the larger branch — around both branches on a tie."
        ),
        process=(
            "Tint the left branch's node interiors pale blue and the right branch's pale orange, all tints fading in "
            "together; then fade in one orange ring around every node of the larger branch, or around both branches on a tie."
        ),
        moving_nouns="the appearing branch tints and orange rings",
    ),
}


NEW_GRAPH_TASK_NAMES = tuple(SPECS)


# one nameable cumulative evidence state that later reasoning causally builds
# on.  `text_cot` tasks have an empty tuple — their
# think block stays text-only and the terminal image is the sole delivery
# image.  The mapping is cross-checked against the registry mode at access.




def get_unified_graph_spec(task_name: str) -> UnifiedGraphSpec:
    try:
        return SPECS[task_name]
    except KeyError as exc:
        raise ValueError(f"unknown unified Graph Theory task: {task_name!r}") from exc


__all__ = [
    "NEW_GRAPH_TASK_NAMES",
    "SPECS",
    "UnifiedGraphSpec",
    "get_unified_graph_spec",
]

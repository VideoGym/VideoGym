TASK_NAME = 'two_hop_neighbor_count'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Tree-plus-chords graph with a purple-ringed query. Candidate queries need >= 2 exact-distance-2 vertices, and (above min difficulty) at least one two-hop vertex reachable through two different one-hop neighbors.',
    "design_intent": 'Layered-expansion member of the unified engine; its merge requirement makes deduplication the load-bearing skill.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

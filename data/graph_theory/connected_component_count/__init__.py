TASK_NAME = 'connected_component_count'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": '2-4 components sampled with sizes summing to 6-10 nodes (singletons allowed, larger islands get trees plus optional chords). The solver partitions by reachability and counts; the count is the typed answer.',
    "design_intent": "Reachability-partition member of the unified engine with the family's group-tint evidence language and packed component layout.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

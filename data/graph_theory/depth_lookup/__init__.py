TASK_NAME = 'depth_lookup'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A rooted tree whose purple-ringed query sits at depth >= 2 (>= 3 at max difficulty). Depth counts parent-child edges from the root at depth 0 (an explicit measure stated in the prompt, not an index).',
    "design_intent": 'Root-to-query chain readout in the unified labelled-graph family, isolating tree depth counting along a unique ancestor path.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'subtree_size'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A rooted tree whose purple-ringed non-root query has a subtree of >= 3 nodes (>= 4 at max, >= 2 at min); the closure count includes the query itself.',
    "design_intent": 'Closure-counting member of the unified engine; its generator filter counts the closure itself so leaf queries can never slip through.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'path_between_nodes'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A rooted tree with blue-ringed and red-ringed query endpoints >= 2 edges apart (>= 3 at max difficulty); 40% of samples navigate root-to-deep-node. The unique connecting path, delivered blue-to-red, is the answer.',
    "design_intent": 'Tree-navigation member of the unified engine covering upward-and-downward edge walking under a fixed delivery direction.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

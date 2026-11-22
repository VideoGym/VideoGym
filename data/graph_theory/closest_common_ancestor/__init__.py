TASK_NAME = 'closest_common_ancestor'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A rooted tree with two purple-ringed queries whose deepest common ancestor is neither query (and not the root at max difficulty). Ancestors include the node itself; the deepest shared one is the answer.',
    "design_intent": 'Find the closest common ancestor through chain intersection, with two successive local fades exposing the chains and selected ancestor.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

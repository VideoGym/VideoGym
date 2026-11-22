TASK_NAME = 'route_via_checkpoint'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Same route source as shortest_path_choice plus one gold checkpoint role. Generation requires the checkpointed optimum to be unique and strictly longer than the unconstrained optimum, keeping the constraint load-bearing.',
    "design_intent": 'Constrained shortest-route search through a mandatory checkpoint node whose optimal simple path is strictly longer than the unconstrained shortest path.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

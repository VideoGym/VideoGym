TASK_NAME = 'grid_bfs'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Open cells of a 4-6 grid receive the shortest 4-directional distance to any source cell; walls block movement and walls plus unreachable open cells output -1. Sampling requires a maximum distance of 3-6 and walls that provably change at least one distance (obstacle ablation non-empty).',
    "design_intent": 'Obstacle-aware multi-source shortest distance with unreachability.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

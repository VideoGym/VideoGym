TASK_NAME = 'binary_matrix'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Every cell of a 4-6 binary matrix receives the Manhattan distance to its nearest 0 under orthogonal steps; all 0s are simultaneous sources. Sampling requires a maximum distance of at least 3 and at least two zero sources that each uniquely determine some cell's distance.",
    "design_intent": 'Multi-source Manhattan distance with per-sample influential sources.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'cyclic_threshold_wave'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A 5-7 lattice of 3-5 cyclic colour states advances a cell to its successor state exactly when enough old-state neighbours (orthogonal or Moore, threshold 1-3) already hold that successor; 2 or 3 synchronous steps. Sampling rejects cases whose alternate threshold or alternate neighbourhood would reproduce the same terminal grid, so both parameters are load-bearing.',
    "design_intent": 'Cyclic successor adoption is neither binary survival nor unordered majority.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

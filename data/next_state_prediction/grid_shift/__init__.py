TASK_NAME = 'grid_shift'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Six numbered tokens on a fixed 4x4 grid translate one cell in the direction shown by the arrow badge; tokens shifted past the boundary disappear (1-2 per sample), the rest keep identity.',
    "design_intent": 'Identity-preserving translation with boundary loss, not wraparound.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

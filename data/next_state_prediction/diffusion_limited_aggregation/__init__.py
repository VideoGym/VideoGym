TASK_NAME = 'diffusion_limited_aggregation'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'For one tick on a wrapped 8-10 grid, every walker orthogonally adjacent to the old aggregate freezes into it first; remaining walkers propose one cell each, proposals into the expanded aggregate wait, shared destinations move only the lowest ID, and distinct destinations move simultaneously. Sampling draws a random 2-3 cell aggregate and 4-6 walkers and rejects ticks lacking a freeze, a multi-walker conflict, and an unaffected mover.',
    "design_intent": 'It grows a persistent structure rather than recolouring every cell.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

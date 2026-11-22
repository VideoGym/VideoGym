TASK_NAME = 'sandpile_avalanche'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A stable 5-6 height grid receives one grain on a named edge-adjacent cell; then, while any cell holds at least 4, all unstable cells topple together in one synchronous wave (each loses 4, sends 1 to each orthogonal neighbour, off-grid grains are permanently lost).',
    "design_intent": 'The terminal state is a fixed point rather than a fixed tick count.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

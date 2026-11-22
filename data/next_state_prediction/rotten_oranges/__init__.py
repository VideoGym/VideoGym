TASK_NAME = 'rotten_oranges'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Fresh oranges rot when orthogonally adjacent to an old rotten orange, one simultaneous wave per minute; the answer is the first minute with no fresh orange, or -1 when some fresh orange is unreachable. Sampling requires the last reachable minute to land in 2-7.',
    "design_intent": 'Expiration-free infection isolates pure front propagation plus reachability.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

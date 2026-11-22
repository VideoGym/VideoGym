TASK_NAME = 'parking_lot_swap_minimal_moves'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A single row of 5-6 slots holds numbered cars and one empty bay. A move drives a car adjacent to the bay into it, leaving its old slot empty. Sampling shuffles initial and target rows and keeps BFS optima of 3..8 moves. The payload lists moved car identities, so no index conversion exists.',
    "design_intent": 'The minimal one-bay corridor world: plans are forced bay walks, training precise adjacency bookkeeping rather than search breadth.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

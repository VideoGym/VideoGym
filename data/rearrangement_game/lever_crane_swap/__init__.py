TASK_NAME = 'lever_crane_swap'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Two stacks hold 3-5 numbered blocks under a lever crane with a one-block buffer. Moves are PL/PR (pick a side's top into the empty buffer) and DL/DR (drop the buffer block onto a side). Successive picks must alternate sides. Sampling requires the alternation-respecting optimum to be strictly longer than the unrestricted optimum, drawing balanced targets of 6, 10, or 14 moves via breadth-first search from the start state. Any legal replay reaching the target at the recomputed optimal length is accepted.",
    "design_intent": 'Alternation-constrained buffer shuttling: the same-side pick ban forces side-switching plans that an unrestricted crane would not need.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

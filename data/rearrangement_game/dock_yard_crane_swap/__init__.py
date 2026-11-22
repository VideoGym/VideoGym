TASK_NAME = 'dock_yard_crane_swap'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Three lanes hold 4-6 numbered boxes. Crane L lifts the top PAIR and lands it flipped; crane R lifts a single top box unflipped. Sampling requires 3..7-move optima where neither crane alone reaches the optimum. Lanes are published 1-based, internal 0-based; box numerals are identities.',
    "design_intent": 'Asymmetric-crane planning: the pair-flip crane is faster but inverts order, so optimal plans must weigh speed against inversion, and the sampler guarantees neither crane suffices alone.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

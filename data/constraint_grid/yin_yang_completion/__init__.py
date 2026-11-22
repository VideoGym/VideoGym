TASK_NAME = 'yin_yang_completion'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Every cell takes a black or white stone; each color forms one edge-connected group and no 2x2 square is monochrome; ringed given stones are fixed. Boards are sampled fresh per instance: an interlocking comb base randomized by legality-preserving single-stone flips (the old five hardcoded 5x5 templates are gone), then givens are trimmed monotonically. Forcedness implies uniqueness. Difficulty gates: at least 3 waves, at least half the board open, both given colors present.',
    "design_intent": 'Trains pure two-color connectivity topology with a local density law; every deduction is one visible stone, making the long forced chains directly watchable.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

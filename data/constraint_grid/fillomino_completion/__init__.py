TASK_NAME = 'fillomino_completion'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "The finished grid is partitioned into orthogonally connected groups; every cell of a group holds the group's size as its digit and two different equal-sized groups never share an edge. The input shows the finished tiling with some digits erased; every blank must be restored. Construction is monotone trimming: starting from the fully visible tiling, digits are erased one at a time in random order and an erasure is kept only if the forced-growth engine still completes the board, which guarantees zero guessing and a unique restoration by construction. Difficulty gates: at least 8 erased digits and at least 2 causal waves.",
    "design_intent": 'Trains self-describing region arithmetic where a digit is simultaneously label and area; the trimming construction keeps every blank on a visible forced-growth chain instead of a search frontier.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

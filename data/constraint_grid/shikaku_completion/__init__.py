TASK_NAME = 'shikaku_completion'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A square grid must be divided into non-overlapping axis-aligned rectangles that jointly cover every cell; each rectangle contains exactly one circled clue and its cell count equals that clue. Some rectangles are pre-drawn in the input and fixed. The generator samples a rectangle partition, places one clue per rectangle, pre-draws the most ambiguous rectangle, reveals further ones only when its deduction engine stalls, and proves per sample that the two forced rules complete the partition with zero guessing; a fully forced chain also implies the completion is unique. Difficulty gates: at least 5 undrawn rectangles, at least 2 causal waves, givens at most half of all rectangles.',
    "design_intent": 'Isolates exact-cover style spatial commitment where each decision is a visible rectangle with area arithmetic; the forced-chain construction turns a search puzzle into a readable chain of visual commitments.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

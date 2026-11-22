TASK_NAME = 'masyu_completion'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'One closed loop through cell centers passes every circle: white circles run straight with a turn in at least one along-loop neighbour, black circles turn between two straight two-cell arms; visited cells carry exactly two segments; drawn givens are fixed. Forced rules: a white circle with one viable through-direction draws it; a black circle with one viable arm per axis draws both arm segments; a loop cell with two remaining viable sides extends; segments into dead cells are eliminated; early closure is eliminated unless it completes the unique loop. Construction perturbs a perimeter loop, samples pearls from corner patterns, and trims givens monotonically; forcedness implies uniqueness. Gates: at least 6 open segments, 2 waves, givens at most half and at least one.',
    "design_intent": 'Isolates local turn/straight grammar propagating over a global cycle; the deepest forced chains of the family, watched one stroke at a time.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

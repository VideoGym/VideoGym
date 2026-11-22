TASK_NAME = 'yajilin_completion'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Arrow clues count the dark cells beyond them; dark cells never touch edgewise; one closed loop passes through every remaining empty cell; shown shades and segments are fixed. Forced rules: a satisfied arrow turns its ray's blanks into loop cells; an arrow whose remaining count equals its non-adjacent open spots shades them; neighbours of dark cells join the loop; a cell with fewer than two open sides must be dark; loop cells with a single open continuation extend; early closure is eliminated unless it completes the unique loop. Construction perturbs a perimeter loop, places darks and clues, and trims the mark set monotonically; forcedness implies uniqueness. Gates: at least 6 open marks, 2 waves, given segments at most half the loop.",
    "design_intent": "Couples counting-driven cell classification with global loop routing so that two visually distinct commitment kinds force each other; the family's richest deduction interleaving.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

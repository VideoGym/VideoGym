TASK_NAME = 'slitherlink_completion'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "One closed non-crossing loop runs along lattice lines; a digit counts its cell's drawn sides exactly; corners join zero or two segments; drawn givens are fixed. Forced rules: a digit whose remaining count equals its open sides draws them all (satisfied digits close their other sides); a corner with one segment and one open continuation extends the line (a lone open side at an empty corner is eliminated); an edge that would close the line early is eliminated unless it completes the unique all-digit loop. Construction samples a simply connected inside region, computes and hides clues, then trims givens monotonically; forcedness implies uniqueness. Gates: at least 6 open segments, 2 waves, givens at most half and at least one.",
    "design_intent": "Isolates boundary-drawing deduction on the lattice dual of the family's cell puzzles; every deduction is one visible stroke of the final loop.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

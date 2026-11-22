TASK_NAME = 'campsite_completion'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Tents and trees pair one to one with each tree orthogonally adjacent to its own tent; no two tents touch, even diagonally; boxed row and column counts are exact; given tents are fixed. Open cells are those adjacent to a tree, within unmet counts, and not touching a placed tent. Forced rules: a line whose remaining count equals its open cells forces tents on all of them; an unserved tree with a single free side forces its tent. Construction samples non-touching tents, matches trees, then un-gives tents monotonically; forcedness implies uniqueness. Difficulty gates: at least 3 open tents, at least 2 waves, givens at most half the tents and at least one.',
    "design_intent": 'Isolates bipartite pairing under marginal counts and adjacency exclusion, with every deduction a visible object placed in a scene-styled grid.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

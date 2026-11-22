TASK_NAME = 'star_battle_completion'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Every row, column, and colored region must contain exactly one star and no two stars may touch, even diagonally; some stars are pre-placed and fixed. Forced rules: a unit whose open cells shrink to one takes a star; a unit whose open cells all lie in one line or one region eliminates the rest of that line or region. Construction samples a star pattern, grows one region around each star, then un-gives stars in random order keeping only removals the engine still completes; forcedness implies uniqueness. Difficulty gates: at least 4 open stars, at least 2 waves, givens at most N/2, and at least one given star so the input legend is always true.',
    "design_intent": 'Isolates short-range adjacency exclusion across three overlapping unit systems, contrasting with the long-range diagonal attacks of the queen placement task.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

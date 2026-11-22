TASK_NAME = 'nurikabe_completion'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Numbered cells seed white islands whose sizes must equal their numbers; different islands never share an edge; the dark sea is one edge-connected body with no 2x2 block fully dark; given sea and island cells are fixed. Forced rules: a complete island seals its border; a cell beyond every island's remaining reach (walking blanks and unclued island ground, one budget unit per cell) is sea; an island with one growth cell grows there; a cell touching two clued islands is sea; three sea cells in a 2x2 square keep the fourth as island ground; a blank whose removal separates the committed sea is sea. Difficulty gates: at least 2 waves, at least half the board open, reveals capped at 40 percent, at least one given of each kind.",
    "design_intent": 'Trains joint region-size and global-connectivity reasoning where every deduction is a visible sea shade or island dot; with yin_yang it represents the shading-topology cluster of the family.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

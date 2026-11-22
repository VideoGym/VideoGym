TASK_NAME = 'sliding_tile_15'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "A 4x4 board holds tiles 1..15 and one blank. A move swaps the blank with one orthogonally adjacent tile; the goal is ascending row-major order with the blank bottom-right. The generator scrambles from the goal 4..8 reverse moves and keeps only boards whose BFS-shortest solution is UNIQUE and exactly the reversed scramble. The formal answer is the blank's complete grid-position path, including its visible start and solved bottom-right endpoint.",
    "design_intent": 'Classic slide-puzzle planning with a uniqueness gate: the model must recover the one shortest blank walk, so near-miss orders and alternate-length walks all fail.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

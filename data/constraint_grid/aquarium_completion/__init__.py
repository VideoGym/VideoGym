TASK_NAME = 'aquarium_completion'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "A square grid is divided into tanks by bold borders. Water settles: if a cell holds water, every cell of the same tank in that row or below also holds water, so each tank's state is one level. Dark numbers above columns and left of rows count that line's water cells and must be met exactly. Given water cells and given dry-ring cells are fixed. The generator samples tanks (target 2-5 cells) and levels, seeds one tank's top water band and another tank's lowest dry band as givens so both glyphs always exist, reveals further settling-consistent bands only when its interval engine stalls, and proves per sample that count-driven interval tightening decides every cell with zero guessing. Difficulty gates: at least 3 causal waves, at least half of the board left open, reveals capped at half the board.",
    "design_intent": 'Isolates joint marginal-count and monotone-level reasoning where every deduction is a visible water or dry commitment inside a fixed tank geometry.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

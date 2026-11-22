TASK_NAME = 'pipeline_hold_cells_3'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Five or six numbered products queue on an infeed; three LIFO hold cells stand between it and an outfeed that must finish in a sampled target order. PUSH moves the infeed front onto a chosen cell; POP moves a cell top to the next outfeed slot. Every product is pushed once and popped once, so every solution is exactly 2N actions; sampling additionally requires that no two-cell routing works, so the third cell is load-bearing. Cells are numbered 1..3 by position; product numerals are identities.',
    "design_intent": 'Three-buffer sequence routing: choosing which LIFO cell receives each product is the whole difficulty, and the minimum-cells gate keeps all three cells honest.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

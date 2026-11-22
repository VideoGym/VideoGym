TASK_NAME = 'lightsout_single_press'
TASK_FAMILY = 'board_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'In a square grid (5x5 by default), choose distinct cells up to the public press limit, each pressed exactly once; the cell itself and its in-bounds up, down, left, and right neighbors toggle synchronously, diagonals stay unchanged, and out-of-bounds neighbors are ignored. Presses are commutative and their order does not change the terminal state. The generator enumerates first-row presses and forces every following row from the visible initial state and keeps only the samples in which exactly one set turns the whole board OFF.',
    "design_intent": 'Isolates single-action neighborhood inversion on a finite grid while keeping the pressed cell visually distinct from the toggled lights.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

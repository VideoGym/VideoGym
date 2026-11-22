TASK_NAME = 'chess_pawn_swap'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A 5x5 board holds 2-3 white pawns on the bottom row and 2-3 black pawns on the top row. White steps straight up into empty squares or captures one square diagonally up; black mirrors downward. The goal is the displayed exact target with every white survivor on the top row and every black survivor on the bottom row. Sampling requires at least one capture, verifies with an independent no-capture BFS that no capture-free route exists, and enforces 8..14 optimal moves. Formal squares use `(x, y)`, where `x` is the 1-based top-to-bottom row and `y` the 1-based left-to-right column; internal indices stay 0-based.',
    "design_intent": 'Directional interlock with forced captures: straight movers block each other, so the model must plan capture order and column clearing across 8-14 forced-tempo moves.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

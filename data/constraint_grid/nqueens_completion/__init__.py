TASK_NAME = 'nqueens_completion'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'An NxN chessboard holds several fixed queens; one queen must be added per missing row so the finished board has exactly one queen per row and column and no two queens share a diagonal. Forced rules: a row with a single unattacked square left takes its queen there; a column coverable from only one open row pins that queen. Construction samples a full solution, then un-fixes queens in random order keeping only removals the engine still completes; a fully forced chain also implies uniqueness. Difficulty gates: at least 4 open rows, at least 2 waves, fixed queens at most N/2+1.',
    "design_intent": 'Isolates long-range diagonal attack geometry as a forced-hand chain, contrasting with the eight-neighbour-only exclusion of the star placement task.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

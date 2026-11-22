TASK_NAME = 'mancala_distribute_predict'
TASK_FAMILY = 'board_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Six ordinary slots P1-P6 and B form the fixed cycle P1→P2→P3→P4→P5→P6→B→P1; the slot names are 1-based indices, and the blue integer inside each slot is its token count (which may be 0). First read and empty the gold source, then distribute token by token starting from its successor; each landing adds one to a single destination slot and leaves the others unchanged. Samples restrict the source to P4-P6 and the source count to 4-6, so the B→P1 wraparound is load-bearing in every sample, yet the process ends before it wraps back to the source. The total token count is conserved.',
    "design_intent": 'The task trains a finite cyclic cumulative update triggered by a single local emptying.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

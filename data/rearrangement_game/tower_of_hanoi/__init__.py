TASK_NAME = 'tower_of_hanoi'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Three pegs carry 3-5 disks of strictly distinct widths. A move lifts the top disk of one peg onto another peg, and a disk may never rest on a strictly narrower disk. The generator samples random legal initial and target configurations (not the classical full-tower transfer) whose BFS-optimal distance is 4..12 moves, requiring search rather than a fixed recursion formula. Pegs are numbered 1..3 left to right, and disk identity is carried by drawn width and color.',
    "design_intent": 'Size-order, top-only access, and the three-peg buffer jointly force lookahead over 4..12 moves between arbitrary legal configurations, exercising shortest-plan search rather than recital of the classical recursive transfer.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'maze_square'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'An 11x11 grid of walkable and wall cells (carved by DFS on a temporary ringed grid at generation time, then the outer ring is stripped). The start (1, 1) and goal (11, 11) are both red cells. Movement is allowed only between side-adjacent white cells, and waiting in place is permitted; any complete legal route reaching the goal is accepted. The generation threshold requires a route of at least max(18, rows+cols+4) cells. Internally cells are 0-based (row, col), converted to 1-based (x, y) only at the answer boundary.',
    "design_intent": "Provides an orthogonal-maze baseline with no extra state, as the reference implementation of the family's light visuals and (x, y) answer space.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

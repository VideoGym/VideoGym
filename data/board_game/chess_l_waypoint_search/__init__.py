TASK_NAME = 'chess_l_waypoint_search'
TASK_FAMILY = 'board_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'The visible L token moves on a 5x5 board only by a (plus/minus 2, plus/minus 1) offset. It must land on both green waypoints in either order before entering the red target; target entry is forbidden earlier and ends the route immediately. Ordinary cells may be revisited. The source solver enumerates every shortest route and construction accepts exactly one, so the full path is semantically privileged rather than an arbitrary solver order. Difficulty spans routes of 4 (minimum), 5-6 (typical), and 7 (maximum) moves.',
    "design_intent": 'Unordered two-waypoint knight tour before entering a terminal target cell, testing non-adjacent jump geometry and shortest-route planning.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'maze_colored_zone'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'On a multi-path maze (DFS-carved, with a proportion of loops opened before the outer ring is stripped), every walkable cell is colored blue or green. Starting from the blue start, each step must land on a differently colored cell (alternating blue-green); waiting does not advance the color phase. Both red-framed endpoints count as blue, and the goal must be entered on a blue phase. Generation guarantees at least 2 geometric routes and that the constrained shortest route is strictly longer than the geometric shortest route (an ordinary BFS shortest path necessarily violates the color rule).',
    "design_intent": 'Overlays a color-phase constraint on the same maze semantic space, giving a mechanic-load-bearing contrast against the pure-topology baseline.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

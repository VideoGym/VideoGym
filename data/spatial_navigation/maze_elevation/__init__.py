TASK_NAME = 'maze_elevation'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'On an open grid (random walls, no outer ring), every walkable cell has a height of 0..4; a single step may only go to a side-adjacent cell whose height differs by at most 1; both red-framed endpoints are height 0. Generation guarantees that the constrained shortest route is strictly longer than the geometric shortest route: the geometric shortcut is flanked on both sides by 0-to-4 cliff cells, so an ordinary BFS shortest path necessarily violates the height limit.',
    "design_intent": 'Replaces the discrete color phase with a continuous quantity constraint (height gradient), testing process reasoning when the visual encoding is an ordered color band.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

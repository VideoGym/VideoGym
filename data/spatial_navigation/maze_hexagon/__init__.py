TASK_NAME = 'maze_hexagon'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A radius-3 hexagonal honeycomb (37 cells); movement is limited to adjacent hexagons sharing an open edge, and corner contact does not constitute adjacency; the start is on the boundary and the goal is at the center. The generation threshold requires a route of at least max(10, 4x radius) cells and picks the longest boundary start.',
    "design_intent": 'Represents six-way adjacency on a non-square topology; the staggered coordinate reading tests robust understanding of a non-Cartesian labeling system.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

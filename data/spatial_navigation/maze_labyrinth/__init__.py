TASK_NAME = 'maze_labyrinth'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A concentric-ring maze: the center disc is ring 1 with no wedge, and each outer ring is divided evenly into 8 wedges; legal moves are through an open angular gap within the same ring or in/out through a radial opening; the start is on the outermost ring and the goal is at the center.',
    "design_intent": 'The representative task for a polar-coordinate world representation; tests spatial reasoning over mixed circumferential/radial movement.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

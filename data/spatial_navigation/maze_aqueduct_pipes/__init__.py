TASK_NAME = 'maze_aqueduct_pipes'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'On a 9x9 stone-slab grid, straight/elbow pipes (two ports) and T/cross (decoy) pipes are placed, each shown in one of four orientations. Water enters the west port of the source pipe on the left edge; it can cross an edge only when the two adjacent ports face each other, cannot revisit a cell, and must leave through the east port of the fountain pipe on the right edge. The unique non-branching live path uses only two-port pipes, and the displayed orientations are not pre-connected.',
    "design_intent": "A process-type maze combining mechanism rotation and fluid propagation; the family's representative of 'modify the scene to make a path valid' rather than 'find a path in a fixed scene'.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

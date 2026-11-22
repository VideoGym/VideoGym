TASK_NAME = 'maze_layered'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Two 9x9 maze layers (each carved then stripped of its outer ring) are placed side by side as L1/L2, sharing yellow staircase cells at matching positions; a staircase is walkable and optionally switches layers; the start is on L1 and the goal is on L2, and one must climb at least once. Generation hard gates: with no staircase there is no solution; exactly one staircase is within the optimal budget (the others are unreachable or add a detour of >= 6 steps), and the reachability/detour facts of each staircase are retained.',
    "design_intent": "The representative task for a layered world representation and a cross-layer causal commit; the staircase discrimination design eliminates the 'any staircase works' one-glance degeneration.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

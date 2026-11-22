TASK_NAME = 'sokoban'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Plan a complete legal route on a 7x7 board with three to six interior walls that pushes the single box onto the green target. Each move steps the agent one cell up/right/down/left; entering the box's cell pushes it one cell onward, legal only when the destination is in-bounds floor. The box cannot be pulled or moved twice per step. Instances are rejection-sampled so the shortest route takes 8-24 moves with 2-5 pushes and is at least two moves longer than the same layout without interior walls, making the walls load-bearing. Any shortest valid full agent-position path is accepted.",
    "design_intent": 'Exercise higher-difficulty box planning where interior walls genuinely reroute the push plan: the wall-detour rejection rule guarantees the walls change the feasible route rather than decorating the board.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

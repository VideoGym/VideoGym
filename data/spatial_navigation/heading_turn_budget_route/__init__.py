TASK_NAME = 'heading_turn_budget_route'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "The agent carries a heading state: each step it can go straight, turn left/right then advance, or stay in place (staying does not spend a turn); it must reach G with <=3 turns and the final heading matching G's hollow-arrow badge. Sampling guarantees: 2 turns is unsolvable (the budget is tight), and the reference route uses exactly 3 turns and is strictly longer than the geometric shortest path (the heading constraint genuinely reroutes). Any route satisfying all heading constraints is accepted.",
    "design_intent": 'The heading state and turn budget make the shortest path infeasible: tests route construction and budget accounting in the (cell, heading) expanded space.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'maze_rush_hour'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A 6x6 lot holds rigid 2- and 3-cell vehicles; each carries a C-numbered badge, and the red vehicle shares its row with the east exit. A move slides one vehicle along its own axis through empty cells; only the red vehicle may leave, fully through the exit. Layout sampling enforces load-bearing blockers (no one-move exits, at least one vertical blocker whose relocation is required) and the published plan is the BFS-recomputed canonical shortest plan. Vehicle ids are printed identities assigned row-major from 1. Every head position is written as (x, y), with x the 1-based row and y the 1-based column; the fully exited red head has y = 7 while internal geometry stays 0-based.',
    "design_intent": 'Classic blocked-exit planning: fixed-axis vehicles create long dependency chains, so the shortest plan requires ordering several enabling slides before the red vehicle can travel.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

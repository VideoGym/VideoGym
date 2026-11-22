TASK_NAME = 'tic_tac_toe_rule_simulator'
TASK_FAMILY = 'board_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'From a non-terminal 3x3 initial state, strictly execute an explicit X/O placement sequence; each step may land only on an empty cell, the operation itself names its mark, and turns are not inferred from the initial count. The winning line has 0-1 of its cells preset in the initial state, the winner fills the remaining 2-3 cells at varying schedule positions, and the last cell, played at the 3rd or 4th actually executed step, first forms the unique three-in-a-row and terminates immediately; the two explicit suffix operations that follow must be ignored.',
    "design_intent": '3-4 steps of pure execution on 3x3 with a single stop rule and a short dependency chain, contrasting with simulators that require liberty and capture checks or 9x9 multi-directional run checks.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

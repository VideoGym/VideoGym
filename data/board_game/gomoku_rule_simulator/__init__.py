TASK_NAME = 'gomoku_rule_simulator'
TASK_FAMILY = 'board_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "From a non-terminal 9x9 intersection initial state, strictly execute an explicit B/W placement sequence, each step landing on an empty point; the schedule's colors are authoritative, and turns are not inferred from the initial stone count. The winning line initially leaves 2-3 gaps, the winner fills them one by one at varying schedule positions, and the last gap, filled at the 3rd through 5th actually executed step, first forms a straight line of exactly five same-colored stones and ends immediately; the two explicit suffix placements that follow must be ignored.",
    "design_intent": "The explicit operation program makes every placement, the winning stop point, and the five-stone evidence unique; the varying gap count and prefix length prevent a 'step three always wins' template, and the legal suffix tests that the model does not play past the endgame.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

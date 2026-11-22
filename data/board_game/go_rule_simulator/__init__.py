TASK_NAME = 'go_rule_simulator'
TASK_FAMILY = 'board_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Over the 7x7 intersection initial state, strictly execute an explicit B/W placement sequence. Each step first places the specified color on an empty point, then from that same placed state finds and simultaneously removes every opponent group that is orthogonally adjacent to the new stone and has zero liberties, and finally checks that the player's own group still has liberties; suicide is forbidden, and there is no pass, ko, or history. The color schedule is authoritative, and turns are not inferred from stone counts. The schedule varies per sample: 3-5 operation steps, exactly one capturing step (its position varying among steps 1-3), and a capture count equal to the proven maximum for that position (1-4 stones in 1-2 groups).",
    "design_intent": 'Simulates an explicit alternating Go stone-placement schedule on a 7x7 board, requiring sequential liberty tracking and simultaneous group capture.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'lifelike_rule_rollout'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A 5-7 binary grid evolves for 2 or 3 generations under sampled birth/survival count sets over the eight-cell neighbourhood with a fixed-zero or wraparound boundary; all cells update simultaneously. Sampling rejects cases with fewer than 6 total cell updates or whose alternate boundary leaves the terminal grid unchanged, so the counts and the boundary both change the answer.',
    "design_intent": 'It consolidates named fixed B/S variants into one rule-sampled task.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

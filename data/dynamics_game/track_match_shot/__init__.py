TASK_NAME = 'track_match_shot'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Move the launcher ball into the unique gold-outlined empty well without shifting existing balls. From the inserted ball, extend left and right only through consecutive same-color occupied wells. If that single run has length at least three, remove its entire run; otherwise remove nothing and the final track equals the filled track. Instances exercise both outcomes (roughly one quarter stay below threshold), so the threshold branch is load-bearing. Cleared wells become pale empty wells. No survivor moves, compacts, falls, refills, or triggers a second/separate run clear.',
    "design_intent": 'Test a load-bearing physical insertion followed by one-dimensional local connectivity, distinct from hex components, whole-row clears, and gravity-based match boards.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

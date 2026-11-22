TASK_NAME = 'balance_comparison'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A rooted tree whose root has exactly two child branches with sizes >= 2 and |left - right| <= 2 (EQUAL possible), left/right taken from rendered positions with seeded mirroring keeping outcomes symmetric.',
    "design_intent": 'Compare node values using shared tint evidence and a ring around the selected answer.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'office_inbox_priority_pull'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Four or five numbered papers stack in the inbox; only the inbox top or the buffer top is reachable. Every paper must reach the middle work stack, which must stay strictly decreasing bottom-to-top; the side buffer is LIFO. Sampling enumerates permutations whose shortest plan is UNIQUE and buffer-dependent. Paper numerals are identities; the goal panel is derived, not drawn.',
    "design_intent": 'Priority discharge through a LIFO buffer where the decreasing-priority output rule forces intermediate parking decisions.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'set_inclusion_diagram'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Compare exact numbered memberships of X and Y and classify the relation as one proper subset direction, partial overlap, or disjointness. The four relation hypotheses are complete and unique.',
    "design_intent": 'Test discrete membership comparison followed by direct abstract topology construction.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'venn_intersection_count'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Count each dot whose centre lies inside both circles exactly once. Four distinct integers encode the intersection and three disjoint-region/union distractors.',
    "design_intent": 'Test spatial membership counting while moving numeric choices into prompt text.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

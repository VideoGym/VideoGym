TASK_NAME = 'napoleon_vertex'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'The input visibly fixes one black triangle and one complete blue equilateral triangle constructed outward on a shared side. The answer is the unique labeled candidate at the centroid of that visible blue triangle, equivalently the common intersection of its three medians. No hidden side choice is part of the task state.',
    "design_intent": 'Constructs outward equilateral triangles on the sides of an asymmetric triangle and locates the missing Napoleon center from their visible geometry.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

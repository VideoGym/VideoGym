TASK_NAME = 'incenter'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Select the unique interior candidate whose perpendicular distances to all three finite triangle sides are equal. It is the common internal angle-bisector intersection and incircle center.',
    "design_intent": 'Incenter makes this exact relation carry correctness: Select the unique interior candidate whose perpendicular distances to all three finite triangle sides are equal. It is the common internal angle-bisector intersection and incircle center. The visual construction exposes the three bisectors, center marker, and circle radius relation without drawing distance spokes.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

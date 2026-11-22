TASK_NAME = 'homothety_center'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Determine which candidate is the center of dilation: the unique point where lines connecting each pair of corresponding vertices all intersect.',
    "design_intent": 'Homothety Center makes this exact relation carry correctness: Determine which candidate is the center of dilation: the unique point where lines connecting each pair of corresponding vertices all intersect. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

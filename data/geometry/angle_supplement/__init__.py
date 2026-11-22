TASK_NAME = 'angle_supplement'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Identify the candidate whose wedge equals the supplement of the reference angle, that is 180 degrees minus the reference.',
    "design_intent": 'Angle Supplement makes this exact relation carry correctness: Identify the candidate whose wedge equals the supplement of the reference angle, that is 180 degrees minus the reference. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

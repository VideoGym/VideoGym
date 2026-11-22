TASK_NAME = 'vertical_angle_pair'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Identify the candidate that sits inside the angle region vertically opposite to the highlighted one (the vertical-angle pair sharing the same opening).',
    "design_intent": 'Vertical Angle Pair makes this exact relation carry correctness: Identify the candidate that sits inside the angle region vertically opposite to the highlighted one (the vertical-angle pair sharing the same opening). Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

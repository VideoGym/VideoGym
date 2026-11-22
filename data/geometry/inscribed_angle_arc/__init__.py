TASK_NAME = 'inscribed_angle_arc'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Identify the candidate whose wedge equals the central angle on the blue highlighted arc, that is twice the inscribed angle.',
    "design_intent": 'Inscribed Angle Arc makes this exact relation carry correctness: Identify the candidate whose wedge equals the central angle on the blue highlighted arc, that is twice the inscribed angle. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

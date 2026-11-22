TASK_NAME = 'arc_connect'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Select the unique right fragment whose continuation through the hidden band lies on the same circle as the left fragment, preserving position, tangent direction, and curvature.',
    "design_intent": 'Arc Connect makes this exact relation carry correctness: Select the unique right fragment whose continuation through the hidden band lies on the same circle as the left fragment, preserving position, tangent direction, and curvature. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

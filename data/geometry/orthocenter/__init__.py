TASK_NAME = 'orthocenter'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'For each vertex, use the line through that vertex perpendicular to the opposite side or its extension. Select the unique candidate where all three altitudes meet.',
    "design_intent": 'Orthocenter makes this exact relation carry correctness: For each vertex, use the line through that vertex perpendicular to the opposite side or its extension. Select the unique candidate where all three altitudes meet. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

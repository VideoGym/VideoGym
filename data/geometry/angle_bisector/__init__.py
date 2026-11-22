TASK_NAME = 'angle_bisector'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Use the smaller, non-reflex angle. Select the unique candidate center on its internal bisector, equally inclined to both ray directions; marker outlines and letters have no geometric extent.',
    "design_intent": 'Angle Bisector makes this exact relation carry correctness: Use the smaller, non-reflex angle. Select the unique candidate center on its internal bisector, equally inclined to both ray directions; marker outlines and letters have no geometric extent. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'perpendicular_bisector'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Determine which candidate lies on the perpendicular bisector of the segment, meaning it is on the line through the midpoint that is perpendicular to the segment.',
    "design_intent": 'Perpendicular Bisector makes this exact relation carry correctness: Determine which candidate lies on the perpendicular bisector of the segment, meaning it is on the line through the midpoint that is perpendicular to the segment. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

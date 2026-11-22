TASK_NAME = 'trapezoid_median'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Identify which candidate is the midpoint of the segment that joins the two leg midpoints (the trapezoid's median).",
    "design_intent": "Trapezoid Median makes this exact relation carry correctness: Identify which candidate is the midpoint of the segment that joins the two leg midpoints (the trapezoid's median). Visual overlays are permitted only when they expose a necessary construction or the terminal commit.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

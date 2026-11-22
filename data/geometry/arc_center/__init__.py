TASK_NAME = 'arc_center'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a single black circular arc spanning part of a full circle, with no center marked, plus five labeled candidate circles A-E. The required relation is: Determine which candidate is the exact center of the full circle that the arc belongs to.',
    "design_intent": 'Arc Center makes this exact relation carry correctness: A white canvas shows a single black circular arc spanning part of a full circle, with no center marked, plus five labeled candidate circles A-E. The required relation is: Determine which candidate is the exact center of the full circle that the arc belongs to. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

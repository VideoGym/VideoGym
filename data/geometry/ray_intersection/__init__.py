TASK_NAME = 'ray_intersection'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows three dark gray line fragments that should be mentally extended inward and five labeled candidate circles A-E near the hidden center. The required relation is: Determine which candidate is the single point where the three full lines intersect.',
    "design_intent": 'Ray Intersection makes this exact relation carry correctness: A white canvas shows three dark gray line fragments that should be mentally extended inward and five labeled candidate circles A-E near the hidden center. The required relation is: Determine which candidate is the single point where the three full lines intersect. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

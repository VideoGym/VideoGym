TASK_NAME = 'secant_intersection'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a black circle with two chord segments inside, each connecting two hollow circumference anchors. Five labeled candidate markers A-E are grouped outside the circle. The required relation is: Identify the candidate that sits at the unique external point where the two chord lines meet when extended.',
    "design_intent": 'Secant Intersection makes this exact relation carry correctness: A white canvas shows a black circle with two chord segments inside, each connecting two hollow circumference anchors. Five labeled candidate markers A-E are grouped outside the circle. The required relation is: Identify the candidate that sits at the unique external point where the two chord lines meet when extended. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

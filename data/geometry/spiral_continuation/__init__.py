TASK_NAME = 'spiral_continuation'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a black polyline forming a growing square spiral where each segment turns 90 degrees and grows by a constant length increment, plus five labeled candidate circles A-E. The required relation is: Determine which candidate is the endpoint of the next segment that continues the spiral pattern.',
    "design_intent": 'Spiral Continuation makes this exact relation carry correctness: A white canvas shows a black polyline forming a growing square spiral where each segment turns 90 degrees and grows by a constant length increment, plus five labeled candidate circles A-E. The required relation is: Determine which candidate is the endpoint of the next segment that continues the spiral pattern. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

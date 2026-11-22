TASK_NAME = 'external_tangent_point'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Identify the point on the larger circle where the common external tangent line (a straight line touching both circles on the same side without crossing between them) touches it.',
    "design_intent": 'External Tangent Point makes this exact relation carry correctness: Identify the point on the larger circle where the common external tangent line (a straight line touching both circles on the same side without crossing between them) touches it. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'symmetric_completion'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a thin solid reflection axis and an incomplete convex hexagon with five visible vertices, missing one vertex on the reflected side. The intended hexagon is mirror-symmetric about the axis. Five labeled candidate circles A-E are nearby. The required relation is: Identify the missing vertex that completes the symmetric hexagon.',
    "design_intent": 'Symmetric Completion makes this exact relation carry correctness: A white canvas shows a thin solid reflection axis and an incomplete convex hexagon with five visible vertices, missing one vertex on the reflected side. The intended hexagon is mirror-symmetric about the axis. Five labeled candidate circles A-E are nearby. The required relation is: Identify the missing vertex that completes the symmetric hexagon. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

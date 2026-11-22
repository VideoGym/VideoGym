TASK_NAME = 'glide_reflection'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A blue tail-to-head arrow visibly defines both glide direction and distance on the black mirror axis. Reflect the source perpendicularly across the axis, then apply that arrow vector and select the final candidate.',
    "design_intent": 'Glide Reflection makes the visible arrow direction and length carry correctness instead of hidden endpoint order. The required relation is: Identify the final position. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

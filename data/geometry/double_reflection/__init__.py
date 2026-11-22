TASK_NAME = 'double_reflection'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A blue axis is explicitly first and a green axis explicitly second. Reflect the source perpendicularly across blue, then reflect that intermediate point perpendicularly across green; select the final candidate.',
    "design_intent": 'Double Reflection makes the visible blue-then-green order carry correctness instead of relying on hidden axis ordering. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

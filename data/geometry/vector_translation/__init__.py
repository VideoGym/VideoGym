TASK_NAME = 'vector_translation'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a black arrow representing a translation vector from point V1 to point V2, a separate hollow source point, and five labeled candidate circles A-E. Apply the same translation to the source point. The required relation is: Determine which candidate is the translated position.',
    "design_intent": 'Vector Translation makes this exact relation carry correctness: A white canvas shows a black arrow representing a translation vector from point V1 to point V2, a separate hollow source point, and five labeled candidate circles A-E. Apply the same translation to the source point. The required relation is: Determine which candidate is the translated position. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

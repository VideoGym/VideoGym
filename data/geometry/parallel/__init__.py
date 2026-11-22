TASK_NAME = 'parallel'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Construct the unique line through the hollow point parallel to the reference direction and select the only candidate center on it. Segment endpoint extent and marker outlines do not count.',
    "design_intent": "Parallel makes this exact relation carry correctness: Construct the unique line through the hollow point parallel to the reference direction and select the only candidate center on it. Segment endpoint extent and marker outlines do not count. The five markers are seated as one arrangement around the answer's slot rather than as a row square to the reference direction, so distance to the hollow point does not determine the answer; every distractor clears the required line by the runner-up margin. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

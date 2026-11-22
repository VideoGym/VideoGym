TASK_NAME = 'closest_pair'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Identify the two labeled points that form the closest pair, defined as the two markers whose straight-line distance is the smallest among all pairs of the five. Answer with the two option labels in alphabetical order.',
    "design_intent": 'Closest Pair makes this exact relation carry correctness: Identify the two labeled points that form the closest pair, defined as the two markers whose straight-line distance is the smallest among all pairs of the five. Answer with the two point labels in alphabetical order. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

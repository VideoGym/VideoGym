TASK_NAME = 'angle_trisection'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Starting at the ray whose endpoint carries the blue reference ring, sweep through the smaller interior angle toward the other ray and identify the candidate exactly one third of that sweep from the reference.',
    "design_intent": 'Angle Trisection makes the visibly oriented one-third sweep carry correctness; the reference ray is not an invisible array-order convention. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

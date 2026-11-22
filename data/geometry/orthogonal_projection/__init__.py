TASK_NAME = 'orthogonal_projection'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Identify which candidate is the foot of the perpendicular dropped from the source point onto the line, i.e. the nearest point on the line to the source. Presentation parity: all five markers lie on the drawn line, so incidence with the line is answer-free. A distractor is wrong because the connector from the source would not meet the line at a right angle there; its along-line offset from the foot grows with the source distance so the tilt away from perpendicular stays at least about 14 degrees.',
    "design_intent": 'Orthogonal Projection makes this exact relation carry correctness: Identify which candidate is the foot of the perpendicular dropped from the source point onto the line, i.e. the nearest point on the line to the source. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

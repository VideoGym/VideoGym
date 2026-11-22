TASK_NAME = 'nine_point_center'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A black triangle outline is shown on a white canvas with five labeled candidate circles A-E inside it. The nine-point circle of the triangle is the unique circle that passes through the midpoints of the three sides and the feet of the three altitudes (an altitude foot is the point where the perpendicular from a vertex meets the opposite side). The required relation is: Identify its center.',
    "design_intent": 'Nine Point Center makes this exact relation carry correctness: A black triangle outline is shown on a white canvas with five labeled candidate circles A-E inside it. The nine-point circle of the triangle is the unique circle that passes through the midpoints of the three sides and the feet of the three altitudes (an altitude foot is the point where the perpendicular from a vertex meets the opposite side). The required relation is: Identify its center. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

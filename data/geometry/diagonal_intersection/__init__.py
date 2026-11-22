TASK_NAME = 'diagonal_intersection'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a convex quadrilateral as a black outline, plus five labeled candidate circles A-E inside it. The required relation is: Identify which candidate is the intersection point of the two diagonals.',
    "design_intent": 'Diagonal Intersection makes this exact relation carry correctness: A white canvas shows a convex quadrilateral as a black outline, plus five labeled candidate circles A-E inside it. The required relation is: Identify which candidate is the intersection point of the two diagonals. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'excircle_center'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a black triangle outline with one vertex marked by a filled dot, plus five labeled candidate circles A-E outside the triangle. The excircle opposite the marked vertex is tangent to the opposite side and to the extensions of the two adjacent sides. The required relation is: Identify its center.',
    "design_intent": 'Excircle Center makes this exact relation carry correctness: A white canvas shows a black triangle outline with one vertex marked by a filled dot, plus five labeled candidate circles A-E outside the triangle. The excircle opposite the marked vertex is tangent to the opposite side and to the extensions of the two adjacent sides. The required relation is: Identify its center. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

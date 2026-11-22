TASK_NAME = 'midline_of_triangle'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A black triangle outline is shown on a white canvas with a filled black disc marking the midpoint of one side, plus five labeled candidate circles A-E along a different side. The midsegment of a triangle connects the midpoints of two sides and is parallel to the third side. The required relation is: Identify which candidate is the OTHER endpoint of this midsegment (i.e. the midpoint of the second side).',
    "design_intent": 'Midline Of Triangle makes this exact relation carry correctness: A black triangle outline is shown on a white canvas with a filled black disc marking the midpoint of one side, plus five labeled candidate circles A-E along a different side. The midsegment of a triangle connects the midpoints of two sides and is parallel to the third side. The required relation is: Identify which candidate is the OTHER endpoint of this midsegment (i.e. the midpoint of the second side). Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

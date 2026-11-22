TASK_NAME = 'arc_bisector'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a single black arc and two hollow endpoint anchors on its endpoints, plus five labeled candidate circles A-E lying directly on the arc. The required relation is: Identify the candidate that bisects the arc, splitting it into two equal-length sub-arcs.',
    "design_intent": 'Arc Bisector makes this exact relation carry correctness: A white canvas shows a single black arc and two hollow endpoint anchors on its endpoints, plus five labeled candidate circles A-E lying directly on the arc. The required relation is: Identify the candidate that bisects the arc, splitting it into two equal-length sub-arcs. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'parabola_vertex'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "A white canvas shows a black parabolic curve (a smooth U-shaped arc at some rotation), plus five labeled candidate circles A-E near the curve's turning point. The required relation is: Identify the vertex, the point where the parabola changes direction (the tip of the U). Presentation parity: all five markers ride the drawn curve, so incidence with the parabola is answer-free. A distractor is wrong because the arc still slopes there; its arc-length offset is at least the distance at which the curve has already fallen a marker radius plus air away from the tip along the axis, and the curvature is bounded from below so the turning point stays sharply localized.",
    "design_intent": "Parabola Vertex makes this exact relation carry correctness: A white canvas shows a black parabolic curve (a smooth U-shaped arc at some rotation), plus five labeled candidate circles A-E near the curve's turning point. The required relation is: Identify the vertex, the point where the parabola changes direction (the tip of the U). Visual overlays are permitted only when they expose a necessary construction or the terminal commit.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

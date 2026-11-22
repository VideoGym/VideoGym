TASK_NAME = 'triangle_center'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Determine which candidate is the triangle's center of mass, namely the centroid — the unique point where the three medians (each running from a vertex to the midpoint of the opposite side) meet.",
    "design_intent": "Triangle Center makes this exact relation carry correctness: Determine which candidate is the triangle's center of mass, namely the centroid — the unique point where the three medians (each running from a vertex to the midpoint of the opposite side) meet. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

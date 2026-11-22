TASK_NAME = 'square_vertex_from_diagonal'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Identify which candidate is one of the other two vertices of the square. Each missing vertex sits on a perpendicular through the diagonal's midpoint, at a distance equal to half the diagonal length.",
    "design_intent": "Square Vertex From Diagonal makes this exact relation carry correctness: Identify which candidate is one of the other two vertices of the square. Each missing vertex sits on a perpendicular through the diagonal's midpoint, at a distance equal to half the diagonal length. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

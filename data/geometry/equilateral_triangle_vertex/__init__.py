TASK_NAME = 'equilateral_triangle_vertex'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Identify which candidate completes a perfect equilateral triangle with the base segment (all three sides equal, all angles 60 degrees).',
    "design_intent": 'Equilateral Triangle Vertex makes this exact relation carry correctness: Identify which candidate completes a perfect equilateral triangle with the base segment (all three sides equal, all angles 60 degrees). Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

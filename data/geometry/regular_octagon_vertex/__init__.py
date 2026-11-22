TASK_NAME = 'regular_octagon_vertex'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Identify which candidate is the next vertex that continues the regular octagon (all sides equal, all interior angles 135 degrees).',
    "design_intent": 'Regular Octagon Vertex makes this exact relation carry correctness: Identify which candidate is the next vertex that continues the regular octagon (all sides equal, all interior angles 135 degrees). Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

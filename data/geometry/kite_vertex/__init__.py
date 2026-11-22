TASK_NAME = 'kite_vertex'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Identify the missing mirror-wing vertex, which is the reflection of the visible wing across the symmetry axis.',
    "design_intent": 'Kite Vertex makes this exact relation carry correctness: Identify the missing mirror-wing vertex, which is the reflection of the visible wing across the symmetry axis. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

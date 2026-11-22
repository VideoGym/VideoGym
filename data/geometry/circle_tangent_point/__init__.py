TASK_NAME = 'circle_tangent_point'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Select the displayed contact where the segment to the external point is perpendicular to the radius; the other mathematical tangent contact is not an option.',
    "design_intent": 'Circle Tangent Point makes this exact relation carry correctness: Select the displayed contact where the segment to the external point is perpendicular to the radius; the other mathematical tangent contact is not an option. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

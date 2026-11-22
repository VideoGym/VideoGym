TASK_NAME = 'right_triangle'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows five labeled candidate circles A-E and no initial connecting lines. Exactly three of the five points form a right triangle. The required relation is: Determine which labeled point is the vertex where the right angle occurs.',
    "design_intent": 'Right Triangle makes this exact relation carry correctness: A white canvas shows five labeled candidate circles A-E and no initial connecting lines. Exactly three of the five points form a right triangle. The required relation is: Determine which labeled point is the vertex where the right angle occurs. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

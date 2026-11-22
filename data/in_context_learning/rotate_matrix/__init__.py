TASK_NAME = 'rotate_matrix'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Write a cell as `(x, y)`, where `x` is the 1-based top-to-bottom row and `y` is the 1-based left-to-right column. One clockwise quarter-turn moves its value to `(y, n + 1 - x)`. Apply that movement to every cell simultaneously for exactly the requested one, two, or three quarter-turns; each committed matrix is the input to the next turn, and no printed value may be changed, lost, or duplicated.',
    "design_intent": 'Test repeated global spatial permutation with object identity preservation.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

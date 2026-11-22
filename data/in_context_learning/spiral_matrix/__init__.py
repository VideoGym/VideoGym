TASK_NAME = 'spiral_matrix'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Start at the top-left and traverse the current top edge left-to-right, right edge downward, bottom edge right-to-left, and left edge upward. Shrink all four boundaries inward after each ring and continue until every cell is read exactly once; odd grids finish at the center and even grids on the inner clockwise ring.',
    "design_intent": 'Test long cumulative visual traversal and boundary-state maintenance.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

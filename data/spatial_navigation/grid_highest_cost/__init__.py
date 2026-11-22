TASK_NAME = 'grid_highest_cost'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "A 4x4 weighted grid, allowing only rightward, downward, or staying in place; go from the top-left corner to the bottom-right corner, with total score = the start cell's value + each newly entered cell's value (staying does not double-count). It requires submitting any legal route that reaches the highest total score; all optimal routes are accepted, and route length is not scored.",
    "design_intent": 'Monotone path optimization under visible weights: the model must compare accumulated cost across alternative routes, with all positions readable from the edge 1-based labels.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

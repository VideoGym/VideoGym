TASK_NAME = 'triple_intersection_red'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Circle membership is evaluated in the displayed task-coordinate plane, including boundaries. Form A∩B∩C symmetrically, without privileging an operand pair. The construction makes every circle load-bearing and places the coordinate formed by averaging the three visible center coordinates inside the unique triple overlap. Every center is snapped to a labeled grid intersection, so the calculation never depends on an unprinted fractional coordinate.',
    "design_intent": 'Test region-level set construction and coordinate/image agreement.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

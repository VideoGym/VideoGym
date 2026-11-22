TASK_NAME = 'object_subtraction'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Each demonstration removes the unique largest visible object by geometric area. Apply that same rule to the target; survivors retain their fixed slots and the removed slot becomes empty. Half of the targets carry the first demonstration's extent trap (a surviving triangle or diamond spans a wider bounding box than the removed square), so bounding-box ranking stays uninformative on the query.",
    "design_intent": 'Test visual rule induction and object-fate tracking with direct target mutation.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'shape_outline_fill'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A maps to B by changing only style between filled and outline. Shape, color, size, and placement remain invariant. C begins with the same source style but a different shape, so D must preserve every C attribute and apply the same unique style direction.',
    "design_intent": 'Test single-attribute analogy transfer with explicit invariant preservation.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

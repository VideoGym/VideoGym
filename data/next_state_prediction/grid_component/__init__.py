TASK_NAME = 'grid_component'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A 7-9 binary grid with 3-5 components and a unique largest one asks for the size of that largest 4-connected component.',
    "design_intent": 'Largest-component size isolates region tracing from counting.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

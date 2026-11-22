TASK_NAME = 'connected_components'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'An 8x8 binary grid built from 2-5 grown regions, always including a diagonal-contact pair, asks for the number of 4-connected components; diagonal contact never connects.',
    "design_intent": 'The persistent diagonal-contact pair makes 4-connectivity load-bearing in every sample.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

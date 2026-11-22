TASK_NAME = 'lever_unlock_sequence'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Three levers must be first reached in the order L1->L2->L3 (stepping onto a later lever early is immediately illegal), and after pulling L3 the door D opens permanently; sampling guarantees that with the door always locked S-G is disconnected (the door is the only passage) and the reference path's first-arrival order is legal. Reaching the next required lever pulls it once, and waiting on an already-pulled lever is just waiting. Any complete route satisfying the order is accepted.",
    "design_intent": 'A strong order constraint overlaid on a single passage: the model must plan the visit order by number and avoid the later levers that would immediately violate the rule.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

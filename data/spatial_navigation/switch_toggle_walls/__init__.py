TASK_NAME = 'switch_toggle_walls'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Wall group A is initially solid (a full-row or full-column barrier, opening only the S/G/switch cells), and wall group B is initially walkable; stepping onto the switch X swaps the two groups' phases once and for all, lasting to the end of the episode. Sampling guarantees: without flipping the phase S-G is disconnected (the A barrier is load-bearing), the switch is reachable on the S side, and after flipping G is reachable. Any legal complete route is accepted.",
    "design_intent": 'A single global phase commit changes reachability across the whole board: the model must split route planning into two different wall-group worlds, before and after the flip.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

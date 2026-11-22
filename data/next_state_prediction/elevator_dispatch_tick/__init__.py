TASK_NAME = 'elevator_dispatch_tick'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Two cars with floors, loads, and capacities serve two ordered hall calls for one controller tick: each call is assigned to the nearest still-unassigned car with lowest-ID tie-break; an assigned car moves one floor toward its call, or at the call opens only if load is under capacity (a full car stays closed). Sampling constructs an exact distance tie on the first call and a full car blocked at its own floor every sample. Floors are the 0-based coordinate axis the user exempted from 1-based numbering.',
    "design_intent": 'It predicts a controller successor instead of an arbitrary service plan.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

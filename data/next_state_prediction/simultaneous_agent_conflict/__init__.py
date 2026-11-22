TASK_NAME = 'simultaneous_agent_conflict'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Three to five lettered agents on a 6-7 wrapped grid each propose one cell in their arrow direction for a single tick. Direct swaps make both wait; shared destinations follow the sampled all-wait or lowest-ID policy; a mover into a cell whose occupant ultimately waits is cancelled, repeated until stable. Sampling rejects ticks without at least one conflict group and one unaffected mover.',
    "design_intent": 'The burden is simultaneous arbitration and identity preservation.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

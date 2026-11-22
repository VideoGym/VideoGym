TASK_NAME = 'clocked_logic_register_tick'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Three D expressions over old Q1-Q3 and inputs X/Y capture simultaneously at each of 1-2 clock edges; sampling requires an expression reading an earlier-indexed register so that sequential in-place evaluation would provably differ from the correct simultaneous capture, and rejects fixed points.',
    "design_intent": 'Synchronous hardware dataflow differs from sequential instructions.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

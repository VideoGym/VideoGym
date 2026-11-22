TASK_NAME = 'finite_tape_machine_rollout'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A three-control machine on a circular 8-11 bit tape runs 6-7 ticks of read, write, control change, then move, driven by a complete sampled 6-entry table. Sampling requires at least one rewrite, at least two control changes, a wrap crossing, and both bits read; the visible-state budget caps ticks at 7.',
    "design_intent": 'Finite-control tape execution differs from the two-dimensional turn/write agent.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'resource_pick_drop_swarm'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Agents A (empty-handed, on a positive pile) and B (carrying) act in ID order for exactly 2 ticks: drop one unit if carrying, else pick one if the pile is positive, else nothing, then move one wrapped cell in their fixed direction; later agents see earlier pile updates. Sampling rejects traces without at least one pick and one drop or with any state where the two badges share a cell.',
    "design_intent": 'Inventory spans agents and cells, unlike movement-only arbitration.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

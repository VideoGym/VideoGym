TASK_NAME = 'directional_arrow_entry'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": '1-2 complete wall barriers each embed one directional gate: it can only be entered along the arrow direction (from the tail side), and after entering it is immediately forced one more cell forward; the gate cell cannot be stopped on. Sampling guarantees: with the gate cell treated as a wall, S-G is disconnected (the gate is the only passage), the minimum number of gate uses equals the number of barriers, and the gate tail/gate exit cell stay clear. Any legal complete trajectory is accepted, and route length is not scored.',
    "design_intent": 'The direction constraint makes the same cell asymmetrically reachable from the two directions, so the model must complete an alignment maneuver before the barrier rather than crossing arbitrarily.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

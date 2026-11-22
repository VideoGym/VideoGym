TASK_NAME = 'rail_signal_interlocking'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Route definitions over numbered blocks meet occupancy and a priority list: scanning requests in priority order, a route is unsafe if any of its blocks is occupied; only the first safe request is accepted, its signal turns GREEN, its switch position is set, and every other signal is RED. Sampling always occupies a block of one requested route so at least one request stays blocked.',
    "design_intent": 'The output is a control successor, not a train trajectory.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

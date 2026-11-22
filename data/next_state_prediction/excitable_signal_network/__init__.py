TASK_NAME = 'excitable_signal_network'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'On a 9-10 grid of empty, conductor, head, refractory tail, and blocker cells, each synchronous step turns heads to tails, tails to conductors, and a conductor to a head exactly when it has exactly one orthogonal old-state head; 2 or 3 steps. Sampling places 2-3 travelling signals on a random conductor field and rejects cases without at least one single-head fanout to two conductors, at least one multi-head collision retention, and at least one blocker whose replacement by a conductor would change the terminal grid.',
    "design_intent": 'Persistent conductors and refractory state distinguish it from infection spread.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

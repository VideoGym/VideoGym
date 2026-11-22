TASK_NAME = 'predator_prey_cycle'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Empty cells birth prey with at least two prey-like Moore neighbours and zero predators; prey with at least two predator neighbours become predators; predators with zero prey-like neighbours starve to empty; sheltered prey and rock are immutable. Exactly 2 synchronous ticks. Sampling draws a random population and rejects cases lacking a birth, a predation, a starvation, a second-tick change, a sheltered prey beside at least two predators (shelter counterfactually load-bearing), and a rock beside a predator.',
    "design_intent": 'Coupled bidirectional species dynamics are not a one-way spread.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

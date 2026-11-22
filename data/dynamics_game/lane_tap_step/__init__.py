TASK_NAME = 'lane_tap_step'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Four lanes have visible speeds of one or two vertical units per tick. Advance every dot simultaneously for exactly two ticks. The unique dot that reaches the bottom judgment line exactly on tick two changes to the hit state; all other dots retain identity and occupy their speed-derived positions. No lane updates early or independently.',
    "design_intent": 'Train synchronized multi-object rollout under heterogeneous visible velocities rather than a static distance comparison.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

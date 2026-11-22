TASK_NAME = 'jumping_frog_swap'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A lane of 5-6 numbered pads holds green frogs (move right only), orange frogs (move left only), and exactly one empty pad. A move steps one pad into the empty pad or leaps two pads into it over a single opposite-colour frog. Instances are drawn from a deterministic catalog of state pairs whose shortest plans need 5..10 moves and at least two leaps. The payload lists the 1-based source pad of each moved frog; all shortest plans are accepted.',
    "design_intent": 'The classic two-colour passing corridor: directional constraints force leap choreography, and accepting every shortest plan keeps the multi-solution space honest.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

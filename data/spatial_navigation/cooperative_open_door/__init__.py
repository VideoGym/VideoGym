TASK_NAME = 'cooperative_open_door'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Blue and orange agents act in sync (one action each per tick, two equal-length sequences committed at the same index). Blue and orange must step onto their same-color pressure plates on the same tick to open the door permanently and at once; after opening, the brown door cells turn white and walkable, orange stays on its plate, and blue goes straight down through the door to the green flag. The two agents may not share a cell or swap positions. Sampling guarantees that with the door always closed the goal is unreachable (the door is load-bearing).',
    "design_intent": 'Synchronized two-agent cooperative door-opening tests multi-agent timing coordination and the pass-through planning that depends on the door being open.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

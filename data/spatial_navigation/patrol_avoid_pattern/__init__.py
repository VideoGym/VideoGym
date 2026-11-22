TASK_NAME = 'patrol_avoid_pattern'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A red patrol shuttles back and forth between two cells (along the red arrow on the first tick, then reversing each tick); after the agent and the patrol commit on the same tick they may not share a cell or swap with it. The only wall gap is exactly a patrol-covered cell, and sampling guarantees that without waiting (allow_stay=False) it is unsolvable and the reference solution necessarily includes stay -- that is, one must offset the patrol beat by waiting to pass through the gap. Any safe route is accepted, and length is not scored.',
    "design_intent": 'Patrol-beat avoidance tests timing reasoning about a moving adversary: one must offset the rhythm by waiting to cross the only gap.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

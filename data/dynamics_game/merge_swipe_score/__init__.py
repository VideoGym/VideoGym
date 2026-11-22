TASK_NAME = 'merge_swipe_score'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Apply badge 1 then badge 2. For each swipe, remove gaps while preserving tile order toward the indicated edge, then from that leading edge fuse the first available adjacent equal pairs. Each pre-round tile participates at most once and a newly created tile cannot merge again in the same round, but may merge in round two. The formal numeric answer is the sum of fused pairs across both rounds.',
    "design_intent": 'Test genuine cross-round dependency and a numeric summary while retaining an independently checkable terminal visual state; preserve its direct count answer kind rather than forcing it into A-D.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

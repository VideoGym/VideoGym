TASK_NAME = 'mr_driller_drop_predict'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A drill in one fixed column executes four ticks, entering rows 1 through 4 in order in the visible board coordinate system. When it enters a colored block, delete that block and every same-color block connected to it through shared edges on the current board. Diagonal contact does not connect. Recompute after each prior deletion; unaffected blocks never fall, refill, or shift. Samples make every tick consequential and include at least one multi-block component.',
    "design_intent": 'Train sequential updates where each removal changes the topology used by the next step, distinct from one-shot hex or linear component clearing.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

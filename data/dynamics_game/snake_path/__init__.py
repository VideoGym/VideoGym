TASK_NAME = 'snake_path'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Move the snake on the visible grid until it has eaten every visible food pellet. The pellets may be eaten in any order. On an ordinary move the tail cell is released; on a food move the pellet disappears and the tail stays, so the snake grows by one. Any complete legal head-position path is accepted; optimality is not required.',
    "design_intent": 'Exercise multi-object route planning under cumulative body growth: every sample requires every visible consumption event, and ignoring either remaining pellets or growth produces an invalid terminal state.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'jewel2_future_board'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Swap exactly the two orthogonally adjacent gold-bordered jewels. On the post-swap board, take the union of every maximal horizontal or vertical same-color run of length at least three, including intersections, and remove that union simultaneously. Only then compact each column downward while preserving survivor order. Do not move sideways, refill, or run a second match/clear cycle.',
    "design_intent": 'Exercise a load-bearing local exchange followed by a global union computation and stable gravity, unlike fixed-scope specials or direct placement clears.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

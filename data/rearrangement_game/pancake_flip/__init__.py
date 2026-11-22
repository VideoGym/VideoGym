TASK_NAME = 'pancake_flip'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'One stack of 4-6 pancakes with strictly distinct widths and colors rests on a plate. The only legal operation flips the top-K prefix (2 <= K <= stack height) in place, reversing that prefix while the rest never moves. The generator shuffles random initial and target orders and keeps instances whose BFS-optimal flip count is 2..4. K is a plain count of pancakes, not a position ordinal.',
    "design_intent": 'Unlike swap-any-two sorting tasks, the only legal operation is a whole-prefix reversal, so planning must reason about how one compound operation permutes several identities at once; width and color jointly carry identity and the rotation itself provides visible stack-operation supervision.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

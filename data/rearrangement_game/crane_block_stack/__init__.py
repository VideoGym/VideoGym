TASK_NAME = 'crane_block_stack'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": '3-5 columns hold 4-6 numbered blocks scattered across random stacks. A move picks the topmost block of any non-empty column and places it on top of any other column; capacity is unlimited and there is no size order. The generator samples random initial and target distributions of the same block set whose BFS-optimal distance is 2..8 moves. Columns are published 1-based and kept 0-based internally; block numerals are identities and never shift.',
    "design_intent": 'Trains multi-column LIFO top-block rearrangement and shortest-path search; unlike the size-ordered tower task it constrains only stack-top access and the exact goal arrangement, so uncovering order is the whole difficulty.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

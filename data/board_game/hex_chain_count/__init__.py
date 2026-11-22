TASK_NAME = 'hex_chain_count'
TASK_FAMILY = 'board_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'In a 7x7 pointy-top hexagonal rhombus grid, two red stones are connected only when their cells share a complete edge; corner contact does not count, and a blue stone or an empty cell both block. Partition all red connected components across the visible board; the answer is the stone count of the largest component (4-6). Every sample must contain one runner-up red component that is exactly one stone smaller, 1-2 small components, and at least two blue blocking stones pressed against red stones, so that the count comparison and the blue blocking are genuinely load-bearing; the sample guarantees a unique largest component.',
    "design_intent": "Trains shared-edge connectivity on a non-square grid; forcing the runner-up to be exactly one stone smaller makes 'find the maximum' require comparing real component sizes rather than spotting a single large cluster.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

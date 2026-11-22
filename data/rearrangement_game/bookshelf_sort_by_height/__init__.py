TASK_NAME = 'bookshelf_sort_by_height'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A shelf holds 5-6 books of strictly distinct heights in random order. A move swaps the books in any two chosen slots (not necessarily adjacent); the goal is always tallest at the left down to shortest at the right. Sampling keeps BFS optima of 2..4 swaps. Slots are numbered from 1 in the payload and spoken 1-based; internal indices stay 0-based. Height and colour identify books; no numerals are printed.',
    "design_intent": 'Free-pair swap sorting as permutation-cycle reading: optimal plans are cycle decompositions, so the model must see the permutation structure rather than bubble neighbours.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

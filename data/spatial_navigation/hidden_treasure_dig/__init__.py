TASK_NAME = 'hidden_treasure_dig'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Two parchment clues, on different rows and columns, each cast one directed ray; the treasure is the unique cell lying on both forward rays at once, and cells behind a ray are not on it. Only the two clue positions leave two row-column crossing candidates, and the arrow directions decide which is the treasure. Sampling guarantees that the intersection of the two rays is exactly that treasure cell and that both candidate routes are reachable. Any legal route to the treasure is accepted, and length is not scored.',
    "design_intent": 'Two-ray intersection deduction tests excluding ambiguous candidates from visual clues: the positions leave two solutions and the directions fix one, and the hidden treasure forces genuine reasoning rather than a direct read-out.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

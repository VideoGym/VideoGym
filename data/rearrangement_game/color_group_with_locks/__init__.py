TASK_NAME = 'color_group_with_locks'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A row of 7-8 slots holds red, green, and blue orbs plus 1-2 locked grey orbs at fixed slots. A move swaps two adjacent non-locked orbs; nothing crosses a lock, so locks partition the row into independent segments that must each end red before green before blue. Sampling keeps BFS optima of 3..8 swaps and rejects instances whose segment target equals the unlocked global grouping, so the lock structure is load-bearing. The payload lists the 1-based left slot of each swapped pair.',
    "design_intent": 'Segmented adjacent-swap sorting: the locks make optimal plans segment-local inversion fixes, and the anti- degeneration gate stops the segmentation from ever being decorative.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

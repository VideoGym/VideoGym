TASK_NAME = 'tile_color_sequence'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Six per-position cards in one row. Cards 1-5 each hold one solid tile whose colors repeat a fixed cycle of length 2, 3, or 4 with pairwise-distinct cycle colors; the generator proves exactly one such cycle fits all five tiles. Card 6 holds only a question mark and must continue the cycle by one tile. Colors come from a fixed six-word master palette (red, blue, yellow, green, orange, purple) whose names are the typed vocabulary.',
    "design_intent": 'Minimal periodic-pattern induction with a nameable color payload and direct completion delivery.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'dot_count_progression'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Four per-position cards: cards 1-3 hold 1-36 colored dots each on an invisible six-by-six grid, filled in reading order; card 4 holds only a question mark. The three visible counts match exactly one allowed family (the generator rejects ambiguous triples), and card 4 continues that family by one more in-range term. The canonical 1,2,4 sequence is reserved for powers of two so the families stay disjoint.',
    "design_intent": 'Visual counting plus numeric sequence induction with direct missing-state completion; the typed payload and the completed card are the same fact in two carriers.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

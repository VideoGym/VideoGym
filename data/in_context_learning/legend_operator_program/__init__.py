TASK_NAME = 'legend_operator_program'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Three visible before/after demonstrations bijectively bind tokens A, B, and C to clockwise quarter-turn, horizontal reflection, and transpose. Apply the two to four visible program tokens strictly from left to right. Generated examples make each legend binding unique and make the forward program result differ from reversed execution.',
    "design_intent": 'Test visual legend induction composed with a short non-commutative program.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

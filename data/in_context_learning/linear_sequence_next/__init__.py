TASK_NAME = 'linear_sequence_next'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Four per-position cards on a 2x2 layout, each holding the same six-by-six grid with 1-based row and column numbers along its edges. A grid position is written `(x, y)`, where `x` is its top-to-bottom row number and `y` is its left-to-right column number. Cards 1-3 hold a 2-3 cell pattern; one fixed change maps card 1 to 2 and card 2 to 3: shift one column right, shift one row down, shift one row down and one column right, or a quarter-turn clockwise about the grid center. The generator proves exactly one candidate change fits both transitions and that all four states stay in bounds with a visible card-3-to-4 difference. Card 4 holds only a question mark over the empty labeled grid and must apply the change once more.',
    "design_intent": 'Rule induction over spatial transformations with an exact coordinate-typed payload and direct completion delivery.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

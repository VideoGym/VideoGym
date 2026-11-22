TASK_NAME = 'growing_polygon_step'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Four per-position cards in one row. Cards 1-3 each hold one regular polygon (vertex counts 3-7 visible) with a small dot on every vertex; counts grow by one constant whole-number step of 1 or 2 and the fourth count stays at most 9. Card 4 holds only a question mark and must continue the increase once. Orientation and drawing size never change a vertex count.',
    "design_intent": 'Discrete geometric counting plus constant-step induction with direct completion delivery.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

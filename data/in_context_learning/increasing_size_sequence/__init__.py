TASK_NAME = 'increasing_size_sequence'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Four per-position cards on a 2x2 layout. Cards 1-3 hold the same shape and color standing on a shared baseline under ruled unit lines numbered 1-8; a shape's height is the numbered line its top edge lands on. The three visible heights increase by one constant whole-number step of 1 or 2 with the fourth height still within the ruler; card 4 holds only a question mark over the same empty ruler and must continue the increase once.",
    "design_intent": 'Minimal equal-step extrapolation whose visual carrier (ruled height) is discrete and typed-answer-faithful, keeping the payload and the completed card one fact.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

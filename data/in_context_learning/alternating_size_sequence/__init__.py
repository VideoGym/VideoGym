TASK_NAME = 'alternating_size_sequence'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Four per-position cards on a 2x2 layout. Cards 1-3 hold the same shape and color standing on a shared baseline under ruled unit lines numbered 1-8; the shape's top edge lands exactly on one numbered line, so its height IS that line's number. The three visible heights follow an a,b,a alternation with |a-b| >= 2 (either phase may come first); card 4 holds only a question mark over the same empty ruler and must continue the alternation with height b.",
    "design_intent": 'Minimal alternation induction where the visual carrier (ruled height) is discrete and typed-answer-faithful, so the completed card and the payload express one fact.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

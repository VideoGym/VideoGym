TASK_NAME = 'analogy_completion'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Infer the unique rigid rotation or reflection mapping A to B and apply it to C. Occupied-cell count and colour are preserved; exactly one of four hypotheses equals the transformed query.',
    "design_intent": 'Test rigid visual analogy with independent hypotheses and a directly constructed result.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

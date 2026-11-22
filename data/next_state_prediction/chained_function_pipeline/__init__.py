TASK_NAME = 'chained_function_pipeline'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'An integer packet passes left to right through 4-5 visible blocks drawn from ADD/MUL/XOR/NEG plus exactly one boundary block (canonical MOD or CLAMP); each block commits its output before the next. Sampling rejects pipelines whose boundary block never actually clamps or wraps and traces with fewer than 4 distinct values.',
    "design_intent": 'All rules are explicit; no hidden-rule induction is required.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'xor_parity_lattice'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A 5-7 binary grid updates every cell to the XOR of its wrapped orthogonal neighbours, optionally including itself, for 2 or 3 synchronous steps. Sampling rejects cases where the wrap contributes nothing at the edge on the first step or where fewer than 4 cells differ at the end, so the wrap and the self-inclusion flag are load-bearing.',
    "design_intent": 'Exact parity differs from count thresholds and block permutations.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

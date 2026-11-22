TASK_NAME = 'arc1d_compact_nonzero'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A row of 10-14 slots holds 3-7 colour-digit tokens; each synchronous round moves every token with an empty left neighbour one slot left, repeating until no empty slot precedes any token (3-6 rounds, at least one round moving two or more tokens, at least one internal gap). Token identity and order are preserved.',
    "design_intent": 'Synchronous stable compaction with preserved token identity, order, and per-round parallelism.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

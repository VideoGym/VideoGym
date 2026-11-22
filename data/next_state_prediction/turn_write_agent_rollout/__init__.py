TASK_NAME = 'turn_write_agent_rollout'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'An oriented head on a 5-6 wrapped binary board runs 5-7 ticks of read, table-driven turn, write, then move. The sampled 2-entry table must rewrite at least one bit; the trace must read both bits and cross a wrapped edge. The visible-state budget caps ticks at 7 successors.',
    "design_intent": 'Orientation plus a sampled table prevents named-system memorization.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

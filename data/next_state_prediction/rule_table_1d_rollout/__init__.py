TASK_NAME = 'rule_table_1d_rollout'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'One binary row of 9-12 cells evolves for a stated 2 or 3 generations under a sample-specific 8-entry left-centre-right truth table with a fixed-zero or wraparound boundary. All cells update synchronously from the old row. The sampler rejects tables whose alternate boundary would leave the terminal row unchanged and rows with fewer than 2 changed cells, so both the table and the boundary mode are load-bearing in every sample.',
    "design_intent": 'The rule table varies per sample, replacing nine fixed rule-number duplicates.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

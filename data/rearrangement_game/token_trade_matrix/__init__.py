TASK_NAME = 'token_trade_matrix'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Five ring sockets hold tokens 1..5; visible undirected edges (5-7 of the 10 pairs) mark legal trades, and one move swaps both endpoint tokens simultaneously. Admitted instances use connected, non-complete graphs whose restricted optimum strictly exceeds the complete-graph optimum, so the wiring is load-bearing. Token numerals are identities; socket positions are printed and published 1-based, internal 0-based.',
    "design_intent": 'Permutation sorting under a visible trade graph: missing edges force multi-hop routing that a complete graph would not need, so reading the wiring is load-bearing.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

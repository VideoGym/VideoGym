TASK_NAME = 'trace_all_edges_once'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A branched semi-Eulerian graph built from a base path plus attached ears (attachment degree capped at 5), guaranteeing exactly two odd-degree endpoints, at least one two-deep ear, and at least two distinct legal trails. Any legal trail is accepted — a multi-certificate task.',
    "design_intent": 'Global-coverage search member of the unified engine; its verifier accepts the full legal solution space rather than one reference.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

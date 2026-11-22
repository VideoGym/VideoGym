TASK_NAME = 'bridge_edge_choice'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Two cycle-carrying blobs (local chords sampled inside each blob so no second crossing appears; both blobs >= 4 nodes at max difficulty) joined by exactly one edge. Generation certifies that edge as the unique bridge; the solver retests every edge by removal.',
    "design_intent": 'Connectivity-criticality member of the unified engine; its generator builds the certificate structurally instead of hoping for one.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

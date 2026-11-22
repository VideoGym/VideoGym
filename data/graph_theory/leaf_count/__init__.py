TASK_NAME = 'leaf_count'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A rooted tree (6-10 nodes, layered layout, seeded mirroring) resampled until it has >= 3 leaves. A leaf is a non-root node without children; the count is the answer.',
    "design_intent": 'Simplest rooted-tree member of the unified engine, anchoring the layered-tree visual language.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

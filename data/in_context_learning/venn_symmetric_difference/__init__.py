TASK_NAME = 'venn_symmetric_difference'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A symmetric difference contains exactly the A-only and B-only atomic regions, excluding overlap and exterior. Four complete region masks contain that result once.',
    "design_intent": 'Test set algebra with independent hypothesis media and direct region construction.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

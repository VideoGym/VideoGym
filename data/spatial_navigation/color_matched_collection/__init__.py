TASK_NAME = 'color_matched_collection'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Four colored balls (red, blue, green, yellow) and same-color empty slot rings; stepping on a ball cell picks it up and stepping on a same-color slot cell drops it in, capacity 1 (while carrying a ball, stepping on another ball cell does not pick it up). A single-ball carry-state search guarantees the real automatic-pickup semantics, the four pairs can be completed in any legal order, and length is not scored.',
    "design_intent": 'Same-color pair collection tests exchangeable multi-target ordering under a capacity constraint; the order is free and the Verifier does not dictate a unique collection order.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

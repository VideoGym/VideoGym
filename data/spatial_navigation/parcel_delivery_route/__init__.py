TASK_NAME = 'parcel_delivery_route'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Two lettered parcel boxes and same-letter target squares; stepping on a box picks it up and stepping on the same-letter target delivers it, capacity 1 (the current box must be delivered before picking up the next). A single-parcel carry-state search guarantees the real automatic-pickup semantics, the two pairs can be completed in any legal order, and length is not scored.',
    "design_intent": 'Labeled pair delivery tests exchangeable multi-target ordering under a capacity constraint, with one fewer pair and a shorter route than color_matched_collection.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

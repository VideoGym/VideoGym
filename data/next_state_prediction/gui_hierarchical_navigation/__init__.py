TASK_NAME = 'gui_hierarchical_navigation'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "A three-level page hierarchy (ROOT, three top pages, three leaves each) executes five prescribed actions: two clicks descend to a leaf, two BACKs return to ROOT, one click opens a sibling. CLICK pushes the tile's target; BACK pops above ROOT and otherwise retains it. The sequence always contains a sibling distinction and a load-bearing BACK.",
    "design_intent": 'It executes interface state rather than recovering a static mapping.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

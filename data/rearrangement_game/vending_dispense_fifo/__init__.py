TASK_NAME = 'vending_dispense_fifo'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "All 4-6 numbered cans start in the chute; DISPENSE moves the chute front to the tray's right end, REINSERT sends a chosen tray can back to the chute tail. The target fixes the final tray order. Sampling requires at least one reinsertion in the shortest plan. Can numerals are identities; no ordinal shifting exists.",
    "design_intent": 'FIFO resequencing through a return loop: out-of-order fronts force the longer reinsert cycle, training queue lookahead rather than direct placement.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

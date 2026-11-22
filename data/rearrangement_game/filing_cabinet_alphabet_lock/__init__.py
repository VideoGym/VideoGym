TASK_NAME = 'filing_cabinet_alphabet_lock'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Five drawers A-E hold 3-4 numbered papers; 1-2 locked drawers are empty and frozen. A move pulls the top paper of an unlocked drawer onto another unlocked drawer (LIFO). Sampling requires the locked-drawer optimum to be strictly longer than the all-unlocked optimum, so the lock pattern is load-bearing. Drawer letters are identities; paper numerals are identities; no ordinal shifting exists.',
    "design_intent": 'Lock-restricted LIFO routing: losing drawers as workspace lengthens the shuffle, so plans must route papers through the remaining open drawers.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

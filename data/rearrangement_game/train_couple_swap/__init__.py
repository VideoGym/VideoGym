TASK_NAME = 'train_couple_swap'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A main track couples 4-6 numbered railcars; a single-car siding sits below. A park move uncouples one named car into the empty siding (survivors close up); a place move splices the parked car back at a chosen boundary slot. Moves therefore alternate park/place and plans have even length. Sampling keeps recomputed optima of 4..6 moves. Car numerals are identities; insertion boundaries are ordinals published 1..len+1, internal 0-based.',
    "design_intent": 'Single-siding train reordering: each relocation is a coupled remove-close-splice-shift event, training selection of which car to pull and where to splice it.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

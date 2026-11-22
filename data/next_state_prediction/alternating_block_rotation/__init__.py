TASK_NAME = 'alternating_block_rotation'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Numbered tokens on an even 6 or 8 grid rotate one quarter-turn inside disjoint 2x2 blocks whose partition alternates with a phase bit each of the 2 steps; phase-1 edge blocks wrap. Sampling guarantees a token in the wrapping partition edge and rejects nothing else, and the phase counterfactual (starting in the other phase) always yields a different first successor.',
    "design_intent": 'The update unit is a disjoint block partition, not an overlapping stencil.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

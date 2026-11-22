TASK_NAME = 'lattice_gas_collision'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Directional particle channels on a 7-9 grid first turn exact head-on pairs (E+W to N+S and vice versa), then every arrow streams one cell, reflecting back into its source cell with reversed direction when the destination is a wall or outside; 2 or 3 ticks. Sampling constructs one head-on pair and one immediate wall reflection, adds 1-4 extra particles, and rejects any case where channel aliasing would break particle conservation.',
    "design_intent": 'Velocity identity and particle conservation are absent from scalar recolouring.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

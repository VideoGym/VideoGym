TASK_NAME = 'bubble_shooter_chain_predict'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'The launcher bubble travels to the unique hollow landing ring, occupies that empty hex cell, and joins its six-neighbor same-color component. If that component has at least three members, the entire component—including the shot—clears simultaneously. Diagonal/non-neighbor tokens and separate same-color components survive, and no gravity, refill, or second clear occurs. The pre-shot neighborhood is generated below threshold so the shot is load-bearing.',
    "design_intent": 'Combine continuous launch/contact with a discrete hex-component threshold update, distinct from rectangular special-area clears and one-dimensional runs.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

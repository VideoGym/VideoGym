TASK_NAME = 'register_machine_tick'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Registers A-C, a PC over a four-line program, and a ZERO flag execute a stated 2-4 instructions: ADD/XOR/MOV write modulo 16 and set ZERO; JZ jumps on ZERO; DECJNZ decrements modulo 16 and jumps on nonzero. Sampled variants guarantee a modulo overflow, a taken/untaken branch, or a decrement loop each sample. Program lines are 1-based everywhere the model sees them.',
    "design_intent": 'Register-program state differs from a spatial tape head.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

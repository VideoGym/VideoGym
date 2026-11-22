TASK_NAME = 'majority_consensus'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A two-opinion grid with immutable vacancies adopts the strict local majority of its orthogonal or Moore neighbours per synchronous step (2 or 3 steps); exact ties retain the old opinion and vacancies never vote or fill. Sampling requires at least 2 first-step flips, at least 1 retained tie, at least one vacancy, and a boundary counterfactual that changes the terminal grid.',
    "design_intent": 'The tie contract and unordered categories distinguish it from cyclic adoption.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

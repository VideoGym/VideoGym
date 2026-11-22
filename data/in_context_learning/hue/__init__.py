TASK_NAME = 'hue'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Read the five displayed legend colors as cyclic positions 1 through 5 from left to right. Moving down advances by one hidden row step and moving right by one different hidden column step; both are selected from one, two, and three, and advancing past position 5 continues from position 1. The fixed cells uniquely determine both steps and the single missing color.',
    "design_intent": 'Test two-dimensional modular pattern induction with strict local-state preservation.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

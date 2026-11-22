TASK_NAME = 'diameter_endpoints'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'An unrooted tree (force layout, no role rings) whose unique longest simple route has >= 3 edges (>= 4 at max difficulty). The sorted endpoint pair is the answer.',
    "design_intent": 'The graph diameter requires a global maximum over endpoint pairs. The complete path highlight and simultaneous endpoint rings express the solution without assigning an order to the two endpoints.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

TASK_NAME = 'euler_trail_endpoints'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A connected graph (tree plus chords) resampled until exactly two vertices have odd degree and some vertex has degree >= 3, so the endpoints are never readable from a bare chain. The parity rule is stated in the prompt, not assumed from memory.',
    "design_intent": 'Degree-parity member of the unified engine: the rule is explicit in the prompt and the visual verdict is one synchronized atomic update.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

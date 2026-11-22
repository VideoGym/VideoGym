TASK_NAME = 'avoid_blocked_vertex'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Route source plus one solid-gray blocked node. Generation guarantees every unconstrained fewest-edge route uses the blocked node and that the legal detour optimum is unique and strictly longer.',
    "design_intent": 'Forbidden-vertex variant of the route engine proving the exception is load-bearing rather than decorative.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

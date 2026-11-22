TASK_NAME = 'bipartite_color_completion'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "A connected bipartite graph (tree plus same-parity-respecting chords) colored red/blue except one white query node of degree >= 2. Consistency of the visible coloring is proved at generation; the solver propagates the constraint and the query's forced color is the answer.",
    "design_intent": 'Constraint-propagation member of the unified engine reduced to its minimal honest form: white unknown, one forced fill.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]

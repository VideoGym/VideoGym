"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

BipartiteColorCompletionGenerator = make_graph_task_class('bipartite_color_completion')

__all__ = ['BipartiteColorCompletionGenerator']

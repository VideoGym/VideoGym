"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

PathBetweenNodesGenerator = make_graph_task_class('path_between_nodes')

__all__ = ['PathBetweenNodesGenerator']

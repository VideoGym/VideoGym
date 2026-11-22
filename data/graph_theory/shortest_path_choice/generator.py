"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

ShortestPathChoiceGenerator = make_graph_task_class('shortest_path_choice')

__all__ = ['ShortestPathChoiceGenerator']

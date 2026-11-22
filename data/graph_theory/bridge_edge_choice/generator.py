"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

BridgeEdgeChoiceGenerator = make_graph_task_class('bridge_edge_choice')

__all__ = ['BridgeEdgeChoiceGenerator']

"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

TwoHopNeighborCountGenerator = make_graph_task_class('two_hop_neighbor_count')

__all__ = ['TwoHopNeighborCountGenerator']

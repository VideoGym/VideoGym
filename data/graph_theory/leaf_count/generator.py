"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

LeafCountGenerator = make_graph_task_class('leaf_count')

__all__ = ['LeafCountGenerator']

"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

SubtreeSizeGenerator = make_graph_task_class('subtree_size')

__all__ = ['SubtreeSizeGenerator']

"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

ClosestCommonAncestorGenerator = make_graph_task_class('closest_common_ancestor')

__all__ = ['ClosestCommonAncestorGenerator']

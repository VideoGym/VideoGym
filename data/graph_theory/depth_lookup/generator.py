"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

DepthLookupGenerator = make_graph_task_class('depth_lookup')

__all__ = ['DepthLookupGenerator']

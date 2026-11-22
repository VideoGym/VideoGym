"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

TraceAllEdgesOnceGenerator = make_graph_task_class('trace_all_edges_once')

__all__ = ['TraceAllEdgesOnceGenerator']

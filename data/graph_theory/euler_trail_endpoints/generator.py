"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

EulerTrailEndpointsGenerator = make_graph_task_class('euler_trail_endpoints')

__all__ = ['EulerTrailEndpointsGenerator']

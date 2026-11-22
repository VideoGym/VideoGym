"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

DiameterEndpointsGenerator = make_graph_task_class('diameter_endpoints')

__all__ = ['DiameterEndpointsGenerator']

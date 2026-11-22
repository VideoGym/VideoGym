"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

RouteViaCheckpointGenerator = make_graph_task_class('route_via_checkpoint')

__all__ = ['RouteViaCheckpointGenerator']

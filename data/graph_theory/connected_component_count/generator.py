"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

ConnectedComponentCountGenerator = make_graph_task_class('connected_component_count')

__all__ = ['ConnectedComponentCountGenerator']

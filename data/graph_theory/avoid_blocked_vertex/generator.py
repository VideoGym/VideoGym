"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

AvoidBlockedVertexGenerator = make_graph_task_class('avoid_blocked_vertex')

__all__ = ['AvoidBlockedVertexGenerator']

"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

DegreeMissingValueGenerator = make_graph_task_class('degree_missing_value')

__all__ = ['DegreeMissingValueGenerator']

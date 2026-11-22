"""Thin task route into the shared unified labelled-graph generator."""

from data.graph_theory.core.generator import make_graph_task_class

BalanceComparisonGenerator = make_graph_task_class('balance_comparison')

__all__ = ['BalanceComparisonGenerator']

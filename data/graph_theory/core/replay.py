"""Independent source replay for the unified Graph Theory tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from data.graph_theory.core.rendering import PHASE_SPLIT, render_unified_graph
from data.graph_theory.core.semantics import parse_graph_source, solve_graph_source


def semantic_progress_checkpoints(
    task_name: str,
    solution: Mapping[str, Any],
) -> tuple[float, ...]:
    """Return sample-specific commits that end on graph states, not mid-edge ticks.

    Two-phase local-fade tasks anchor their single interior checkpoint at the
    shared :data:`PHASE_SPLIT` boundary, where the evidence overlay is fully
    committed and the answer overlay has not yet begun.  The checkpoint task
    anchors at the node commit where its route reaches the gold checkpoint.
    Everything else runs one uninterrupted process from source to terminal.
    """

    if task_name == "route_via_checkpoint":
        paths = [list(path) for path in solution.get("paths") or [] if len(path) >= 2]
        if not paths:
            raise ValueError("route_via_checkpoint solution has no route to anchor")
        path = [str(value) for value in paths[0]]
        checkpoint = str((solution.get("support") or {}).get("checkpoint") or "")
        if checkpoint not in path[1:-1]:
            raise ValueError("route_via_checkpoint solution does not expose one internal checkpoint commit")
        return (0.0, path.index(checkpoint) / (len(path) - 1), 1.0)
    split = PHASE_SPLIT.get(task_name)
    if split is not None:
        return (0.0, float(split), 1.0)
    return (0.0, 1.0)


@dataclass(frozen=True)
class UnifiedGraphReplay:
    task_name: str
    source: Mapping[str, Any]
    solution: Mapping[str, Any]

    @property
    def semantic_checkpoints(self) -> tuple[float, ...]:
        return semantic_progress_checkpoints(self.task_name, self.solution)

    def render_progress(self, progress: float) -> Image.Image:
        return render_unified_graph(self.source, self.solution, progress=progress)

    def render_input(self) -> Image.Image:
        return render_unified_graph(self.source)


def replay_unified_graph(task_name: str, semantic_input: Mapping[str, Any]) -> UnifiedGraphReplay:
    source = parse_graph_source(task_name, semantic_input)
    solution = solve_graph_source(task_name, source)
    accepts_multi_trace = (
        task_name == "trace_all_edges_once"
        and solution.get("multi_certificate")
        and len(solution.get("solutions") or []) >= 2
    )
    if not solution.get("valid") or not (solution.get("unique") or accepts_multi_trace):
        raise ValueError(f"unified graph source does not replay to one formal answer: {solution}")
    return UnifiedGraphReplay(task_name=task_name, source=source, solution=solution)


__all__ = ["UnifiedGraphReplay", "replay_unified_graph", "semantic_progress_checkpoints"]

"""Canonical source-only semantic inputs for the Spatial 3D family.

Spatial 3D generators retain rich solved state (unfolded matrices, state
chains, pose orbits, view grids) so they can author reference text, terminal
images, and videos.  None of those derived values belongs in the canonical
verifier source.  This module deliberately uses per-task allowlists instead
of a denylist: a newly added outcome, trace, render checkpoint, or solver
diagnostic therefore cannot silently enter ``semantic_input``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Spatial 3D canonical source lacks {label}")
    return value


def _required(source: Mapping[str, Any], *keys: str, label: str) -> dict[str, Any]:
    missing = [key for key in keys if key not in source]
    if missing:
        raise ValueError(f"Spatial 3D canonical source {label} lacks fields: {missing}")
    return {key: deepcopy(source[key]) for key in keys}


def _native_state(task_name: str, state: Mapping[str, Any]) -> dict[str, Any]:
    if task_name == "paper_fold_hole_matrix":
        inner = _mapping(state.get("semantic_state"), "paper-fold semantic_state")
        return _required(inner, "grid_size", "fold_sequence", "punched_cells", label="paper-fold program")
    if task_name == "cube_net_fold":
        inner = _mapping(state.get("semantic_state"), "cube-net semantic_state")
        return _required(inner, "net_cells", "base_cell", "pip_by_cell", label="cube-net state")
    if task_name == "polycube_rotation":
        inner = _mapping(state.get("semantic_state"), "polycube semantic_state")
        return _required(
            inner, "cells", "rotations", "options", label="polycube rotation state"
        )
    if task_name == "voxel_three_views":
        inner = _mapping(state.get("semantic_state"), "voxel-views semantic_state")
        return _required(inner, "heights", "view", label="voxel view state")
    if task_name == "stack_orbit_view":
        inner = _mapping(state.get("semantic_state"), "stack-orbit semantic_state")
        return _required(inner, "cell_colors", "floor_size", "view", label="stack orbit state")
    if task_name == "slice_cross_section":
        inner = _mapping(state.get("semantic_state"), "slice semantic_state")
        return _required(inner, "solid", "cut", label="slice state")
    raise ValueError(f"No source-only Spatial 3D canonicalization for {task_name!r}")


def canonical_spatial_3d_semantic_input(
    task_name: str,
    structured_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the minimal raw-source domain for one Spatial 3D sample."""

    state = _mapping(structured_state, f"{task_name} structured state")
    return {"semantic_state": _native_state(task_name, state)}


__all__ = ["canonical_spatial_3d_semantic_input"]

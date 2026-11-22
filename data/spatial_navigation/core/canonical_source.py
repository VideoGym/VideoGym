"""Canonical source-only semantic input for Spatial Navigation allowlisted tasks.

Most Spatial Navigation records pass through the generic canonical
sanitization.  ``maze_aqueduct_pipes`` retains solved rotations and flow
schedules beside its raw source, so it keeps a family-owned allowlist: only
the visible geometry and the displayed pipe states may enter
``semantic_input``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Spatial Navigation canonical source lacks {label}")
    return value


def _required(source: Mapping[str, Any], *keys: str, label: str) -> dict[str, Any]:
    missing = [key for key in keys if key not in source]
    if missing:
        raise ValueError(f"Spatial Navigation canonical source {label} lacks fields: {missing}")
    return {key: deepcopy(source[key]) for key in keys}


def canonical_aqueduct_semantic_input(structured_state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the minimal raw-source domain for one aqueduct sample.

    The allowlist keeps the visible grid geometry plus, per pipe, only its
    position, kind, and displayed rotation; solved rotations, flow orders,
    and render schedules stay out.  The source stays top-level because the
    shared path evaluator consumes the grid geometry directly.
    """

    state = _mapping(structured_state, "aqueduct structured state")
    specs = state.get("pipe_specs")
    if not isinstance(specs, list) or not specs:
        raise ValueError("Spatial Navigation canonical aqueduct source lacks visible pipe specs")
    visible_specs = [
        _required(
            _mapping(spec, "aqueduct pipe spec"),
            "grid_row",
            "grid_col",
            "kind",
            "rotation_displayed",
            label="aqueduct visible pipe spec",
        )
        for spec in specs
    ]
    source = _required(
        state,
        "canvas_dimensions",
        "grid_size",
        "cell_size",
        "padding",
        label="aqueduct geometry",
    )
    source["pipe_specs"] = visible_specs
    return source


__all__ = ["canonical_aqueduct_semantic_input"]

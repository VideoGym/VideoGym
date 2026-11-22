"""Canonical source-only semantic inputs for the Physics family.

Physics generators retain rich solved state so they can author reference text,
terminal images, and videos.  None of those derived values belongs in the
canonical verifier source.  This module deliberately uses per-task allowlists
instead of a denylist: a newly added outcome, trace, render checkpoint, or
solver diagnostic therefore cannot silently enter ``semantic_input``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Physics canonical source lacks {label}")
    return value


def _required(source: Mapping[str, Any], *keys: str, label: str) -> dict[str, Any]:
    missing = [key for key in keys if key not in source]
    if missing:
        raise ValueError(f"Physics canonical source {label} lacks fields: {missing}")
    return {key: deepcopy(source[key]) for key in keys}


def _native_state(task_name: str, state: Mapping[str, Any]) -> dict[str, Any]:
    if task_name == "ball_bounces_given_time":
        evidence = _mapping(state.get("source_evidence"), "ball source_evidence")
        return {
            "source_evidence": _required(
                evidence,
                "start",
                "initial_velocity",
                "bounce_count",
                "center_bounds",
                "candidates",
                label="ball source_evidence",
            )
        }
    if task_name == "domino_reach_cascade":
        inner = _mapping(state.get("semantic_state"), "domino semantic_state")
        return _required(inner, "nodes", "edges", "px_per_unit", label="domino network state")
    if task_name == "chamber_flow_fill":
        inner = _mapping(state.get("semantic_state"), "chamber semantic_state")
        return _required(inner, "widths", "crests", "source", label="chamber state")
    if task_name == "elastic_collision_track":
        inner = _mapping(state.get("semantic_state"), "collision semantic_state")
        return _required(inner, "starts", "velocities", "horizon", label="collision state")
    if task_name == "gear_train_direction":
        inner = _mapping(state.get("semantic_state"), "gear semantic_state")
        return _required(
            inner, "gears", "belts", "driver_direction", "target", label="gear train state"
        )
    if task_name == "water_pour_levels":
        inner = _mapping(state.get("semantic_state"), "pour semantic_state")
        return _required(inner, "capacities", "levels", "pours", label="pour state")
    if task_name == "mirror_aim_detector":
        inner = _mapping(state.get("semantic_state"), "mirror semantic_state")
        evidence = _mapping(inner.get("source_evidence"), "mirror source_evidence")
        return {
            "source_evidence": _required(
                evidence,
                "source",
                "entry_dir",
                "mirrors",
                "detectors",
                label="mirror source_evidence",
            )
        }

    raise ValueError(f"No source-only Physics canonicalization for {task_name!r}")


def canonical_physics_semantic_input(
    task_name: str,
    structured_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the minimal raw-source domain for one Physics sample.

    Native Physics tasks use the normal ``semantic_state`` envelope consumed
    by ``source_replay``.
    """

    state = _mapping(structured_state, f"{task_name} structured state")
    return {"semantic_state": _native_state(task_name, state)}


__all__ = ["canonical_physics_semantic_input"]

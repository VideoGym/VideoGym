from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from data.task_spec import TaskFamily, TaskRoute, TaskSpec

from data.board_game.core.family import FAMILY as BOARD_GAME
from data.constraint_grid.core.family import FAMILY as CONSTRAINT_GRID
from data.dynamics_game.core.family import FAMILY as DYNAMICS_GAME
from data.geometry.core.family import FAMILY as GEOMETRY
from data.graph_theory.core.family import FAMILY as GRAPH_THEORY
from data.in_context_learning.core.family import FAMILY as IN_CONTEXT_LEARNING
from data.next_state_prediction.core.family import FAMILY as NEXT_STATE_PREDICTION
from data.physics.core.family import FAMILY as PHYSICS
from data.rearrangement_game.core.family import FAMILY as REARRANGEMENT_GAME
from data.spatial_3d.core.family import FAMILY as SPATIAL_3D
from data.spatial_navigation.core.family import FAMILY as SPATIAL_NAVIGATION

FAMILIES: Tuple[TaskFamily, ...] = (
    BOARD_GAME,
    CONSTRAINT_GRID,
    DYNAMICS_GAME,
    GEOMETRY,
    GRAPH_THEORY,
    IN_CONTEXT_LEARNING,
    NEXT_STATE_PREDICTION,
    PHYSICS,
    REARRANGEMENT_GAME,
    SPATIAL_3D,
    SPATIAL_NAVIGATION,
)


def _build_registry() -> tuple[Dict[str, TaskSpec], Dict[str, TaskFamily]]:
    family_map: Dict[str, TaskFamily] = {}
    routes: Dict[str, tuple[TaskFamily, TaskRoute]] = {}
    for family in FAMILIES:
        if family.name in family_map:
            raise ValueError(f"Duplicate Task Family: {family.name}")
        family_map[family.name] = family
        for route in family.tasks:
            if route.name in routes:
                previous = routes[route.name][0].name
                raise ValueError(
                    f"Task {route.name!r} is assigned to both {previous!r} and {family.name!r}"
                )
            routes[route.name] = (family, route)
    specs = {
        name: TaskSpec(
            name=name,
            task_family=family.name,
            module=route.module,
            class_name=route.class_name,
            verifier=route.verifier,
            requires=route.requires,
            answer_kind=route.answer_kind,
            answer_codec=route.answer_codec,
            position_collection_order=route.position_collection_order,
        )
        for name, (family, route) in routes.items()
    }
    return specs, family_map


TASK_SPECS, TASK_FAMILY_SPECS = _build_registry()
TASK_FAMILIES: Tuple[str, ...] = tuple(family.name for family in FAMILIES)


def get_task_spec(task_name: str) -> Optional[TaskSpec]:
    spec = TASK_SPECS.get(task_name)
    if spec is None:
        from evaluation.catalog.registry import EVAL_TASK_SPECS

        spec = EVAL_TASK_SPECS.get(task_name)
    return spec


def resolve_requested_tasks(
    tasks: Optional[List[str]],
    task_families: Optional[List[str]],
) -> List[str]:
    requested_names = list(tasks or ["all"])
    unknown_tasks = sorted(set(requested_names) - {"all"} - set(TASK_SPECS))
    if unknown_tasks:
        raise ValueError(f"Unknown Tasks: {', '.join(unknown_tasks)}")
    selected_families = set(task_families or [])
    requested = {task for task in requested_names if task != "all"}
    if "all" in requested_names:
        requested.update(TASK_SPECS)
    if selected_families:
        invalid = sorted(selected_families - set(TASK_FAMILIES))
        if invalid:
            raise ValueError(f"Unknown Task Families: {', '.join(invalid)}")
        requested = {
            task
            for task in requested
            if TASK_SPECS[task].task_family in selected_families
        }
    return sorted(requested)


__all__ = [
    "TASK_FAMILIES",
    "TASK_FAMILY_SPECS",
    "TASK_SPECS",
    "TaskFamily",
    "TaskRoute",
    "TaskSpec",
    "get_task_spec",
    "resolve_requested_tasks",
]

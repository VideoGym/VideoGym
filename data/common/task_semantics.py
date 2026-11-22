from __future__ import annotations

import json
from pathlib import Path

TASK_GROUPS = {
    "single_step": {
        "label": "Single-Step Group",
        "description": "Marks a fixed answer or construction on an unchanging puzzle scene, or applies one simultaneous state update; visual execution is grouped independently of internal search depth.",
    },
    "multi_step": {
        "label": "Multi-Step Group",
        "description": "Executes sequential state transitions or chained transformations where each stage updates the visual state for the next.",
    },
}

TASK_GROUP_BY_TASK = {
    task: group
    for group, tasks in json.loads(Path(__file__).with_name("task_groups.json").read_text()).items()
    for task in tasks
}


def get_task_group(task_name: str, *, evaluation_task: str | None = None) -> str | None:
    """Resolve the evaluated contract before falling back to its source template."""
    return TASK_GROUP_BY_TASK.get(evaluation_task, TASK_GROUP_BY_TASK.get(task_name))


TASK_FAMILY_LABELS = {
    "geometry": "Geometry",
    "graph_theory": "Graph Theory",
    "in_context_learning": "In-Context Learning",
    "next_state_prediction": "Next State Prediction",
    "spatial_navigation": "Spatial Navigation",
    "rearrangement_game": "Rearrangement Game",
    "dynamics_game": "Dynamics Game",
    "constraint_grid": "Constraint Grid",
    "board_game": "Board Game",
    "physics": "Physics",
    "spatial_3d": "Spatial 3D",
}

VALID_SPLITS = ("train", "eval")


def normalize_split(value: object, *, default: str = "train") -> str:
    text = str(value or "").strip().lower()
    return text if text in VALID_SPLITS else default

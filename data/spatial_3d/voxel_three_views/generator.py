"""Voxel orthographic views: fill the asked view grid of a column structure.

A small structure of stacked unit blocks (columns rising from a 1-based
labeled floor) is shown in the family three-quarter camera; a bold blue 3D
arrow names the asked orthographic view (top, or one of two side directions).
The answer is the exact binary view grid, typed as JSON and painted into the
blank answer panel.  Sampling gates guarantee the structure is fully readable
from the single picture: every column top, every empty floor cell, and at
least one side face of every block stay visible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract
from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    cell_to_xy,
    format_xy_position,
    format_xy_positions_payload,
)
from data.common.pipeline.structured_grid import matrix_to_json
from data.spatial_3d.core.iso_geometry import silhouette_view
from data.spatial_3d.core.scene_base import Spatial3DScenePattern
from data.spatial_3d.core.view_scene import (
    build_geometry,
    render_view_scene,
    structure_visibility,
)

_MAX_HEIGHT = 3
_VIEWS = ("top", "front", "right")
def heightmap_cells(heights: Sequence[Sequence[int]]) -> List[Tuple[int, int, int]]:
    return [
        (x, y, z)
        for y, row in enumerate(heights)
        for x, value in enumerate(row)
        for z in range(int(value))
    ]


def replay_voxel_views(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Independent forward replay from the visible structure and view only."""

    heights = [[int(v) for v in row] for row in source["heights"]]
    view = str(source["view"])
    size = len(heights)
    if size not in (3, 4) or any(len(row) != size for row in heights):
        raise ValueError("voxel view heights must form a 3x3 or 4x4 floor")
    if any(not 0 <= v <= _MAX_HEIGHT for row in heights for v in row):
        raise ValueError("voxel view heights must lie in 0..3")
    if view not in _VIEWS:
        raise ValueError(f"unsupported voxel view: {view!r}")
    cells = heightmap_cells(heights)
    if not cells:
        raise ValueError("voxel view structure must contain at least one block")
    grid = silhouette_view(cells, view, size_x=size, size_y=size, height=_MAX_HEIGHT)
    return {"heights": heights, "view": view, "cells": cells, "grid": grid}


class VoxelThreeViewsPattern(Spatial3DScenePattern):
    task_name = "voxel_three_views"
    task_family = "spatial_3d"
    task_mechanic = "orthographic_view_projection"
    motion = False
    verifier_profile = "voxel_views_visible_structure_v1"

    # ------------------------------------------------------------- sampling

    def _sample_task(self) -> Dict[str, Any]:
        profile = {
            "min": ((3,), (4, 5)),
            "typical": ((3, 4), (5, 7)),
            "max": ((4,), (7, 9)),
        }.get(self.difficulty or "", ((3, 3, 4), (4, 9)))
        for _ in range(4000):
            size = self.rng.choice(profile[0])
            target_blocks = self.rng.randint(*profile[1])
            heights = [[0] * size for _ in range(size)]
            placed = 0
            columns = [(x, y) for x in range(size) for y in range(size)]
            self.rng.shuffle(columns)
            for x, y in columns:
                if placed >= target_blocks:
                    break
                height = self.rng.choice((0, 1, 1, 2, 2, 3))
                height = min(height, target_blocks - placed)
                heights[y][x] = height
                placed += height
            task = self._build_task(heights)
            if task is not None:
                return task
        raise RuntimeError("voxel_three_views could not sample a readable structure")

    def _build_task(self, heights: List[List[int]]) -> Optional[Dict[str, Any]]:
        size = len(heights)
        values = [v for row in heights for v in row if v > 0]
        if len(values) < 2 or len(set(values)) < 2:
            return None
        empty_cells = sum(1 for row in heights for v in row if v == 0)
        if empty_cells < 2:
            return None
        # Side views exercise the full height reasoning; the top view is the
        # lighter footprint read, so it appears with reduced weight.
        view = self.rng.choice(("top", "front", "front", "right", "right"))
        replayed = replay_voxel_views({"heights": heights, "view": view})
        grid = replayed["grid"]
        filled = sum(sum(row) for row in grid)
        if filled == 0 or filled == len(grid) * len(grid[0]):
            return None
        geometry = build_geometry(size, view, len(grid), len(grid[0]))
        evidence = structure_visibility(replayed["cells"], size, geometry)
        if (
            evidence["min_empty_floor_visibility"] < 0.45
            or evidence["min_column_top_visibility"] < 0.55
            or evidence["min_cube_side_visibility"] < 0.25
        ):
            return None
        return {
            "heights": heights,
            "size": size,
            "view": view,
            "cells": replayed["cells"],
            "grid": grid,
            "geometry": geometry,
            "visibility_evidence": evidence,
        }

    # ------------------------------------------------------------- rendering

    def _panel_fill(self, task: Mapping[str, Any], filled: bool) -> Optional[List[List[Optional[str]]]]:
        if not filled:
            return None
        return [["#" if value else None for value in row] for row in task["grid"]]

    def _stage_count(self, task: Mapping[str, Any]) -> int:
        return 2

    def _render_stage(self, task: Mapping[str, Any], index: int) -> Image.Image:
        return render_view_scene(
            task["cells"],
            task["size"],
            task["geometry"],
            panel_fill=self._panel_fill(task, index == 1),
        )

    # ------------------------------------------------------------- semantics

    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        return matrix_to_json(task["grid"])

    def _question(self, task: Mapping[str, Any]) -> str:
        return "Fill the answer panel with the view of the structure seen along the bold arrow."

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[Any, str]]:
        heights = task["heights"]
        size = task["size"]
        view = task["view"]
        grid = task["grid"]
        sentences: List[str] = []
        if view == "top":
            sentences.append(
                "The arrow looks straight down, so the view is the floor plan: a view cell is filled "
                "exactly where a stack of any height stands."
            )
            for y in range(size):
                stacks = [
                    cell_to_xy((y, x))
                    for x in range(size)
                    if heights[y][x] > 0
                ]
                if stacks:
                    sentences.append(
                        f"Floor positions {format_xy_positions_payload(stacks)} hold stacks."
                    )
                else:
                    sentences.append(f"Floor row {y + 1} is empty.")
        else:
            axis_word = "column" if view == "front" else "row"
            sentences.append(
                f"The arrow looks along the floor, so each view column collects one floor {axis_word}: "
                f"view column i lines up with floor {axis_word} i, and a view cell is filled up to the "
                f"tallest stack in that {axis_word}, top layer in view row 1."
            )
            for index in range(size):
                if view == "front":
                    line = [heights[y][index] for y in range(size)]
                else:
                    line = [heights[index][x] for x in range(size)]
                tallest = max(line)
                if tallest == 0:
                    sentences.append(f"Floor {axis_word} {index + 1} is empty, so view column {index + 1} stays blank.")
                else:
                    positions = [i for i, value in enumerate(line) if value == tallest]
                    floor_cell = (
                        (positions[0], index)
                        if view == "front"
                        else (index, positions[0])
                    )
                    where = format_xy_position(cell_to_xy(floor_cell))
                    sentences.append(
                        f"Floor {axis_word} {index + 1} rises to height {tallest} (tallest stack at {where}), "
                        f"so view column {index + 1} fills its bottom {tallest} cell{'s' if tallest > 1 else ''}."
                    )
        filled_count = sum(sum(row) for row in grid)
        sentences.append(
            f"That fills {filled_count} view cell{'s' if filled_count != 1 else ''}, and the panel rows from "
            f"top to bottom give the answer grid."
        )
        return [(1, " ".join(sentences))]

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "semantic_state": {
                "heights": [list(row) for row in task["heights"]],
                "view": task["view"],
            },
            "view_grid": [list(row) for row in task["grid"]],
            "visibility_evidence": dict(task["visibility_evidence"]),
            "multiple_solution_policy": "exactly one binary view grid follows from the visible structure",
            "difficulty_evidence": {
                "floor_size": task["size"],
                "block_count": len(task["cells"]),
                "distinct_heights": len({v for row in task["heights"] for v in row if v > 0}),
            },
            "answer_schema": {
                "schema_version": "voxel_view_grid_v1",
                "kind": "json_binary_grid",
                "parser_profile": "strict_json_binary_grid_with_period",
                "format_template": "Answer: [[<0-or-1>, ...], ...].",
                "format_example": "Answer: [[0, 1, ...], [1, 0, ...], ...].",
                "judged_artifacts": {
                    "ti2v": ["full_video"],
                },
                "accepts_multiple_valid_answers": False,
                "extra_text_allowed": False,
            },
            "verifier_contract": {
                "profile": self.verifier_profile,
                "semantic_source": "visible column structure, floor labels, and the bold view arrow",
                "strict_binary_correctness": True,
                "strict_text_parser": "strict_json_binary_grid_with_period",
                "untrusted_redundant_fields": ["view_grid"],
            },
        }

    def _glyph_inventory(self, task: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
        size = int(task["size"])
        rows = len(task["grid"])
        cols = len(task["grid"][0])
        labels = sorted({str(v + 1) for v in range(max(size, rows, cols))}, key=lambda v: (len(v), v))
        inventory = {
            "schema_version": "voxel_views_rendered_glyph_inventory_v1",
            "fixed_labels": labels,
            "dynamic_labels": [],
            "all_rendered_text": labels,
        }
        schema = {
            "fixed_labels": "1-based floor row/column numbers and answer-panel row/column numbers only",
            "dynamic_labels": "none: the view direction is a drawn arrow, not text",
            "exclusions": "no task title, rule prose, status sentence, or answer banner",
        }
        return inventory, schema

    # -------------------------------------------------------------- contract

    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        view = task["view"]
        grid_rows = len(task["grid"])
        if view == "top":
            mapping_sentence = (
                "For this arrow the view is seen from straight above: view row i is floor row i and view "
                "column j is floor column j, and a view cell is filled exactly where a stack of any height "
                "stands."
            )
        elif view == "front":
            mapping_sentence = (
                "For this arrow the view is seen from beside the floor: view column j lines up with floor "
                "column j, view row 1 is the highest layer, and a view cell is filled when any stack in "
                "that floor column reaches that layer."
            )
        else:
            mapping_sentence = (
                "For this arrow the view is seen from beside the floor: view column j lines up with floor "
                "row j, view row 1 is the highest layer, and a view cell is filled when any stack in that "
                "floor row reaches that layer."
            )
        return VisualReasoningContract(
            input_description=(
                "The single image shows a structure of identical unit blocks standing in stacks on a floor "
                "grid, one bold blue arrow floating beside the scene, and a blank answer panel at the "
                "bottom. Blue numbers along two floor edges are the 1-based floor column and row numbers, "
                f"and matching blue numbers label the answer panel's columns and rows. {GRID_XY_CONVENTION_SENTENCE}"
            ),
            symbol_legend=(
                "Every block is a full unit cube resting on the floor or on another block.",
                "The bold blue arrow points the way the required view looks at the structure.",
            ),
            rules=(
                "The required view is the flat picture of the structure seen along the arrow: each view "
                "cell is filled if any block of the structure lies behind it in the viewing direction, and "
                "empty otherwise. "
                + mapping_sentence,
                f"The answer panel has exactly {grid_rows} rows and {len(task['grid'][0])} columns; its "
                "numbers give the 1-based view rows and columns.",
            ),
            modality_outputs={
                "ti2v": (
                    "The answer panel completes in one commit: all filled view cells darken together, "
                    "fading from transparent to fully dark inside their exact cells, and the finished view "
                    "then holds."
                ),
                "solution_image": (
                    "Complete the answer panel: paint every filled view cell dark and leave every empty "
                    "cell white; the structure, arrow, and labels stay as given."
                ),
            },
            preservation_constraints=(
                "the block structure, the floor grid and its numbers, the blue arrow, the answer panel "
                "frame and its numbers, and the background unchanged; only answer-panel cells may darken.",
            ),
            transition_constraints=(
                "Commit the whole view as one synchronized element-local fade of its filled cells at fixed "
                "geometry; never fill cells one by one, move a cell, or dissolve the full frame.",
            ),
            ti2v_continuity={
                "moving_nouns": "the dark view cells appearing in the answer panel",
                "static_nouns": "the block structure, the floor grid and numbers, the blue arrow, the "
                "answer-panel frame and numbers, the background, and the camera",
                "hold_noun": "the completed view panel",
            },
            answer=AnswerContract(
                kind="grid",
                format_template="Answer: [[<0-or-1>, ...], ...].",
                format_example="Answer: [[0, 1, ...], [1, 0, ...], ...].",
                parser_profile="strict_json_binary_grid_with_period",
                verifier_profile=self.verifier_profile,
            ),
            render_profile="voxel_three_views_v1",
        )


__all__ = ["VoxelThreeViewsPattern", "heightmap_cells", "replay_voxel_views"]

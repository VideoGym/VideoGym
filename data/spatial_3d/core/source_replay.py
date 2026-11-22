"""Source-only semantic replay and rendering for the Spatial 3D family. Only fields that describe the visible initial state or an explicitly stated
rule are read here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Optional, Tuple

from PIL import Image

from data.common.schemas import CanonicalSample


@dataclass(frozen=True)
class Spatial3DReplay:
    # Comparable Python value of the unique correct final answer.
    answer: Any
    source_variant: str
    # True: continuous-motion process checked against a progress bank.
    # False: ordered discrete states joined by element-local fades.
    motion: bool
    render_progress: Optional[Callable[[float], Image.Image]] = None
    semantic_checkpoints: Tuple[float, ...] = ()
    render_stage: Optional[Callable[[int], Image.Image]] = None
    stage_count: int = 0
    # Source-derived per-member evidence boxes ``(name, (x0, y0, x1, y1))`` in
    # 512-space, each hard-gated individually against the rendered terminal
    # so one wrong member can never be averaged away.
    member_regions: Tuple[Tuple[str, Tuple[float, float, float, float]], ...] = ()


def source_state(sample: CanonicalSample) -> Mapping[str, Any]:
    semantic = sample.semantic_input
    if not isinstance(semantic, Mapping):
        return {}
    for key in ("semantic_state", "source_contract", "structured_input"):
        value = semantic.get(key)
        if isinstance(value, Mapping):
            return value
    return semantic


def _paper_fold_replay(state: Mapping[str, Any]) -> Spatial3DReplay:
    from data.spatial_3d.paper_fold_hole_matrix.generator import (
        PaperFoldHoleMatrixPattern,
        replay_paper_fold,
    )

    replayed = replay_paper_fold(state)
    renderer = PaperFoldHoleMatrixPattern(Path(tempfile.gettempdir()), seed=0, video=False)
    task = {
        "size": replayed["grid_size"],
        "folds": replayed["fold_sequence"],
        "punched": [tuple(cell) for cell in replayed["punched_cells"]],
        "unfolded": replayed["unfolded_hole_matrix"],
        "states": replayed["states"],
        "final_shape": tuple(replayed["final_folded_shape"]),
    }
    left, top, cell, _extent = renderer.answer_grid_geometry(replayed["grid_size"])
    member_regions = tuple(
        (
            f"hole_r{row + 1}c{col + 1}",
            (
                float(left + col * cell),
                float(top + row * cell),
                float(left + (col + 1) * cell),
                float(top + (row + 1) * cell),
            ),
        )
        for row, line in enumerate(replayed["unfolded_hole_matrix"])
        for col, value in enumerate(line)
        if value
    )
    return Spatial3DReplay(
        answer=[list(row) for row in replayed["unfolded_hole_matrix"]],
        source_variant="paper_fold_visible_program_v4",
        motion=False,
        render_stage=lambda index: renderer._render_stage(task, index),
        stage_count=len(replayed["states"]),
        member_regions=member_regions,
    )


def _cube_net_replay(state: Mapping[str, Any]) -> Spatial3DReplay:
    from data.spatial_3d.cube_net_fold.generator import CubeNetFoldPattern, replay_cube_net

    replayed = replay_cube_net(state)
    renderer = CubeNetFoldPattern(Path(tempfile.gettempdir()), seed=0, video=False)
    cells = [(int(c), int(r)) for c, r in state["net_cells"]]
    base = (int(state["base_cell"][0]), int(state["base_cell"][1]))
    pips = {(int(c), int(r)): int(p) for c, r, p in state["pip_by_cell"]}
    task = renderer._build_task(cells, base, pips)
    if task is None:
        raise ValueError("cube net source does not fit the family camera")
    # the lid and both camera-facing walls of the assembled box carry
    # their own evidence boxes, projected through the same fixed camera.
    import math as _math

    net = task["net"]
    final_faces = {
        face.cell: face
        for face in net.folded_faces({cell: _math.pi / 2.0 for cell in cells if cell != base})
    }
    roles = replayed["roles"]
    projection = task["projection"]
    member_regions = []
    for role_name in ("top", "north", "east"):
        cell = next((c for c, role in roles.items() if role == role_name), None)
        if cell is None:
            continue
        quad = [projection.project(point) for point in final_faces[cell].corners]
        xs = [point[0] for point in quad]
        ys = [point[1] for point in quad]
        member_regions.append(
            (f"box_{role_name}_{pips[cell]}pips", (min(xs) - 2.0, min(ys) - 2.0, max(xs) + 2.0, max(ys) + 2.0))
        )
    return Spatial3DReplay(
        answer=int(replayed["answer"]),
        source_variant="cube_net_visible_fold_program_v1",
        motion=True,
        render_progress=lambda progress: renderer._render_progress(task, progress),
        semantic_checkpoints=renderer.semantic_checkpoints(task),
        member_regions=tuple(member_regions),
    )


def _polycube_replay(state: Mapping[str, Any]) -> Spatial3DReplay:
    from data.spatial_3d.polycube_rotation.generator import (
        PolycubeRotationPattern,
        _bbox_center,
        replay_polycube,
    )

    replayed = replay_polycube(state)
    renderer = PolycubeRotationPattern(Path(tempfile.gettempdir()), seed=0, video=False)
    cells = replayed["states"][0]
    rotations = [(str(axis), int(sign)) for axis, sign in state["rotations"]]
    options = [
        {"label": option["label"], "cells": option["cells"]}
        for option in replayed["options"]
    ]
    task = renderer._build_task(cells, rotations, replayed["states"], options, replayed["answer"])
    if task is None:
        raise ValueError("polycube source does not fit the family camera")
    final = replayed["states"][-1]
    center = _bbox_center(final)
    projection = task["projection"]
    member_regions = []
    for index, cell in enumerate(sorted(final)):
        corners = [
            projection.project((cell[0] + dx - center[0], cell[1] + dy - center[1], cell[2] + dz - center[2]))
            for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)
        ]
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        member_regions.append(
            (f"cube_{index + 1}", (min(xs) - 2.0, min(ys) - 2.0, max(xs) + 2.0, max(ys) + 2.0))
        )
    return Spatial3DReplay(
        answer=str(replayed["answer"]),
        source_variant="polycube_rotation_visible_program_v1",
        motion=True,
        render_progress=lambda progress: renderer._render_progress(task, progress),
        semantic_checkpoints=renderer.semantic_checkpoints(task),
        member_regions=tuple(member_regions),
    )


def _panel_member_regions(geometry, grid, prefix: str):
    """One evidence box per non-empty answer-panel cell."""

    regions = []
    for row, line in enumerate(grid):
        for col, token in enumerate(line):
            if token in (0, ".", None):
                continue
            box = geometry.panel_cell_box(row, col)
            regions.append(
                (f"{prefix}_r{row + 1}c{col + 1}", tuple(float(v) for v in box))
            )
    return tuple(regions)


def _voxel_views_replay(state: Mapping[str, Any]) -> Spatial3DReplay:
    from data.spatial_3d.core.view_scene import build_geometry
    from data.spatial_3d.voxel_three_views.generator import (
        VoxelThreeViewsPattern,
        replay_voxel_views,
    )

    replayed = replay_voxel_views(state)
    renderer = VoxelThreeViewsPattern(Path(tempfile.gettempdir()), seed=0, video=False)
    grid = replayed["grid"]
    geometry = build_geometry(len(replayed["heights"]), replayed["view"], len(grid), len(grid[0]))
    task = {
        "heights": replayed["heights"],
        "size": len(replayed["heights"]),
        "view": replayed["view"],
        "cells": replayed["cells"],
        "grid": grid,
        "geometry": geometry,
    }
    return Spatial3DReplay(
        answer=[list(row) for row in grid],
        source_variant="voxel_views_visible_structure_v1",
        motion=False,
        render_stage=lambda index: renderer._render_stage(task, index),
        stage_count=2,
        member_regions=_panel_member_regions(geometry, grid, "view"),
    )


def _stack_orbit_replay(state: Mapping[str, Any]) -> Spatial3DReplay:
    from data.spatial_3d.core.view_scene import build_geometry
    from data.spatial_3d.stack_orbit_view.generator import (
        StackOrbitViewPattern,
        replay_stack_orbit,
    )

    replayed = replay_stack_orbit(state)
    renderer = StackOrbitViewPattern(Path(tempfile.gettempdir()), seed=0, video=False)
    grid = replayed["grid"]
    geometry = build_geometry(int(state["floor_size"]), replayed["view"], len(grid), len(grid[0]))
    task = {
        "cell_colors": replayed["cell_colors"],
        "view": replayed["view"],
        "grid": grid,
        "geometry": geometry,
    }
    return Spatial3DReplay(
        answer=[list(row) for row in grid],
        source_variant="stack_orbit_visible_structure_v1",
        motion=False,
        render_stage=lambda index: renderer._render_stage(task, index),
        stage_count=2,
        member_regions=_panel_member_regions(geometry, grid, "view"),
    )


def _slice_replay(state: Mapping[str, Any]) -> Spatial3DReplay:
    from data.spatial_3d.slice_cross_section.generator import (
        SliceCrossSectionPattern,
        replay_slice,
    )

    replayed = replay_slice(state)
    renderer = SliceCrossSectionPattern(Path(tempfile.gettempdir()), seed=0, video=False)
    task = {
        "solid": replayed["solid"],
        "cut": replayed["cut"],
        "word": replayed["word"],
        "section": replayed["section"],
    }
    task["projection"] = renderer._fit_camera(task)
    # members: the traced outline band on the solid and the panel shape.
    projection = task["projection"]
    loop = [projection.project(p) for p in replayed["section"]["loop_3d"]]
    xs = [p[0] for p in loop]
    ys = [p[1] for p in loop]
    from data.spatial_3d.slice_cross_section.generator import _PANEL_BOX

    member_regions = (
        ("cut_outline", (min(xs) - 4.0, min(ys) - 4.0, max(xs) + 4.0, max(ys) + 4.0)),
        ("panel_shape", tuple(float(v) for v in _PANEL_BOX)),
    )
    return Spatial3DReplay(
        answer=str(replayed["word"]),
        source_variant="slice_cross_section_visible_cut_v1",
        motion=False,
        render_stage=lambda index: renderer._render_stage(task, index),
        stage_count=3,
        member_regions=member_regions,
    )


def replay_spatial_3d(sample: CanonicalSample, *, modality: str | None = None) -> Spatial3DReplay:
    del modality
    state = source_state(sample)
    if sample.task_name == "paper_fold_hole_matrix":
        return _paper_fold_replay(state)
    if sample.task_name == "cube_net_fold":
        return _cube_net_replay(state)
    if sample.task_name == "polycube_rotation":
        return _polycube_replay(state)
    if sample.task_name == "voxel_three_views":
        return _voxel_views_replay(state)
    if sample.task_name == "stack_orbit_view":
        return _stack_orbit_replay(state)
    if sample.task_name == "slice_cross_section":
        return _slice_replay(state)
    raise ValueError(f"unsupported Spatial 3D source replay task: {sample.task_name}")


__all__ = ["Spatial3DReplay", "replay_spatial_3d", "source_state"]

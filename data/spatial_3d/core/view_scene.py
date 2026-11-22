"""Shared voxel-scene-with-view-panel rendering for the Spatial 3D family.

Both orthographic-view tasks (silhouette and colored nearest-face) draw the
same world: a small column structure on a 1-based labeled floor, one bold
blue 3D view arrow naming the asked viewing direction, and a blank answer
view panel with its own 1-based labels.  Geometry (projection, floor label
anchors, arrow endpoints, panel cell boxes) lives here as the single truth
consumed by generators, reference authoring, and verifier pixel decodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from data.common.pipeline.structured_grid import load_font
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.spatial_3d.core.iso_geometry import (
    Cell,
    Face,
    IsoProjection,
    face_visibility,
    fit_projection,
    floor_grid_faces,
    render_faces,
    voxel_faces,
)

BG = (246, 248, 251)
CARD = (255, 255, 253)
CARD_EDGE = (216, 221, 226)
BLOCK_COLOR = (196, 210, 226)
ARROW_COLOR = (57, 115, 182)
AXIS_LABEL_COLOR = (30, 90, 220)
PANEL_LINE = (170, 176, 184)
PANEL_FRAME = (90, 96, 108)
FILL_DARK = (46, 54, 68)

# Answer-letter palette shared by the colored-view task; prompt legends and
# verifier decodes both read from here.
COLOR_LETTERS: Dict[str, Tuple[int, int, int]] = {
    "R": (214, 84, 84),
    "G": (96, 178, 102),
    "B": (92, 146, 214),
    "Y": (226, 196, 80),
    "P": (164, 116, 204),
}
COLOR_WORDS: Dict[str, str] = {
    "R": "red",
    "G": "green",
    "B": "blue",
    "Y": "yellow",
    "P": "purple",
}

_SCENE_BOX = (36.0, 30.0, 476.0, 322.0)
_MAX_HEIGHT = 3


@dataclass(frozen=True)
class ViewSceneGeometry:
    projection: IsoProjection
    floor_size: int
    view: str
    panel_rows: int
    panel_cols: int
    panel_left: int
    panel_top: int
    panel_cell: int

    def panel_cell_box(self, row: int, col: int) -> Tuple[int, int, int, int]:
        return (
            self.panel_left + col * self.panel_cell,
            self.panel_top + row * self.panel_cell,
            self.panel_left + (col + 1) * self.panel_cell,
            self.panel_top + (row + 1) * self.panel_cell,
        )


def arrow_segment(view: str, floor_size: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """3D tail and head of the bold view arrow for one named view."""

    center = floor_size / 2.0
    if view == "top":
        return (center, center, _MAX_HEIGHT + 2.1), (center, center, _MAX_HEIGHT + 0.9)
    if view == "front":
        return (center, -2.1, 0.5), (center, -0.9, 0.5)
    if view == "right":
        return (floor_size + 2.1, center, 0.5), (floor_size + 0.9, center, 0.5)
    if view == "back":
        return (center, floor_size + 2.1, 0.5), (center, floor_size + 0.9, 0.5)
    if view == "left":
        return (-2.1, center, 0.5), (-0.9, center, 0.5)
    raise ValueError(f"unsupported view: {view!r}")


def build_geometry(floor_size: int, view: str, panel_rows: int, panel_cols: int) -> ViewSceneGeometry:
    tail, head = arrow_segment(view, floor_size)
    cloud = [
        (x, y, z)
        for x in (0, floor_size)
        for y in (0, floor_size)
        for z in (0, _MAX_HEIGHT)
    ]
    # Label anchors sit just outside the two near floor edges.
    cloud.extend((x + 0.5, floor_size + 0.9, 0.0) for x in range(floor_size))
    cloud.extend((floor_size + 0.9, y + 0.5, 0.0) for y in range(floor_size))
    cloud.extend((tail, head))
    projection = fit_projection(cloud, box=_SCENE_BOX, max_scale=46.0)
    cell = min(30, 150 // max(panel_rows, 1), 300 // max(panel_cols, 1))
    panel_width = cell * panel_cols
    panel_left = (512 - panel_width) // 2 + 12
    panel_top = 500 - 12 - cell * panel_rows
    return ViewSceneGeometry(
        projection=projection,
        floor_size=floor_size,
        view=view,
        panel_rows=panel_rows,
        panel_cols=panel_cols,
        panel_left=panel_left,
        panel_top=panel_top,
        panel_cell=cell,
    )


def scene_faces(
    cells: Sequence[Cell],
    floor_size: int,
    colors: Optional[Mapping[Cell, str]] = None,
) -> List[Face]:
    rgb = (
        {cell: COLOR_LETTERS[colors[cell]] for cell in colors}
        if colors is not None
        else None
    )
    return [
        *floor_grid_faces(floor_size, floor_size),
        *voxel_faces(list(cells), rgb, default_color=BLOCK_COLOR),
    ]


def structure_visibility(
    cells: Sequence[Cell],
    floor_size: int,
    geometry: ViewSceneGeometry,
) -> Dict[str, float]:
    """evidence: per-face visibility fractions grouped by role."""

    floor = floor_grid_faces(floor_size, floor_size)
    voxels = voxel_faces(list(cells), None, default_color=BLOCK_COLOR)
    fractions = face_visibility([*floor, *voxels], geometry.projection)
    floor_count = len(floor)
    occupied_floor = {(x, y) for x, y, _z in cells}
    empty_floor_fractions = [
        fractions[index]
        for index, face in enumerate(floor)
        if (int(face.points[0][0]), int(face.points[0][1])) not in occupied_floor
    ]
    top_fractions: List[float] = []
    side_best: Dict[Cell, float] = {cell: 0.0 for cell in cells}
    tops = {}
    for x, y, _z in cells:
        column_height = max(z for cx, cy, z in cells if (cx, cy) == (x, y)) + 1
        tops[(x, y)] = column_height
    occupied = set(cells)
    for index, face in enumerate(voxels):
        normal = face.normal()
        base = tuple(int(round(min(p[axis] for p in face.points))) for axis in range(3))
        if abs(normal[2] - 1.0) < 1e-6:
            owner = (base[0], base[1], base[2] - 1)
            if owner in occupied and tops.get((owner[0], owner[1])) == owner[2] + 1:
                top_fractions.append(fractions[floor_count + index])
        else:
            candidates = (base, (base[0] - 1, base[1], base[2]), (base[0], base[1] - 1, base[2]))
            owner = next((cell for cell in candidates if cell in occupied), None)
            if owner is not None:
                side_best[owner] = max(side_best.get(owner, 0.0), fractions[floor_count + index])
    return {
        "min_empty_floor_visibility": min(empty_floor_fractions) if empty_floor_fractions else 1.0,
        "min_column_top_visibility": min(top_fractions) if top_fractions else 0.0,
        "min_cube_side_visibility": min(side_best.values()) if side_best else 0.0,
        "min_diagonal_separation_px": geometry.projection.min_diagonal_separation(
            max_span=2 * (floor_size - 1), max_height=_MAX_HEIGHT
        ),
    }


def per_cube_best_visibility(
    cells: Sequence[Cell],
    floor_size: int,
    geometry: ViewSceneGeometry,
) -> Dict[Cell, float]:
    """Best visible-face fraction per cube (any camera-facing face)."""

    floor = floor_grid_faces(floor_size, floor_size)
    voxels = voxel_faces(list(cells), None, default_color=BLOCK_COLOR)
    fractions = face_visibility([*floor, *voxels], geometry.projection)
    best: Dict[Cell, float] = {cell: 0.0 for cell in cells}
    for index, face in enumerate(voxels):
        base = tuple(int(round(min(p[axis] for p in face.points))) for axis in range(3))
        # A face's minimum corner may sit outside its own cube for the -x/-y
        # faces; recover the owning cube by testing both candidates.
        candidates = [base, (base[0] - 1, base[1], base[2]), (base[0], base[1] - 1, base[2]), (base[0], base[1], base[2] - 1)]
        owner = next((cell for cell in candidates if cell in best), None)
        if owner is None:
            continue
        best[owner] = max(best[owner], fractions[len(floor) + index])
    return best


def render_view_scene(
    cells: Sequence[Cell],
    floor_size: int,
    geometry: ViewSceneGeometry,
    *,
    colors: Optional[Mapping[Cell, str]] = None,
    panel_fill: Optional[Sequence[Sequence[Optional[str]]]] = None,
) -> Image.Image:
    """Render the full 512x512 scene plus the answer view panel.

    ``panel_fill`` is the panel state: ``None`` cells stay white, ``"#"``
    fills the family dark slate, and a colour letter fills that palette
    colour.  The blank input uses ``panel_fill=None``.
    """

    supersample = DOCTRINE_SUPERSAMPLE
    canvas, draw = new_scaled_canvas(512, 512, supersample, background=BG)
    draw.rounded_rectangle(
        (8, 8, 504, 504),
        radius=20,
        fill=CARD,
        outline=CARD_EDGE,
        width=2,
    )
    render_faces(canvas, scene_faces(cells, floor_size, colors), geometry.projection, supersample)

    # Floor labels on the two near edges (never occludable by the structure).
    label_font = load_font(12 * supersample)
    for x in range(floor_size):
        u, v = geometry.projection.project((x + 0.5, floor_size + 0.55, 0.0))
        draw.text((u, v), str(x + 1), fill=AXIS_LABEL_COLOR, font=label_font, anchor="mm")
    for y in range(floor_size):
        u, v = geometry.projection.project((floor_size + 0.55, y + 0.5, 0.0))
        draw.text((u, v), str(y + 1), fill=AXIS_LABEL_COLOR, font=label_font, anchor="mm")

    # Bold blue view arrow.  The two head barbs are one mitered polyline through
    # the head point, so the barbs meet in a point instead of as two stroke ends
    # butting against each other.
    tail, head = arrow_segment(geometry.view, floor_size)
    tail2 = geometry.projection.project(tail)
    head2 = geometry.projection.project(head)
    draw.line((tail2[0], tail2[1], head2[0], head2[1]), fill=ARROW_COLOR, width=5)
    angle = math.atan2(head2[1] - tail2[1], head2[0] - tail2[0])
    barbs = [
        (head2[0] + 13 * math.cos(angle + spread), head2[1] + 13 * math.sin(angle + spread))
        for spread in (math.radians(150.0), math.radians(-150.0))
    ]
    draw.open_path([barbs[0], (head2[0], head2[1]), barbs[1]], fill=ARROW_COLOR, width=5)

    # Answer view panel with 1-based labels.
    panel_font = load_font(11 * supersample)
    for col in range(geometry.panel_cols):
        box = geometry.panel_cell_box(0, col)
        draw.text(
            ((box[0] + box[2]) / 2.0, geometry.panel_top - 10),
            str(col + 1),
            fill=AXIS_LABEL_COLOR,
            font=panel_font,
            anchor="mm",
        )
    for row in range(geometry.panel_rows):
        box = geometry.panel_cell_box(row, 0)
        draw.text(
            (geometry.panel_left - 12, (box[1] + box[3]) / 2.0),
            str(row + 1),
            fill=AXIS_LABEL_COLOR,
            font=panel_font,
            anchor="mm",
        )
    for row in range(geometry.panel_rows):
        for col in range(geometry.panel_cols):
            token = panel_fill[row][col] if panel_fill is not None else None
            if token is None or token == ".":
                fill = (255, 255, 255)
            elif token == "#":
                fill = FILL_DARK
            else:
                fill = COLOR_LETTERS[str(token)]
            draw.rectangle(
                geometry.panel_cell_box(row, col),
                fill=fill,
                outline=PANEL_LINE,
                width=1,
            )
    draw.rectangle(
        (
            geometry.panel_left,
            geometry.panel_top,
            geometry.panel_left + geometry.panel_cols * geometry.panel_cell,
            geometry.panel_top + geometry.panel_rows * geometry.panel_cell,
        ),
        outline=PANEL_FRAME,
        width=2,
    )
    return finish_scaled_canvas(canvas, (512, 512))


__all__ = [
    "AXIS_LABEL_COLOR",
    "COLOR_LETTERS",
    "COLOR_WORDS",
    "FILL_DARK",
    "ViewSceneGeometry",
    "arrow_segment",
    "build_geometry",
    "render_view_scene",
    "scene_faces",
    "structure_visibility",
]

"""Shared 3D geometry and axonometric rendering for the Spatial 3D family.

This module is the family's single geometric truth source: every task renderer AND every verifier pixel
decode derives voxel projections, face meshes, orthographic views, cube-net
folds, and rotation semantics from here.  Nothing in this module reads answers
or references; it is pure geometry plus painter-ordered drawing on the shared
supersampled canvas stack (``data.common.render_primitives``).

Projection
----------
The family camera is a fixed axonometric basis with +z up:

``u = (x - y) * UX * s``  and  ``v = (x + y) * UY * s - z * UZ * s``

with ``UZ / UY = 2.25`` deliberately different from the true-isometric ratio 2.
Under ratio 2 the lattice offset ``(+1, +1, +1)`` projects to the exact same
point, which is the classic hidden-cube ambiguity.  With ``UZ/UY = 9/4`` two
distinct lattice cube positions can only share a projection when
``2 * (dx + dy) == 9 * dz / 2``; over the family's scene box (|dx+dy| <= 6,
|dz| <= 3) the smallest nonzero screen separation is ``0.125 * scale`` —
the maximum achievable over that box — so every projected top-face position
pins a unique lattice cell once the footprint is readable (certified per
scene by :meth:`IsoProjection.min_diagonal_separation` plus the actual-cell
margins each sampler reports).

Screen orientation: +x runs toward the lower-right, +y toward the lower-left,
+z up.  The iso camera therefore looks from beyond the (x+, y+) corner; the
floor edges nearest the viewer are the (y = size_y) and (x = size_x) edges,
where the 1-based floor labels live (never occludable by the structure).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.pipeline.structured_grid import load_font
from data.common.render_primitives import DOCTRINE_SUPERSAMPLE, ScaledDraw

Vec3 = Tuple[float, float, float]
Cell = Tuple[int, int, int]

# Axonometric basis (unitless; multiplied by the per-scene pixel scale).
UX = 0.90
UY = 0.50
UZ = 1.125

# Camera direction: +x, +y, and the z rate that projects to a zero screen
# offset.  Larger dot product with this vector means nearer to the viewer.
_DEPTH_Z = 2.0 * UY / UZ

# Fixed light for face shading.  Axis-aligned faces receive exactly the three
# family shades (top brightest, +y left face medium, +x right face darkest);
# rotated faces interpolate through the same formula, so animation frames and
# static renders share one shading language.
_LIGHT = (-0.199, 0.348, 0.915)

EDGE_COLOR = (58, 66, 78)
EDGE_WIDTH = 2.0


def shade_factor(normal: Vec3) -> float:
    """Brightness factor for one outward face normal under the family light.

    Top faces render at (almost exactly) the base color and side faces darken,
    so light paper and pastel palettes never wash out to white.
    """

    norm = math.sqrt(sum(component * component for component in normal))
    if norm <= 1e-12:
        return 1.0
    dot = sum(a * b for a, b in zip(normal, _LIGHT)) / norm
    return max(0.42, min(1.06, 0.62 + 0.42 * dot))


def shade_color(base: Tuple[int, int, int], normal: Vec3) -> Tuple[int, int, int]:
    factor = shade_factor(normal)
    return tuple(max(0, min(255, int(round(channel * factor)))) for channel in base)


@dataclass(frozen=True)
class IsoProjection:
    """Linear axonometric projection with a fixed screen anchor."""

    scale: float
    origin: Tuple[float, float]

    def project(self, point: Sequence[float]) -> Tuple[float, float]:
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        return (
            self.origin[0] + (x - y) * UX * self.scale,
            self.origin[1] + (x + y) * UY * self.scale - z * UZ * self.scale,
        )

    @staticmethod
    def depth(point: Sequence[float]) -> float:
        return float(point[0]) + float(point[1]) + float(point[2]) * _DEPTH_Z

    def min_diagonal_separation(self, max_span: int = 6, max_height: int = 3) -> float:
        """Smallest nonzero screen distance between distinct lattice cube origins.

        Certifies per scene that no two candidate cube positions inside the
        scene box project to the same point: only cells sharing a
        screen diagonal (equal ``x - y``) can compete, and their vertical
        separation is ``|ds * UY - dz * UZ| * scale``.  Returned in pixels.
        """

        best = float("inf")
        for ds in range(-max_span, max_span + 1):
            for dz in range(-max_height, max_height + 1):
                if ds == 0 and dz == 0:
                    continue
                dv = abs(ds * UY - dz * UZ) * self.scale
                best = min(best, dv if dv > 1e-9 else float("inf"))
        return best


def fit_projection(
    points: Iterable[Sequence[float]],
    *,
    box: Tuple[float, float, float, float],
    max_scale: float = 60.0,
) -> IsoProjection:
    """Fit a projection so all 3D points land inside ``box`` (x0, y0, x1, y1)."""

    pts = [tuple(float(v) for v in p) for p in points]
    if not pts:
        raise ValueError("fit_projection requires at least one point")
    probe = IsoProjection(1.0, (0.0, 0.0))
    us = [probe.project(p)[0] for p in pts]
    vs = [probe.project(p)[1] for p in pts]
    span_u = max(us) - min(us)
    span_v = max(vs) - min(vs)
    x0, y0, x1, y1 = box
    scale = min(
        max_scale,
        (x1 - x0) / span_u if span_u > 1e-9 else max_scale,
        (y1 - y0) / span_v if span_v > 1e-9 else max_scale,
    )
    center_u = (max(us) + min(us)) / 2.0
    center_v = (max(vs) + min(vs)) / 2.0
    origin = (
        (x0 + x1) / 2.0 - center_u * scale,
        (y0 + y1) / 2.0 - center_v * scale,
    )
    return IsoProjection(scale, origin)


# ---------------------------------------------------------------------------
# Faces and painter-ordered mesh rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decal:
    """A glyph or dice-pip pattern painted onto a planar face patch.

    ``origin``/``axis_u``/``axis_v`` span the patch in 3D.  When the projected
    frame is mirror-handed the source tile is flipped first, so the mark reads
    correctly from whichever side of the sheet faces the camera ("printed on
    both sides" semantics).  ``kind="pips"`` draws the standard die layout for
    ``int(text)`` pips — rotation-safe marks for faces that fold through
    arbitrary orientations.
    """

    text: str
    origin: Vec3
    axis_u: Vec3
    axis_v: Vec3
    color: Tuple[int, int, int] = (40, 47, 56)
    rel_size: float = 0.58
    kind: str = "text"


# Standard die pip anchors on a 3x3 patch grid (fractions of the tile).
_PIP_ANCHORS: Dict[int, Tuple[Tuple[float, float], ...]] = {
    1: ((0.5, 0.5),),
    2: ((0.25, 0.25), (0.75, 0.75)),
    3: ((0.25, 0.25), (0.5, 0.5), (0.75, 0.75)),
    4: ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)),
    5: ((0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)),
    6: ((0.25, 0.25), (0.75, 0.25), (0.25, 0.5), (0.75, 0.5), (0.25, 0.75), (0.75, 0.75)),
}


@dataclass(frozen=True)
class Face:
    points: Tuple[Vec3, ...]
    color: Tuple[int, int, int]
    shaded: bool = True
    edge_color: Optional[Tuple[int, int, int]] = EDGE_COLOR
    edge_width: float = EDGE_WIDTH
    decal: Optional[Decal] = None
    # Painter-order bias for coplanar overlays (larger draws later).
    bias: float = 0.0

    def normal(self) -> Vec3:
        a, b, c = self.points[0], self.points[1], self.points[2]
        u = tuple(b[i] - a[i] for i in range(3))
        v = tuple(c[i] - a[i] for i in range(3))
        return (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )

    def centroid_depth(self) -> float:
        n = len(self.points)
        cx = sum(p[0] for p in self.points) / n
        cy = sum(p[1] for p in self.points) / n
        cz = sum(p[2] for p in self.points) / n
        return IsoProjection.depth((cx, cy, cz))


def _visible_color(face: Face) -> Tuple[int, int, int]:
    if not face.shaded:
        return face.color
    normal = face.normal()
    # Shade by the side actually facing the camera so a sheet seen from
    # behind is lit for its visible side, not its hidden one.
    facing = normal if (normal[0] + normal[1] + normal[2] * _DEPTH_Z) > 0 else tuple(-c for c in normal)
    return shade_color(face.color, facing)


def _paste_decal(
    canvas: Image.Image,
    decal: Decal,
    projection: IsoProjection,
    factor: int,
) -> None:
    corners3d = [
        decal.origin,
        tuple(decal.origin[i] + decal.axis_u[i] for i in range(3)),
        tuple(decal.origin[i] + decal.axis_u[i] + decal.axis_v[i] for i in range(3)),
        tuple(decal.origin[i] + decal.axis_v[i] for i in range(3)),
    ]
    quad = [
        tuple(coord * factor for coord in projection.project(corner))
        for corner in corners3d
    ]
    ax, ay = quad[1][0] - quad[0][0], quad[1][1] - quad[0][1]
    bx, by = quad[3][0] - quad[0][0], quad[3][1] - quad[0][1]
    det = ax * by - ay * bx
    if abs(det) < 1e-6:
        return  # patch seen edge-on: nothing readable to draw
    tile_px = 96
    tile = Image.new("RGBA", (tile_px, tile_px), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    from data.common.prompt_paraphrase import active_profile
    numbered = decal.kind == "pips" and active_profile("numbered_faces")
    if decal.kind == "pips" and not numbered:
        count = int(decal.text)
        if count not in _PIP_ANCHORS:
            raise ValueError(f"pip decals support 1..6 pips, got {count}")
        radius = tile_px * 0.085
        for fx, fy in _PIP_ANCHORS[count]:
            tile_draw.ellipse(
                (fx * tile_px - radius, fy * tile_px - radius, fx * tile_px + radius, fy * tile_px + radius),
                fill=(*decal.color, 255),
            )
    else:
        font = load_font(int(tile_px * decal.rel_size))
        tile_draw.text(
            (tile_px / 2.0, tile_px / 2.0),
            decal.text,
            font=font,
            fill=(*decal.color, 255),
            anchor="mm",
        )
    if numbered:
        tile_draw.rounded_rectangle((8,7,88,89),radius=9,outline=(*decal.color,255),width=5)
        for x,y in ((15,15),(81,15),(15,81),(81,81)):
            tile_draw.ellipse((x-2,y-2,x+2,y+2),fill=(*decal.color,255))
        tile_draw.line((tile_px*.28,tile_px*.82,tile_px*.72,tile_px*.82),fill=(*decal.color,255),width=3)
    if det < 0:
        # Mirror-handed frame: the camera sees the back of the sheet.  The
        # same glyph is printed readable on both sides, so paint the flipped
        # source through the unchanged frame; on screen it reads correctly.
        tile = tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x_min, y_min = int(math.floor(min(xs))) - 2, int(math.floor(min(ys))) - 2
    x_max, y_max = int(math.ceil(max(xs))) + 2, int(math.ceil(max(ys))) + 2
    width, height = x_max - x_min, y_max - y_min
    if width <= 0 or height <= 0:
        return
    # Inverse affine: output bbox pixel -> source tile pixel.
    ox, oy = quad[0][0] - x_min, quad[0][1] - y_min
    inv_a = by / det * tile_px
    inv_b = -bx / det * tile_px
    inv_d = -ay / det * tile_px
    inv_e = ax / det * tile_px
    coeffs = (
        inv_a,
        inv_b,
        -(inv_a * ox + inv_b * oy),
        inv_d,
        inv_e,
        -(inv_d * ox + inv_e * oy),
    )
    warped = tile.transform((width, height), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BILINEAR)
    canvas.paste(warped, (x_min, y_min), warped)


def render_faces(
    canvas: Image.Image,
    faces: Sequence[Face],
    projection: IsoProjection,
    factor: int,
) -> None:
    """Painter-ordered face rendering onto a supersampled canvas."""

    draw = ScaledDraw(ImageDraw.Draw(canvas), factor)
    ordered = sorted(faces, key=lambda face: (face.centroid_depth(), face.bias))
    for face in ordered:
        polygon = [projection.project(point) for point in face.points]
        draw.polygon(polygon, fill=_visible_color(face))
        if face.edge_color is not None:
            draw.closed_outline(
                polygon,
                fill=face.edge_color,
                width=face.edge_width,
            )
        if face.decal is not None:
            _paste_decal(canvas, face.decal, projection, factor)


# ---------------------------------------------------------------------------
# Voxel meshes
# ---------------------------------------------------------------------------

# Outward face templates for the unit cube with min corner at the origin:
# (normal, four corners in outward counter-clockwise order).
_CUBE_FACES: Tuple[Tuple[Vec3, Tuple[Vec3, ...]], ...] = (
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
)

VISIBLE_NORMALS: Tuple[Vec3, ...] = ((0, 0, 1), (1, 0, 0), (0, 1, 0))


def voxel_faces(
    cells: Sequence[Cell],
    colors: Optional[Mapping[Cell, Tuple[int, int, int]]] = None,
    *,
    default_color: Tuple[int, int, int] = (196, 210, 226),
    edge_color: Tuple[int, int, int] = EDGE_COLOR,
    edge_width: float = EDGE_WIDTH,
    only_camera_facing: bool = True,
) -> List[Face]:
    """Outward faces of a voxel set with internal shared faces culled."""

    occupied = set(cells)
    faces: List[Face] = []
    for cell in cells:
        base = colors.get(cell, default_color) if colors else default_color
        for normal, corners in _CUBE_FACES:
            neighbor = (cell[0] + int(normal[0]), cell[1] + int(normal[1]), cell[2] + int(normal[2]))
            if neighbor in occupied:
                continue
            if only_camera_facing and (normal[0] + normal[1] + normal[2] * _DEPTH_Z) <= 0:
                continue
            faces.append(
                Face(
                    points=tuple(
                        (cell[0] + dx, cell[1] + dy, cell[2] + dz) for dx, dy, dz in corners
                    ),
                    color=base,
                    edge_color=edge_color,
                    edge_width=edge_width,
                )
            )
    return faces


def floor_grid_faces(
    size_x: int,
    size_y: int,
    *,
    fill: Tuple[int, int, int] = (244, 246, 250),
    line: Tuple[int, int, int] = (150, 158, 170),
) -> List[Face]:
    """Floor cells at z=0 drawn as unshaded tiles below every voxel."""

    faces: List[Face] = []
    for ix in range(size_x):
        for iy in range(size_y):
            faces.append(
                Face(
                    points=((ix, iy, 0), (ix + 1, iy, 0), (ix + 1, iy + 1, 0), (ix, iy + 1, 0)),
                    color=fill,
                    shaded=False,
                    edge_color=line,
                    edge_width=1.2,
                    bias=-1000.0,
                )
            )
    return faces


# ---------------------------------------------------------------------------
# Face visibility measurement (sampling gates and margins)
# ---------------------------------------------------------------------------


def face_visibility(
    faces: Sequence[Face],
    projection: IsoProjection,
    canvas_size: Tuple[int, int] = (512, 512),
    factor: int = DOCTRINE_SUPERSAMPLE,
) -> List[float]:
    """Visible-pixel fraction per face under painter occlusion.

    Renders face indices in painter order and compares each face's surviving
    pixels with its unoccluded footprint.  Used by sampling gates ("every
    column top must stay readable") and reported as a per-sample margin.

    The index buffers are rasterized at ``factor`` times the delivered canvas —
    the same resolution :func:`render_faces` paints at — so a sliver that
    survives into the shipped image as a real (if faint) LANCZOS blend is
    measured instead of being quantized away by an aliased 512-square buffer.

    Both buffers share one rasterizer and one resolution, so the fraction stays
    a ratio of comparable pixel counts.  The solo footprint reuses a single
    buffer, cleared by re-filling the same polygon with 0, and is counted inside
    the face's own bounding box, which keeps the cost proportional to face area
    rather than to canvas area.
    """

    if not faces:
        return []
    width, height = int(canvas_size[0]) * factor, int(canvas_size[1]) * factor
    order = sorted(range(len(faces)), key=lambda i: (faces[i].centroid_depth(), faces[i].bias))
    index_image = Image.new("I", (width, height), 0)
    index_draw = ScaledDraw(ImageDraw.Draw(index_image), factor)
    solo_image = Image.new("I", (width, height), 0)
    solo_draw = ScaledDraw(ImageDraw.Draw(solo_image), factor)
    solo_areas: Dict[int, int] = {}
    for face_index in order:
        polygon = [projection.project(point) for point in faces[face_index].points]
        solo_draw.polygon(polygon, fill=1)
        box = (
            max(0, int(math.floor(min(p[0] for p in polygon) * factor)) - 1),
            max(0, int(math.floor(min(p[1] for p in polygon) * factor)) - 1),
            min(width, int(math.ceil(max(p[0] for p in polygon) * factor)) + 2),
            min(height, int(math.ceil(max(p[1] for p in polygon) * factor)) + 2),
        )
        if box[0] < box[2] and box[1] < box[3]:
            solo_areas[face_index] = int(np.count_nonzero(np.asarray(solo_image.crop(box))))
        else:
            solo_areas[face_index] = 0
        solo_draw.polygon(polygon, fill=0)
        index_draw.polygon(polygon, fill=face_index + 1)
    counts = np.bincount(np.asarray(index_image).ravel(), minlength=len(faces) + 1)
    fractions: List[float] = []
    for face_index in range(len(faces)):
        area = solo_areas.get(face_index, 0)
        if area <= 0:
            fractions.append(0.0)
            continue
        fractions.append(int(counts[face_index + 1]) / area)
    return fractions


# ---------------------------------------------------------------------------
# Orthographic views (single documented convention)
# ---------------------------------------------------------------------------

# World axes: floor columns count 1-based along +x, floor rows 1-based along
# +y, layers along +z from the floor.  Each named view fixes the viewer's
# stand point, gaze direction, and the exact left-to-right / top-to-bottom
# grid mapping a standing viewer sees.  Generators quote the mapping of the
# sampled view in their prompts and the verifier replays it from here.

# * ``top``   — looking straight down (−z).  View row r = floor row r, view
# column c = floor column c (up-screen in the view = smaller floor rows).
# * ``front`` — viewer beyond the y = 0 edge looking along +y.  Left-to-right
# view columns = ascending floor columns; view row 1 = the top layer.
# * ``right`` — viewer beyond the x = size_x edge looking along −x.
# Left-to-right view columns = ascending floor rows; view row 1 = top layer.
# * ``back``  — viewer beyond the y = size_y edge looking along −y.
# Left-to-right view columns = DESCENDING floor columns (mirror of front);
# view row 1 = the top layer.
# * ``left``  — viewer beyond the x = 0 edge looking along +x.  Left-to-right
# view columns = DESCENDING floor rows (mirror of right); view row 1 = top.

VIEW_NAMES = ("top", "front", "right", "back", "left")


def view_grid_shape(view: str, size_x: int, size_y: int, height: int) -> Tuple[int, int]:
    if view == "top":
        return (size_y, size_x)
    if view in ("front", "back"):
        return (height, size_x)
    if view in ("right", "left"):
        return (height, size_y)
    raise ValueError(f"unsupported view: {view!r}")


def _ray_cells(view: str, row: int, col: int, size_x: int, size_y: int, height: int) -> List[Cell]:
    """Cells along one view ray, ordered nearest-to-farthest for the viewer."""

    if view == "top":
        return [(col, row, z) for z in range(height - 1, -1, -1)]
    if view == "front":
        z = height - 1 - row
        return [(col, y, z) for y in range(size_y)]
    if view == "right":
        z = height - 1 - row
        return [(x, col, z) for x in range(size_x - 1, -1, -1)]
    if view == "back":
        z = height - 1 - row
        x = size_x - 1 - col
        return [(x, y, z) for y in range(size_y - 1, -1, -1)]
    if view == "left":
        z = height - 1 - row
        y = size_y - 1 - col
        return [(x, y, z) for x in range(size_x)]
    raise ValueError(f"unsupported view: {view!r}")


def silhouette_view(
    cells: Sequence[Cell],
    view: str,
    *,
    size_x: int,
    size_y: int,
    height: int,
) -> List[List[int]]:
    occupied = set(cells)
    rows, cols = view_grid_shape(view, size_x, size_y, height)
    return [
        [
            1 if any(cell in occupied for cell in _ray_cells(view, r, c, size_x, size_y, height)) else 0
            for c in range(cols)
        ]
        for r in range(rows)
    ]


def visible_color_view(
    cell_colors: Mapping[Cell, str],
    view: str,
    *,
    size_x: int,
    size_y: int,
    height: int,
    empty: str = ".",
) -> List[List[str]]:
    """Nearest visible colour letter along each view ray (``empty`` if none)."""

    rows, cols = view_grid_shape(view, size_x, size_y, height)
    grid: List[List[str]] = []
    for r in range(rows):
        line: List[str] = []
        for c in range(cols):
            token = empty
            for cell in _ray_cells(view, r, c, size_x, size_y, height):
                if cell in cell_colors:
                    token = cell_colors[cell]
                    break
            line.append(token)
        grid.append(line)
    return grid


# ---------------------------------------------------------------------------
# Polycube rotations (90-degree lattice moves)
# ---------------------------------------------------------------------------

AXIS_NAMES = ("X", "Y", "Z")


def rotate_cell_90(cell: Cell, axis: str, sign: int) -> Cell:
    """Rotate one lattice cell 90 degrees about the named axis through the origin.

    ``sign=+1`` is counterclockwise as seen from the axis arrow tip looking
    back toward the origin (right-hand rule about +axis); ``sign=-1`` is
    clockwise from that same viewpoint.
    """

    x, y, z = cell
    if sign not in (1, -1):
        raise ValueError("rotation sign must be +1 or -1")
    if axis == "X":
        return (x, -z, y) if sign == 1 else (x, z, -y)
    if axis == "Y":
        return (z, y, -x) if sign == 1 else (-z, y, x)
    if axis == "Z":
        return (-y, x, z) if sign == 1 else (y, -x, z)
    raise ValueError(f"unsupported axis: {axis!r}")


def normalize_cells(cells: Iterable[Cell]) -> Tuple[Cell, ...]:
    """Translate a cell set so its minimum corner sits at the origin, sorted."""

    cell_list = [tuple(int(v) for v in cell) for cell in cells]
    if not cell_list:
        return ()
    mins = tuple(min(cell[i] for cell in cell_list) for i in range(3))
    return tuple(sorted((c[0] - mins[0], c[1] - mins[1], c[2] - mins[2]) for c in cell_list))


def rotate_cells_90(cells: Iterable[Cell], axis: str, sign: int) -> Tuple[Cell, ...]:
    return normalize_cells(rotate_cell_90(cell, axis, sign) for cell in cells)


def rotation_orbit(cells: Iterable[Cell]) -> List[Tuple[Cell, ...]]:
    """All distinct normalized orientations of a cell set under rotations."""

    start = normalize_cells(cells)
    seen = {start}
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for axis in AXIS_NAMES:
            for sign in (1, -1):
                rotated = rotate_cells_90(current, axis, sign)
                if rotated not in seen:
                    seen.add(rotated)
                    frontier.append(rotated)
    return sorted(seen)


def rotation_matrix(axis: str, angle: float) -> Tuple[Tuple[float, float, float], ...]:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "X":
        return ((1, 0, 0), (0, c, -s), (0, s, c))
    if axis == "Y":
        return ((c, 0, s), (0, 1, 0), (-s, 0, c))
    if axis == "Z":
        return ((c, -s, 0), (s, c, 0), (0, 0, 1))
    raise ValueError(f"unsupported axis: {axis!r}")


def apply_matrix(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> Vec3:
    return tuple(
        sum(matrix[row][col] * float(point[col]) for col in range(3)) for row in range(3)
    )


# ---------------------------------------------------------------------------
# Cube nets: enumeration, hinge trees, and folding
# ---------------------------------------------------------------------------


def _net_transforms(cells: Sequence[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], ...]]:
    variants = []
    current = [tuple(cell) for cell in cells]
    for _ in range(4):
        current = [(-r, c) for c, r in current]  # rotate 90
        for mirrored in (current, [(-c, r) for c, r in current]):
            min_c = min(c for c, _ in mirrored)
            min_r = min(r for _, r in mirrored)
            variants.append(tuple(sorted((c - min_c, r - min_r) for c, r in mirrored)))
    return variants


def _canonical_net(cells: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    return min(_net_transforms(cells))


@dataclass(frozen=True)
class FoldedFace:
    """One net cell after folding: its 3D quad and readable-frame axes."""

    cell: Tuple[int, int]
    corners: Tuple[Vec3, Vec3, Vec3, Vec3]
    origin: Vec3
    axis_u: Vec3
    axis_v: Vec3

    def center(self) -> Vec3:
        return tuple(sum(corner[i] for corner in self.corners) / 4.0 for i in range(3))

    def normal(self) -> Vec3:
        u = tuple(self.corners[1][i] - self.corners[0][i] for i in range(3))
        v = tuple(self.corners[3][i] - self.corners[0][i] for i in range(3))
        return (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )


class CubeNet:
    """A hexomino net with a hinge tree rooted at a base cell.

    Folding every hinge to 90 degrees lifts the five flaps up off the plane
    and assembles the unit box above the base cell.  The terminal pose depends
    only on the final hinge angles, never on the temporal fold order.
    """

    def __init__(self, cells: Sequence[Tuple[int, int]], base: Tuple[int, int]) -> None:
        self.cells: Tuple[Tuple[int, int], ...] = tuple((int(c), int(r)) for c, r in cells)
        if len(self.cells) != 6 or len(set(self.cells)) != 6:
            raise ValueError("a cube net requires six distinct cells")
        base = (int(base[0]), int(base[1]))
        if base not in self.cells:
            raise ValueError("net base cell must be one of the six cells")
        self.base = base
        occupied = set(self.cells)
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        order = [base]
        seen = {base}
        cursor = 0
        while cursor < len(order):
            current = order[cursor]
            cursor += 1
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (current[0] + dc, current[1] + dr)
                if neighbor in occupied and neighbor not in seen:
                    seen.add(neighbor)
                    parent[neighbor] = current
                    order.append(neighbor)
        if len(order) != 6:
            raise ValueError("net cells must be edge-connected")
        self.parent = parent
        self.bfs_order = tuple(order)

    def hinge(self, child: Tuple[int, int]) -> Tuple[Vec3, Vec3]:
        """Flat-coordinate hinge segment (point, direction) toward the parent."""

        parent = self.parent[child]
        # Shared edge between parent and child cells in flat coordinates
        # (cell (c, r) occupies [c, c+1] x [r, r+1] at z=0).
        if child[0] == parent[0] + 1:  # child east of parent -> hinge x = child c
            point = (float(child[0]), float(child[1]), 0.0)
            direction = (0.0, 1.0, 0.0)
        elif child[0] == parent[0] - 1:  # child west -> hinge x = parent c
            point = (float(parent[0]), float(child[1]), 0.0)
            direction = (0.0, 1.0, 0.0)
        elif child[1] == parent[1] + 1:  # child north (larger r)
            point = (float(child[0]), float(child[1]), 0.0)
            direction = (1.0, 0.0, 0.0)
        elif child[1] == parent[1] - 1:
            point = (float(child[0]), float(parent[1]), 0.0)
            direction = (1.0, 0.0, 0.0)
        else:
            raise ValueError("hinge requires 4-adjacent parent and child")
        return point, direction

    def _lift_sign(self, child: Tuple[int, int]) -> float:
        point, direction = self.hinge(child)
        probe_angle = math.pi / 2.0
        rotate = _axis_rotation(point, direction, probe_angle)
        probe = rotate((child[0] + 0.5, child[1] + 0.5, 0.0))
        return 1.0 if probe[2] > 1e-9 else -1.0

    def folded_faces(self, angles: Mapping[Tuple[int, int], float]) -> List[FoldedFace]:
        """Fold with per-flap hinge angles (radians, 0..pi/2), base fixed.

        Every non-base cell owns exactly one hinge (toward its BFS parent).
        A face's world transform composes the hinge rotations along its path
        root-to-leaf, all expressed in flat coordinates.
        """

        transforms: Dict[Tuple[int, int], Callable[[Vec3], Vec3]] = {self.base: lambda p: p}
        for cell in self.bfs_order[1:]:
            angle = float(angles.get(cell, 0.0))
            point, direction = self.hinge(cell)
            local = _axis_rotation(point, direction, self._lift_sign(cell) * angle)
            parent_transform = transforms[self.parent[cell]]
            transforms[cell] = (
                lambda p, outer=parent_transform, inner=local: outer(inner(p))
            )
        faces: List[FoldedFace] = []
        for cell in self.cells:
            transform = transforms[cell]
            c, r = cell
            corners = (
                transform((c, r, 0.0)),
                transform((c + 1, r, 0.0)),
                transform((c + 1, r + 1, 0.0)),
                transform((c, r + 1, 0.0)),
            )
            pad = 0.16
            side = 1.0 - 2.0 * pad
            origin = transform((c + pad, r + pad, 0.0))
            u_end = transform((c + pad + side, r + pad, 0.0))
            v_end = transform((c + pad, r + pad + side, 0.0))
            faces.append(
                FoldedFace(
                    cell=cell,
                    corners=corners,
                    origin=origin,
                    axis_u=tuple(u_end[i] - origin[i] for i in range(3)),
                    axis_v=tuple(v_end[i] - origin[i] for i in range(3)),
                )
            )
        return faces

    def fold_roles(self) -> Dict[Tuple[int, int], str]:
        """Terminal box role per cell after a complete fold.

        Roles: ``bottom``, ``top``, ``north`` (+y wall), ``south`` (-y wall),
        ``east`` (+x wall), ``west`` (-x wall), relative to the base cell's
        unit box.
        """

        faces = self.folded_faces({cell: math.pi / 2.0 for cell in self.cells if cell != self.base})
        bx, by = self.base
        roles: Dict[Tuple[int, int], str] = {}
        for face in faces:
            cx, cy, cz = face.center()
            local = (cx - bx, cy - by, cz)
            if abs(local[2]) < 1e-6:
                role = "bottom"
            elif abs(local[2] - 1.0) < 1e-6 and abs(local[0] - 0.5) < 1e-6 and abs(local[1] - 0.5) < 1e-6:
                role = "top"
            elif abs(local[0] - 1.0) < 1e-6:
                role = "east"
            elif abs(local[0]) < 1e-6:
                role = "west"
            elif abs(local[1] - 1.0) < 1e-6:
                role = "north"
            elif abs(local[1]) < 1e-6:
                role = "south"
            else:
                raise ValueError("folded face does not land on the unit box")
            if role in roles.values():
                raise ValueError("two net cells fold onto the same box face")
            roles[face.cell] = role
        return roles

    def is_valid_cube_net(self) -> bool:
        try:
            roles = self.fold_roles()
        except ValueError:
            return False
        return sorted(roles.values()) == ["bottom", "east", "north", "south", "top", "west"]


def _axis_rotation(point: Vec3, direction: Vec3, angle: float) -> Callable[[Vec3], Vec3]:
    norm = math.sqrt(sum(d * d for d in direction))
    ux, uy, uz = (d / norm for d in direction)
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    m = (
        (c + ux * ux * one_c, ux * uy * one_c - uz * s, ux * uz * one_c + uy * s),
        (uy * ux * one_c + uz * s, c + uy * uy * one_c, uy * uz * one_c - ux * s),
        (uz * ux * one_c - uy * s, uz * uy * one_c + ux * s, c + uz * uz * one_c),
    )
    px, py, pz = point

    def rotate(p: Vec3) -> Vec3:
        dx, dy, dz = p[0] - px, p[1] - py, p[2] - pz
        return (
            px + m[0][0] * dx + m[0][1] * dy + m[0][2] * dz,
            py + m[1][0] * dx + m[1][1] * dy + m[1][2] * dz,
            pz + m[2][0] * dx + m[2][1] * dy + m[2][2] * dz,
        )

    return rotate


@lru_cache(maxsize=1)
def all_cube_nets() -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """The 11 free hexomino cube nets, derived by folding every free hexomino.

    Enumerated programmatically so the net list and the folding code can never
    drift apart; the count is asserted to the known value 11.
    """

    shapes = {((0, 0),)}
    for _ in range(5):
        grown = set()
        for shape in shapes:
            for c, r in shape:
                for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cell = (c + dc, r + dr)
                    if cell in shape:
                        continue
                    grown.add(_canonical_net((*shape, cell)))
        shapes = grown
    nets = []
    for shape in sorted(shapes):
        net = CubeNet(shape, shape[0])
        if net.is_valid_cube_net():
            nets.append(tuple(shape))
    if len(nets) != 11:
        raise AssertionError(f"cube-net enumeration found {len(nets)} nets, expected 11")
    return tuple(nets)


__all__ = [
    "AXIS_NAMES",
    "Cell",
    "CubeNet",
    "Decal",
    "EDGE_COLOR",
    "EDGE_WIDTH",
    "Face",
    "FoldedFace",
    "IsoProjection",
    "UX",
    "UY",
    "UZ",
    "VIEW_NAMES",
    "VISIBLE_NORMALS",
    "all_cube_nets",
    "apply_matrix",
    "face_visibility",
    "fit_projection",
    "floor_grid_faces",
    "normalize_cells",
    "render_faces",
    "rotate_cell_90",
    "rotate_cells_90",
    "rotation_matrix",
    "rotation_orbit",
    "shade_color",
    "shade_factor",
    "silhouette_view",
    "view_grid_shape",
    "visible_color_view",
    "voxel_faces",
]

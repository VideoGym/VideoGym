"""Shared light Spatial Navigation maze styling and square-grid scene helpers.

Every maze task in this family renders on the same light palette that the
non-maze grid tasks already use: a white canvas, slate impassable cells, thin
dark cell outlines, red endpoints and route, and royal-blue 1-based axis
numbers in the gutters.  Square-grid scenes are drawn supersampled through the
shared render primitives (factor 4) and LANCZOS-downsampled to 512x512.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

from PIL import Image
from data.common.visual_style import current_style

from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)

Color = Tuple[int, int, int]

# Family-wide light maze palette.  The slate pair matches the blocked-cell
# style of the non-maze grid tasks so "impassable" reads identically across
# the family.
MAZE_BACKGROUND: Color = (255, 255, 255)
MAZE_WALL_FILL: Color = (54, 64, 82)
MAZE_WALL_OUTLINE: Color = (32, 38, 52)
MAZE_CELL_OUTLINE: Color = (25, 25, 25)
MAZE_OPEN_FILL: Color = (255, 255, 255)
# Light body tint for non-square mazes (hexagon plates, labyrinth rings) so
# the maze body stays visible against the white canvas.
MAZE_BODY_FILL: Color = (232, 236, 243)
# Stroke walls used by edge-wall mazes (hexagon, labyrinth).
MAZE_EDGE_WALL: Color = (32, 38, 52)
MAZE_ENDPOINT_COLOR: Color = (220, 30, 30)
MAZE_ROUTE_COLOR: Color = (220, 30, 30)


def strip_outer_ring(grid: Sequence[Sequence[int]]) -> list[list[int]]:
    """Return the maze grid without its all-wall boundary ring.

    Carved mazes bound their passages with a solid ring of wall cells.  The
    rendered task drops that ring entirely: the playable grid starts at the
    first interior cell, and stepping outside the grid is illegal by the grid
    bounds alone, so the ring carries no information a solver needs.
    """

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows < 3 or cols < 3:
        raise ValueError("grid must be at least 3x3 to strip its boundary ring")
    return [list(row[1 : cols - 1]) for row in grid[1 : rows - 1]]


def begin_maze_scene(
    canvas_dims: Tuple[int, int],
    *,
    factor: int = DOCTRINE_SUPERSAMPLE,
    background: Color = MAZE_BACKGROUND,
) -> Tuple[Image.Image, ScaledDraw]:
    """Allocate the supersampled scene canvas for one maze render."""

    image, draw = new_scaled_canvas(int(canvas_dims[0]), int(canvas_dims[1]), factor, background)
    if style := current_style():
        style.surface(draw, (0, 0, canvas_dims[0], canvas_dims[1]), style.background)
    from data.common.prompt_paraphrase import active_profile
    if active_profile("node_maze"):
        draw.rectangle((0, 0, canvas_dims[0], canvas_dims[1]),fill=(243,224,181))
    return image, draw


def finish_maze_scene(image: Image.Image, canvas_dims: Tuple[int, int]) -> Image.Image:
    """Downsample the supersampled scene back to its target size."""

    return finish_scaled_canvas(image, (int(canvas_dims[0]), int(canvas_dims[1])))


def route_run_breakdown(
    path: Sequence[Tuple[int, int]],
) -> list[Tuple[str, list[Tuple[int, int]]]]:
    """Split an orthogonal cell route into straight runs.

    Returns ``(direction, cells)`` pairs where ``cells`` includes both the run
    entry and exit cell, so consecutive runs share their turning cell.  The
    direction names follow the on-screen reading: ``down`` increases the row,
    ``right`` increases the column.
    """

    cells = [(int(row), int(col)) for row, col in path]
    if len(cells) < 2:
        return []

    def step_direction(a: Tuple[int, int], b: Tuple[int, int]) -> str:
        dr, dc = b[0] - a[0], b[1] - a[1]
        if dr > 0 and dc == 0:
            return "down"
        if dr < 0 and dc == 0:
            return "up"
        if dc > 0 and dr == 0:
            return "right"
        if dc < 0 and dr == 0:
            return "left"
        return "across"

    runs: list[Tuple[str, list[Tuple[int, int]]]] = []
    current_direction = step_direction(cells[0], cells[1])
    current_cells = [cells[0], cells[1]]
    for previous, cell in zip(cells[1:], cells[2:]):
        direction = step_direction(previous, cell)
        if direction == current_direction:
            current_cells.append(cell)
        else:
            runs.append((current_direction, current_cells))
            current_direction = direction
            current_cells = [previous, cell]
    runs.append((current_direction, current_cells))
    return runs


def draw_square_cell_grid(
    sdraw: ScaledDraw,
    *,
    rows: int,
    cols: int,
    cell_size: int,
    pad_left: int,
    pad_top: int,
    is_wall: Callable[[int, int], bool],
    open_fill: Optional[Callable[[int, int], Color]] = None,
) -> None:
    """Draw every cell of a square maze grid in the shared family style.

    Walls use the slate blocked-cell pair; open cells use ``open_fill`` (white
    by default) with the thin dark family outline.
    """

    from data.common.prompt_paraphrase import active_profile, current_prompt_paraphrase
    if active_profile("node_maze"):
        def centre(r,c):return pad_left+(c+.5)*cell_size,pad_top+(r+.5)*cell_size
        if (current_prompt_paraphrase() or {}).get("task")=="maze_keys_and_locks":
            # A chamber floor plan exposes the same walkable positions and
            # passage incidence as the source, with keycards and vault doors.
            floor=(250,248,239);edge=(128,150,139)
            for r in range(int(rows)):
                for c in range(int(cols)):
                    if is_wall(r,c):continue
                    x,y=centre(r,c)
                    for rr,cc in ((r+1,c),(r,c+1)):
                        if rr<rows and cc<cols and not is_wall(rr,cc):
                            xx,yy=centre(rr,cc)
                            sdraw.line((x,y,xx,yy),fill=edge,width=cell_size*.58)
                            sdraw.line((x,y,xx,yy),fill=floor,width=cell_size*.47)
            for r in range(int(rows)):
                for c in range(int(cols)):
                    if not is_wall(r,c):
                        x,y=centre(r,c);radius=cell_size*.42
                        sdraw.rounded_rectangle((x-radius,y-radius,x+radius,y+radius),radius=cell_size*.18,fill=floor,outline=edge,width=1.5)
            return
        for r in range(int(rows)):
            for c in range(int(cols)):
                if is_wall(r,c):continue
                x,y=centre(r,c)
                for rr,cc in ((r+1,c),(r,c+1)):
                    if rr<rows and cc<cols and not is_wall(rr,cc):
                        xx,yy=centre(rr,cc)
                        sdraw.line((x,y,xx,yy),fill=(78,81,91),width=max(7,cell_size*.23))
                        sdraw.line((x,y,xx,yy),fill=(255,248,217),width=max(2,cell_size*.08))
        for r in range(int(rows)):
            for c in range(int(cols)):
                if not is_wall(r,c):
                    x,y=centre(r,c);radius=cell_size*.27
                    sdraw.polygon([(x,y-radius),(x+radius,y),(x,y+radius),(x-radius,y)],fill=(255,251,228),outline=(78,81,91),width=2)
        return
    for row in range(int(rows)):
        for col in range(int(cols)):
            left = pad_left + col * cell_size
            top = pad_top + row * cell_size
            box = (left, top, left + cell_size - 1, top + cell_size - 1)
            wall = is_wall(row, col)
            if style := current_style():
                fill = MAZE_WALL_FILL if wall else (open_fill(row, col) if open_fill else MAZE_OPEN_FILL)
                style.maze_cell(sdraw, box, wall=wall, fill=fill,
                                outline=MAZE_WALL_OUTLINE if wall else MAZE_CELL_OUTLINE)
            elif wall:
                sdraw.rectangle(box, fill=MAZE_WALL_FILL, outline=MAZE_WALL_OUTLINE, width=1)
            else:
                fill = open_fill(row, col) if open_fill is not None else MAZE_OPEN_FILL
                sdraw.rectangle(box, fill=fill, outline=MAZE_CELL_OUTLINE, width=1)


__all__ = [
    "MAZE_BACKGROUND",
    "MAZE_BODY_FILL",
    "MAZE_CELL_OUTLINE",
    "MAZE_EDGE_WALL",
    "MAZE_ENDPOINT_COLOR",
    "MAZE_OPEN_FILL",
    "MAZE_ROUTE_COLOR",
    "MAZE_WALL_FILL",
    "MAZE_WALL_OUTLINE",
    "begin_maze_scene",
    "draw_square_cell_grid",
    "finish_maze_scene",
    "route_run_breakdown",
    "strip_outer_ring",
]

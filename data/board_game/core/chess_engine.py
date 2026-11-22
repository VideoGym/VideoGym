"""Executable Logic and Vision contract for 5x5 L-waypoint search.

The deleted neutral rule-simulator branch is intentionally absent.  This
module owns only the retained search task's numeric coordinates, L-offset
solver, deterministic scenario construction, stable token renderer, and
committed waypoint state. Internal cells are zero-based; every model-facing
position is a one-based ``(x, y)`` pair with ``(1, 1)`` at the upper-left,
where ``x`` is the row and ``y`` is the column.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from data.board_game.core.coordinates import from_visible_cell, to_visible_cell
from data.common.pipeline.board_motion import CANVAS_SIZE
from data.common.pipeline.board_rendering import load_font
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)


BOARD_SIZE = 5

Cell = Tuple[int, int]
ExternalCell = Tuple[int, int]
Piece = Optional[Tuple[str, bool]]
Board = Tuple[Tuple[Piece, ...], ...]

KIND_LABEL: Mapping[str, str] = {"N": "L"}
L_DELTAS: Tuple[Cell, ...] = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)


@dataclass(frozen=True)
class LWaypointScenario:
    """A two-waypoint L search whose shortest destination sequence is unique."""

    start: Cell
    waypoints: Tuple[Cell, Cell]
    target: Cell
    shortest_routes: Tuple[Tuple[Cell, ...], ...]
    difficulty: str

    @property
    def optimal_length(self) -> int:
        return len(self.shortest_routes[0])

    @property
    def unique_route(self) -> Tuple[Cell, ...]:
        if len(self.shortest_routes) != 1:
            raise ValueError("scenario does not have a unique shortest route")
        return self.shortest_routes[0]

def inside(cell: Cell) -> bool:
    row, column = cell
    return 0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE


def to_external_cell(cell: Cell) -> ExternalCell:
    return to_visible_cell(cell, size=BOARD_SIZE)


def from_external_cell(cell: ExternalCell) -> Cell:
    return from_visible_cell(cell, size=BOARD_SIZE)


def freeze_board(board: Sequence[Sequence[Piece]]) -> Board:
    """Validate the search renderer's empty-or-L board."""

    if len(board) != BOARD_SIZE:
        raise ValueError(f"board must have exactly {BOARD_SIZE} rows")
    frozen_rows: List[Tuple[Piece, ...]] = []
    for row in board:
        if len(row) != BOARD_SIZE:
            raise ValueError(f"every board row must have exactly {BOARD_SIZE} cells")
        frozen_row: List[Piece] = []
        for piece in row:
            if piece is None:
                frozen_row.append(None)
                continue
            if not isinstance(piece, (tuple, list)) or len(piece) != 2:
                raise ValueError(f"malformed L token: {piece!r}")
            kind, light = piece
            if kind != "N" or type(light) is not bool:
                raise ValueError(f"unsupported search token: {piece!r}")
            frozen_row.append(("N", light))
        frozen_rows.append(tuple(frozen_row))
    frozen = tuple(frozen_rows)
    if sum(piece is not None for row in frozen for piece in row) != 1:
        raise ValueError("L-waypoint board must contain exactly one L token")
    return frozen


def empty_board() -> Board:
    return tuple(tuple(None for _ in range(BOARD_SIZE)) for _ in range(BOARD_SIZE))


def _replace_cells(board: Board, updates: Mapping[Cell, Piece]) -> Board:
    mutable = [list(row) for row in board]
    for cell, piece in updates.items():
        if not inside(cell):
            raise ValueError(f"update cell is outside the board: {cell!r}")
        mutable[cell[0]][cell[1]] = piece
    return freeze_board(mutable)


def l_destinations(cell: Cell) -> Tuple[Cell, ...]:
    """Return stable row-major L-offset destinations on the 5x5 board."""

    if not inside(cell):
        raise ValueError(f"cell is outside the board: {cell!r}")
    row, column = cell
    return tuple(
        sorted(
            (row + dr, column + dc)
            for dr, dc in L_DELTAS
            if inside((row + dr, column + dc))
        )
    )


LSearchState = Tuple[Cell, int]


def _normalise_internal_cell(cell: Sequence[int], *, name: str) -> Cell:
    if len(cell) != 2:
        raise ValueError(f"{name} must contain an internal row and column")
    result = int(cell[0]), int(cell[1])
    if not inside(result):
        raise ValueError(f"{name} lies outside the 5x5 board: {cell!r}")
    return result


def _validate_l_problem(
    start: Cell,
    waypoints: Sequence[Cell],
    target: Cell,
) -> Tuple[Cell, Tuple[Cell, Cell], Cell]:
    if len(waypoints) != 2:
        raise ValueError("L search requires exactly two waypoints")
    start_cell = _normalise_internal_cell(start, name="start")
    waypoint_pair = (
        _normalise_internal_cell(waypoints[0], name="first waypoint"),
        _normalise_internal_cell(waypoints[1], name="second waypoint"),
    )
    target_cell = _normalise_internal_cell(target, name="target")
    if len({start_cell, waypoint_pair[0], waypoint_pair[1], target_cell}) != 4:
        raise ValueError("start, both waypoints, and target must be distinct")
    return start_cell, waypoint_pair, target_cell


def _l_shortest_dag(
    start: Cell,
    waypoints: Sequence[Cell],
    target: Cell,
) -> Tuple[
    LSearchState,
    LSearchState,
    Dict[LSearchState, int],
    Dict[LSearchState, List[LSearchState]],
]:
    start_cell, waypoint_pair, target_cell = _validate_l_problem(start, waypoints, target)
    waypoint_bits = {waypoint_pair[0]: 1, waypoint_pair[1]: 2}
    start_state: LSearchState = start_cell, 0
    goal_state: LSearchState = target_cell, 3
    distances: Dict[LSearchState, int] = {start_state: 0}
    parents: Dict[LSearchState, List[LSearchState]] = {start_state: []}
    queue = deque([start_state])
    goal_distance: Optional[int] = None

    while queue:
        state = queue.popleft()
        distance = distances[state]
        if goal_distance is not None and distance >= goal_distance:
            continue
        position, mask = state
        for destination in l_destinations(position):
            next_mask = mask | waypoint_bits.get(destination, 0)
            if destination == target_cell and next_mask != 3:
                continue
            next_state = destination, next_mask
            next_distance = distance + 1
            if next_state not in distances:
                distances[next_state] = next_distance
                parents[next_state] = [state]
                queue.append(next_state)
            elif distances[next_state] == next_distance:
                parents[next_state].append(state)
            if next_state == goal_state:
                goal_distance = next_distance if goal_distance is None else goal_distance
    return start_state, goal_state, distances, parents


def count_shortest_l_routes(start: Cell, waypoints: Sequence[Cell], target: Cell) -> int:
    """Count every shortest route without choosing an arbitrary representative."""

    start_state, goal_state, distances, parents = _l_shortest_dag(start, waypoints, target)
    if goal_state not in distances:
        return 0
    counts: Dict[LSearchState, int] = {start_state: 1}
    for state in sorted(distances, key=lambda item: distances[item]):
        if state != start_state:
            counts[state] = sum(counts[parent] for parent in parents[state])
    return counts[goal_state]


def all_shortest_l_routes(
    start: Cell,
    waypoints: Sequence[Cell],
    target: Cell,
) -> Tuple[Tuple[Cell, ...], ...]:
    """Enumerate all shortest destination sequences, excluding the source."""

    start_state, goal_state, distances, parents = _l_shortest_dag(start, waypoints, target)
    if goal_state not in distances:
        return ()
    routes: List[Tuple[Cell, ...]] = []

    def collect(state: LSearchState, reversed_cells: List[Cell]) -> None:
        if state == start_state:
            routes.append(tuple(reversed(reversed_cells)))
            return
        for parent in parents[state]:
            collect(parent, reversed_cells + [state[0]])

    collect(goal_state, [])
    return tuple(sorted(set(routes)))


def build_l_waypoint_scenario(
    rng: random.Random,
    *,
    difficulty: str = "typical",
    max_attempts: int = 6000,
) -> LWaypointScenario:
    """Build a search instance with a proven unique shortest route."""

    length_ranges = {
        "minimum": (4, 4),
        "typical": (5, 6),
        "maximum": (7, 7),
    }
    if difficulty not in length_ranges:
        raise ValueError("difficulty must be minimum, typical, or maximum")
    minimum_length, maximum_length = length_ranges[difficulty]
    cells = [(row, column) for row in range(BOARD_SIZE) for column in range(BOARD_SIZE)]
    for _ in range(max_attempts):
        start, waypoint_a, waypoint_b, target = rng.sample(cells, 4)
        waypoints = waypoint_a, waypoint_b
        if count_shortest_l_routes(start, waypoints, target) != 1:
            continue
        routes = all_shortest_l_routes(start, waypoints, target)
        if len(routes) == 1 and minimum_length <= len(routes[0]) <= maximum_length:
            return LWaypointScenario(start, waypoints, target, routes, difficulty)
    raise RuntimeError(
        f"could not build a unique {difficulty} L-waypoint scenario after {max_attempts} attempts"
    )


_BG = (247, 249, 252)
_LIGHT_CELL = (240, 244, 249)
_DARK_CELL = (219, 228, 239)
_GRID = (55, 68, 86)
_TEXT = (41, 50, 65)
_LIGHT_TOKEN = (255, 255, 255)
_LIGHT_OUTLINE = (48, 65, 86)
_DARK_TOKEN = (49, 65, 88)
_DARK_OUTLINE = (25, 35, 51)
_GREEN = (65, 151, 104)
_PALE_GREEN = (214, 240, 224)
_RED = (218, 64, 64)


class ChessBoardRenderer:
    """Fixed edge-to-edge renderer for the L-waypoint search scene.

    The cell grid fills the canvas except for the family-wide 44-pixel
    top/left label strip that hosts the 1-based coordinate numbers; the
    right/bottom pad is only the residue of an integer cell size.
    """

    LABEL_STRIP = 44
    CELL_SIZE = 93

    def __init__(self) -> None:
        self.size = BOARD_SIZE
        self.cell_size = self.CELL_SIZE
        self.board_pixels = self.size * self.cell_size
        self.origin_x = self.LABEL_STRIP
        self.origin_y = self.LABEL_STRIP

    def centre(self, cell: Cell) -> Tuple[float, float]:
        if not inside(cell):
            raise ValueError(f"cell is outside the board: {cell!r}")
        row, column = cell
        return (
            self.origin_x + (column + 0.5) * self.cell_size,
            self.origin_y + (row + 0.5) * self.cell_size,
        )

    def _draw_piece(
        self,
        draw: ScaledDraw,
        piece: Tuple[str, bool],
        centre: Tuple[float, float],
    ) -> None:
        kind, light = piece
        cx, cy = centre
        radius = self.cell_size * 0.30
        fill = _LIGHT_TOKEN if light else _DARK_TOKEN
        outline = _LIGHT_OUTLINE if light else _DARK_OUTLINE
        label_fill = _TEXT if light else (255, 255, 255)
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=fill,
            outline=outline,
            width=3,
        )
        highlight_radius = max(2.0, radius * 0.12)
        highlight_x = cx - radius * 0.38
        highlight_y = cy - radius * 0.38
        highlight_fill = (255, 255, 255) if not light else (232, 238, 246)
        draw.ellipse(
            (
                highlight_x - highlight_radius,
                highlight_y - highlight_radius,
                highlight_x + highlight_radius,
                highlight_y + highlight_radius,
            ),
            fill=highlight_fill,
        )
        draw.text(
            (cx, cy - 1),
            KIND_LABEL[kind],
            fill=label_fill,
            font=load_font(max(19, int(self.cell_size * 0.32)) * draw.factor),
            anchor="mm",
        )

    def render(
        self,
        board: Sequence[Sequence[Piece]],
        *,
        waypoints: Sequence[Cell] = (),
        visited_waypoints: Sequence[Cell] = (),
        target: Optional[Cell] = None,
        moving_piece: Optional[Tuple[str, bool]] = None,
        moving_from: Optional[Cell] = None,
        moving_to: Optional[Cell] = None,
        progress: Optional[float] = None,
    ) -> Image.Image:
        """Render one state without a movement-trail API.

        A visited waypoint has exactly one encoding: its cell background is
        the pale-green fill.  The waypoint outline itself never changes and no
        extra glyph is added.
        """

        canonical = freeze_board(board)
        waypoint_set = set(waypoints)
        visited_set = set(visited_waypoints)
        for cell in waypoint_set | visited_set:
            if not inside(cell):
                raise ValueError(f"waypoint is outside the board: {cell!r}")
        if not visited_set.issubset(waypoint_set):
            raise ValueError("visited_waypoints must be a subset of waypoints")
        if target is not None and not inside(target):
            raise ValueError(f"target is outside the board: {target!r}")

        supersampled, draw = new_scaled_canvas(
            CANVAS_SIZE, CANVAS_SIZE, DOCTRINE_SUPERSAMPLE, _BG
        )
        for row in range(self.size):
            for column in range(self.size):
                x0 = self.origin_x + column * self.cell_size
                y0 = self.origin_y + row * self.cell_size
                fill = _LIGHT_CELL if (row + column) % 2 == 0 else _DARK_CELL
                if (row, column) in visited_set:
                    fill = _PALE_GREEN
                draw.rectangle((x0, y0, x0 + self.cell_size, y0 + self.cell_size), fill=fill)

        for index in range(self.size + 1):
            x = self.origin_x + index * self.cell_size
            y = self.origin_y + index * self.cell_size
            width = 4 if index in {0, self.size} else 2
            draw.line(
                (x, self.origin_y, x, self.origin_y + self.board_pixels),
                fill=_GRID,
                width=width,
            )
            draw.line(
                (self.origin_x, y, self.origin_x + self.board_pixels, y),
                fill=_GRID,
                width=width,
            )

        label_font = load_font(max(16, int(self.cell_size * 0.24)) * draw.factor)
        for index in range(self.size):
            x, _ = self.centre((0, index))
            _, y = self.centre((index, 0))
            label = str(index + 1)
            draw.text((x, self.origin_y - 22), label, fill=_TEXT, font=label_font, anchor="mm")
            draw.text((self.origin_x - 22, y), label, fill=_TEXT, font=label_font, anchor="mm")

        for cell in waypoint_set:
            row, column = cell
            x0 = self.origin_x + column * self.cell_size
            y0 = self.origin_y + row * self.cell_size
            inset = 8
            draw.rounded_rectangle(
                (x0 + inset, y0 + inset, x0 + self.cell_size - inset, y0 + self.cell_size - inset),
                radius=12,
                outline=_GREEN,
                width=5,
            )
        if target is not None:
            row, column = target
            x0 = self.origin_x + column * self.cell_size
            y0 = self.origin_y + row * self.cell_size
            draw.rectangle(
                (x0 + 5, y0 + 5, x0 + self.cell_size - 5, y0 + self.cell_size - 5),
                outline=_RED,
                width=5,
            )

        moving_active = (
            moving_piece is not None
            and moving_from is not None
            and moving_to is not None
            and progress is not None
        )
        if moving_active and (not inside(moving_from) or not inside(moving_to)):
            raise ValueError("moving cells must remain inside the board")
        for row in range(self.size):
            for column in range(self.size):
                piece = canonical[row][column]
                if piece is None:
                    continue
                if moving_active and (row, column) == moving_from:
                    continue
                self._draw_piece(draw, piece, self.centre((row, column)))

        if moving_active:
            assert moving_piece is not None
            assert moving_from is not None
            assert moving_to is not None
            assert progress is not None
            depth = max(0.0, min(1.0, float(progress)))
            smooth = depth * depth * (3.0 - 2.0 * depth)
            source_x, source_y = self.centre(moving_from)
            target_x, target_y = self.centre(moving_to)
            centre = (
                source_x + (target_x - source_x) * smooth,
                source_y + (target_y - source_y) * smooth,
            )
            self._draw_piece(draw, moving_piece, centre)
        return finish_scaled_canvas(supersampled, (CANVAS_SIZE, CANVAS_SIZE))


def l_search_board(token: Cell, *, light: bool = True) -> Board:
    """Create the one-token board consumed by the fixed renderer."""

    if not inside(token):
        raise ValueError(f"token cell is outside the board: {token!r}")
    return _replace_cells(empty_board(), {token: ("N", light)})


def visited_waypoints_for_prefix(
    waypoints: Sequence[Cell],
    route_prefix: Sequence[Cell],
) -> Tuple[Cell, ...]:
    """Derive cumulative visited markers from an executed route prefix."""

    if len(waypoints) != 2:
        raise ValueError("L search requires exactly two waypoints")
    waypoint_pair = (
        (int(waypoints[0][0]), int(waypoints[0][1])),
        (int(waypoints[1][0]), int(waypoints[1][1])),
    )
    if waypoint_pair[0] == waypoint_pair[1] or any(not inside(cell) for cell in waypoint_pair):
        raise ValueError("waypoints must be two distinct cells on the board")
    if any(not inside(cell) for cell in route_prefix):
        raise ValueError("route prefix contains a cell outside the board")
    visited = set(route_prefix)
    return tuple(cell for cell in waypoint_pair if cell in visited)


def run_self_check() -> Dict[str, int]:
    """Run an in-memory search and renderer check without persisting assets."""

    search = build_l_waypoint_scenario(random.Random(2718), difficulty="typical")
    assert len(search.shortest_routes) == 1
    assert count_shortest_l_routes(search.start, search.waypoints, search.target) == 1

    tied_start = (0, 0)
    tied_waypoints = ((0, 1), (0, 2))
    tied_target = (1, 3)
    tied_routes = all_shortest_l_routes(tied_start, tied_waypoints, tied_target)
    assert count_shortest_l_routes(tied_start, tied_waypoints, tied_target) == 2
    assert len(tied_routes) == 2

    renderer = ChessBoardRenderer()
    search_image = renderer.render(
        l_search_board(search.start),
        waypoints=search.waypoints,
        target=search.target,
    )
    assert search_image.size == (CANVAS_SIZE, CANVAS_SIZE)
    assert to_external_cell((0, 0)) == (1, 1)
    assert from_external_cell((5, 5)) == (4, 4)
    return {
        "search_length": search.optimal_length,
        "search_solution_count": len(search.shortest_routes),
    }


__all__ = [
    "BOARD_SIZE",
    "Board",
    "Cell",
    "ChessBoardRenderer",
    "ExternalCell",
    "KIND_LABEL",
    "LWaypointScenario",
    "L_DELTAS",
    "Piece",
    "all_shortest_l_routes",
    "build_l_waypoint_scenario",
    "count_shortest_l_routes",
    "empty_board",
    "freeze_board",
    "from_external_cell",
    "inside",
    "l_destinations",
    "l_search_board",
    "run_self_check",
    "to_external_cell",
    "visited_waypoints_for_prefix",
]

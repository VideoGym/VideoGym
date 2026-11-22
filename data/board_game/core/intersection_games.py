"""Pure semantics and one numeric renderer for Go and terminal Gomoku tasks.

This module is deliberately limited to Gate A concerns.  It owns the board
geometry, legal state transitions, deterministic scenario construction and
the semantic step history from which cumulative reasoning images can later be
derived.  Prompt, answer, profile and verifier policy do not belong here.

Every operation explicitly names its stone colour.  That supplied schedule is
authoritative: no hidden alternation rule or inference from the initial stone
counts changes which colour is placed next.

Coordinates are zero-based internally.  :class:`NumericIntersectionRenderer`
is the single model-facing geometry and labels its top row and left column
with the numbers 1 through ``size``.  Thus internal ``(0, 0)`` is visibly
``(1, 1)``.
"""

from __future__ import annotations

from data.common.assets import render_font

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple

from PIL import Image, ImageFont

from data.board_game.core.coordinates import (
    from_visible_cell,
    to_visible_cell,
    visible_cell_text,
)
from data.common.pipeline.board_motion import CANVAS_SIZE
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)


Cell = Tuple[int, int]
Stone = Literal["B", "W"]
Board = Tuple[Tuple[Optional[Stone], ...], ...]

_STONES = frozenset({"B", "W"})
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_LINE_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
_MAX_SCENARIO_ATTEMPTS = 256


class IllegalPlacement(ValueError):
    """A supplied operation cannot legally be applied to its current board."""

    def __init__(self, move_index: int, move: "StoneMove", reason: str) -> None:
        self.move_index = int(move_index)
        self.move = move
        self.reason = str(reason)
        super().__init__(f"move {self.move_index + 1} {move}: {self.reason}")


@dataclass(frozen=True)
class StoneMove:
    """One explicit coloured placement at an internal zero-based cell."""

    colour: Stone
    cell: Cell

    def visible_cell(self, *, size: Literal[7, 9]) -> Cell:
        """Return this move's coordinate through the shared model-facing codec."""

        return to_visible_cell(self.cell, size=size)


@dataclass(frozen=True)
class GoStep:
    """One fully determined Go transition. ``placed_board`` retains the newly placed stone and all not-yet-removed opponent groups. ``final_board`` applies all captures simultaneously."""

    move_index: int
    move: StoneMove
    board_before: Board
    placed_board: Board
    captured_groups: Tuple[Tuple[Cell, ...], ...]
    captured_cells: Tuple[Cell, ...]
    own_group: Tuple[Cell, ...]
    own_liberties: Tuple[Cell, ...]
    final_board: Board

    @property
    def captured_count(self) -> int:
        return len(self.captured_cells)


@dataclass(frozen=True)
class GoSimulation:
    initial_board: Board
    moves: Tuple[StoneMove, ...]
    steps: Tuple[GoStep, ...]
    final_board: Board


@dataclass(frozen=True)
class GoSearchCandidate:
    cell: Cell
    step: GoStep

    @property
    def captured_count(self) -> int:
        return self.step.captured_count


@dataclass(frozen=True)
class GoMaxCaptureSolution:
    """Exhaustive black-placement result, without an arbitrary tie-break."""

    legal_candidates: Tuple[GoSearchCandidate, ...]
    suicidal_cells: Tuple[Cell, ...]
    max_captured: int
    best_cells: Tuple[Cell, ...]


@dataclass(frozen=True)
class GoSimulatorScenario:
    initial_board: Board
    moves: Tuple[StoneMove, ...]
    simulation: GoSimulation


@dataclass(frozen=True)
class GoSearchScenario:
    initial_board: Board
    solution: GoMaxCaptureSolution


@dataclass(frozen=True)
class LineStep:
    move_index: int
    move: StoneMove
    board_before: Board
    board_after: Board
    winning_cells: Tuple[Cell, ...]
    board_full: bool


@dataclass(frozen=True)
class GomokuSimulation:
    initial_board: Board
    moves: Tuple[StoneMove, ...]
    steps: Tuple[LineStep, ...]
    ignored_moves: Tuple[StoneMove, ...]
    final_board: Board
    winner: Optional[Stone]
    winning_cells: Tuple[Cell, ...]
    stop_reason: Literal["win", "full", "moves_exhausted"]


@dataclass(frozen=True)
class GomokuScenario:
    mode: Literal["terminal_win"]
    initial_board: Board
    moves: Tuple[StoneMove, ...]
    simulation: GomokuSimulation


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    return render_font(size, bold=bold)


class NumericIntersectionRenderer:
    """The shared 7x7 Go / 9x9 Gomoku intersection-board geometry.

    The wood board surface fills the entire canvas with no outer margin band.
    The line grid is inset by one symmetric margin per size, which hosts the
    1-based coordinate numbers above and left of the grid while keeping edge
    stones fully on the wood.  Optional rings are overlays and never move or
    resize the board.
    """

    BACKGROUND = (226, 191, 130)
    WOOD = (226, 191, 130)
    GRID = (73, 58, 43)
    LABEL = (48, 50, 57)
    BLACK = (39, 43, 50)
    BLACK_EDGE = (10, 13, 18)
    WHITE = (246, 244, 237)
    WHITE_EDGE = (111, 105, 94)
    HIGHLIGHT = (230, 178, 42)
    CANDIDATE = (52, 116, 190)
    _GEOMETRY = {7: (58, 66), 9: (56, 50)}

    def __init__(self, size: Literal[7, 9]) -> None:
        if size not in self._GEOMETRY:
            raise ValueError("intersection boards are fixed to Go 7x7 or Gomoku 9x9")
        self.size = int(size)
        margin, cell = self._GEOMETRY[self.size]
        from data.common.prompt_paraphrase import current_prompt_paraphrase
        plan = current_prompt_paraphrase()
        if plan and plan.get("board_spacing") == "compact":
            cell -= 6
            margin = (CANVAS_SIZE - cell * (self.size - 1)) // 2
        self.cell = cell
        self.board_span = cell * (self.size - 1)
        self.origin_x = margin
        self.origin_y = margin
        self.stone_radius = int(round(self.cell * (0.40 if self.size == 7 else 0.38)))
        self.label_gap = max(34, self.stone_radius + 16)
        self.diamond = bool(plan and plan.get("profile") == "tile_go")
        if self.diamond:
            self.cell = 44 if self.size == 7 else 34
            self.stone_radius = 16 if self.size == 7 else 12

    def intersection_xy(self, cell: Cell) -> Tuple[int, int]:
        row, col = cell
        if not _inside(self.size, cell):
            raise ValueError(f"cell {cell} is outside {self.size}x{self.size}")
        if self.diamond:
            step = self.cell * .72
            return round(256 + (col-row)*step), round(256 + (col+row-(self.size-1))*step)
        return self.origin_x + col * self.cell, self.origin_y + row * self.cell

    def render(
        self,
        board: Sequence[Sequence[Optional[str]]],
        *,
        ring_cells: Iterable[Cell] = (),
        ring_colour: Tuple[int, int, int] = HIGHLIGHT,
        winning_line: Sequence[Cell] = (),
    ) -> Image.Image:
        frozen = freeze_board(board, size=self.size, allowed=_STONES)
        rings = frozenset(ring_cells)
        if any(not _inside(self.size, cell) for cell in rings):
            raise ValueError("ring cell is outside the board")
        ordered_line = tuple(sorted(set(winning_line)))
        if ordered_line:
            if self.size != 9 or len(ordered_line) != 5:
                raise ValueError("winning_line must contain exactly five Gomoku intersections")
            if any(not _inside(self.size, cell) for cell in ordered_line):
                raise ValueError("winning_line contains an intersection outside the board")
            deltas = tuple(
                (right[0] - left[0], right[1] - left[1])
                for left, right in zip(ordered_line, ordered_line[1:])
            )
            if not deltas or len(set(deltas)) != 1 or deltas[0] not in _LINE_DIRECTIONS:
                raise ValueError("winning_line must be one contiguous straight run")
            stones = {frozen[row][col] for row, col in ordered_line}
            if len(stones) != 1 or None in stones:
                raise ValueError("winning_line must cross five equal visible stones")

        supersampled, draw = new_scaled_canvas(
            CANVAS_SIZE, CANVAS_SIZE, DOCTRINE_SUPERSAMPLE, self.WOOD
        )
        from data.common.visual_style import current_style
        style = current_style()
        if style:
            style.surface(draw, (0, 0, CANVAS_SIZE, CANVAS_SIZE), style.background)
        grid_color = (116, 151, 178) if style and style.name == "machined" else self.GRID
        label_color = (225, 235, 245) if style and style.name == "machined" else self.LABEL

        from data.common.prompt_paraphrase import active_profile
        if active_profile("tile_go"):
            draw.rectangle((0,0,512,512),fill=(245,249,245))
            label_color=(45,78,77)
            for row in range(self.size):
                for col in range(self.size):
                    cx,cy=self.intersection_xy((row,col));half=self.cell*.67
                    fill=(223,237,237) if (row+col)%2==0 else (139,177,180)
                    draw.polygon([(cx,cy-half),(cx+half,cy),(cx,cy+half),(cx-half,cy)],fill=fill,outline=(70,111,121),width=2)
        else:
            for index in range(self.size):
                offset = index * self.cell
                draw.line(
                    (self.origin_x, self.origin_y + offset,
                     self.origin_x + self.board_span, self.origin_y + offset),
                    fill=grid_color,
                    width=2,
                )
                draw.line(
                    (self.origin_x + offset, self.origin_y,
                     self.origin_x + offset, self.origin_y + self.board_span),
                    fill=grid_color,
                    width=2,
                )
        label_font = _font(18 * draw.factor, bold=True)
        for index in range(self.size):
            if self.diamond:
                x,y=self.intersection_xy((0,index))
                draw.text((x+21,y-21),str(index+1),fill=label_color,font=label_font,anchor="mm")
                x,y=self.intersection_xy((index,0))
                draw.text((x-21,y-21),str(index+1),fill=label_color,font=label_font,anchor="mm")
                continue
            x, _ = self.intersection_xy((0, index))
            _, y = self.intersection_xy((index, 0))
            # Model-facing coordinates are 1-based and numeric on both axes.
            draw.text(
                (x, self.origin_y - self.label_gap),
                str(index + 1),
                fill=label_color,
                font=label_font,
                anchor="mm",
            )
            draw.text(
                (self.origin_x - self.label_gap, y),
                str(index + 1),
                fill=label_color,
                font=label_font,
                anchor="mm",
            )

        for row in range(self.size):
            for col in range(self.size):
                stone = frozen[row][col]
                if stone is None:
                    continue
                cx, cy = self.intersection_xy((row, col))
                radius = self.stone_radius
                body = self.BLACK if stone == "B" else self.WHITE
                edge = self.BLACK_EDGE if stone == "B" else self.WHITE_EDGE
                if style:
                    rim = (160, 174, 188) if style.name == "machined" and stone == "B" else edge
                    style.token(draw, (cx - radius, cy - radius, cx + radius, cy + radius), body, rim, width=3)
                    continue
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=body,
                    outline=edge,
                    width=3,
                )
                highlight = (93, 97, 104) if stone == "B" else (255, 255, 250)
                dot = max(3, radius // 5)
                draw.ellipse(
                    (cx - radius * 0.45, cy - radius * 0.45,
                     cx - radius * 0.45 + dot, cy - radius * 0.45 + dot),
                    fill=highlight,
                )

        for cell in sorted(rings):
            cx, cy = self.intersection_xy(cell)
            radius = self.stone_radius + 5
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                outline=ring_colour,
                width=5,
            )
        if ordered_line:
            draw.line(
                (self.intersection_xy(ordered_line[0]), self.intersection_xy(ordered_line[-1])),
                fill=self.HIGHLIGHT,
                width=7,
            )
        return finish_scaled_canvas(supersampled, (CANVAS_SIZE, CANVAS_SIZE))


def freeze_board(
    board: Sequence[Sequence[Optional[str]]],
    *,
    size: Optional[int] = None,
    allowed: Iterable[str] = _STONES,
) -> Board:
    """Validate and copy a square board into an immutable representation."""

    if isinstance(board, (str, bytes)) or not isinstance(board, Sequence):
        raise ValueError("board must be a square sequence of rows")
    frozen = tuple(tuple(row) for row in board)
    expected = len(frozen) if size is None else int(size)
    if expected < 2 or len(frozen) != expected or any(len(row) != expected for row in frozen):
        raise ValueError(f"board must be exactly {expected}x{expected}")
    valid = frozenset(allowed)
    if any(cell is not None and cell not in valid for row in frozen for cell in row):
        raise ValueError("board contains an unsupported piece code")
    return frozen  # type: ignore[return-value]


def _replace(board: Board, cell: Cell, value: Optional[Stone]) -> Board:
    row, col = cell
    mutable = [list(values) for values in board]
    mutable[row][col] = value
    return tuple(tuple(values) for values in mutable)


def _inside(size: int, cell: Cell) -> bool:
    return 0 <= cell[0] < size and 0 <= cell[1] < size


def visible_cell(cell: Cell, *, size: Literal[7, 9]) -> Cell:
    """Return the numeric one-based coordinate shown by the renderer."""

    return to_visible_cell(cell, size=size)


def internal_cell(cell: Cell, *, size: Literal[7, 9]) -> Cell:
    """Convert a visible numeric coordinate back to an array index."""

    return from_visible_cell(cell, size=size)


def cell_text(cell: Cell, *, size: Literal[7, 9]) -> str:
    """Serialize an internal cell only through the shared visible codec."""

    return visible_cell_text(cell, size=size)


def _neighbours(size: int, cell: Cell) -> Tuple[Cell, ...]:
    row, col = cell
    return tuple(
        (row + dr, col + dc)
        for dr, dc in _DIRECTIONS
        if _inside(size, (row + dr, col + dc))
    )


def _group_and_liberties(board: Board, start: Cell) -> Tuple[Tuple[Cell, ...], Tuple[Cell, ...]]:
    colour = board[start[0]][start[1]]
    if colour not in _STONES:
        return (), ()
    group = {start}
    liberties: set[Cell] = set()
    stack = [start]
    while stack:
        cell = stack.pop()
        for neighbour in _neighbours(len(board), cell):
            value = board[neighbour[0]][neighbour[1]]
            if value is None:
                liberties.add(neighbour)
            elif value == colour and neighbour not in group:
                group.add(neighbour)
                stack.append(neighbour)
    return tuple(sorted(group)), tuple(sorted(liberties))


def _validate_move(move: StoneMove, *, size: int, move_index: int) -> None:
    if not isinstance(move, StoneMove):
        raise TypeError("moves must be StoneMove instances with explicit colours")
    if move.colour not in _STONES:
        raise IllegalPlacement(move_index, move, "colour must be B or W")
    if (
        not isinstance(move.cell, tuple)
        or len(move.cell) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in move.cell)
        or not _inside(size, move.cell)
    ):
        raise IllegalPlacement(move_index, move, "cell is outside the board")


def _validate_go_position(board: Board) -> None:
    inspected: set[Cell] = set()
    for row in range(len(board)):
        for col in range(len(board)):
            cell = (row, col)
            if board[row][col] is None or cell in inspected:
                continue
            group, liberties = _group_and_liberties(board, cell)
            inspected.update(group)
            if not liberties:
                raise ValueError("initial Go board contains a zero-liberty group")


def apply_go_move(board: Sequence[Sequence[Optional[str]]], move: StoneMove, *, move_index: int = 0) -> GoStep:
    """Apply one move using capture-before-suicide Go semantics.

    There is no pass, ko or history rule.  Every adjacent opposing group is
    inspected on the same placed board, and all zero-liberty groups are then
    removed simultaneously.  Only after those removals is self-capture tested.
    """

    frozen = freeze_board(board, allowed=_STONES)
    _validate_go_position(frozen)
    _validate_move(move, size=len(frozen), move_index=move_index)
    row, col = move.cell
    if frozen[row][col] is not None:
        raise IllegalPlacement(move_index, move, "intersection is occupied")

    placed = _replace(frozen, move.cell, move.colour)
    opponent: Stone = "W" if move.colour == "B" else "B"
    inspected: set[Cell] = set()
    captured_groups: list[Tuple[Cell, ...]] = []
    for neighbour in _neighbours(len(placed), move.cell):
        if placed[neighbour[0]][neighbour[1]] != opponent or neighbour in inspected:
            continue
        group, liberties = _group_and_liberties(placed, neighbour)
        inspected.update(group)
        if not liberties:
            captured_groups.append(group)

    captured_groups.sort(key=lambda group: group[0])
    captured = tuple(sorted(cell for group in captured_groups for cell in group))
    final = placed
    for cell in captured:
        final = _replace(final, cell, None)

    own_group, own_liberties = _group_and_liberties(final, move.cell)
    if not own_liberties:
        raise IllegalPlacement(move_index, move, "suicide is forbidden")
    return GoStep(
        move_index=int(move_index),
        move=move,
        board_before=frozen,
        placed_board=placed,
        captured_groups=tuple(captured_groups),
        captured_cells=captured,
        own_group=own_group,
        own_liberties=own_liberties,
        final_board=final,
    )


def simulate_go(board: Sequence[Sequence[Optional[str]]], moves: Sequence[StoneMove]) -> GoSimulation:
    """Execute an explicit coloured Go operation sequence without mutation."""

    initial = freeze_board(board, allowed=_STONES)
    _validate_go_position(initial)
    operation_tuple = tuple(moves)
    current = initial
    steps: list[GoStep] = []
    for index, move in enumerate(operation_tuple):
        step = apply_go_move(current, move, move_index=index)
        steps.append(step)
        current = step.final_board
    return GoSimulation(initial, operation_tuple, tuple(steps), current)


def solve_go_max_capture(board: Sequence[Sequence[Optional[str]]]) -> GoMaxCaptureSolution:
    """Exhaustively solve the black maximum-capture placement problem.

    All legal empty intersections are retained in row-major order.  Tied
    maxima are all returned; this function never selects a private tie-break.
    A task source with no positive capture is rejected.
    """

    initial = freeze_board(board, allowed=_STONES)
    _validate_go_position(initial)
    legal: list[GoSearchCandidate] = []
    suicidal: list[Cell] = []
    for row in range(len(initial)):
        for col in range(len(initial)):
            if initial[row][col] is not None:
                continue
            cell = (row, col)
            try:
                step = apply_go_move(initial, StoneMove("B", cell))
            except IllegalPlacement as exc:
                if exc.reason != "suicide is forbidden":
                    raise
                suicidal.append(cell)
                continue
            legal.append(GoSearchCandidate(cell, step))
    if not legal:
        raise ValueError("Go position has no legal black placement")
    maximum = max(candidate.captured_count for candidate in legal)
    if maximum <= 0:
        raise ValueError("Go position has no positive black capture")
    best = tuple(candidate.cell for candidate in legal if candidate.captured_count == maximum)
    return GoMaxCaptureSolution(tuple(legal), tuple(suicidal), maximum, best)


def _winning_run(board: Board, cell: Cell, colour: Stone, target: int) -> Tuple[Cell, ...]:
    winning: set[Cell] = set()
    for dr, dc in _LINE_DIRECTIONS:
        run = [cell]
        row, col = cell[0] - dr, cell[1] - dc
        while _inside(len(board), (row, col)) and board[row][col] == colour:
            run.insert(0, (row, col))
            row, col = row - dr, col - dc
        row, col = cell[0] + dr, cell[1] + dc
        while _inside(len(board), (row, col)) and board[row][col] == colour:
            run.append((row, col))
            row, col = row + dr, col + dc
        if len(run) >= target:
            winning.update(run)
    return tuple(sorted(winning))


def _board_has_line(board: Board, colour: Stone, target: int) -> bool:
    return any(
        board[row][col] == colour and _winning_run(board, (row, col), colour, target)
        for row in range(len(board))
        for col in range(len(board))
    )


def simulate_gomoku(board: Sequence[Sequence[Optional[str]]], moves: Sequence[StoneMove]) -> GomokuSimulation:
    """Place B/W stones until the first run of at least five or a full board.

    Each operation explicitly supplies B or W; the simulator does not infer a
    turn from the initial stone counts.  This keeps arbitrary visible puzzle
    states closed under the operation sequence stated in the prompt.

    Operations after that first terminal step are retained as ``ignored_moves``
    and do not mutate the terminal board.  Their colours and coordinates are
    still syntax-checked so a source sequence cannot hide malformed metadata.
    """

    initial = freeze_board(board, size=9, allowed=_STONES)
    if _board_has_line(initial, "B", 5) or _board_has_line(initial, "W", 5) or all(
        cell is not None for row in initial for cell in row
    ):
        raise ValueError("initial Gomoku board must be nonterminal")
    operation_tuple = tuple(moves)
    for index, move in enumerate(operation_tuple):
        _validate_move(move, size=9, move_index=index)

    current = initial
    steps: list[LineStep] = []
    winner: Optional[Stone] = None
    winning_cells: Tuple[Cell, ...] = ()
    stop_reason: Literal["win", "full", "moves_exhausted"] = "moves_exhausted"
    for index, move in enumerate(operation_tuple):
        row, col = move.cell
        if current[row][col] is not None:
            raise IllegalPlacement(index, move, "intersection is occupied")
        after = _replace(current, move.cell, move.colour)
        winning = _winning_run(after, move.cell, move.colour, 5)
        full = all(cell is not None for values in after for cell in values)
        steps.append(LineStep(index, move, current, after, winning, full))
        current = after
        if winning:
            winner = move.colour
            winning_cells = winning
            stop_reason = "win"
            break
        if full:
            stop_reason = "full"
            break
    ignored = operation_tuple[len(steps):]
    return GomokuSimulation(
        initial,
        operation_tuple,
        tuple(steps),
        ignored,
        current,
        winner,
        winning_cells,
        stop_reason,
    )


def _seeded_rng(namespace: str, seed: int) -> random.Random:
    material = f"{namespace}:{int(seed)}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))


def _empty(size: int) -> Board:
    return tuple(tuple(None for _ in range(size)) for _ in range(size))


def _set_many(board: Board, entries: Iterable[Tuple[Cell, Stone]]) -> Board:
    result = board
    for cell, colour in entries:
        if not _inside(len(result), cell) or result[cell[0]][cell[1]] is not None:
            raise ValueError("scenario motif overlaps or leaves the board")
        result = _replace(result, cell, colour)
    return result


def build_go_max_capture_scenario(seed: int) -> GoSearchScenario:
    """Build a 7x7 search source with a proved unique positive maximum.

    The doomed White material varies per sample: one connected group of one
    to three stones, or two separate groups whose only shared last liberty is
    the target so a single placement removes both simultaneously.  A separate
    motif always offers a strictly smaller alternative — a capturable smaller
    group when the maximum exceeds one, or a two-liberty near-miss group when
    the maximum is a single stone.  Random sparse stones are accepted only
    after the exhaustive solver still proves the intended cell is the sole
    maximum.
    """

    rng = _seeded_rng("go-max-capture", seed)
    for _ in range(_MAX_SCENARIO_ATTEMPTS * 4):
        double_group = rng.random() < 0.35
        if double_group:
            maximum = rng.randint(2, 4)
            first_size = rng.randint(1, maximum - 1)
            group_sizes = (first_size, maximum - first_size)
        else:
            maximum = rng.choice((1, 2, 2, 3))
            group_sizes = (maximum,)

        target = (rng.randint(1, 5), rng.randint(1, 5))
        blocked: set[Cell] = {target}
        victim_groups: list[Tuple[Cell, ...]] = []
        feasible = True
        for group_size in group_sizes:
            attached = None
            seed_cells = [cell for cell in _neighbours(7, target) if cell not in blocked]
            rng.shuffle(seed_cells)
            for seed_cell in seed_cells:
                # Grow from a target neighbour so the group touches the target,
                # which is what makes the target its shared last liberty.
                group = [seed_cell]
                while len(group) < group_size:
                    frontier = [
                        neighbour
                        for cell in group
                        for neighbour in _neighbours(7, cell)
                        if neighbour not in blocked and neighbour not in group
                    ]
                    if not frontier:
                        break
                    group.append(rng.choice(frontier))
                if len(group) == group_size:
                    attached = tuple(sorted(group))
                    break
            if attached is None:
                feasible = False
                break
            victim_groups.append(attached)
            blocked.update(attached)
            # Later groups must not touch this group, or they would merge.
            for cell in attached:
                blocked.update(_neighbours(7, cell))
            blocked.add(target)
        if not feasible or len(victim_groups) != len(group_sizes):
            continue

        board = _empty(7)
        entries: dict[Cell, Stone] = {}
        for group in victim_groups:
            for cell in group:
                entries[cell] = "W"
        for group in victim_groups:
            for cell in group:
                for neighbour in _neighbours(7, cell):
                    if neighbour == target or entries.get(neighbour) == "W":
                        continue
                    entries[neighbour] = "B"
        protected = set(entries) | {target}
        for cell in list(protected):
            protected.update(_neighbours(7, cell))

        # Strictly smaller alternative motif, well separated from the target.
        decoy_region = [
            (row, col)
            for row in range(7)
            for col in range(7)
            if (row, col) not in protected
            and all(
                abs(row - cell[0]) + abs(col - cell[1]) >= 3
                for cell in list(entries) + [target]
            )
        ]
        rng.shuffle(decoy_region)
        decoy_built = False
        decoy_liberty_count = 1 if maximum > 1 else 2
        decoy_size = rng.randint(1, max(1, maximum - 1)) if maximum > 1 else rng.randint(1, 2)
        for decoy_seed in decoy_region:
            decoy_blocked = protected | set(entries)
            group = [decoy_seed]
            while len(group) < decoy_size:
                frontier = [
                    neighbour
                    for cell in group
                    for neighbour in _neighbours(7, cell)
                    if neighbour not in decoy_blocked and neighbour not in group
                ]
                if not frontier:
                    break
                group.append(rng.choice(frontier))
            if len(group) != decoy_size:
                continue
            liberty_candidates = sorted({
                neighbour
                for cell in group
                for neighbour in _neighbours(7, cell)
                if neighbour not in group and neighbour not in decoy_blocked
            })
            if len(liberty_candidates) < decoy_liberty_count:
                continue
            open_liberties = rng.sample(liberty_candidates, decoy_liberty_count)
            overlaps = False
            for cell in group:
                if entries.get(cell) is not None:
                    overlaps = True
            for cell in liberty_candidates:
                if cell not in open_liberties and entries.get(cell) == "W":
                    overlaps = True
            if overlaps:
                continue
            for cell in group:
                entries[cell] = "W"
            for cell in liberty_candidates:
                if cell not in open_liberties:
                    entries[cell] = "B"
            protected.update(group)
            protected.update(liberty_candidates)
            decoy_built = True
            break
        if not decoy_built:
            continue

        try:
            board = _set_many(board, entries.items())
        except ValueError:
            continue

        noise_candidates = [
            (row, col)
            for row in range(7)
            for col in range(7)
            if board[row][col] is None and (row, col) not in protected
        ]
        rng.shuffle(noise_candidates)
        noisy = board
        for cell in noise_candidates[: rng.randint(0, 3)]:
            noisy = _replace(noisy, cell, rng.choice(("B", "W")))
        try:
            solution = solve_go_max_capture(noisy)
        except ValueError:
            continue
        if solution.max_captured != maximum or solution.best_cells != (target,):
            continue
        best_candidate = next(
            candidate for candidate in solution.legal_candidates if candidate.cell == target
        )
        if len(best_candidate.step.captured_groups) != len(group_sizes):
            continue
        if maximum > 1 and not any(
            0 < candidate.captured_count < maximum
            for candidate in solution.legal_candidates
        ):
            continue
        return GoSearchScenario(noisy, solution)
    raise RuntimeError("unable to build a uniquely solved Go max-capture scenario")


def _legal_non_capture_moves(board: Board, colour: Stone, excluded: set[Cell]) -> list[GoStep]:
    choices: list[GoStep] = []
    for row in range(len(board)):
        for col in range(len(board)):
            cell = (row, col)
            if cell in excluded or board[row][col] is not None:
                continue
            try:
                step = apply_go_move(board, StoneMove(colour, cell))
            except IllegalPlacement:
                continue
            if step.captured_count == 0:
                choices.append(step)
    return choices


def build_go_simulator_scenario(
    seed: int, *, program_length: int | None = None
) -> GoSimulatorScenario:
    """Build a legal explicit-colour sequence containing a simultaneous capture.

    The schedule varies per sample: three to five operations, the Black
    capture placed anywhere from the first to the third position, and the
    other operations legal capture-free placements of freely mixed colours.
    Exactly one operation captures, and its count equals the source position's
    proved maximum (one to four stones in one or two groups).
    """

    rng = _seeded_rng("go-simulator", seed)
    for attempt in range(_MAX_SCENARIO_ATTEMPTS):
        search = build_go_max_capture_scenario(seed * 257 + attempt)
        target = search.solution.best_cells[0]
        board = search.initial_board
        capture_position = rng.randint(1, 3)
        suffix_count = rng.randint(1, 2)
        if capture_position + suffix_count < 3:
            suffix_count = 3 - capture_position

        if program_length is not None:
            if not 3 <= int(program_length) <= 9:
                raise ValueError("Go program length must lie in 3..9")
            suffix_count = int(program_length) - capture_position
        moves: list[StoneMove] = []
        used: set[Cell] = {target}
        current = board
        feasible = True
        for _ in range(capture_position - 1):
            choices = _legal_non_capture_moves(current, rng.choice(("B", "W")), used)
            rng.shuffle(choices)
            step = next(
                (
                    choice
                    for choice in choices
                    if abs(choice.move.cell[0] - target[0])
                    + abs(choice.move.cell[1] - target[1])
                    >= 3
                ),
                None,
            )
            if step is None:
                feasible = False
                break
            moves.append(step.move)
            used.add(step.move.cell)
            current = step.final_board
        if not feasible:
            continue
        try:
            capture = apply_go_move(
                current, StoneMove("B", target), move_index=len(moves)
            )
        except IllegalPlacement:
            continue
        if capture.captured_count != search.solution.max_captured:
            continue
        moves.append(StoneMove("B", target))
        current = capture.final_board
        for _ in range(suffix_count):
            choices = _legal_non_capture_moves(current, rng.choice(("B", "W")), used)
            rng.shuffle(choices)
            if not choices:
                feasible = False
                break
            step = choices[0]
            moves.append(step.move)
            used.add(step.move.cell)
            current = step.final_board
        if not feasible:
            continue
        try:
            simulation = simulate_go(board, tuple(moves))
        except (IllegalPlacement, ValueError):
            continue
        capture_counts = [step.captured_count for step in simulation.steps]
        if (
            len(simulation.steps) == len(moves)
            and capture_counts[capture_position - 1] == search.solution.max_captured
            and sum(1 for count in capture_counts if count > 0) == 1
        ):
            return GoSimulatorScenario(board, tuple(moves), simulation)
    raise RuntimeError("unable to build a Go simulator scenario within 256 attempts")


def _five_windows(size: int = 9) -> Tuple[Tuple[Cell, ...], ...]:
    windows: list[Tuple[Cell, ...]] = []
    for dr, dc in _LINE_DIRECTIONS:
        for row in range(size):
            for col in range(size):
                cells = tuple((row + dr * index, col + dc * index) for index in range(5))
                if all(_inside(size, cell) for cell in cells):
                    windows.append(cells)
    return tuple(windows)


def _build_gomoku_terminal_win(rng: random.Random) -> GomokuScenario:
    """Build an executed prefix of three to five placements ending on the win.

    The winning line starts with two or three gaps, the winner fills them at
    varying schedule positions whose last one completes the run, and the
    remaining executed positions are decoy placements of the other colour.
    Two scheduled suffix operations after the win must stay ignored.
    """

    windows = _five_windows()
    for _ in range(_MAX_SCENARIO_ATTEMPTS):
        line = rng.choice(windows)
        winner: Stone = rng.choice(("B", "W"))
        other: Stone = "W" if winner == "B" else "B"
        gap_count = rng.choice((2, 2, 3))
        executed = gap_count + rng.randint(1, 2)
        gaps = sorted(rng.sample(range(5), gap_count))
        gap_cells = [line[index] for index in gaps]
        rng.shuffle(gap_cells)
        board = _empty(9)
        board = _set_many(board, ((cell, winner) for index, cell in enumerate(line) if index not in gaps))

        outside = [
            (row, col)
            for row in range(9)
            for col in range(9)
            if (row, col) not in line
        ]
        rng.shuffle(outside)
        initial_noise = outside[: rng.randint(3, 7)]
        try:
            for index, cell in enumerate(initial_noise):
                colour: Stone = other if index % 2 == 0 else rng.choice(("B", "W"))
                candidate = _replace(board, cell, colour)
                if _board_has_line(candidate, colour, 5):
                    continue
                board = candidate
        except ValueError:
            continue
        if _board_has_line(board, "B", 5) or _board_has_line(board, "W", 5):
            continue

        remaining = [cell for cell in outside if board[cell[0]][cell[1]] is None]
        rng.shuffle(remaining)
        decoy_count = executed - gap_count
        if len(remaining) < decoy_count + 2:
            continue
        decoys = [remaining.pop() for _ in range(decoy_count)]
        suffix_cells = (remaining.pop(), remaining.pop())
        winner_positions = sorted(rng.sample(range(1, executed), gap_count - 1)) + [executed]
        schedule: list[StoneMove] = []
        gap_iter = iter(gap_cells)
        decoy_iter = iter(decoys)
        for position in range(1, executed + 1):
            if position in winner_positions:
                schedule.append(StoneMove(winner, next(gap_iter)))
            else:
                schedule.append(StoneMove(other, next(decoy_iter)))
        schedule.append(StoneMove(other, suffix_cells[0]))
        schedule.append(StoneMove(winner, suffix_cells[1]))
        moves = tuple(schedule)
        try:
            simulation = simulate_gomoku(board, moves)
        except (IllegalPlacement, ValueError):
            continue
        if (
            simulation.stop_reason == "win"
            and simulation.winner == winner
            and len(simulation.steps) == executed
            and simulation.winning_cells == tuple(sorted(line))
            and len(simulation.ignored_moves) == 2
            and all(simulation.final_board[cell[0]][cell[1]] is None for cell in suffix_cells)
        ):
            return GomokuScenario("terminal_win", board, moves, simulation)
    raise RuntimeError("unable to build a terminal-win Gomoku scenario within 256 attempts")


def build_gomoku_simulator_scenario(seed: int) -> GomokuScenario:
    """Build a deterministic trace whose third executed placement wins."""

    return _build_gomoku_terminal_win(_seeded_rng("gomoku-simulator", seed))


def _self_check() -> None:
    renderer7 = NumericIntersectionRenderer(7)
    renderer9 = NumericIntersectionRenderer(9)
    for renderer in (renderer7, renderer9):
        last = renderer.origin_x + renderer.board_span
        assert renderer.origin_x - renderer.label_gap >= 8
        assert last + renderer.stone_radius + 4 <= CANVAS_SIZE

    # Capture two distinct adjacent groups simultaneously.
    board = _empty(5)
    board = _set_many(
        board,
        (
            ((2, 1), "W"), ((1, 1), "B"), ((3, 1), "B"), ((2, 0), "B"),
            ((2, 3), "W"), ((1, 3), "B"), ((3, 3), "B"), ((2, 4), "B"),
        ),
    )
    step = apply_go_move(board, StoneMove("B", (2, 2)))
    assert len(step.captured_groups) == 2 and step.captured_count == 2

    # Capturing into a point with no pre-capture liberty is legal; bare
    # self-capture is not.
    assert step.own_liberties
    suicide = _empty(3)
    suicide = _set_many(
        suicide,
        (((0, 1), "W"), ((1, 0), "W"), ((1, 2), "W"), ((2, 1), "W")),
    )
    try:
        apply_go_move(suicide, StoneMove("B", (1, 1)))
    except IllegalPlacement as exc:
        assert exc.reason == "suicide is forbidden"
    else:
        raise AssertionError("suicide must be rejected")

    # A tie is reported in full, never privately resolved.
    tie = _empty(5)
    tie = _set_many(
        tie,
        (
            ((0, 0), "W"), ((1, 0), "B"),
            ((4, 4), "W"), ((3, 4), "B"),
        ),
    )
    tie_solution = solve_go_max_capture(tie)
    assert tie_solution.max_captured == 1
    assert tie_solution.best_cells == ((0, 1), (4, 3))

    search_maxima: set[int] = set()
    search_group_counts: set[int] = set()
    simulator_lengths: set[int] = set()
    simulator_capture_positions: set[int] = set()
    for seed in range(24):
        search = build_go_max_capture_scenario(seed)
        assert len(search.solution.best_cells) == 1
        assert 1 <= search.solution.max_captured <= 4
        best = next(
            candidate
            for candidate in search.solution.legal_candidates
            if candidate.cell == search.solution.best_cells[0]
        )
        search_maxima.add(search.solution.max_captured)
        search_group_counts.add(len(best.step.captured_groups))
        simulator = build_go_simulator_scenario(seed)
        counts = [step.captured_count for step in simulator.simulation.steps]
        assert 3 <= len(counts) <= 5
        assert sum(1 for count in counts if count > 0) == 1
        simulator_lengths.add(len(counts))
        simulator_capture_positions.add(next(i for i, c in enumerate(counts) if c > 0))
    assert len(search_maxima) >= 2, "capture maxima must vary across seeds"
    assert len(search_group_counts) >= 2, "single and double victim groups must both occur"
    assert len(simulator_lengths) >= 2, "schedule lengths must vary across seeds"
    assert len(simulator_capture_positions) >= 2, "capture positions must vary across seeds"

    gomoku_lengths: set[int] = set()
    for seed in range(24):
        scenario = build_gomoku_simulator_scenario(seed)
        assert scenario.mode == "terminal_win"
        assert scenario.simulation.stop_reason == "win"
        assert 3 <= len(scenario.simulation.steps) <= 5
        assert len(scenario.simulation.winning_cells) == 5
        assert len(scenario.simulation.ignored_moves) == 2
        gomoku_lengths.add(len(scenario.simulation.steps))
        renderer9.render(
            scenario.simulation.final_board,
            winning_line=scenario.simulation.winning_cells,
        )
    assert len(gomoku_lengths) >= 2, "executed prefix lengths must vary across seeds"


if __name__ == "__main__":
    _self_check()


__all__ = [
    "Board",
    "Cell",
    "GoMaxCaptureSolution",
    "GoSearchCandidate",
    "GoSearchScenario",
    "GoSimulation",
    "GoSimulatorScenario",
    "GoStep",
    "GomokuScenario",
    "GomokuSimulation",
    "IllegalPlacement",
    "LineStep",
    "NumericIntersectionRenderer",
    "Stone",
    "StoneMove",
    "apply_go_move",
    "build_go_max_capture_scenario",
    "build_go_simulator_scenario",
    "build_gomoku_simulator_scenario",
    "cell_text",
    "freeze_board",
    "internal_cell",
    "simulate_go",
    "simulate_gomoku",
    "solve_go_max_capture",
    "visible_cell",
]

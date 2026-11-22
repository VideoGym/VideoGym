"""Gate-A semantics and numeric board rendering for terminal Tic-Tac-Toe traces. Every generated scenario starts from a nonterminal position and contains an
explicit X/O placement that creates the first three-in-a-row. The remaining
scheduled suffix is retained so the first-terminal stopping rule is visible
and load-bearing. The supplied mark on every operation is authoritative. There is no hidden
alternation rule and no inference of the next mark from the initial X/O counts."""

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
Mark = Literal["X", "O"]
Board = Tuple[Tuple[Optional[Mark], ...], ...]

_MARKS = frozenset({"X", "O"})
_LINES: Tuple[Tuple[Cell, Cell, Cell], ...] = (
    ((0, 0), (0, 1), (0, 2)),
    ((1, 0), (1, 1), (1, 2)),
    ((2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 1), (2, 1)),
    ((0, 2), (1, 2), (2, 2)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 2), (1, 1), (2, 0)),
)
_MAX_SCENARIO_ATTEMPTS = 128


class IllegalTicTacToePlacement(ValueError):
    def __init__(self, move_index: int, move: "MarkMove", reason: str) -> None:
        self.move_index = int(move_index)
        self.move = move
        self.reason = str(reason)
        super().__init__(f"move {self.move_index + 1} {move}: {self.reason}")


@dataclass(frozen=True)
class MarkMove:
    mark: Mark
    cell: Cell

    @property
    def visible_cell(self) -> Cell:
        return to_visible_cell(self.cell, size=3)


@dataclass(frozen=True)
class TicTacToeStep:
    move_index: int
    move: MarkMove
    board_before: Board
    board_after: Board
    winning_cells: Tuple[Cell, ...]
    board_full: bool


@dataclass(frozen=True)
class TicTacToeSimulation:
    initial_board: Board
    moves: Tuple[MarkMove, ...]
    steps: Tuple[TicTacToeStep, ...]
    ignored_moves: Tuple[MarkMove, ...]
    final_board: Board
    winner: Optional[Mark]
    winning_cells: Tuple[Cell, ...]
    stop_reason: Literal["win", "full", "moves_exhausted"]


@dataclass(frozen=True)
class TicTacToeScenario:
    mode: Literal["terminal_win"]
    initial_board: Board
    moves: Tuple[MarkMove, ...]
    simulation: TicTacToeSimulation


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    return render_font(size, bold=bold)


class TicTacToeRenderer:
    """One fixed edge-to-edge 3x3 layout with numeric 1-based labels.

    The panel surface fills the whole canvas.  The cell grid is inset only by
    the family-wide 44-pixel top/left label strip plus a small residual pad on
    the right and bottom.
    """

    LABEL_STRIP = 44
    PANEL = (231, 237, 245)
    CELL_FILL = (255, 255, 255)
    GRID = (47, 56, 72)
    LABEL = (46, 52, 65)
    X_COLOUR = (55, 124, 218)
    O_COLOUR = (222, 67, 67)
    GOLD = (229, 177, 42)

    def __init__(self) -> None:
        self.size = 3
        self.cell = (CANVAS_SIZE - self.LABEL_STRIP - 8) // 3
        self.board_span = self.cell * 3
        self.origin_x = self.LABEL_STRIP
        self.origin_y = self.LABEL_STRIP

    def cell_box(self, cell: Cell) -> Tuple[int, int, int, int]:
        if not _inside(cell):
            raise ValueError(f"cell {cell} is outside 3x3")
        row, col = cell
        return (
            self.origin_x + col * self.cell,
            self.origin_y + row * self.cell,
            self.origin_x + (col + 1) * self.cell,
            self.origin_y + (row + 1) * self.cell,
        )

    def render(
        self,
        board: Sequence[Sequence[Optional[str]]],
        *,
        winning_cells: Iterable[Cell] = (),
    ) -> Image.Image:
        frozen = _freeze_board(board)
        winners = frozenset(winning_cells)
        if winners:
            if len(winners) != 3 or not any(winners == frozenset(line) for line in _LINES):
                raise ValueError("winning_cells must be one complete row, column, or corner-to-corner diagonal")
            marks = {frozen[row][col] for row, col in winners}
            if len(marks) != 1 or None in marks:
                raise ValueError("winning_cells must contain three equal visible marks")
        supersampled, draw = new_scaled_canvas(
            CANVAS_SIZE, CANVAS_SIZE, DOCTRINE_SUPERSAMPLE, self.PANEL
        )
        from data.common.visual_style import current_style
        style = current_style()
        if style:
            style.surface(draw, (0, 0, CANVAS_SIZE, CANVAS_SIZE), style.background)
        for row in range(3):
            for col in range(3):
                if style:
                    style.surface(draw, self.cell_box((row, col)), self.CELL_FILL)
                else:
                    draw.rectangle(self.cell_box((row, col)), fill=self.CELL_FILL)
        for index in range(4):
            offset = index * self.cell
            draw.line(
                (self.origin_x, self.origin_y + offset,
                 self.origin_x + self.board_span, self.origin_y + offset),
                fill=self.GRID,
                width=6,
            )
            draw.line(
                (self.origin_x + offset, self.origin_y,
                 self.origin_x + offset, self.origin_y + self.board_span),
                fill=self.GRID,
                width=6,
            )

        label_font = _font(21 * draw.factor, bold=True)
        label_color = (225, 235, 245) if style and style.name == "machined" else self.LABEL
        for index in range(3):
            centre = self.origin_x + index * self.cell + self.cell // 2
            draw.text(
                (centre, self.origin_y - 22),
                str(index + 1),
                fill=label_color,
                font=label_font,
                anchor="mm",
            )
            centre_y = self.origin_y + index * self.cell + self.cell // 2
            draw.text(
                (self.origin_x - 22, centre_y),
                str(index + 1),
                fill=label_color,
                font=label_font,
                anchor="mm",
            )

        inset = int(round(self.cell * 0.25))
        for row in range(3):
            for col in range(3):
                mark = frozen[row][col]
                if mark is None:
                    continue
                x0, y0, x1, y1 = self.cell_box((row, col))
                if mark == "X":
                    for points in (((x0 + inset, y0 + inset), (x1 - inset, y1 - inset)),
                                   ((x0 + inset, y1 - inset), (x1 - inset, y0 + inset))):
                        if style:
                            draw.line(points, fill=self.X_COLOUR, width=18)
                            draw.line(points, fill=(123, 172, 235), width=3)
                        else:
                            draw.line(points, fill=self.X_COLOUR, width=15)
                else:
                    draw.ellipse(
                        (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
                        outline=self.O_COLOUR,
                        width=15,
                    )

        for cell in sorted(winners):
            x0, y0, x1, y1 = self.cell_box(cell)
            draw.rounded_rectangle((x0 + 10, y0 + 10, x1 - 10, y1 - 10), radius=14, outline=self.GOLD, width=8)
        return finish_scaled_canvas(supersampled, (CANVAS_SIZE, CANVAS_SIZE))


def _inside(cell: Cell) -> bool:
    return 0 <= cell[0] < 3 and 0 <= cell[1] < 3


def visible_cell(cell: Cell) -> Cell:
    """Return the numeric one-based coordinate shown by the renderer."""

    return to_visible_cell(cell, size=3)


def internal_cell(cell: Cell) -> Cell:
    """Convert a visible numeric coordinate back to an array index."""

    return from_visible_cell(cell, size=3)


def cell_text(cell: Cell) -> str:
    """Serialize an internal cell only through the shared visible codec."""

    return visible_cell_text(cell, size=3)


def _freeze_board(board: Sequence[Sequence[Optional[str]]]) -> Board:
    if isinstance(board, (str, bytes)) or not isinstance(board, Sequence):
        raise ValueError("board must be a 3x3 row sequence")
    frozen = tuple(tuple(row) for row in board)
    if len(frozen) != 3 or any(len(row) != 3 for row in frozen):
        raise ValueError("board must be exactly 3x3")
    if any(cell is not None and cell not in _MARKS for row in frozen for cell in row):
        raise ValueError("board contains an unsupported mark")
    return frozen  # type: ignore[return-value]


def _replace(board: Board, cell: Cell, mark: Mark) -> Board:
    mutable = [list(row) for row in board]
    mutable[cell[0]][cell[1]] = mark
    return tuple(tuple(row) for row in mutable)


def _winning_cells(board: Board, mark: Mark) -> Tuple[Cell, ...]:
    return tuple(sorted({cell for line in _LINES if all(board[row][col] == mark for row, col in line) for cell in line}))


def _validate_move(move: MarkMove, index: int) -> None:
    if not isinstance(move, MarkMove):
        raise TypeError("moves must be MarkMove instances with explicit X/O marks")
    if move.mark not in _MARKS:
        raise IllegalTicTacToePlacement(index, move, "mark must be X or O")
    if (
        not isinstance(move.cell, tuple)
        or len(move.cell) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in move.cell)
        or not _inside(move.cell)
    ):
        raise IllegalTicTacToePlacement(index, move, "cell is outside the board")


def simulate_ttt(board: Sequence[Sequence[Optional[str]]], moves: Sequence[MarkMove]) -> TicTacToeSimulation:
    """Execute explicitly marked placements until the first win or full board."""

    initial = _freeze_board(board)
    if _winning_cells(initial, "X") or _winning_cells(initial, "O") or all(
        cell is not None for row in initial for cell in row
    ):
        raise ValueError("initial Tic-Tac-Toe board must be nonterminal")
    operation_tuple = tuple(moves)
    for index, move in enumerate(operation_tuple):
        _validate_move(move, index)

    current = initial
    steps: list[TicTacToeStep] = []
    winner: Optional[Mark] = None
    winning: Tuple[Cell, ...] = ()
    stop_reason: Literal["win", "full", "moves_exhausted"] = "moves_exhausted"
    for index, move in enumerate(operation_tuple):
        row, col = move.cell
        if current[row][col] is not None:
            raise IllegalTicTacToePlacement(index, move, "cell is occupied")
        after = _replace(current, move.cell, move.mark)
        winning = _winning_cells(after, move.mark)
        full = all(cell is not None for values in after for cell in values)
        steps.append(TicTacToeStep(index, move, current, after, winning, full))
        current = after
        if winning:
            winner = move.mark
            stop_reason = "win"
            break
        if full:
            stop_reason = "full"
            break
    ignored = operation_tuple[len(steps):]
    return TicTacToeSimulation(
        initial,
        operation_tuple,
        tuple(steps),
        ignored,
        current,
        winner,
        winning,
        stop_reason,
    )


def _seeded_rng(seed: int) -> random.Random:
    material = f"tic-tac-toe-simulator:{int(seed)}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))


def _empty() -> Board:
    return tuple(tuple(None for _ in range(3)) for _ in range(3))


def _build_terminal_win(rng: random.Random) -> TicTacToeScenario:
    """Build an executed prefix of three or four placements ending on the win.

    Sampled configurations vary how much of the winning line pre-exists and
    where the winner's placements sit in the schedule: the winner completes
    the line with its second or third executed placement, decoy placements of
    the other mark fill the remaining executed positions, and two scheduled
    suffix operations after the win must stay ignored.
    """

    for _ in range(_MAX_SCENARIO_ATTEMPTS * 2):
        winner: Mark = rng.choice(("X", "O"))
        other: Mark = "O" if winner == "X" else "X"
        line = rng.choice(_LINES)
        line_order = list(line)
        rng.shuffle(line_order)
        # (winner line placements during execution, executed prefix length)
        winner_moves, executed = rng.choice(((2, 3), (2, 4), (3, 4)))
        initial_line_cells = line_order[: 3 - winner_moves]
        executed_line_cells = line_order[3 - winner_moves:]
        outside = [cell for row in range(3) for col in range(3) if (cell := (row, col)) not in line]
        rng.shuffle(outside)
        initial_other_count = rng.randint(1, 2)
        if (len(initial_line_cells) + initial_other_count) % 2 and rng.random() < 0.5:
            initial_other_count = 3 - initial_other_count
        decoy_count = executed - winner_moves
        if initial_other_count + decoy_count + 2 > len(outside):
            continue
        initial_other_cells = outside[:initial_other_count]
        decoy_cells = outside[initial_other_count:initial_other_count + decoy_count]

        board = _empty()
        for cell in initial_line_cells:
            board = _replace(board, cell, winner)
        for cell in initial_other_cells:
            board = _replace(board, cell, other)
        if _winning_cells(board, winner) or _winning_cells(board, other):
            continue

        winner_positions = sorted(rng.sample(range(1, executed), winner_moves - 1)) + [executed]
        schedule: list[MarkMove] = []
        line_iter = iter(executed_line_cells)
        decoy_iter = iter(decoy_cells)
        for position in range(1, executed + 1):
            if position in winner_positions:
                schedule.append(MarkMove(winner, next(line_iter)))
            else:
                schedule.append(MarkMove(other, next(decoy_iter)))
        occupied = set(initial_line_cells) | set(initial_other_cells) | set(line) | set(decoy_cells)
        suffix = [cell for row in range(3) for col in range(3) if (cell := (row, col)) not in occupied]
        rng.shuffle(suffix)
        if len(suffix) < 2:
            continue
        schedule.append(MarkMove(other, suffix[0]))
        schedule.append(MarkMove(winner, suffix[1]))
        moves = tuple(schedule)
        try:
            simulation = simulate_ttt(board, moves)
        except (IllegalTicTacToePlacement, ValueError):
            continue
        if (
            simulation.stop_reason == "win"
            and simulation.winner == winner
            and len(simulation.steps) == executed
            and len(simulation.ignored_moves) == 2
            and all(simulation.final_board[cell[0]][cell[1]] is None for cell in suffix[:2])
        ):
            return TicTacToeScenario("terminal_win", board, moves, simulation)
    raise RuntimeError("unable to build a terminal-win Tic-Tac-Toe scenario within 256 attempts")


def build_ttt_simulator_scenario(seed: int) -> TicTacToeScenario:
    """Build a deterministic trace whose third executed placement wins."""

    return _build_terminal_win(_seeded_rng(seed))


def _self_check() -> None:
    renderer = TicTacToeRenderer()
    assert renderer.cell_box((0, 0))[0:2] == (renderer.origin_x, renderer.origin_y)
    assert renderer.origin_x + renderer.board_span <= CANVAS_SIZE
    executed_lengths: set[int] = set()
    for seed in range(32):
        scenario = build_ttt_simulator_scenario(seed)
        assert scenario.mode == "terminal_win"
        # Replay proves that the trace is determined only by the visible
        # initial board and explicit operation sequence.
        assert simulate_ttt(scenario.initial_board, scenario.moves) == scenario.simulation
        assert scenario.simulation.stop_reason == "win"
        assert len(scenario.simulation.steps) in {3, 4}
        assert len(scenario.simulation.winning_cells) == 3
        assert len(scenario.simulation.ignored_moves) == 2
        executed_lengths.add(len(scenario.simulation.steps))
        renderer.render(
            scenario.simulation.final_board,
            winning_cells=scenario.simulation.winning_cells,
        )
    assert executed_lengths == {3, 4}, "executed prefix lengths must vary across seeds"


if __name__ == "__main__":
    _self_check()


__all__ = [
    "Board",
    "Cell",
    "IllegalTicTacToePlacement",
    "Mark",
    "MarkMove",
    "TicTacToeRenderer",
    "TicTacToeScenario",
    "TicTacToeSimulation",
    "TicTacToeStep",
    "build_ttt_simulator_scenario",
    "cell_text",
    "internal_cell",
    "simulate_ttt",
    "visible_cell",
]

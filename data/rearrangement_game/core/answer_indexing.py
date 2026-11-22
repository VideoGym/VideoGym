"""Visible position contract for Rearrangement Game formal answers.

Every model-facing sequence ordinal is counted from 1. Grid cells instead use
the shared one-based ``(x, y)`` convention, with x as row and y as column.
Solver state, semantic replay, and renderer geometry stay 0-based, so the two
systems meet only in this module. A task generator serializes its internal plan through
:func:`to_visible_plan`; every consumer of a *submitted* answer recovers
internal positions through :func:`from_visible_plan`.

Two kinds of integers must not be confused.  An **ordinal** names a position in
a visible sequence and is therefore shifted.  An **identity** is a label the
image already prints from 1 — a car token, a railcar number, a block numeral —
and is never shifted, exactly like a value stored in a cell.  The per-task
tables below record those distinctions explicitly, so no task can inherit the
wrong one by accident.

The submitted direction fails closed: an ordinal that is missing, non-integer,
or below 1 (an old zero-based payload, for instance) yields ``None`` rather
than an exception, so a verifier can report a rejected candidate without
crashing the batch.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from data.common.grid_coordinates import cell_to_xy, xy_to_cell


# The payload item is itself one ordinal.
_SCALAR_ORDINAL_TASKS = frozenset({
    "color_group_with_locks",     # left slot of the adjacent swap pair
    "jumping_frog_swap",          # current pad of the frog that moves
})

# The payload item is a tuple; the listed positions inside it are ordinals and
# every other position (an action keyword or a printed identity) is preserved.
_TUPLE_ORDINAL_POSITIONS: dict[str, tuple[int, ...]] = {
    "bookshelf_sort_by_height": (0, 1),     # (left slot, right slot)
    # Block numbers printed on the crane blocks are identities and never
    # appear in the payload; only the two column positions shift.
    "crane_block_stack": (0, 1),            # (source column, destination column)
    "dock_yard_crane_swap": (1, 2),         # (crane name, source lane, destination lane)
    # Block numbers printed on the stack blocks are identities and never appear
    # in the payload; only the two column positions shift.
    "stack_reversal_minimal_moves": (0, 1),  # (source column, destination column)
    "token_trade_matrix": (0, 1),           # (position, position)
    # Disk sizes are read from the drawn widths and never appear in the
    # payload; only the two peg positions shift.
    "tower_of_hanoi": (0, 1),               # (source peg, destination peg)
}

# These plans retain their subject/action structure but expose every grid cell
# through the shared visible ``(x, y)`` convention instead of a row-major id.
_GRID_COORDINATE_TASKS = frozenset({
    "chess_pawn_swap",
    "parking_2d_n_by_n",
})

# The payload item is an ``(action, value)`` tuple whose value means different
# things per action.  Only the listed actions carry an ordinal.
_ACTION_ORDINAL_POSITIONS: dict[str, dict[str, tuple[int, ...]]] = {
    # ("park", car identity) names a printed railcar; ("place", slot) names an
    # insertion boundary of the current main track.
    "train_couple_swap": {"place": (1,)},
}

ONE_BASED_ANSWER_TASKS = frozenset(
    _SCALAR_ORDINAL_TASKS
    | set(_TUPLE_ORDINAL_POSITIONS)
    | set(_ACTION_ORDINAL_POSITIONS)
    | _GRID_COORDINATE_TASKS
)

_SCALAR = "scalar"
_POSITIONS = "positions"
_UNSHIFTED = "unshifted"
_MALFORMED = "malformed"


def _move_contract(task_name: str, move: Any) -> tuple[str, tuple[int, ...]]:
    """Classify one move against this task's ordinal contract."""

    if task_name in _SCALAR_ORDINAL_TASKS:
        return _SCALAR, ()
    keyed = _ACTION_ORDINAL_POSITIONS.get(task_name)
    if keyed is not None:
        if not isinstance(move, (list, tuple)) or not move:
            return _MALFORMED, ()
        action = move[0]
        key = action.strip().lower() if isinstance(action, str) else action
        positions = keyed.get(key)
        # An action without an ordinal, and an unknown action, both leave the
        # payload untouched: naming a wrong action is a semantic error for the
        # task rule to reject, not an index-base error.
        return (_POSITIONS, positions) if positions else (_UNSHIFTED, ())
    positions = _TUPLE_ORDINAL_POSITIONS.get(task_name)
    if positions is None:
        return _UNSHIFTED, ()
    if not isinstance(move, (list, tuple)):
        return _MALFORMED, ()
    return _POSITIONS, positions


def _rebuilt(move: Any, values: list[Any]) -> Any:
    return values if isinstance(move, list) else tuple(values)


def _to_visible_ordinal(value: Any, *, task_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{task_name} internal answer position must be a 0-based integer")
    return value + 1


def to_visible_move(task_name: str, move: Any) -> Any:
    """Convert one internal move into its model-facing one-based form."""

    if task_name in _GRID_COORDINATE_TASKS:
        if not isinstance(move, (list, tuple)) or len(move) != 2:
            raise ValueError(f"{task_name} internal move does not match its coordinate contract")
        if task_name == "parking_2d_n_by_n":
            car_id, destination = move
            if isinstance(destination, bool) or not isinstance(destination, int) or not 0 <= destination < 9:
                raise ValueError("parking destination must be a zero-based 3x3 cell")
            visible = (car_id, cell_to_xy(divmod(destination, 3)))
        else:
            source, destination = move
            if any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 25
                for value in (source, destination)
            ):
                raise ValueError("chess-pawn move must use zero-based 5x5 cells")
            visible = (
                cell_to_xy(divmod(source, 5)),
                cell_to_xy(divmod(destination, 5)),
            )
        return _rebuilt(move, list(visible))
    kind, positions = _move_contract(task_name, move)
    if kind == _MALFORMED:
        raise ValueError(f"{task_name} internal move does not match its answer contract")
    if kind == _UNSHIFTED:
        return move
    if kind == _SCALAR:
        return _to_visible_ordinal(move, task_name=task_name)
    values = list(move)
    for index in positions:
        if index >= len(values):
            raise ValueError(f"{task_name} internal move is missing an answer position")
        values[index] = _to_visible_ordinal(values[index], task_name=task_name)
    return _rebuilt(move, values)


def to_visible_plan(task_name: str, plan: Sequence[Any]) -> list[Any]:
    """Convert an internal plan into the model-facing one-based payload."""

    return [to_visible_move(task_name, move) for move in plan]


def _from_visible_ordinal(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value - 1


def from_visible_move(task_name: str, move: Any) -> Any:
    """Recover one internal move from a submitted one-based move.

    Returns ``None`` — never raises — when a submitted ordinal is absent,
    non-integer, or below 1, so an out-of-range candidate fails closed.
    """

    if task_name in _GRID_COORDINATE_TASKS:
        if not isinstance(move, (list, tuple)) or len(move) != 2:
            return None
        if task_name == "parking_2d_n_by_n":
            car_id, raw_destination = move
            destination = xy_to_cell(raw_destination)
            if isinstance(car_id, bool) or not isinstance(car_id, int) or destination is None:
                return None
            if not (0 <= destination[0] < 3 and 0 <= destination[1] < 3):
                return None
            recovered = (car_id, destination[0] * 3 + destination[1])
        else:
            source = xy_to_cell(move[0])
            destination = xy_to_cell(move[1])
            if source is None or destination is None:
                return None
            if not all(
                0 <= row < 5 and 0 <= col < 5
                for row, col in (source, destination)
            ):
                return None
            recovered = (
                source[0] * 5 + source[1],
                destination[0] * 5 + destination[1],
            )
        return _rebuilt(move, list(recovered))
    kind, positions = _move_contract(task_name, move)
    if kind == _MALFORMED:
        return None
    if kind == _UNSHIFTED:
        return move
    if kind == _SCALAR:
        return _from_visible_ordinal(move)
    values = list(move)
    for index in positions:
        if index >= len(values):
            return None
        recovered = _from_visible_ordinal(values[index])
        if recovered is None:
            return None
        values[index] = recovered
    return _rebuilt(move, values)


def from_visible_plan(task_name: str, plan: Sequence[Any]) -> Optional[list[Any]]:
    """Recover an internal plan from a submitted one-based payload.

    Returns ``None`` when any submitted ordinal falls outside the published
    one-based domain.  The whole plan is rejected rather than partially
    repaired, because a single misread position would silently replay a
    different arrangement.
    """

    if task_name not in ONE_BASED_ANSWER_TASKS:
        return list(plan)
    recovered: list[Any] = []
    for move in plan:
        value = from_visible_move(task_name, move)
        if value is None:
            return None
        recovered.append(value)
    return recovered


__all__ = [
    "ONE_BASED_ANSWER_TASKS",
    "from_visible_move",
    "from_visible_plan",
    "to_visible_move",
    "to_visible_plan",
]

"""One model-facing contract for grid positions and grid paths.

Internal solvers and renderers use zero-based ``(row, col)`` tuples.  At the
model boundary every grid position is written ``(x, y)``: ``x`` is the
one-based row counted from the top and ``y`` is the one-based column counted
from the left.  This module owns the wording, canonical spacing, serializers,
and strict parsers so task families cannot drift into direction strings,
linear cell IDs, pipe separators, or list-of-list coordinate spellings.
"""

from __future__ import annotations

from numbers import Integral
import re
from typing import Any, Mapping, Optional, Sequence

from data.common.typed_answers import (
    ANSWER_SYNTAX_EXAMPLE_CUE,
    format_answer_line,
    parse_answer_line,
)


GridCell = tuple[int, int]
GridPath = list[GridCell]

GRID_XY_CONVENTION_SENTENCE = (
    "Write each grid position as (x, y), where x is the row number counted from the top and y is the "
    "column number counted from the left; both start at 1, so the top-left position is (1, 1)."
)

GRID_PATH_ANSWER_SCHEMA = "Answer: [(x1, y1), (x2, y2), ...]."
GRID_PATH_ANSWER_EXAMPLE = "Answer: [(1, 2), (3, 4), ...]."
GRID_PATH_ANSWER_FORMAT_INSTRUCTION = (
    "Write the complete ordered path as a bracketed list of (x, y) positions, including its first "
    f"and final positions. {ANSWER_SYNTAX_EXAMPLE_CUE} `{GRID_PATH_ANSWER_EXAMPLE}` Use one space after every comma."
)
GRID_PATH_STAY_ENCODING_SENTENCE = (
    "Encode one stay by writing the same (x, y) position in two consecutive path entries."
)

GRID_POSITION_LIST_ANSWER_SCHEMA = GRID_PATH_ANSWER_SCHEMA
GRID_POSITION_LIST_ANSWER_EXAMPLE = GRID_PATH_ANSWER_EXAMPLE

_POSITIVE_INTEGER = r"[1-9][0-9]*"
_XY_PAIR_PATTERN = rf"\({_POSITIVE_INTEGER}, {_POSITIVE_INTEGER}\)"
_XY_PAIR_RE = re.compile(rf"\(({_POSITIVE_INTEGER}), ({_POSITIVE_INTEGER})\)")
STRICT_XY_PATH_ANSWER_RE = re.compile(
    rf"^Answer: \[({_XY_PAIR_PATTERN}(?:, {_XY_PAIR_PATTERN})*)\]\.$"
)
STRICT_XY_POSITION_ANSWER_RE = re.compile(
    rf"^Answer: ({_XY_PAIR_PATTERN})\.$"
)


def cell_to_xy(cell: Sequence[int]) -> GridCell:
    """Convert an internal zero-based ``(row, col)`` cell to visible ``(x, y)``."""

    if (
        not isinstance(cell, Sequence)
        or len(cell) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in cell
        )
    ):
        raise ValueError("internal grid cells must be exact integer pairs")
    row, col = int(cell[0]), int(cell[1])
    return row + 1, col + 1


def xy_to_cell(value: Any) -> Optional[GridCell]:
    """Convert a visible one-based ``(x, y)`` pair to internal ``(row, col)``."""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    coordinates: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            return None
        coordinates.append(item)
    return coordinates[0] - 1, coordinates[1] - 1


def format_xy_position(position: Sequence[int]) -> str:
    """Format one already-visible ``(x, y)`` position with canonical spacing."""

    if (
        not isinstance(position, Sequence)
        or len(position) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in position
        )
    ):
        raise ValueError("model-facing grid positions must be exact integer pairs")
    x, y = int(position[0]), int(position[1])
    if x < 1 or y < 1:
        raise ValueError("model-facing grid positions must be positive and 1-based")
    return f"({x}, {y})"


def format_xy_positions_payload(positions: Sequence[Sequence[int]]) -> str:
    """Format visible positions as ``[(x1, y1), (x2, y2), ...]``."""

    if not isinstance(positions, (list, tuple)):
        raise ValueError("grid-position payload must be a list or tuple")
    if not positions:
        raise ValueError("a grid-position payload must contain at least one position")
    return "[" + ", ".join(format_xy_position(position) for position in positions) + "]"


def format_xy_path_payload(cells: Sequence[Sequence[int]]) -> str:
    """Format internal zero-based cells as a model-facing grid-position list."""

    if not isinstance(cells, (list, tuple)):
        raise ValueError("internal grid path must be a list or tuple")
    return format_xy_positions_payload([cell_to_xy(cell) for cell in cells])


def format_xy_position_list_payload(cells: Sequence[Sequence[int]]) -> str:
    """Format internal cells as a non-path list of model-facing positions."""

    return format_xy_path_payload(cells)


def format_xy_path_answer(cells: Sequence[Sequence[int]]) -> str:
    """Format internal cells as the complete canonical ``Answer:`` line."""

    return format_answer_line(format_xy_path_payload(cells))


def format_xy_position_list_answer(cells: Sequence[Sequence[int]]) -> str:
    """Format internal cells as a canonical position-list ``Answer:`` line."""

    return format_answer_line(format_xy_position_list_payload(cells))


def parse_xy_path_answer(text: object) -> Optional[GridPath]:
    """Strictly parse one canonical grid-path Answer into internal cells."""

    positions = parse_visible_xy_positions_answer(text)
    if positions is None:
        return None
    cells = [xy_to_cell(position) for position in positions]
    if any(cell is None for cell in cells):
        return None
    return [cell for cell in cells if cell is not None]


def parse_visible_xy_positions_answer(text: object) -> Optional[list[GridCell]]:
    """Strictly parse canonical pairs while retaining their visible 1-based values."""

    if not isinstance(text, str):
        return None
    match = STRICT_XY_PATH_ANSWER_RE.fullmatch(text)
    if match is None:
        return None
    positions = [
        (int(raw_x), int(raw_y))
        for raw_x, raw_y in _XY_PAIR_RE.findall(match.group(1))
    ]
    return positions or None


def parse_xy_positions_answer(text: object) -> Optional[list[GridCell]]:
    """Strictly parse a canonical position-list Answer into internal cells.

    The wire grammar is deliberately identical to a path.  This separate name
    preserves the semantic distinction for unordered or row-major position
    collections, whose members need not be adjacent.
    """

    return parse_xy_path_answer(text)


def format_xy_position_answer(cell: Sequence[int]) -> str:
    """Format one internal cell as ``Answer: (x, y).``."""

    return format_answer_line(format_xy_position(cell_to_xy(cell)))


def parse_xy_position_answer(text: object) -> Optional[GridCell]:
    """Strictly parse one canonical single-position Answer into an internal cell."""

    if not isinstance(text, str):
        return None
    match = STRICT_XY_POSITION_ANSWER_RE.fullmatch(text)
    if match is None:
        return None
    raw_x, raw_y = _XY_PAIR_RE.fullmatch(match.group(1)).groups()
    return xy_to_cell((int(raw_x), int(raw_y)))


def _validate_path_names(names: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(names)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) is None
            for name in normalized
        )
    ):
        raise ValueError("named grid paths require unique identifier-like names")
    return normalized


def format_named_xy_paths_answer(
    paths: Mapping[str, Sequence[Sequence[int]]],
) -> str:
    """Format named internal paths in mapping order as one canonical Answer line."""

    if not paths:
        raise ValueError("at least one named grid path is required")
    parts = [
        f"{name}={format_xy_path_payload(path)}"
        for name, path in paths.items()
    ]
    _validate_path_names(tuple(paths))
    return format_answer_line("; ".join(parts))


def format_visible_named_xy_paths_answer(
    paths: Mapping[str, Sequence[Sequence[int]]],
) -> str:
    """Format already-visible one-based named paths in mapping order."""

    if not paths:
        raise ValueError("at least one named grid path is required")
    _validate_path_names(tuple(paths))
    parts = [
        f"{name}={format_xy_positions_payload(path)}"
        for name, path in paths.items()
    ]
    return format_answer_line("; ".join(parts))


def parse_visible_named_xy_paths_answer(
    text: object,
    names: Optional[Sequence[str]] = None,
) -> Optional[dict[str, GridPath]]:
    """Parse canonical named paths while retaining visible one-based values."""

    payload = parse_answer_line(text)
    if payload is None:
        return None
    pair_list = rf"\[{_XY_PAIR_PATTERN}(?:, {_XY_PAIR_PATTERN})*\]"
    if names is not None:
        normalized_names = _validate_path_names(names)
        body = "; ".join(
            rf"{re.escape(name)}=({pair_list})" for name in normalized_names
        )
        match = re.fullmatch(body, payload)
        if match is None:
            return None
        raw_paths = list(match.groups())
    else:
        item_pattern = re.compile(
            rf"([A-Za-z][A-Za-z0-9_-]*)=({pair_list})"
        )
        raw_items = payload.split("; ")
        matches = [item_pattern.fullmatch(item) for item in raw_items]
        if not matches or any(match is None for match in matches):
            return None
        normalized_names = tuple(match.group(1) for match in matches if match is not None)
        try:
            normalized_names = _validate_path_names(normalized_names)
        except ValueError:
            return None
        raw_paths = [match.group(2) for match in matches if match is not None]

    parsed: dict[str, GridPath] = {}
    for name, raw_path in zip(normalized_names, raw_paths):
        path = parse_visible_xy_positions_answer(format_answer_line(raw_path))
        if path is None:
            return None
        parsed[name] = path
    return parsed


def parse_named_xy_paths_answer(
    text: object,
    names: Sequence[str],
) -> Optional[dict[str, GridPath]]:
    """Strictly parse named canonical grid paths in the requested order."""

    visible_paths = parse_visible_named_xy_paths_answer(text, names)
    if visible_paths is None:
        return None
    parsed: dict[str, GridPath] = {}
    for name, visible_path in visible_paths.items():
        path = [xy_to_cell(position) for position in visible_path]
        if any(cell is None for cell in path):
            return None
        parsed[name] = [cell for cell in path if cell is not None]
    return parsed


__all__ = [
    "GRID_PATH_ANSWER_EXAMPLE",
    "GRID_PATH_ANSWER_FORMAT_INSTRUCTION",
    "GRID_PATH_ANSWER_SCHEMA",
    "GRID_POSITION_LIST_ANSWER_EXAMPLE",
    "GRID_POSITION_LIST_ANSWER_SCHEMA",
    "GRID_XY_CONVENTION_SENTENCE",
    "GridCell",
    "GridPath",
    "STRICT_XY_PATH_ANSWER_RE",
    "STRICT_XY_POSITION_ANSWER_RE",
    "cell_to_xy",
    "format_named_xy_paths_answer",
    "format_visible_named_xy_paths_answer",
    "format_xy_path_answer",
    "format_xy_path_payload",
    "format_xy_position",
    "format_xy_position_answer",
    "format_xy_position_list_answer",
    "format_xy_position_list_payload",
    "format_xy_positions_payload",
    "parse_named_xy_paths_answer",
    "parse_visible_named_xy_paths_answer",
    "parse_visible_xy_positions_answer",
    "parse_xy_path_answer",
    "parse_xy_positions_answer",
    "parse_xy_position_answer",
    "xy_to_cell",
]

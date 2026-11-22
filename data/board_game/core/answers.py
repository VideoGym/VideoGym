"""Typed-answer serializers shared by rebuilt Board Game tasks."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from data.board_game.core.coordinates import visible_cell_text
from data.common.typed_answers import (
    canonical_json_dumps,
    format_matrix_answer,
    parse_matrix_answer,
)


def _board_symbol_matrix(
    board: Sequence[Sequence[Any]],
    *,
    symbols: Mapping[Any, str] | Callable[[Any], str],
) -> list[list[str]]:
    """Map one square semantic board to its row-major symbol matrix."""

    if isinstance(board, (str, bytes)) or not isinstance(board, Sequence) or not board:
        raise ValueError("board must be a non-empty square row sequence")
    size = len(board)
    matrix: list[list[str]] = []
    for raw_row in board:
        if (
            isinstance(raw_row, (str, bytes))
            or not isinstance(raw_row, Sequence)
            or len(raw_row) != size
        ):
            raise ValueError("board must be square")
        row: list[str] = []
        for cell in raw_row:
            symbol = symbols(cell) if callable(symbols) else symbols[cell]
            if not isinstance(symbol, str) or len(symbol) != 1:
                raise ValueError("every board cell must encode to exactly one character")
            row.append(symbol)
        matrix.append(row)
    return matrix


def encode_board_matrix(
    board: Sequence[Sequence[Any]],
    *,
    symbols: Mapping[Any, str] | Callable[[Any], str],
) -> str:
    """Encode a row-major symbol matrix as canonical JSON without its envelope."""

    return canonical_json_dumps(_board_symbol_matrix(board, symbols=symbols))


def format_board_matrix_answer(
    board: Sequence[Sequence[Any]],
    *,
    symbols: Mapping[Any, str] | Callable[[Any], str],
) -> str:
    """Serialize a square board as one canonical JSON-matrix Answer line."""

    return format_matrix_answer(
        _board_symbol_matrix(board, symbols=symbols),
        cell_kind="string",
    )


def parse_board_matrix_answer(
    text: object,
    *,
    size: int,
    symbols: Sequence[str],
) -> Optional[list[list[str]]]:
    """Strictly parse one fixed-size canonical JSON symbol matrix."""

    if size < 1:
        raise ValueError("board size must be positive")
    allowed = frozenset(str(symbol) for symbol in symbols)
    if not allowed or any(len(symbol) != 1 for symbol in allowed):
        raise ValueError("board symbols must be single characters")
    parsed = parse_matrix_answer(
        text,
        rows=size,
        cols=size,
        cell_kind="string",
        allowed_values=allowed,
    )
    return parsed if parsed is not None else None


def encode_visible_path(path: Sequence[Sequence[int]], *, size: int) -> str:
    """Encode an internal path as one canonical list of visible positions."""

    if not path:
        raise ValueError("path must contain at least one cell")
    return "[" + ", ".join(visible_cell_text(cell, size=size) for cell in path) + "]"


__all__ = [
    "encode_board_matrix",
    "encode_visible_path",
    "format_board_matrix_answer",
    "parse_board_matrix_answer",
]

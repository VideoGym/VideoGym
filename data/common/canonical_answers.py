"""Canonical answer reserialization from formal Answer lines."""

from __future__ import annotations

from typing import Optional

from data.common.cartesian_coordinates import (
    parse_cartesian_point_answer,
)
from data.common.grid_coordinates import (
    parse_visible_named_xy_paths_answer,
    parse_visible_xy_positions_answer,
    parse_xy_position_answer,
)
from data.common.scalar_lists import (
    parse_integer_answer,
    parse_nonnegative_integer_answer,
    parse_nonnegative_integer_list_answer,
    parse_string_list_answer,
)
from data.common.typed_answers import (
    ChoiceCodec,
    decode_canonical_json_answer,
    parse_answer_line,
)


def _parse_grid_payload(payload: object) -> Optional[list[list[object]]]:
    if isinstance(payload, dict):
        payload = payload.get("grid")
    if not isinstance(payload, list) or not payload:
        return None
    rows: list[list[object]] = []
    for row in payload:
        if (
            not isinstance(row, list)
            or not row
            or any(
                isinstance(item, bool) or type(item) not in {int, str}
                for item in row
            )
        ):
            return None
        rows.append(list(row))
    if len({len(row) for row in rows}) != 1:
        return None
    return rows


def parse_canonical_answer_line(
    answer_line: object,
    *,
    kind: str,
    choice_codec: Optional[ChoiceCodec] = None,
) -> object:
    """Parse one formal Answer line through the codec selected by the registry."""

    from data.common.schemas import CanonicalAnswer

    if kind == "option":
        parsed = choice_codec.parse(answer_line) if choice_codec else None
        if parsed is not None:
            return CanonicalAnswer(kind=kind, option=parsed)
    elif kind == "grid":
        payload, valid = decode_canonical_json_answer(answer_line)
        parsed_grid = _parse_grid_payload(payload) if valid else None
        if parsed_grid is not None:
            return CanonicalAnswer(kind=kind, answer_grid=parsed_grid)
    elif kind in {"path", "positions"}:
        parsed_positions = parse_visible_xy_positions_answer(answer_line)
        if parsed_positions is not None:
            field_name = "path_positions" if kind == "path" else "positions"
            return CanonicalAnswer(
                kind=kind,
                **{field_name: [[x, y] for x, y in parsed_positions]},
            )
    elif kind == "position":
        parsed_cell = parse_xy_position_answer(answer_line)
        if parsed_cell is not None:
            return CanonicalAnswer(
                kind=kind,
                position=[parsed_cell[0] + 1, parsed_cell[1] + 1],
            )
    elif kind == "cartesian_point":
        parsed_point = parse_cartesian_point_answer(answer_line)
        if parsed_point is not None:
            return CanonicalAnswer(
                kind=kind,
                cartesian_point=list(parsed_point),
            )
    elif kind == "named_paths":
        parsed_paths = parse_visible_named_xy_paths_answer(answer_line)
        if parsed_paths is not None:
            return CanonicalAnswer(
                kind=kind,
                named_path_positions={
                    name: [[x, y] for x, y in positions]
                    for name, positions in parsed_paths.items()
                },
            )
    elif kind == "json_object":
        payload, valid = decode_canonical_json_answer(answer_line)
        if valid and isinstance(payload, dict):
            return CanonicalAnswer(kind=kind, json_object=payload)
    elif kind == "integer":
        parsed_integer = parse_integer_answer(answer_line)
        if parsed_integer is not None:
            return CanonicalAnswer(kind=kind, integer=parsed_integer)
    elif kind == "nonnegative_integer":
        parsed_integer = parse_nonnegative_integer_answer(answer_line)
        if parsed_integer is not None:
            return CanonicalAnswer(kind=kind, nonnegative_integer=parsed_integer)
    elif kind == "nonnegative_integer_list":
        parsed_values = parse_nonnegative_integer_list_answer(answer_line)
        if parsed_values is not None:
            return CanonicalAnswer(kind=kind, nonnegative_integer_list=parsed_values)
    elif kind == "string_list":
        parsed_values = parse_string_list_answer(answer_line)
        if parsed_values is not None:
            return CanonicalAnswer(kind=kind, string_list=parsed_values)
    elif kind == "typed_text":
        parsed_text = parse_answer_line(answer_line)
        if parsed_text is not None:
            return CanonicalAnswer(kind=kind, typed_text=parsed_text)
    raise ValueError(
        f"Answer line {answer_line!r} does not satisfy registered kind {kind!r}"
    )


__all__ = [
    "parse_canonical_answer_line",
]

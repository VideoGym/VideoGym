"""Canonical model-facing grammars for scalar and scalar-list answers."""

from __future__ import annotations

import re
from numbers import Integral
from typing import Optional, Sequence

from data.common.typed_answers import (
    decode_canonical_json_answer,
    format_answer_line,
    format_canonical_json_answer,
)


NONNEGATIVE_INTEGER_LIST_ANSWER_SCHEMA = "Answer: [<nonnegative integer>, ...]."
NONNEGATIVE_INTEGER_LIST_ANSWER_EXAMPLE = "Answer: [98, 99, 100, ...]."
NONNEGATIVE_INTEGER_ANSWER_SCHEMA = "Answer: <nonnegative integer>."

_INTEGER = r"(?:0|[1-9][0-9]*)"
_SIGNED_INTEGER = r"(?:0|-?[1-9][0-9]*)"
STRICT_NONNEGATIVE_INTEGER_LIST_ANSWER_RE = re.compile(
    rf"^Answer: \[({_INTEGER}(?:, {_INTEGER})*)\]\.$"
)
STRICT_NONNEGATIVE_INTEGER_ANSWER_RE = re.compile(rf"^Answer: ({_INTEGER})\.$")
STRICT_INTEGER_ANSWER_RE = re.compile(rf"^Answer: ({_SIGNED_INTEGER})\.$")


def format_nonnegative_integer_list_payload(values: Sequence[int]) -> str:
    """Serialize a non-empty nonnegative sequence with canonical spacing."""

    if not isinstance(values, (list, tuple)):
        raise ValueError(
            "nonnegative-integer-list payload must be a list or tuple"
        )
    raw_values = list(values)
    if not raw_values:
        raise ValueError("a nonnegative-integer-list payload must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in raw_values
    ):
        raise ValueError(
            "nonnegative-integer-list payload values must be exact integers"
        )
    normalized = [int(value) for value in raw_values]
    if any(value < 0 for value in normalized):
        raise ValueError(
            "nonnegative-integer-list payload values must be nonnegative"
        )
    return "[" + ", ".join(str(value) for value in normalized) + "]"


def format_nonnegative_integer_list_answer(values: Sequence[int]) -> str:
    return format_answer_line(format_nonnegative_integer_list_payload(values))


def parse_nonnegative_integer_list_answer(text: object) -> Optional[list[int]]:
    """Parse only the canonical bracketed grammar; legacy CSV fails closed."""

    match = (
        STRICT_NONNEGATIVE_INTEGER_LIST_ANSWER_RE.fullmatch(text)
        if isinstance(text, str)
        else None
    )
    if match is None:
        return None
    return [int(value) for value in match.group(1).split(", ")]


def format_nonnegative_integer_answer(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("integer answer must be an exact integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError("integer answer must be nonnegative")
    return format_answer_line(str(normalized))


def parse_nonnegative_integer_answer(text: object) -> Optional[int]:
    match = (
        STRICT_NONNEGATIVE_INTEGER_ANSWER_RE.fullmatch(text)
        if isinstance(text, str)
        else None
    )
    return int(match.group(1)) if match is not None else None


def format_integer_answer(value: int) -> str:
    """Serialize one signed integer without coercing strings or booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("integer answer must be an exact integer")
    return format_answer_line(str(int(value)))


def parse_integer_answer(text: object) -> Optional[int]:
    match = (
        STRICT_INTEGER_ANSWER_RE.fullmatch(text)
        if isinstance(text, str)
        else None
    )
    return int(match.group(1)) if match is not None else None


def format_string_list_answer(values: Sequence[str]) -> str:
    """Serialize a non-empty string sequence as canonical JSON."""

    if not isinstance(values, (list, tuple)):
        raise ValueError("string-list payload must be a list or tuple")
    normalized = list(values)
    if (
        not normalized
        or any(not isinstance(value, str) or not value for value in normalized)
    ):
        raise ValueError("string-list payload values must be non-empty strings")
    return format_canonical_json_answer(normalized)


def parse_string_list_answer(text: object) -> Optional[list[str]]:
    """Parse only a non-empty canonical JSON array of non-empty strings."""

    value, valid = decode_canonical_json_answer(text)
    if (
        not valid
        or not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        return None
    return value


__all__ = [
    "NONNEGATIVE_INTEGER_LIST_ANSWER_EXAMPLE",
    "NONNEGATIVE_INTEGER_LIST_ANSWER_SCHEMA",
    "NONNEGATIVE_INTEGER_ANSWER_SCHEMA",
    "STRICT_NONNEGATIVE_INTEGER_LIST_ANSWER_RE",
    "STRICT_INTEGER_ANSWER_RE",
    "STRICT_NONNEGATIVE_INTEGER_ANSWER_RE",
    "format_integer_answer",
    "format_nonnegative_integer_list_answer",
    "format_nonnegative_integer_list_payload",
    "format_nonnegative_integer_answer",
    "format_string_list_answer",
    "parse_nonnegative_integer_list_answer",
    "parse_integer_answer",
    "parse_nonnegative_integer_answer",
    "parse_string_list_answer",
]

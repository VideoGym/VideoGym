"""Strict wire grammar for integer Cartesian point answers.

This codec is intentionally separate from ``grid_coordinates``: Cartesian
axes may contain zero or negative values and do not imply the project-wide
1-based row/column convention used by grid cells.
"""

from __future__ import annotations

from numbers import Integral
import re
from typing import Optional, Sequence


CARTESIAN_POINT_ANSWER_SCHEMA = "Answer: (<x integer>, <y integer>)."
CARTESIAN_POINT_ANSWER_EXAMPLE = "Answer: (98, 99)."

_SIGNED_INTEGER = r"(?:0|-?[1-9][0-9]*)"
STRICT_CARTESIAN_POINT_ANSWER_RE = re.compile(
    rf"^Answer: \(({_SIGNED_INTEGER}), ({_SIGNED_INTEGER})\)\.$"
)


def format_cartesian_point_payload(point: Sequence[int]) -> str:
    if (
        not isinstance(point, (list, tuple))
        or len(point) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in point
        )
    ):
        raise ValueError("Cartesian point payload must be one exact integer pair")
    x, y = (int(value) for value in point)
    return f"({x}, {y})"


def parse_cartesian_point_answer(text: object) -> Optional[tuple[int, int]]:
    match = (
        STRICT_CARTESIAN_POINT_ANSWER_RE.fullmatch(text)
        if isinstance(text, str)
        else None
    )
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


__all__ = [
    "CARTESIAN_POINT_ANSWER_EXAMPLE",
    "CARTESIAN_POINT_ANSWER_SCHEMA",
    "STRICT_CARTESIAN_POINT_ANSWER_RE",
    "format_cartesian_point_payload",
    "parse_cartesian_point_answer",
]

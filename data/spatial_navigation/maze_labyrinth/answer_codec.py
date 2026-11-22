"""Strict model-facing answer codec for the circular labyrinth.

The labyrinth is a polar topology, not a rectangular grid.  Its formal path
therefore names ``ring`` and ``wedge`` explicitly instead of overloading the
repository-wide grid ``(x, y)`` grammar.  The centre disc is ring 1 and has no
angular sector, represented by JSON ``null``.
"""

from __future__ import annotations

from numbers import Integral
from typing import Mapping, Optional, Sequence

from data.common.typed_answers import (
    canonical_json_dumps,
    decode_canonical_json_answer,
)


PolarPosition = tuple[int, Optional[int]]

POLAR_PATH_ANSWER_EXAMPLE = (
    'Answer: {"path": [{"ring": 98, "wedge": 99}, '
    '{"ring": 97, "wedge": 99}]}.'
)
POLAR_PATH_ANSWER_FORMAT_INSTRUCTION = (
    "Write the complete ordered polar path as one canonical JSON object with "
    'the sole key "path". Each path item has exactly the keys "ring" and '
    '"wedge"; both are 1-based positive integers on outer rings, while the '
    'centre disc is {"ring": 1, "wedge": null}. Repeat an identical item for '
    "one wait. Sort object keys alphabetically and use one space after every "
    f"comma and colon. As a syntax-only out-of-domain example: "
    f"`{POLAR_PATH_ANSWER_EXAMPLE}`"
)


def _normalize_position(value: object) -> Optional[PolarPosition]:
    if isinstance(value, Mapping):
        if set(value) != {"ring", "wedge"}:
            return None
        ring = value.get("ring")
        wedge = value.get("wedge")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        ring, wedge = value
    else:
        return None
    if isinstance(ring, bool) or not isinstance(ring, Integral) or int(ring) < 1:
        return None
    normalized_ring = int(ring)
    if normalized_ring == 1:
        return (1, None) if wedge is None else None
    if isinstance(wedge, bool) or not isinstance(wedge, Integral) or int(wedge) < 1:
        return None
    return normalized_ring, int(wedge)


def format_polar_path_payload(positions: Sequence[object]) -> str:
    if not isinstance(positions, (list, tuple)) or not positions:
        raise ValueError("polar path must contain at least one position")
    normalized = [_normalize_position(position) for position in positions]
    if any(position is None for position in normalized):
        raise ValueError("polar path contains an invalid ring/wedge position")
    normalized_positions = [
        position for position in normalized if position is not None
    ]
    payload = {
        "path": [
            {"ring": ring, "wedge": wedge}
            for ring, wedge in normalized_positions
        ]
    }
    return canonical_json_dumps(payload)


def parse_polar_path_answer(text: object) -> Optional[list[PolarPosition]]:
    payload, valid = decode_canonical_json_answer(text)
    if not valid or not isinstance(payload, dict) or set(payload) != {"path"}:
        return None
    raw_path = payload.get("path")
    if not isinstance(raw_path, list) or not raw_path:
        return None
    positions = [_normalize_position(position) for position in raw_path]
    if any(position is None for position in positions):
        return None
    return [position for position in positions if position is not None]


__all__ = [
    "POLAR_PATH_ANSWER_EXAMPLE",
    "POLAR_PATH_ANSWER_FORMAT_INSTRUCTION",
    "PolarPosition",
    "format_polar_path_payload",
    "parse_polar_path_answer",
]

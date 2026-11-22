from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def matches_schema(value: object, schema: str) -> bool:
    """Match VideoGym schema identifiers by contract and version."""
    if not isinstance(value, str):
        return False
    namespace, _, contract = value.partition(".")
    return namespace == "videogym" and contract == schema.partition(".")[2]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

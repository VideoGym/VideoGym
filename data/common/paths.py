from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def to_absolute(path_value: str | Path, base_dir: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def to_relative(path_value: str | Path, base_dir: Path) -> str:
    target = to_absolute(path_value, base_dir)
    return Path(os.path.relpath(target, base_dir.expanduser().resolve())).as_posix()


def first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

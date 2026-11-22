from __future__ import annotations
import json
from pathlib import Path
from data.common.io import matches_schema
from data.common.paths import to_absolute, to_relative
from evaluation.catalog.index import PROTOCOL


def read_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def rebase_sample_media(sample: dict, *, from_dir: Path, to_dir: Path) -> dict:
    updated = dict(sample)
    for key in ("image", "video"):
        if updated.get(key):
            updated[key] = to_relative(to_absolute(updated[key], from_dir), to_dir)
    return updated


def load_dataset(root):
    root = Path(root)
    metadata = json.loads((root / "suite.json").read_text())
    if not matches_schema(metadata.get("protocol"), PROTOCOL) or metadata.get("status") != "complete":
        raise ValueError("evaluation set is incomplete or uses an obsolete protocol")
    return metadata, read_rows(root / "benchmark_manifest.jsonl")

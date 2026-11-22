"""Public image/text inputs shared by inference engines."""

import json
from pathlib import Path

from data.common.paths import to_absolute


def load_inputs(path, shard_index=0, shards=1, limit=None):
    if shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("shard index must be in [0, shards)")
    manifest_path = Path(path).expanduser().resolve()
    base_dir = manifest_path.parent
    rows, ids = [], set()
    with manifest_path.open() as stream:
        index = 0
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"id", "evaluation_task", "prompt", "image"}:
                raise ValueError(
                    "inference requires the public model_inputs.jsonl schema"
                )
            if (
                not isinstance(row["prompt"], str)
                or not row["prompt"].strip()
                or not isinstance(row["image"], str)
                or not row["image"].strip()
            ):
                raise ValueError("expected one complete public image/text condition")
            identity = row["id"]
            if (
                not identity
                or identity in {".", ".."}
                or Path(identity).name != identity
                or identity in ids
            ):
                raise ValueError(f"invalid or duplicate sample id: {identity}")
            ids.add(identity)
            if index % shards == shard_index:
                rows.append(
                    {
                        **row,
                        "image": to_absolute(row["image"], base_dir).as_posix(),
                    }
                )
            index += 1
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("shard has no public inputs")
    return rows

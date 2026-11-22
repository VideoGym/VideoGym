"""Measure actual training membership and independently replayed parameter ranges."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from evaluation.extrapolation.catalog import extrapolation_tasks
from evaluation.shared.features import FEATURE_EXTRACTORS, replay_features

MEASURED_TASKS = frozenset(case.task_name for case in extrapolation_tasks())
VISUAL_FIELDS = frozenset(
    {
        "canvas_dimensions",
        "canvas_size",
        "grid_bbox",
        "result_grid_bbox",
        "result_cell_size",
        "cell_size",
        "cell_bboxes",
        "padding",
        "start_point",
        "goal_point",
        "program_token_bboxes",
        "allowed_change_bboxes",
        "task_mechanic",
        "schema_version",
        "task_name",
    }
)


def semantic_key(row: dict) -> str:
    source = row["semantic_input"]
    source = source.get("semantic_state", source)

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in VISUAL_FIELDS}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return json.dumps(
        {"task": row["task_name"], "source": clean(source)},
        sort_keys=True,
        separators=(",", ":"),
    )


def observed_features(row: dict) -> dict:
    """Measure actual training exposure by independently replaying input sources."""
    return (
        replay_features(row)
        if row["task_name"] in MEASURED_TASKS or row["task_name"] in FEATURE_EXTRACTORS
        else {}
    )


def profile_training(
    manifest: Path,
    output: Path,
    *,
    members=None,
    all_training_rows=False,
    dataset_name=None,
) -> dict:
    from data.registry import TASK_SPECS

    if sum((members is not None, bool(all_training_rows))) != 1:
        raise ValueError("choose --all-training-rows or --members-file")
    if not all_training_rows and not members:
        raise ValueError("member list is empty")
    members = set(members or ())
    counts = Counter()
    features = defaultdict(lambda: defaultdict(Counter))
    seeds = defaultdict(set)
    matched = set()
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not all_training_rows and row["id"] not in members:
                continue
            if row["id"] in matched:
                raise ValueError("duplicate training sample ID")
            if row["task_name"] not in TASK_SPECS:
                raise ValueError(
                    "unregistered or evaluation-only goal appears in training membership"
                )
            matched.add(row["id"])
            task = row["task_name"]
            counts[task] += 1
            seeds[task].add(str(row["seed"]))
            for key, value in observed_features(row).items():
                features[task][key][str(value)] += 1

    if not matched:
        raise ValueError("no actual training samples matched")
    if not all_training_rows and matched != members:
        raise ValueError(
            f"{len(members - matched)} training IDs absent from source manifest"
        )
    from data.common.paths import to_relative

    profile = {
        "schema": "videogym.video.training-coverage",
        "model_dataset": dataset_name or manifest.stem,
        "source_manifest": to_relative(manifest, output.expanduser().resolve().parent),
        "membership": "all_manifest_rows" if all_training_rows else "explicit_ids",
        "training_samples": len(matched),
        "task_counts": dict(sorted(counts.items())),
        "features": {
            task: {name: dict(values) for name, values in fields.items()}
            for task, fields in features.items()
        },
        "training_sample_ids": sorted(matched),
        "training_seeds": {task: sorted(values) for task, values in seeds.items()},
    }
    from evaluation.composition.catalog import TRAINING_MECHANISMS

    profile["mechanism_sets"] = {
        task: TRAINING_MECHANISMS.get(task, ()) for task in counts
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    return {
        "training_samples": len(matched),
        "training_tasks": len(counts),
        "features": profile["features"],
    }

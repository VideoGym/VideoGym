from __future__ import annotations

import json
from pathlib import Path
import shutil

from data.common.paths import to_absolute, to_relative
from evaluation.catalog.index import catalog_summary, evaluation_tasks
from evaluation.execution.datasets.budgets import allocate_task_budget
from evaluation.execution.datasets.records import (
    load_dataset,
    read_rows,
    rebase_sample_media,
    write_json,
)
from evaluation.execution.datasets.validation import audit_dataset, resolve_training_profile
from evaluation.shared.features import packed


def subset_dataset(dataset, output, count_per_task=5, *, tasks=None, training_profile=None):
    """Select a deterministic subset, retaining original media and pair IDs."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    metadata, rows = load_dataset(dataset)
    profile = resolve_training_profile(dataset, training_profile)
    config = json.loads(json.dumps(metadata["config"]))
    if tasks is not None:
        if not tasks or set(tasks) - metadata["tasks"].keys():
            raise ValueError("selected tasks must be present in the dataset")
        config.setdefault("catalog", {})["tasks"] = list(tasks)
    config["sampling"]["count_per_task"] = count_per_task
    config.pop("budgets", None)
    definitions = evaluation_tasks(config, training_tasks=profile["task_counts"])
    quotas = {
        name: allocate_task_budget(case, count_per_task)
        for name, case in definitions.items()
    }
    selected_ids = set()
    for name, tiers in quotas.items():
        for tier, count in tiers.items():
            candidates = sorted(
                (
                    row
                    for row in rows
                    if row["evaluation_task"] == name and row["difficulty_tier"] == tier
                ),
                key=lambda row: row["sample"]["seed"],
            )
            if len(candidates) < count:
                raise ValueError(f"insufficient samples for {name}/{tier}")
            selected_ids.update(row["sample"]["id"] for row in candidates[:count])
    if output.exists() and any(output.iterdir()):
        raise ValueError("subset output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    subset_meta = {
        **metadata,
        "config": config,
        "count_per_task": count_per_task,
        "budgets": {name: count_per_task for name in definitions},
        "tier_budgets": quotas,
        "tasks": {name: case.to_dict() for name, case in definitions.items()},
        "catalog": catalog_summary(definitions),
        "subset_of": to_relative(dataset, output),
        "selection_rule": "lowest source seeds within each task/tier; task total balanced across declared tiers, remainder to harder tiers",
    }
    write_json(output / "suite.json", subset_meta)
    if (dataset / "training_profile.json").is_file():
        shutil.copyfile(dataset / "training_profile.json", output / "training_profile.json")
    for filename in (
        "benchmark_manifest.jsonl",
        "canonical_manifest.jsonl",
        "model_inputs.jsonl",
    ):
        if not (dataset / filename).is_file():
            continue
        original = read_rows(dataset / filename)

        def identity(row):
            return (
                row["sample"]["id"]
                if filename == "benchmark_manifest.jsonl"
                else row["id"]
            )

        def rebase_row(row):
            if filename == "benchmark_manifest.jsonl":
                return {**row, "sample": rebase_sample_media(row["sample"], from_dir=dataset, to_dir=output)}
            if filename == "canonical_manifest.jsonl":
                return rebase_sample_media(row, from_dir=dataset, to_dir=output)
            if filename == "model_inputs.jsonl" and row.get("image"):
                return {**row, "image": to_relative(to_absolute(row["image"], dataset), output)}
            return row

        (output / filename).write_text(
            "".join(
                packed(rebase_row(row)) + "\n" for row in original if identity(row) in selected_ids
            )
        )
    return audit_dataset(output, training_profile=training_profile)

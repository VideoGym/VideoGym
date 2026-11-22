from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from data.common.io import matches_schema
from evaluation.catalog.index import catalog_summary, evaluation_tasks
from evaluation.execution.datasets.budgets import frozen_tier_budgets
from evaluation.execution.datasets.records import load_dataset, read_rows
from evaluation.execution.datasets.training_exposure import (
    observed_features,
    semantic_key,
)
from evaluation.shared.features import packed


def validate_relation(case, features, profile):
    trained = profile["task_counts"]
    if case.track in ("id", "extrapolation") and case.task_name not in trained:
        raise ValueError("known-task tier contains an untrained task")
    if case.track == "extrapolation":
        values = (
            profile.get("features", {})
            .get(case.task_name, {})
            .get(case.extrapolation_axis)
        )
        if not values:
            raise ValueError(
                f"{case.name}: actual training range for {case.extrapolation_axis} is missing; rebuild training profile"
            )
        value = features[case.extrapolation_axis]
        if case.novelty == "mechanism_extrapolation":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                exposed = min(map(float, values)) <= value <= max(map(float, values))
            else:
                exposed = str(value) in values
        else:
            exposed = value <= max(map(float, values))
        if exposed:
            raise ValueError(
                "extrapolation parameter was already covered by audited training"
            )
    if case.track in ("composition", "cross_task"):
        if case.task_name in trained:
            raise ValueError("evaluation-only task was in the mixed training corpus")
        if case.track == "composition" and not all(name in trained for name in case.primitives):
            raise ValueError("declared primitive is absent from audited training")
        if case.track == "composition":
            from evaluation.composition.catalog import TRAINING_MECHANISMS

            exposure = profile.get("mechanism_sets", {})
            observed = [
                set(TRAINING_MECHANISMS.get(task, exposure.get(task, ())))
                for task in trained
            ]
            union = set(case.mechanisms)
            if len(union) < 2:
                raise ValueError("composition requires two distinct audited mechanisms")
            if not all(
                any(mechanism in mechanisms for mechanisms in observed)
                for mechanism in union
            ):
                raise ValueError("composition mechanism lacks trained source exposure")
            if any(union.issubset(mechanisms) for mechanisms in observed):
                raise ValueError(
                    "composition mechanism union was already exposed in training"
                )


def _validate_profile(profile):
    if not matches_schema(profile.get("schema"), "videogym.video.training-coverage"):
        raise ValueError("unsupported training coverage")
    if not profile.get("task_counts") or any(
        count < 1 for count in profile["task_counts"].values()
    ):
        raise ValueError("training coverage requires actual positive task counts")


def resolve_training_profile(root: Path, training_profile: Path | str | None = None) -> dict:
    local = Path(root) / "training_profile.json"
    if local.is_file():
        path = local
    elif training_profile is not None:
        path = Path(training_profile).expanduser().resolve()
    else:
        raise FileNotFoundError(
            f"missing training_profile.json in {root}; pass --training-profile"
        )
    profile = json.loads(path.read_text())
    _validate_profile(profile)
    return profile


def audit_dataset(root, training_profile=None):
    from data.common.schemas import CanonicalSample

    root = Path(root)
    metadata, rows = load_dataset(root)
    profile = resolve_training_profile(root, training_profile)
    seen = set(profile["training_sample_ids"])
    identities = set()
    counts = Counter()
    tier_counts = Counter()
    definitions = evaluation_tasks(
        metadata["config"], training_tasks=profile["task_counts"]
    )

    def contract(fields):
        return {
            k: v
            for k, v in fields.items()
            if k not in ("title", "description", "novelty", "paired_group")
        }

    for row in rows:
        definition = definitions.get(row["evaluation_task"])
        case = definition.at_tier(row["difficulty_tier"]) if definition else None
        if case is None or packed(contract(definition.to_dict())) != packed(
            contract(metadata["tasks"][case.name])
        ):
            raise ValueError("evaluation task definition differs from current source")
        if row["role"] != case.role:
            raise ValueError("evaluation role mismatch")
        sample = row["sample"]
        canonical = CanonicalSample.from_dict(sample, base_dir=root)
        if canonical.task_name != case.task_name or row["track"] != case.track:
            raise ValueError("evaluation task/track mismatch")
        features = observed_features(sample)
        key = semantic_key(sample)
        if (
            features != row["features"]
            or key != row["scene_group"]
            or not case.accepts(features)
        ):
            raise ValueError(
                "evaluation parameters disagree with independently replayed source"
            )
        validate_relation(case, features, profile)
        if canonical.id in seen or str(sample["seed"]) in profile[
            "training_seeds"
        ].get(case.task_name, []):
            raise ValueError("evaluation sample reused a training instance or seed")
        if canonical.id in identities:
            raise ValueError("duplicate evaluation case")
        if (
            not Path(canonical.primary_input_image).is_file()
            or not Path(canonical.reference_video).is_file()
        ):
            raise ValueError("missing evaluation media")
        identities.add(canonical.id)
        counts[case.name] += 1
        tier_counts[(case.name, row["difficulty_tier"])] += 1
    expected_tiers = frozen_tier_budgets(metadata, definitions)
    if any(
        counts[name] != sum(expected_tiers[name].values()) for name in metadata["tasks"]
    ):
        raise ValueError("missing evaluation samples")
    if any(
        tier_counts[(name, tier["name"])] != expected_tiers[name][tier["name"]]
        for name, case in definitions.items()
        for tier in case.difficulty_tiers()
    ):
        raise ValueError("missing difficulty-tier samples")
    inputs = read_rows(root / "model_inputs.jsonl")
    expected_inputs = [
        {
            "id": row["sample"]["id"],
            "evaluation_task": row["evaluation_task"],
            "prompt": row["sample"]["prompt"],
            "image": row["sample"]["image"],
        }
        for row in rows
    ]
    if inputs != expected_inputs:
        raise ValueError("model inputs differ from source prompts")
    if read_rows(root / "canonical_manifest.jsonl") != [row["sample"] for row in rows]:
        raise ValueError("canonical manifest differs from benchmark manifest")
    return {
        "passed": True,
        "samples": len(rows),
        "evaluation_tasks": len(counts),
        "catalog": catalog_summary(definitions),
        "counts": dict(counts),
        "tracks": dict(Counter(row["track"] for row in rows)),
        "training_dataset": metadata["training_dataset"],
    }

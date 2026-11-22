from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil

from data.common.paths import to_relative
from evaluation.catalog.index import (
    PROTOCOL,
    catalog_summary,
    evaluation_tasks,
    load_config,
)
from evaluation.execution.datasets.budgets import (
    allocate_task_budget,
    frozen_tier_budgets,
)
from evaluation.execution.datasets.records import (
    load_dataset,
    read_rows,
    rebase_sample_media,
    write_json,
)
from evaluation.execution.datasets.training_exposure import (
    observed_features,
    semantic_key,
)
from evaluation.execution.datasets.validation import (
    _validate_profile,
    audit_dataset,
    resolve_training_profile,
    validate_relation,
)
from evaluation.shared.features import packed


def _retain_declared_assets(root, canonical):
    retained = set()
    for value in (canonical.image, canonical.video):
        if value:
            retained.add(Path(value).resolve())
    for asset in Path(root).rglob("*"):
        if asset.is_file() and asset.resolve() not in retained:
            asset.unlink()


def _generate(task, config, seed, root, split="eval", video=True):
    from data.generate import load_generator_class, run_generation
    from data.manifest import build_canonical_sample
    from data.registry import TASK_FAMILIES, get_task_spec

    spec = get_task_spec(task)
    if spec is None:
        raise ValueError(f"Unknown evaluation task {task}")
    from evaluation.catalog.registry import EVAL_TASK_SPECS

    if task in EVAL_TASK_SPECS:
        generator = load_generator_class(spec)(
            output_dir=root, seed=seed, video=video, **config
        )
        record = generator.create_puzzle()
        sample = build_canonical_sample(
            record,
            task_dir=root,
            source_task_name=task,
            seed=seed,
            split=split,
            record_index=0,
            video_requested=video,
        )
        return sample.to_dict(base_dir=root)
    args = argparse.Namespace(
        tasks=[task],
        task_families=list(TASK_FAMILIES),
        split="train" if split == "train" else "eval",
        output_root=str(root),
        count=1,
        num_workers=1,
        seed=seed,
        video=video,
        task_config=json.dumps({task: config}),
    )
    result = run_generation(args)
    return read_rows(result["manifest_path"])[0]


def _sample_generation(task, config, seed, destination, *, video):
    from data.constraint_grid.core.forced_deduction import ForcedDeductionError

    try:
        return _generate(task, config, seed, destination, video=video)
    except ForcedDeductionError:
        return None
    except RuntimeError as error:
        if str(error) in {
            f"{task}: failed to sample scenario.",
            "grid_bfs could not sample the requested distance/source constraints",
            f"{task.removeprefix('eval_')}: native graph sampling exhausted",
        }:
            return None
        raise
    except ValueError as error:
        if "rendered text bounding boxes overlap" in str(error):
            return None
        raise


def _generate_unit(
    case,
    tier_name,
    dataset_seed,
    count,
    max_attempts,
    root,
    profile_path,
    settings,
    retained=(),
):
    """Sample and qualify one task/tier using its deterministic seed namespace."""
    from data.common.schemas import CanonicalSample

    evaluation_tasks(settings)
    profile = (
        json.loads(Path(profile_path).read_text())
        if profile_path
        else {"training_sample_ids": [], "training_seeds": {}}
    )
    seen = set(profile["training_sample_ids"])
    seen.update(row["sample"]["id"] for row in retained)
    excluded_seeds = {str(row["sample"]["seed"]) for row in retained}
    active = case.at_tier(tier_name)
    rng = random.Random(f"{dataset_seed}:{case.name}:{tier_name}")
    rows = []
    scratch = root / ".candidate" / case.name / tier_name
    accepted = 0
    try:
        for _ in range(count * max_attempts):
            seed = rng.randrange(2**62)
            if str(seed) in excluded_seeds:
                continue
            if str(seed) in profile["training_seeds"].get(active.task_name, []):
                continue
            preview = None
            if active.track != "id":
                shutil.rmtree(scratch, ignore_errors=True)
                preview = _sample_generation(
                    active.task_name,
                    active.config,
                    seed,
                    scratch / active.name,
                    video=False,
                )
                if preview is None:
                    continue
                features = observed_features(preview)
                if not active.accepts(features):
                    continue
                if profile_path:
                    validate_relation(active, features, profile)
                if preview["id"] in seen:
                    continue
            destination = (
                root / "assets" / active.name / tier_name / f"{accepted:06d}"
            )
            sample = _sample_generation(
                active.task_name, active.config, seed, destination, video=True
            )
            if sample is None:
                shutil.rmtree(destination, ignore_errors=True)
                continue
            features = observed_features(sample)
            key = semantic_key(sample)
            if preview is not None and (
                semantic_key(preview) != key or observed_features(preview) != features
            ):
                raise ValueError("video generation changed the semantic case")
            if profile_path:
                validate_relation(active, features, profile)
            canonical = CanonicalSample.from_dict(sample, base_dir=destination)
            if canonical.id in seen:
                shutil.rmtree(destination, ignore_errors=True)
                continue
            _retain_declared_assets(destination, canonical)
            rows.append(
                {
                    "evaluation_task": active.name,
                    "difficulty_tier": tier_name,
                    "track": active.track,
                    "role": active.role,
                    "pair_id": None,
                    "features": features,
                    "scene_group": key,
                    "sample": canonical.to_dict(base_dir=root),
                }
            )
            seen.add(canonical.id)
            accepted += 1
            if accepted == count:
                return rows
        raise RuntimeError(
            f"{case.name}/{tier_name}: generated {accepted}/{count}; sampling budget exhausted"
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def generate_dataset(args):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    settings = load_config(getattr(args, "config", None))
    settings = json.loads(json.dumps(settings))
    selection = settings.setdefault("catalog", {})
    for key in ("tasks", "tracks", "families"):
        if getattr(args, key, None):
            selection[key] = getattr(args, key)
    sampling = settings.get("sampling", {})
    count = getattr(args, "count_per_task", None)
    count = sampling.get("count_per_task", 5) if count is None else count
    workers = getattr(args, "workers", None)
    workers = sampling.get("workers", 4) if workers is None else workers
    seed = getattr(args, "seed", None)
    seed = sampling.get("seed", 20261008) if seed is None else seed
    attempts = getattr(args, "max_attempts", None)
    attempts = sampling.get("max_attempts", 200) if attempts is None else attempts
    if min(count, workers, attempts) < 1:
        raise ValueError("counts and workers must be positive")
    settings["sampling"] = {
        **sampling,
        "count_per_task": count,
        "workers": workers,
        "seed": seed,
        "max_attempts": attempts,
    }
    profile_file = Path(args.training_profile).expanduser().resolve()
    profile = json.loads(profile_file.read_text())
    _validate_profile(profile)
    cases = evaluation_tasks(settings, training_tasks=profile["task_counts"])
    budgets = settings.get("budgets", {})
    if budgets.keys() - cases.keys():
        raise ValueError("budget refers to an unselected task")
    for case in cases.values():
        if case.track in ("composition", "cross_task", "id"):
            validate_relation(case, {}, profile)
        else:
            support = (
                profile.get("features", {})
                .get(case.task_name, {})
                .get(case.extrapolation_axis)
            )
            if not support:
                raise ValueError(
                    f"{case.name}: actual training range is missing; rebuild training profile"
                )
            for tier in case.difficulty_tiers():
                active = case.at_tier(tier["name"])
                declared = active.constraints.get(case.extrapolation_axis, ())
                for value in declared:
                    validate_relation(active, {case.extrapolation_axis: value}, profile)
    units, task_budgets, tier_budgets = [], {}, {}
    for case in cases.values():
        tier_names = [t["name"] for t in case.difficulty_tiers()]
        budget = budgets.get(case.name, count)
        if type(budget) is not int or budget < 1:
            raise ValueError("sampling budgets must be positive integers")
        allocated = allocate_task_budget(case, budget)
        task_budgets[case.name] = budget
        tier_budgets[case.name] = allocated.copy()
        units.extend((case, tier, allocated[tier]) for tier in tier_names)
    root = Path(args.output_root).expanduser().resolve()
    inherited = []
    extension = None
    if getattr(args, "extend_from", None):
        parent = Path(args.extend_from).expanduser().resolve()
        if parent == root:
            raise ValueError("extension output must differ from the parent dataset")
        audit_dataset(parent, training_profile=profile_file)
        parent_metadata, raw_inherited = load_dataset(parent)
        inherited = [
            {
                **row,
                "sample": rebase_sample_media(
                    row["sample"], from_dir=parent, to_dir=root
                ),
            }
            for row in raw_inherited
        ]
        if set(parent_metadata["tasks"]) != set(cases):
            raise ValueError("extension must retain the parent's complete task selection")
        if resolve_training_profile(parent, profile_file) != profile:
            raise ValueError("extension training membership differs from the parent")
        retained_counts = Counter(
            (r["evaluation_task"], r["difficulty_tier"]) for r in inherited
        )
        if any(
            retained_counts[name, tier] > budget
            for name, tiers in tier_budgets.items()
            for tier, budget in tiers.items()
        ):
            raise ValueError(
                "extension budget cannot discard existing difficulty-tier questions"
            )
        extension = {
            "dataset": to_relative(parent, root),
            "samples": len(inherited),
        }
    metadata = {
        "protocol": PROTOCOL,
        "training_dataset": profile["model_dataset"],
        "training_samples": profile["training_samples"],
        "training_tasks": len(profile["task_counts"]),
        "seed": seed,
        "count_per_task": count,
        "status": "generating",
        "config": settings,
        "tasks": {name: case.to_dict() for name, case in cases.items()},
        "catalog": catalog_summary(cases),
        "budgets": task_budgets,
        "tier_budgets": tier_budgets,
        "extension": extension,
    }
    if root.exists() and any(root.iterdir()):
        if not getattr(args, "resume", False):
            raise ValueError(
                "output must be empty; use --resume to continue an existing build"
            )
        previous = json.loads((root / "suite.json").read_text())
        if previous.get("extension") != extension:
            raise ValueError("resume build differs in extension parent")
        for key in ("protocol", "seed", "tasks", "budgets", "count_per_task"):
            if packed(previous[key]) != packed(metadata[key]):
                raise ValueError(f"resume build differs in {key}")
        if frozen_tier_budgets(previous, cases) != tier_budgets:
            raise ValueError("resume build differs in difficulty-tier budgets")
        if resolve_training_profile(root, profile_file) != profile:
            raise ValueError("resume training membership differs")
        if previous["status"] == "complete":
            return audit_dataset(root, training_profile=profile_file)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "suite.json", metadata)
    write_json(root / "training_profile.json", profile)
    rows, errors = list(inherited), {}
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), max_tasks_per_child=1
    ) as executor:
        pending = {}
        for case, tier, budget in units:
            retained = [
                r
                for r in inherited
                if r["evaluation_task"] == case.name and r["difficulty_tier"] == tier
            ]
            budget -= len(retained)
            if not budget:
                continue
            shard = root / ".shards" / case.name / (tier + ".json")
            if shard.is_file():
                saved = json.loads(shard.read_text())
                rows.extend(saved["rows"])
                continue
            shutil.rmtree(root / "assets" / case.name / tier, ignore_errors=True)
            pending[
                executor.submit(
                    _generate_unit,
                    case,
                    tier,
                    seed,
                    budget,
                    attempts,
                    root,
                    root / "training_profile.json",
                    settings,
                    retained,
                )
            ] = (case, tier, shard)
        for future in as_completed(pending):
            case, tier, shard = pending[future]
            try:
                values = future.result()
            except Exception as error:
                errors[f"{case.name}/{tier}"] = str(error)
                print(f"{case.name}/{tier}: failed: {error}", flush=True)
                continue
            shard.parent.mkdir(parents=True, exist_ok=True)
            write_json(shard, {"rows": values})
            rows.extend(values)
            print(
                f"{case.name}/{tier}: {len(values)} generated samples",
                flush=True,
            )
    if errors:
        metadata["status"] = "generation_failed"
        metadata["failed_units"] = errors
        write_json(root / "suite.json", metadata)
        raise RuntimeError(
            f"{len(errors)} generation units failed; successful shards retained for --resume: {errors}"
        )
    order = {name: i for i, name in enumerate(cases)}
    rows.sort(
        key=lambda r: (
            order[r["evaluation_task"]],
            r["difficulty_tier"],
            r["sample"]["seed"],
        )
    )
    shutil.rmtree(root / ".candidate", ignore_errors=True)
    (root / "benchmark_manifest.jsonl").write_text(
        "".join(packed(row) + "\n" for row in rows)
    )
    (root / "canonical_manifest.jsonl").write_text(
        "".join(packed(row["sample"]) + "\n" for row in rows)
    )
    (root / "model_inputs.jsonl").write_text(
        "".join(
            packed(
                {
                    "id": row["sample"]["id"],
                    "evaluation_task": row["evaluation_task"],
                    "prompt": row["sample"]["prompt"],
                    "image": row["sample"]["image"],
                }
            )
            + "\n"
            for row in rows
        )
    )
    metadata["status"] = "complete"
    write_json(root / "suite.json", metadata)
    try:
        result = audit_dataset(root, training_profile=profile_file)
        shutil.rmtree(root / ".shards", ignore_errors=True)
        return result
    except Exception:
        metadata["status"] = "failed_audit"
        write_json(root / "suite.json", metadata)
        raise

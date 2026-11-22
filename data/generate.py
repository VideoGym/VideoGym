from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import multiprocessing as mp
import re
import secrets
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from data.common.base import AbstractPuzzleGenerator
from data.manifest import build_canonical_sample, write_manifest
from data.registry import TASK_FAMILIES, TASK_SPECS, TaskSpec, resolve_requested_tasks
from data.common.output_normalization import normalize_task_output_records
from data.common.sample_identity import index_unique_samples
from data.common.io import read_json
from data.common.task_semantics import VALID_SPLITS, normalize_split

def load_generator_class(spec: TaskSpec):
    module = importlib.import_module(spec.module)
    class_name = spec.class_name
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError(f"Task {spec.name!r} must declare an explicit generator class_name")
    try:
        generator_class = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Generator class {class_name!r} not found in module {spec.module!r}"
        ) from exc
    if not isinstance(generator_class, type):
        raise TypeError(
            f"Task {spec.name!r} class_name {class_name!r} in module {spec.module!r} is not a class"
        )
    return generator_class


def _constructor_contract(generator_class: type) -> tuple[set[str], bool]:
    """Return explicit constructor parameters across the generator MRO.

    Some task classes expose ``*args, **kwargs`` only and delegate to a shared
    base constructor.  Looking at the leaf signature alone therefore makes
    orchestration pass unrelated runtime keys through ``**kwargs`` while also
    missing aliases such as ``record_video``.  The MRO-wide contract keeps
    constructor injection explicit; process-level runtime state is applied by
    ``configure_runtime`` after construction.
    """

    explicit: set[str] = set()
    accepts_custom_kwargs = False
    for candidate in generator_class.__mro__:
        initializer = candidate.__dict__.get("__init__")
        if initializer is None:
            continue
        signature = inspect.signature(initializer)
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_custom_kwargs = True
            elif parameter.kind not in {
                inspect.Parameter.VAR_POSITIONAL,
            }:
                explicit.add(name)
    return explicit, accepts_custom_kwargs

def parse_task_config(config_json: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if config_json:
        payload = json.loads(config_json)
        if not isinstance(payload, dict):
            raise ValueError("task_config must be a JSON object")
        return payload
    return {}

def build_default_kwargs(spec: TaskSpec, args: argparse.Namespace, seed: Optional[int]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "seed": seed,
        "video": bool(args.video),
        "video_fps": 16,
    }
    return kwargs


def _requirement_available(requirement: str, args: argparse.Namespace, kwargs: Dict[str, Any]) -> bool:
    return bool(getattr(args, requirement, None))


def _missing_requirements(spec: TaskSpec, args: argparse.Namespace, kwargs: Dict[str, Any]) -> List[str]:
    return [requirement for requirement in spec.requires if not _requirement_available(requirement, args, kwargs)]

def _prefixed_dest_name(worker_name: str, filename: str) -> str:
    return f"{worker_name}__{filename}"


_UUID_ASSET_TOKEN = re.compile(
    r"(?i)(?<![0-9a-f])(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})(?![0-9a-f])"
)


def _stable_asset_component(value: str) -> str:
    """Remove entropy-only UUID tokens from canonical asset paths."""

    return _UUID_ASSET_TOKEN.sub("sample", value)

def _move_asset_value(
    raw_value: Any,
    *,
    worker_dir: Path,
    task_dir: Path,
    worker_name: str,
) -> Any:
    if not isinstance(raw_value, str):
        return raw_value
    raw_path = Path(raw_value)
    src = raw_path
    if not src.is_absolute():
        src = worker_dir / src
    if src.suffix.lower() == ".json":
        raise RuntimeError(
            f"Generated JSON asset paths are not supported: {src}. "
            "Embed structured state in the canonical record instead."
        )
    if not src.exists():
        return raw_value

    if raw_path.is_absolute():
        try:
            relative_asset_path = src.resolve().relative_to(worker_dir.resolve())
        except ValueError:
            relative_asset_path = Path(src.name)
    else:
        try:
            relative_asset_path = src.resolve().relative_to(worker_dir.resolve())
        except ValueError:
            relative_asset_path = Path(src.name)

    stable_parts = [_stable_asset_component(part) for part in relative_asset_path.parts]
    record_suffix = worker_name.removeprefix("record_")
    canonical_record_dir = f"{task_dir.name}-{record_suffix}"
    task_record_pattern = re.compile(rf"^{re.escape(task_dir.name)}-\d{{5}}$")
    for part_index in range(max(0, len(stable_parts) - 1)):
        part = stable_parts[part_index]
        if part == "sample" or task_record_pattern.fullmatch(part):
            stable_parts[part_index] = canonical_record_dir
        elif part.startswith("sample_"):
            stable_parts[part_index] = f"{canonical_record_dir}{part[len('sample') :]}"
    relative_asset_path = Path(*stable_parts)
    rel_subdir = relative_asset_path.parent
    if str(rel_subdir) == ".":
        rel_subdir = Path("")
    dest_dir = task_dir / rel_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_name = _prefixed_dest_name(worker_name, src.name)
    dest_name = _stable_asset_component(dest_name)
    dest = dest_dir / dest_name
    collision_index = 1
    while dest.exists():
        dest = dest.with_name(f"{Path(dest_name).stem}__{collision_index:02d}{Path(dest_name).suffix}")
        collision_index += 1
    shutil.move(src.as_posix(), dest.as_posix())
    return dest.relative_to(task_dir).as_posix()


def _reject_json_asset_paths(value: Any, *, worker_dir: Path, field_path: str) -> None:
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.suffix.lower() != ".json":
            return
        if not candidate.is_absolute():
            candidate = worker_dir / candidate
        if candidate.exists():
            raise RuntimeError(
                f"Generated record field '{field_path}' references JSON asset {candidate}. "
                "Embed structured state in the canonical record instead."
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_json_asset_paths(
                item,
                worker_dir=worker_dir,
                field_path=f"{field_path}.{key}",
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_json_asset_paths(
                item,
                worker_dir=worker_dir,
                field_path=f"{field_path}[{index}]",
            )


def _merge_worker_records(task_dir: Path, worker_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    path_keys = [
        "image",
        "solution_image_path",
        "solution_video_path",
        "video",
        "puzzle",
        "solution",
    ]
    list_path_keys = [
        "images",
        "option_image_paths",
        "ti2v_input_images",
        "video_keyframe_paths",
    ]
    for worker_dir in worker_dirs:
        worker_meta = worker_dir / "data.json"
        if not worker_meta.exists():
            continue
        payload = read_json(worker_meta)
        if not isinstance(payload, list):
            continue
        worker_name = worker_dir.name
        for record in payload:
            if not isinstance(record, dict):
                continue
            record = dict(record)
            _reject_json_asset_paths(record, worker_dir=worker_dir, field_path="record")
            remapped_paths: Dict[str, Any] = {}
            for key in path_keys:
                if key in record:
                    raw_value = record[key]
                    if isinstance(raw_value, str) and raw_value in remapped_paths:
                        record[key] = remapped_paths[raw_value]
                    else:
                        moved_value = _move_asset_value(
                            raw_value,
                            worker_dir=worker_dir,
                            task_dir=task_dir,
                            worker_name=worker_name,
                        )
                        record[key] = moved_value
                        if isinstance(raw_value, str):
                            remapped_paths[raw_value] = moved_value
            for list_key in list_path_keys:
                if list_key not in record or not isinstance(record[list_key], list):
                    continue
                remapped_images: List[Any] = []
                for item in record[list_key]:
                    if isinstance(item, str) and item in remapped_paths:
                        remapped_images.append(remapped_paths[item])
                        continue
                    moved_item = _move_asset_value(
                        item,
                        worker_dir=worker_dir,
                        task_dir=task_dir,
                        worker_name=worker_name,
                    )
                    remapped_images.append(moved_item)
                    if isinstance(item, str):
                        remapped_paths[item] = moved_item
                record[list_key] = remapped_images
            for structured_key in ("structured_answer", "answer_payload"):
                structured = record.get(structured_key)
                if not isinstance(structured, dict):
                    continue
                raw_video_keyframes = structured.get("video_keyframe_paths")
                if not isinstance(raw_video_keyframes, list):
                    continue
                remapped_video_keyframes: List[Any] = []
                for item in raw_video_keyframes:
                    if isinstance(item, str) and item in remapped_paths:
                        remapped_video_keyframes.append(remapped_paths[item])
                        continue
                    moved_item = _move_asset_value(
                        item,
                        worker_dir=worker_dir,
                        task_dir=task_dir,
                        worker_name=worker_name,
                    )
                    remapped_video_keyframes.append(moved_item)
                    if isinstance(item, str):
                        remapped_paths[item] = moved_item
                structured["video_keyframe_paths"] = remapped_video_keyframes
            merged.append(record)
    return merged

def _assign_ordered_record_ids(
    records: Sequence[Dict[str, Any]],
    task_name: str,
) -> List[Dict[str, Any]]:
    """Assign deterministic generator-local ids used only for asset naming.

    Generator-local UUIDs are implementation details used while rendering.
    They must not leak into semantic sample ids or rollout group keys.
    """

    unique_records: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        normalized = dict(record)
        normalized["id"] = f"{task_name}-{index:05d}"
        unique_records.append(normalized)
    return unique_records

def _annotate_task_record(record: Dict[str, Any], spec: TaskSpec, split: str) -> Dict[str, Any]:
    if isinstance(record, dict):
        annotated = dict(record)
    elif hasattr(record, "to_dict"):
        annotated = dict(record.to_dict())
    elif dataclasses.is_dataclass(record):
        annotated = dataclasses.asdict(record)
    else:
        annotated = dict(record)
    # Ensure canonical registry identity is always present before family annotation.
    annotated["task_name"] = spec.name
    annotated["task_family"] = spec.task_family
    if spec.task_family == "board_game":
        from data.board_game.core.glyph_inventory import annotate_board_glyphs

        annotated = annotate_board_glyphs(annotated)
    elif spec.task_family == "constraint_grid":
        # Normalize constraint-grid prompts at the family boundary without
        # altering task semantics.
        from data.constraint_grid.core.glyph_inventory import annotate_constraint_grid_glyphs
        from data.constraint_grid.core.prompt_contract import normalize_constraint_grid_prompts

        annotated = normalize_constraint_grid_prompts(annotated)
        annotated = annotate_constraint_grid_glyphs(annotated)
    elif spec.task_family == "dynamics_game":
        from data.dynamics_game.core.glyph_inventory import annotate_dynamics_glyphs

        annotated = annotate_dynamics_glyphs(annotated)
    elif spec.task_family == "next_state_prediction":
        from data.next_state_prediction.core.glyph_inventory import annotate_next_state_glyphs

        annotated = annotate_next_state_glyphs(annotated, spec.name)
    elif spec.task_family == "physics":
        from data.physics.core.glyph_inventory import annotate_physics_glyphs

        annotated = annotate_physics_glyphs(annotated)
    elif spec.task_family == "rearrangement_game":
        from data.rearrangement_game.core.glyph_inventory import (
            annotate_rearrangement_glyphs,
        )
        from data.rearrangement_game.core.prompt_contract import (
            normalize_rearrangement_prompts,
        )

        annotated = normalize_rearrangement_prompts(annotated)
        annotated = annotate_rearrangement_glyphs(annotated)
    elif spec.task_family == "graph_theory":
        from data.graph_theory.core.glyph_inventory import annotate_graph_glyphs

        annotated = annotate_graph_glyphs(annotated)
    elif spec.task_family == "geometry":
        from data.geometry.core.glyph_inventory import annotate_geometry_glyphs
        from data.geometry.core.rendered_contract import finalize_geometry_sample

        annotated = dict(finalize_geometry_sample(annotated))
        annotated = annotate_geometry_glyphs(annotated)
    annotated["task_name"] = spec.name
    annotated["task_family"] = spec.task_family
    from data.common.prompt_validation import validate_model_prompt
    from data.common.prompts import normalize_model_prompt_typography

    prompt = annotated.get("ti2v_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{spec.name}: missing prompt")
    prompt = normalize_model_prompt_typography(prompt)
    annotated["ti2v_prompt"] = prompt
    validate_model_prompt(prompt, modality="ti2v")
    # Canonical export wraps raw generator answers into the standard Answer line.
    from data.common.typed_answers import format_answer_line
    raw_answer = annotated.get("answer")
    if raw_answer is not None:
        payload = raw_answer if isinstance(raw_answer, str) else json.dumps(raw_answer)
        if payload and not payload.startswith("Answer:"):
            annotated["answer"] = format_answer_line(payload.strip())
    annotated["split"] = normalize_split(split)
    return annotated


def _write_worker_records(
    generator: AbstractPuzzleGenerator,
    records: Sequence[Dict[str, Any]],
    worker_record_path: Path,
) -> None:
    """Persist normalized raw records only for short-lived worker IPC."""

    payload = [generator.record_to_dict(record) for record in records]
    normalized = normalize_task_output_records(generator.output_dir, payload)
    worker_record_path.parent.mkdir(parents=True, exist_ok=True)
    worker_record_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def _generate_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    spec: TaskSpec = job["spec"]
    output_dir: Path = Path(job["output_dir"])
    count: int = int(job["count"])
    kwargs: Dict[str, Any] = dict(job["kwargs"])
    split = normalize_split(job.get("split"), default="train")

    output_dir.mkdir(parents=True, exist_ok=True)
    generator_class = load_generator_class(spec)
    explicit_parameters, accepts_custom_kwargs = _constructor_contract(generator_class)
    constructor_kwargs: Dict[str, Any] = {"output_dir": output_dir}
    video = bool(kwargs.get("video", False))
    if "seed" in explicit_parameters:
        constructor_kwargs["seed"] = kwargs.get("seed")
    elif "random_seed" in explicit_parameters:
        constructor_kwargs["random_seed"] = kwargs.get("seed")
    elif accepts_custom_kwargs:
        # Pass seed during construction for renderers that accept **kwargs and
        # initialize a task-local RNG inside __init__.
        constructor_kwargs["seed"] = kwargs.get("seed")
    if "video" in explicit_parameters:
        constructor_kwargs["video"] = video
    if "generate_videos" in explicit_parameters:
        constructor_kwargs["generate_videos"] = video
    if "record_video" in explicit_parameters:
        constructor_kwargs["record_video"] = video
    if "video_fps" in explicit_parameters:
        constructor_kwargs["video_fps"] = int(kwargs.get("video_fps", 16))
    runtime_keys = {"seed", "video", "video_fps"}
    for key, value in kwargs.items():
        if key in runtime_keys:
            continue
        if key in explicit_parameters or accepts_custom_kwargs:
            constructor_kwargs[key] = value

    generator = generator_class(**constructor_kwargs)
    if not isinstance(generator, AbstractPuzzleGenerator):
        raise TypeError(
            f"Registered generator {generator_class.__module__}.{generator_class.__name__} "
            "must implement AbstractPuzzleGenerator"
        )
    generator.configure_runtime(
        output_dir=output_dir,
        seed=kwargs.get("seed"),
        video=video,
        video_fps=int(kwargs.get("video_fps", 16)),
    )
    records = []
    if spec.task_family == "spatial_navigation":
        from data.common.rendered_glyphs import (
            capture_pil_text,
            rendered_glyph_contract,
            text_bbox_collisions,
        )

        for _ in range(count):
            with capture_pil_text() as glyph_calls:
                raw_record = generator.create_random_puzzle()
            record = generator.record_to_dict(raw_record)
            inventory, glyph_schema = rendered_glyph_contract(glyph_calls)
            collisions = text_bbox_collisions(glyph_calls)
            if collisions:
                raise ValueError(
                    f"{spec.name}: rendered text bounding boxes overlap: {collisions[:5]}"
                )
            record["rendered_glyph_inventory"] = inventory
            record["rendered_glyph_schema"] = glyph_schema
            records.append(record)
    else:
        records = [generator.create_random_puzzle() for _ in range(count)]
    # Rearrangement annotation rewrites the authored state sequence.  It
    # must run after worker assets have been merged, when it can normalize the
    # final canonical paths exactly once. Other families annotate in the worker.
    if spec.task_family != "rearrangement_game":
        records = [_annotate_task_record(record, spec, split) for record in records]
    worker_record_path = output_dir / "data.json"
    _write_worker_records(generator, records, worker_record_path)

    return {
        "record_index": int(job["record_index"]),
        "worker_dir": output_dir.as_posix(),
    }

def run_generation(args: argparse.Namespace) -> Dict[str, Any]:
    requested_tasks = resolve_requested_tasks(
        args.tasks,
        args.task_families,
    )
    if not requested_tasks:
        raise ValueError("No tasks selected. Check --tasks / --task-families")
    if args.count <= 0:
        raise ValueError("--count must be a positive integer")

    split = normalize_split(getattr(args, "split", "train"), default="train")
    output_root = Path(args.output_root).expanduser().resolve()
    families_root = output_root / "families"
    families_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "canonical_manifest.jsonl"

    task_config = parse_task_config(getattr(args, "task_config", None))

    manifest_samples = []
    skipped_tasks: List[Dict[str, Any]] = []

    for task in requested_tasks:
        spec = TASK_SPECS[task]

        requirement_kwargs = build_default_kwargs(spec, args, args.seed)
        requirement_kwargs.update(task_config.get(task, {}))
        missing = _missing_requirements(spec, args, requirement_kwargs)
        if missing:
            skipped_tasks.append(
                {
                    "task": task,
                    "status": "unmet_requirement",
                    "requirements": missing,
                }
            )
            continue

        task_dir = families_root / spec.task_family / "tasks" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        tmp_root = task_dir / ".tmp_workers"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)

        workers = max(1, min(args.num_workers, args.count))
        override_kwargs = dict(task_config.get(task, {}))
        configured_base_seed = override_kwargs.pop("seed", args.seed)
        if configured_base_seed is None:
            # A seedless run should remain stochastic, but every independently
            # scheduled record still needs its own RNG stream.  Draw the seeds
            # in the parent process so worker scheduling cannot affect which
            # record receives which stream, and reject the vanishingly rare
            # duplicate locally rather than relying on probability.
            record_seeds: List[int] = []
            used_seeds: set[int] = set()
            while len(record_seeds) < args.count:
                candidate = secrets.randbits(63)
                if candidate in used_seeds:
                    continue
                used_seeds.add(candidate)
                record_seeds.append(candidate)
        else:
            base_seed = int(configured_base_seed)
            record_seeds = [base_seed + record_index for record_index in range(args.count)]

        jobs: List[Dict[str, Any]] = []
        for record_index, record_seed in enumerate(record_seeds):
            worker_dir = tmp_root / f"record_{record_index:05d}"
            kwargs = build_default_kwargs(spec, args, record_seed)
            if override_kwargs:
                kwargs.update(override_kwargs)
            missing = _missing_requirements(spec, args, kwargs)
            if missing:
                shutil.rmtree(tmp_root, ignore_errors=True)
                raise RuntimeError(
                    f"Task '{task}' is missing required runtime or args: {missing}"
                )
            jobs.append(
                {
                    "spec": spec,
                    "output_dir": worker_dir,
                    "count": 1,
                    "kwargs": kwargs,
                    "split": split,
                    "record_index": record_index,
                }
            )

        try:
            results: List[Dict[str, Any]] = []
            if workers <= 1:
                for job in jobs:
                    results.append(_generate_worker(job))
            else:
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=workers) as pool:
                    for result in pool.imap_unordered(_generate_worker, jobs):
                        results.append(result)

            results.sort(key=lambda item: int(item["record_index"]))
            worker_dirs = [Path(item["worker_dir"]) for item in results]
            merged_records = _assign_ordered_record_ids(_merge_worker_records(task_dir, worker_dirs), task)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        merged_records = normalize_task_output_records(task_dir, merged_records)
        merged_records = [_annotate_task_record(record, spec, split) for record in merged_records]

        task_samples = []
        for record_index, record in enumerate(merged_records):
            sample = build_canonical_sample(
                record,
                task_dir=task_dir,
                source_task_name=task,
                seed=record_seeds[record_index],
                split=split,
                record_index=record_index,
                video_requested=bool(args.video),
            )
            if sample is not None:
                task_samples.append(sample)

        expected_records = args.count
        if len(merged_records) != expected_records:
            raise RuntimeError(
                f"Task '{task}' generated {len(merged_records)} records, expected {expected_records}. "
                "This usually indicates worker failure or premature termination.",
            )
        if len(task_samples) != expected_records:
            raise RuntimeError(
                f"Task '{task}' canonicalized {len(task_samples)} samples, expected {expected_records}. "
                "Check task metadata and generated assets for dropped records.",
            )

        manifest_samples.extend(task_samples)
    index_unique_samples(manifest_samples, context="data generation")
    write_manifest(manifest_path, manifest_samples)

    return {
        "manifest_path": manifest_path,
        "skipped_tasks": skipped_tasks,
    }

def build_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("generate", help="Generate puzzle datasets and canonical manifest")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--tasks", nargs="+", default=["all"], help="Task names or all")
    parser.add_argument(
        "--task-families",
        nargs="+",
        default=list(TASK_FAMILIES),
        choices=list(TASK_FAMILIES),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=list(VALID_SPLITS),
        help="Dataset split tag for generated samples.",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", help="Generate solution videos when supported")
    parser.add_argument("--task-config", type=str, default=None)
    parser.set_defaults(func=_cmd_generate)

def _cmd_generate(args: argparse.Namespace) -> None:
    result = run_generation(args)
    if result.get("skipped_tasks"):
        print(json.dumps({"skipped_tasks": result["skipped_tasks"]}, sort_keys=True))
    print(result["manifest_path"])

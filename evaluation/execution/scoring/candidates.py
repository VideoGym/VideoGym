from __future__ import annotations
from evaluation.execution.scoring.statistics import DIMENSIONS, summarize
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import tempfile
from pathlib import Path
from evaluation.execution.datasets.validation import audit_dataset
from evaluation.execution.datasets.records import load_dataset, write_json
from data.common.verifiers.keyframes import VERIFIER_PROTOCOL
from evaluation.shared.features import packed
from evaluation.catalog.index import evaluation_tasks, PROTOCOL


def _score_one(work, *, prompt_text=None):
    import cv2
    from data.registry import get_task_spec
    from data.common.schemas import CanonicalSample
    from data.common.verifiers.routes import VerifierRoute
    from data.common.verifiers.candidates import resolve_candidate_video

    cv2.setNumThreads(1)
    if len(work) == 4:
        row, prediction_root, reference, dataset_root = work
    else:
        row, prediction_root, reference = work
        dataset_root = None
    sample = CanonicalSample.from_dict(row["sample"], base_dir=dataset_root)
    if prompt_text is not None:
        # Prompt Paraphrase conditions retain the validated canonical source identity.
        sample.prompt = prompt_text
    if reference:
        VerifierRoute._link_candidate(
            Path(sample.reference_video),
            prediction_root / sample.id / "prediction.webm",
        )
    records = get_task_spec(sample.task_name).verifier([sample], prediction_root)
    if len(records) != 1 or records[0].sample_id != sample.id:
        raise RuntimeError("verifier did not return the requested sample")
    record = records[0]
    vector = record.offline_metrics.get("raw_reward_vector", {})
    if any(key not in vector for key in DIMENSIONS):
        raise RuntimeError("verifier lacks the three canonical reward dimensions")
    rewards = {key: float(vector[key]) for key in DIMENSIONS}
    masks = record.offline_metrics.get("reward_availability_mask", {})
    available = {key: bool(masks.get(key, True)) for key in DIMENSIONS}
    if (
        rewards["media_contract"] == 1
        and record.offline_metrics.get("whole_video_loaded") is False
        and record.offline_metrics.get("whole_video_error")
    ):
        raise RuntimeError(
            f"verifier could not evaluate valid media for {sample.id}: {record.offline_metrics['whole_video_error']}"
        )
    candidate = resolve_candidate_video(
        prediction_root, sample.id, sample.split
    )
    record.prediction_asset = (
        str(Path(sample.reference_video).resolve())
        if reference
        else str(candidate.resolve()) if candidate else None
    )
    result = {
        "sample_id": sample.id,
        "task": sample.task_name,
        "evaluation_task": row["evaluation_task"],
        "difficulty_tier": row["difficulty_tier"],
        "track": row["track"],
        "role": row["role"],
        "pair_id": row["pair_id"],
        "features": row["features"],
        "scene_group": row["scene_group"],
        "rewards": rewards,
        "reward_available": available,
        "success": int(
            all(available.values()) and all(value == 1 for value in rewards.values())
        ),
        "verifier": record.to_dict(),
    }
    return result


def score(args):
    from data.registry import get_task_spec

    if not args.reference and not args.model:
        raise ValueError("--model must identify the fixed checkpoint for model scoring")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap-samples cannot be negative")
    workers = getattr(args, "workers", 1)
    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(args.dataset).expanduser().resolve()
    profile_arg = getattr(args, "training_profile", None)
    audit_dataset(root, training_profile=profile_arg)
    suite, rows = load_dataset(root)
    from evaluation.execution.datasets.validation import resolve_training_profile

    profile = resolve_training_profile(root, profile_arg)
    definitions = evaluation_tasks(
        suite["config"], training_tasks=profile["task_counts"]
    )
    requested = set(args.tasks or ())
    if requested - definitions.keys():
        raise ValueError("score selection includes tasks absent from frozen suite")
    selected = [
        r
        for r in rows
        if r["track"] in args.tracks
        and (not requested or r["evaluation_task"] in requested)
        and (
            not getattr(args, "families", None)
            or get_task_spec(r["sample"]["task_name"]).task_family in args.families
        )
    ]
    if not selected:
        raise ValueError("no evaluation rows selected")
    eligible_count = len(selected)
    ids_path = getattr(args, "sample_ids_file", None)
    requested_ids = None
    if ids_path is not None:
        ids_path = Path(ids_path).expanduser().resolve()
        requested_ids = {
            line.strip()
            for line in ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not requested_ids:
            raise ValueError("sample-ids-file must contain at least one sample ID")
        eligible_ids = {row["sample"]["id"] for row in selected}
        unknown = requested_ids - eligible_ids
        if unknown:
            raise ValueError(
                "sample IDs absent from filtered frozen suite: "
                + ", ".join(sorted(unknown)[:10])
            )
        selected = [row for row in selected if row["sample"]["id"] in requested_ids]
    output = Path(args.output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("score output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with (
        (output / "scores.jsonl").open("w") as stream,
        tempfile.TemporaryDirectory(prefix="videogym-reference-") as temporary,
    ):
        prediction_root = (
            Path(temporary)
            if args.reference
            else Path(args.predictions_root).expanduser().resolve()
        )
        work = [(row, prediction_root, args.reference, root) for row in selected]
        with ExitStack() as stack:
            executor = (
                stack.enter_context(
                    ProcessPoolExecutor(
                        max_workers=workers, mp_context=get_context("spawn")
                    )
                )
                if workers > 1
                else None
            )
            decisions = (
                executor.map(_score_one, work) if executor else map(_score_one, work)
            )
            for i, (row, decision) in enumerate(zip(selected, decisions)):
                result = decision
                results.append(result)
                stream.write(packed(result) + "\n")
                stream.flush()
                print(
                    f"{i + 1}/{len(selected)} {row['evaluation_task']}: {result['success']}",
                    flush=True,
                )
    report = {
        "protocol": PROTOCOL,
        "verifier_protocol": VERIFIER_PROTOCOL,
        "kind": "reference_calibration" if args.reference else "model_evaluation",
        "model": None if args.reference else args.model,
        "training_dataset": suite["training_dataset"],
        "training_samples": suite["training_samples"],
        "training_tasks": suite["training_tasks"],
        "sampling_budget": 1,
        "scoring_workers": workers,
        "frozen_samples": len(rows),
        "eligible_samples": eligible_count,
        "scored_samples": len(results),
        "partial": len(results) < len(rows),
        "partial_kind": (
            "sample_ids"
            if len(results) < eligible_count
            else "task_filters" if eligible_count < len(rows) else None
        ),
        "selection": {
            "tracks": list(args.tracks),
            "tasks": sorted(args.tasks or ()),
            "families": sorted(getattr(args, "families", None) or ()),
            "sample_ids_file": str(ids_path) if ids_path is not None else None,
            "sample_ids": sorted(requested_ids) if requested_ids is not None else None,
        },
        "catalog": suite["catalog"],
        **summarize(results, args.bootstrap_samples),
    }
    write_json(output / "report.json", report)
    if args.reference and not all(r["success"] for r in results):
        raise RuntimeError(
            f"reference calibration failed; inspect {output / 'scores.jsonl'}"
        )
    return report

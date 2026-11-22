"""Verifier controls and candidate scoring for Text Prompt Paraphrase."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import json
from multiprocessing import get_context
from pathlib import Path
import tempfile

from data.common.io import matches_schema
from evaluation.execution.datasets.records import read_rows, write_json
from evaluation.execution.scoring.candidates import _score_one
from evaluation.prompt_paraphrase.text_paraphrase.variants import (
    PROTOCOL,
    compare,
    load_package,
)


def _control(payload):
    from PIL import Image
    import numpy as np
    from data.common.paths import to_absolute
    from data.common.video_encoding import encode_rgb_frames_to_webm
    from evaluation.execution.scoring.positive_reference import qualifying_reference

    row, text, dataset_root = payload
    with tempfile.TemporaryDirectory(prefix="videogym-text-control-") as temporary:
        root = Path(temporary)
        sample = row["sample"]
        positive_sample = qualifying_reference(sample, root, base_dir=dataset_root)
        positive_row = {**row, "sample": positive_sample}
        positive = _score_one((positive_row, root / "positive", True, dataset_root), prompt_text=text)
        with Image.open(to_absolute(sample["image"], dataset_root)) as opened:
            frame = np.array(opened.convert("RGB"))
        destination = root / "negative" / sample["id"] / "prediction.webm"
        encode_rgb_frames_to_webm([frame] * sample["video_num_frames"], destination, fps=sample["video_fps"])
        negative = _score_one((row, root / "negative", False, dataset_root), prompt_text=text)
        return {
            "sample_id": sample["id"],
            "evaluation_task": row["evaluation_task"],
            "positive_success": positive["success"],
            "unchanged_input_success": negative["success"],
            "positive_rewards": positive["rewards"],
            "unchanged_input_rewards": negative["rewards"],
        }


def _write_results(output, filename, work, function, workers):
    output = Path(output)
    if workers < 1:
        raise ValueError("workers must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("score output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with ExitStack() as stack:
        executor = (
            stack.enter_context(
                ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
            )
            if workers > 1
            else None
        )
        decisions = executor.map(function, work) if executor else map(function, work)
        stream = stack.enter_context((output / filename).open("w"))
        for result in decisions:
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{len(results)}/{len(work)} {result['sample_id']}", flush=True)
    return results


def calibrate(dataset, output, workers=1):
    dataset_root = Path(dataset).expanduser().resolve()
    metadata, rows = load_package(dataset_root)
    public = read_rows(dataset_root / "model_inputs.jsonl")
    work = [(row, condition["prompt"], dataset_root) for row, condition in zip(rows, public)]
    results = _write_results(output, "controls.jsonl", work, _control, workers)
    complete = all(r["positive_success"] == 1 and r["unchanged_input_success"] == 0 for r in results)
    report = {
        "protocol": PROTOCOL,
        "dataset": str(dataset_root),
        "samples": metadata["samples"],
        "positive_passes": sum(r["positive_success"] for r in results),
        "unchanged_input_rejections": sum(1 - r["unchanged_input_success"] for r in results),
        "complete": complete,
    }
    write_json(Path(output) / "qualification.json", report)
    if not complete:
        raise RuntimeError("verifier controls failed; inspect controls.jsonl")
    return report


def _score_text(payload):
    row, predictions, text, dataset_root = payload
    return _score_one((row, predictions, False, dataset_root), prompt_text=text)


def score(dataset, output, predictions, model, workers=1, shard_index=0, shards=1):
    dataset_root = Path(dataset).expanduser().resolve()
    metadata, rows = load_package(dataset_root)
    if shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("invalid scoring shard")
    rows = rows[shard_index::shards]
    if not rows:
        raise ValueError("empty scoring shard")
    predictions = Path(predictions).resolve()
    from data.common.verifiers.candidates import resolve_candidate_video

    for row in rows:
        sample = row["sample"]
        if not resolve_candidate_video(predictions, sample["id"], sample["split"]):
            raise ValueError(f"missing candidate: {sample['id']}")
    public = {r["id"]: r["prompt"] for r in read_rows(dataset_root / "model_inputs.jsonl")}
    results = _write_results(
        output,
        "scores.jsonl",
        [(r, predictions, public[r["sample"]["id"]], dataset_root) for r in rows],
        _score_text,
        workers,
    )
    report = {
        "protocol": PROTOCOL,
        "evaluation_protocol": "reference_length",
        "model": model,
        "dataset": str(dataset_root),
        "predictions": str(predictions),
        "samples": len(results),
        "frozen_samples": metadata["samples"],
        "shard_index": shard_index,
        "shards": shards,
        "correct": sum(r["success"] for r in results),
        "complete": True,
    }
    write_json(Path(output) / "report.json", report)
    return report


def _score_baseline(payload):
    from data.common.verifiers.routes import VerifierRoute

    row, candidate, dataset_root = payload
    with tempfile.TemporaryDirectory(prefix="videogym-id-rescore-") as directory:
        root = Path(directory)
        suffix = Path(candidate).suffix or ".webm"
        VerifierRoute._link_candidate(Path(candidate), root / row["sample"]["id"] / f"prediction{suffix}")
        result = _score_one((row, root, False, dataset_root))
        result["verifier"]["prediction_asset"] = candidate
        return result


def _resolve_baseline_candidates(rows, *, dataset_root, predictions_root=None, baseline_scores=None):
    if (predictions_root is None) == (baseline_scores is None):
        raise ValueError("pass exactly one of predictions_root or baseline_scores")
    work = []
    if predictions_root is not None:
        from data.common.verifiers.candidates import resolve_candidate_video

        predictions = Path(predictions_root).resolve()
        for row in rows:
            sample = row["sample"]
            candidate = resolve_candidate_video(
                predictions, sample["id"], sample["split"]
            )
            if candidate is None:
                raise ValueError(f"missing ID candidate: {sample['id']}")
            work.append((row, str(candidate.resolve()), dataset_root))
        return work, str(predictions)
    scores_path = Path(baseline_scores).resolve()
    archived = {
        r["sample_id"]: r
        for r in read_rows(scores_path)
        if r.get("track", "id") == "id"
    }
    if set(archived) != {r["sample"]["id"] for r in rows}:
        raise ValueError("baseline scores do not cover the ID sample set")
    for row in rows:
        previous = archived[row["sample"]["id"]]
        if previous["evaluation_task"] != row["evaluation_task"]:
            raise ValueError("baseline task identity differs")
        candidate = previous.get("verifier", {}).get("prediction_asset")
        if not candidate or not Path(candidate).is_file():
            raise ValueError(f"missing baseline ID candidate: {candidate}")
        work.append((row, str(Path(candidate).resolve()), dataset_root))
    return work, str(scores_path)


def rescore_baseline(
    dataset,
    output,
    model,
    workers=1,
    *,
    predictions_root=None,
    baseline_scores=None,
):
    """Score In-Domain videos with the same verifier used on the text variant."""
    dataset_root = Path(dataset).expanduser().resolve()
    metadata, rows = load_package(dataset_root)
    work, source = _resolve_baseline_candidates(
        rows, dataset_root=dataset_root, predictions_root=predictions_root, baseline_scores=baseline_scores
    )
    results = _write_results(output, "scores.jsonl", work, _score_baseline, workers)
    report = {
        "protocol": PROTOCOL,
        "evaluation_protocol": "reference_length",
        "model": model,
        "dataset": str(Path(dataset).resolve()),
        "samples": metadata["samples"],
        "scored_samples": len(results),
        "complete": True,
        "correct": sum(r["success"] for r in results),
        "candidate_source": source,
        "kind": "id_baseline_rescore",
    }
    write_json(Path(output) / "report.json", report)
    return report


def merge(dataset, shard_roots, output, id_scores):
    metadata, rows = load_package(dataset)
    reports = [json.loads((Path(root) / "report.json").read_text()) for root in shard_roots]
    if not reports:
        raise ValueError("no scoring shards")
    first = reports[0]
    for report in reports:
        if report.get("complete") is not True or any(
            report[key] != first[key]
            for key in ("protocol", "evaluation_protocol", "model", "dataset", "predictions", "shards")
        ):
            raise ValueError("incomplete or incompatible scoring shards")
    if (
        first["dataset"] != str(Path(dataset).resolve())
        or not matches_schema(first["protocol"], PROTOCOL)
        or first["evaluation_protocol"] != "reference_length"
    ):
        raise ValueError("scoring shards belong to a different package")
    if len(reports) != first["shards"] or {r["shard_index"] for r in reports} != set(range(first["shards"])):
        raise ValueError("missing or duplicate scoring shard")
    baseline = json.loads(Path(id_scores).with_name("report.json").read_text())
    if (
        baseline.get("kind") != "id_baseline_rescore"
        or baseline.get("complete") is not True
        or any(baseline.get(key) != first[key] for key in ("model", "evaluation_protocol", "dataset"))
    ):
        raise ValueError("baseline must match model, protocol, and dataset")
    scores = [r for root in shard_roots for r in read_rows(Path(root) / "scores.jsonl")]
    by_id = {r["sample_id"]: r for r in scores}
    if len(scores) != len(by_id) or set(by_id) != {r["sample"]["id"] for r in rows}:
        raise ValueError("scores do not cover the ID set exactly once")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("merge output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "scores.jsonl").write_text("".join(json.dumps(by_id[r["sample"]["id"]]) + "\n" for r in rows))
    comparison = compare(id_scores, output / "scores.jsonl")
    write_json(output / "comparison.json", comparison)
    report = {
        **first,
        "samples": metadata["samples"],
        "scored_samples": metadata["samples"],
        "correct": comparison["overall"]["variant_correct"],
        "overall": {"complete": True},
        "baseline_scores": str(Path(id_scores).resolve()),
    }
    report.pop("shard_index", None)
    write_json(output / "report.json", report)
    return report

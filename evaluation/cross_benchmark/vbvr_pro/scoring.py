from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

from data.common.paths import to_absolute
from evaluation.cross_benchmark.vbvr_pro.benchmark.catalog import enumerate_samples
from evaluation.cross_benchmark.vbvr_pro.reports import (
    summarize_vbvr_pro,
    write_json,
    write_jsonl,
)
from inference.data.inputs import load_inputs


def resolve_vbvr_candidate(
    predictions: Path, sample_id: str, input_id: str | None = None
) -> Path:
    predictions = Path(predictions)
    slug = input_id or ("vbvr-" + sample_id.replace("/", "--"))
    candidates = [
        predictions / slug / "prediction.webm",
        predictions / sample_id / "prediction.webm",
        predictions / f"{sample_id}.mp4",
    ]
    found = [path for path in candidates if path.is_file()]
    if len(found) > 1:
        raise ValueError(
            f"ambiguous candidate videos for {sample_id}: {[str(p) for p in found]}"
        )
    if not found:
        raise FileNotFoundError(f"missing candidate video for {sample_id}")
    return found[0]


def resolve_vbvr_work(rows: list[dict], predictions: Path) -> list[tuple[dict, Path]]:
    seen = set()
    work = []
    for entry in rows:
        sample = entry["sample"] if "sample" in entry else entry
        sample_id = sample["sample_id"]
        if sample_id in seen:
            raise ValueError(f"duplicate VBVR-Pro sample identity: {sample_id}")
        seen.add(sample_id)
        input_id = entry.get("id") if "sample" in entry else None
        path = resolve_vbvr_candidate(predictions, sample_id, input_id)
        work.append((sample, path))
    return work


def _score_vbvr_pro_one(work):
    import cv2
    from evaluation.cross_benchmark.vbvr_pro.verification.engine import evaluate_sample

    sample, path = work
    cv2.setNumThreads(1)
    decision = evaluate_sample(sample, path)
    decision["prediction"] = str(Path(path).resolve())
    return decision


def score_vbvr_pro(dataset, predictions, output, workers):
    dataset, predictions, output = Path(dataset), Path(predictions), Path(output)
    _PATH_FIELDS = ("first_frame", "final_frame", "reference_video", "metadata", "prompt")
    map_path = dataset / "benchmark-map.jsonl"
    if map_path.is_file():
        raw_rows = [
            json.loads(line)
            for line in map_path.read_text().splitlines()
            if line.strip()
        ]
        rows = [
            {
                **entry,
                "sample": {
                    **entry["sample"],
                    **{
                        field: str(to_absolute(entry["sample"][field], dataset))
                        for field in _PATH_FIELDS
                        if field in entry["sample"]
                    },
                },
            }
            for entry in raw_rows
        ]
        inputs = load_inputs(dataset / "model_inputs.jsonl")
        if len(rows) != 500 or {r["id"] for r in rows} != {r["id"] for r in inputs}:
            raise ValueError(
                "cross-benchmark scoring requires every VBVR-Pro sample"
            )
    else:
        rows = [
            {"id": "vbvr-" + row["sample_id"].replace("/", "--"), "sample": row}
            for row in enumerate_samples(dataset)
        ]
    work = resolve_vbvr_work(rows, predictions)
    if output.exists() and any(output.iterdir()):
        raise ValueError("scoring output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as pool:

        def decisions():
            for index, result in enumerate(pool.map(_score_vbvr_pro_one, work), 1):
                results.append(result)
                yield result
                print(
                    f"{index}/{len(work)} {result['sample_id']}: {result['status']}",
                    flush=True,
                )

        write_jsonl(output / "scores.jsonl", decisions())
    summary = summarize_vbvr_pro(results)
    write_json(output / "report.json", summary)
    if not summary["overall"]["complete"]:
        raise RuntimeError(
            "scoring has unresolved or missing samples; not a complete result"
        )
    return summary["overall"]

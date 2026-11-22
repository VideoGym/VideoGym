"""Run the upstream VBVR-Pro continuous partial-credit evaluator on candidate videos."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys

from evaluation.cross_benchmark.vbvr_pro.benchmark.catalog import resolve_gt_root


def evaluate(item):
    row, upstream, gt_root = item
    sys.path.insert(0, upstream)
    import cv2

    cv2.setNumThreads(1)
    from run_evaluation_video import evaluate_single_video, find_gt_info

    prediction = Path(row["prediction"])
    if not prediction.is_file():
        raise FileNotFoundError(f"Missing candidate video: {prediction}")
    folder, task, instance = row["sample_id"].split("/")
    result = evaluate_single_video(
        str(prediction), task, find_gt_info(task, int(instance), gt_root), "cpu"
    )
    return {
        "folder": folder,
        "task_name": task,
        "video_file": f"{instance}{prediction.suffix}",
        "score": float(result.get("score", 0)),
        "error": result.get("error"),
        "dimensions": result.get("dimensions", {}),
    }


def run_continuous(
    strict: Path, upstream: Path, gt_root: Path, output: Path, workers: int = 1
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(upstream.resolve()))
    from run_evaluation_video import NumpyEncoder

    rows = [json.loads(line) for line in strict.read_text().splitlines() if line.strip()]
    resolved_gt_root = str(resolve_gt_root(gt_root))
    items = [(row, str(upstream.resolve()), resolved_gt_root) for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.with_suffix(".jsonl")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        with progress.open("w") as stream:
            samples = []
            for result in pool.map(evaluate, items):
                samples.append(result)
                stream.write(
                    json.dumps(result, allow_nan=False, cls=NumpyEncoder) + "\n"
                )
                stream.flush()
                print(f"{len(samples)}/{len(rows)}", flush=True)
    output.write_text(
        json.dumps(
            {
                "python": sys.version,
                "device": "cpu",
                "workers": workers,
                "samples": samples,
            },
            indent=2,
            cls=NumpyEncoder,
        )
        + "\n"
    )
    progress.unlink()
    if any(row["error"] for row in samples):
        raise RuntimeError(
            "Upstream execution errors; inspect results before comparing scores"
        )

"""Replay all VBVR-Pro Bench reference videos and static-input negatives at 512px."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from evaluation.cross_benchmark.vbvr_pro.benchmark.catalog import (
    KNOWN_INCORRECT_REFERENCES,
    enumerate_samples,
)
from evaluation.cross_benchmark.vbvr_pro.media import video as media
from evaluation.cross_benchmark.vbvr_pro.reports import _json_value, write_json
from evaluation.cross_benchmark.vbvr_pro.verification.engine import verify_frames


def _control(row):
    cv2.setNumThreads(1)
    first, final = media.image(row["first_frame"]), media.image(row["final_frame"])
    reference = media.decode(row["reference_video"])
    metadata = json.loads(Path(row["metadata"]).read_text())
    size = (512, round(first.shape[0] * 512 / first.shape[1]))
    with TemporaryDirectory(prefix="vbvr-control-") as temporary:
        path = Path(temporary) / "reference.mp4"
        media.encode_h264(
            [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in reference.frames],
            path,
            reference.fps,
        )
        observed = media.decode(path).frames
    replay = verify_frames(
        row["task"],
        observed,
        first,
        final,
        metadata,
        reference.frames,
        observation_fps=reference.fps,
    )
    still = verify_frames(
        row["task"],
        [observed[0]] * len(observed),
        first,
        final,
        metadata,
        reference.frames,
        observation_fps=reference.fps,
    )
    return {"sample_id": row["sample_id"], "reference": replay, "unchanged_input": still}


def run_controls(gt_root: Path, output: Path, workers: int = 1) -> int:
    if workers < 1:
        raise ValueError("workers must be positive")
    rows = enumerate_samples(gt_root)
    output.mkdir(parents=True, exist_ok=False)
    controls = []
    with (
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool,
        (output / "controls.jsonl").open("w") as stream,
    ):
        futures = [pool.submit(_control, row) for row in rows]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            expected = (
                "incorrect"
                if row["sample_id"] in KNOWN_INCORRECT_REFERENCES
                else "correct"
            )
            row["expected_reference_status"] = expected
            row["passed"] = (
                row["reference"]["status"] == expected
                and row["unchanged_input"]["status"] == "incorrect"
            )
            controls.append(row)
            stream.write(json.dumps(row, allow_nan=False, default=_json_value) + "\n")
            stream.flush()
            print(f"{i}/{len(rows)} {row['sample_id']}: {row['passed']}", flush=True)
    summary = {
        "samples": len(controls),
        "expected_positive_references": sum(
            r["expected_reference_status"] == "correct" for r in controls
        ),
        "reference_label_matches": sum(
            r["reference"]["status"] == r["expected_reference_status"]
            for r in controls
        ),
        "unchanged_input_rejections": sum(
            r["unchanged_input"]["status"] == "incorrect" for r in controls
        ),
        "failures": [r["sample_id"] for r in controls if not r["passed"]],
        "complete": all(r["passed"] for r in controls),
    }
    write_json(output / "report.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["complete"] else 2

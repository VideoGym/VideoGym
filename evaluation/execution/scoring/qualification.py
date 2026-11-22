"""Positive and negative counterfactual video controls for task verifiers."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path
import tempfile


def qualify(row, controls=None, frozen_reference=False, base_dir=None):
    import cv2
    import numpy as np
    from data.common.paths import to_absolute
    from data.common.schemas import CanonicalSample
    from data.common.video_encoding import encode_rgb_frames_to_webm
    from data.registry import get_task_spec
    from evaluation.shared.task_io import read_video

    cv2.setNumThreads(1)
    sample = CanonicalSample.from_dict(row, base_dir=base_dir)
    result = {"id": sample.id, "task": sample.task_name, "controls": {}}
    try:
        from evaluation.execution.scoring.positive_reference import qualifying_reference

        with tempfile.TemporaryDirectory(prefix="videogym-reference-") as replay_directory:
            positive = (
                row
                if frozen_reference
                else qualifying_reference(row, Path(replay_directory), base_dir=base_dir)
            )
            video_path = (
                to_absolute(positive["video"], base_dir)
                if base_dir is not None
                else Path(positive["video"])
            )
            reference = read_video(video_path)
        middle = len(reference) // 2
        clips = {
            "reference": reference,
            "unchanged_input": [reference[0]] * len(reference),
            "reverse": reference[::-1],
            "terminal_cut": [reference[0]] * middle
            + [reference[-1]] * (len(reference) - middle),
            "erased_scene": [reference[0]] * 5
            + [np.full_like(reference[0], 255)] * (len(reference) - 5),
        }
        corruption = [frame.copy() for frame in reference]
        corruption[middle][8:48, 8:48] = (255, 0, 255)
        clips["transient_foreign_object"] = corruption
        if sample.task_name.startswith("eval_"):
            from evaluation.shared.scenes.catalog import TASKS

            source = sample.semantic_input.get("semantic_state", sample.semantic_input)
            task = TASKS.get(source.get("key"))
            if task is not None and task.track == "composition":
                from evaluation.shared.scenes.ablation import evidence
                from evaluation.shared.scenes.media import negatives, positives

                result["ablation"] = evidence(source)
                if not result["ablation"]["passed"]:
                    raise ValueError("composition fails same-input nondegeneracy")
                clips = {
                    "reference": reference,
                    "erased_scene": clips["erased_scene"],
                    **dict(negatives(source, reference)),
                    **dict(positives(source)),
                }
        if controls is not None:
            unknown = set(controls) - clips.keys()
            if unknown:
                raise ValueError(
                    f"controls unavailable for {sample.task_name}: {sorted(unknown)}"
                )
            clips = {name: clips[name] for name in controls}
        with tempfile.TemporaryDirectory(prefix="videogym-controls-") as directory:
            root = Path(directory)
            output = root / sample.id / "prediction.webm"
            output.parent.mkdir()
            for name, frames in clips.items():
                source = sample.semantic_input.get(
                    "semantic_state", sample.semantic_input
                )
                single_commit = sample.task_name == "grid_component" or (
                    sample.task_family == "next_state_prediction"
                    and source.get("params", {}).get("steps") == 1
                )
                if sample.task_family == "geometry":
                    from data.geometry.core.construction_paths import (
                        NO_CONSTRUCTION_TASKS,
                    )

                    single_commit = (
                        sample.task_name == "midpoint"
                        or sample.task_name in NO_CONSTRUCTION_TASKS
                    )
                if sample.task_name.startswith("eval_cross_"):
                    from evaluation.cross_task.engine import (
                        semantic_mark_groups,
                        solve,
                    )

                    single_commit = (
                        len(semantic_mark_groups(source, solve(source)[0])) == 1
                    )
                expected = (
                    name == "reference"
                    or name.startswith("valid_")
                    or (name == "terminal_cut" and single_commit)
                )
                if not encode_rgb_frames_to_webm(frames, output, fps=16):
                    raise RuntimeError("control encoding failed")
                record = get_task_spec(sample.task_name).verifier(
                    [sample], root
                )[0]
                observed = bool(record.offline_pass)
                result["controls"][name] = {
                    "accepted": observed,
                    "expected": expected,
                    "passed": observed == expected,
                    "error": record.error,
                    **({"metrics": record.offline_metrics} if observed != expected else {}),
                }
        result["passed"] = all(v["passed"] for v in result["controls"].values())
    except Exception:
        import traceback

        result.update(passed=False, error=traceback.format_exc())
    return result


def run(manifest, output, workers, *, controls=None, frozen_reference=False):
    manifest_path = Path(manifest).expanduser().resolve()
    base_dir = manifest_path.parent
    rows = [
        json.loads(line)
        for line in manifest_path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("qualification requires nonempty input")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with (
        output.open("w") as stream,
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool,
    ):
        for future in as_completed(
            [pool.submit(qualify, row, controls, frozen_reference, base_dir) for row in rows]
        ):
            result = future.result()
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"{'PASS' if result['passed'] else 'FAIL'} {result['task']} {len(results)}/{len(rows)}",
                flush=True,
            )
    summary = {
        "samples": len(results),
        "passed": sum(r["passed"] for r in results),
        "failed_tasks": sorted({r["task"] for r in results if not r["passed"]}),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--frozen-reference",
        action="store_true",
        help="Use existing reference media directly.",
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=(
            "reference",
            "unchanged_input",
            "reverse",
            "terminal_cut",
            "erased_scene",
            "transient_foreign_object",
        ),
        help="Select controls to evaluate; defaults to all controls.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    summary = run(
        args.manifest,
        args.output,
        args.workers,
        controls=args.controls,
        frozen_reference=args.frozen_reference,
    )
    print(json.dumps(summary), flush=True)
    raise SystemExit(bool(summary["failed_tasks"]))

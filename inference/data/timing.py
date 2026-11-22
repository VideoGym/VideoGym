"""Timing-only duration sidecars for Reference Length and Fixed Length evaluation."""

import argparse
import json
from pathlib import Path

from inference.data.inputs import load_inputs


def load_timing(path, inputs):
    expected = {row["id"] for row in load_inputs(inputs)}
    timings = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != {"id", "frames", "fps"}:
            raise ValueError("reference timing sidecar accepts only id, frames, fps")
        if row["id"] not in expected or row["id"] in timings:
            raise ValueError("unknown or duplicate reference timing ID")
        if type(row["frames"]) is not int or row["frames"] < 1 or row["fps"] != 16:
            raise ValueError("timing requires positive integer frames at 16 FPS")
        timings[row["id"]] = row
    if timings.keys() != expected:
        raise ValueError("reference timing must cover every public input exactly once")
    return timings


def resolve_timing(inputs, *, frames=None, reference_timing=None):
    if reference_timing is not None:
        if frames is not None:
            raise ValueError("choose fixed frames or reference timing, never both")
        return "reference_length", load_timing(reference_timing, inputs)
    if type(frames) is not int or frames < 5 or (frames - 1) % 4:
        raise ValueError("fixed timing requires at least 5 frames in 4N+1 form")
    return "fixed_length", {
        row["id"]: {"id": row["id"], "frames": frames, "fps": 16}
        for row in load_inputs(inputs)
    }


def wan_native_frames(delivery_frames):
    return max(5, 1 + 4 * ((delivery_frames - 1 + 3) // 4))


def export_timing(dataset, output):
    dataset, output = Path(dataset), Path(output)
    inputs = load_inputs(dataset / "model_inputs.jsonl")
    timings = {}
    for line in (dataset / "benchmark_manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)["sample"]
        identity = sample["id"]
        if identity in timings:
            raise ValueError("duplicate frozen sample ID")
        frames, fps = sample["video_num_frames"], sample["video_fps"]
        if type(frames) is not int or frames < 1 or fps != 16:
            raise ValueError(
                "reference timing requires positive integer frames at 16 FPS"
            )
        timings[identity] = {"id": identity, "frames": frames, "fps": int(fps)}
    if timings.keys() != {row["id"] for row in inputs}:
        raise ValueError("frozen timing and public inputs differ")
    content = "".join(json.dumps(timings[row["id"]]) + "\n" for row in inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text() != content:
        raise ValueError("refusing to replace a different reference timing sidecar")
    output.write_text(content)
    return {
        "samples": len(timings),
        "evaluation_protocol": "reference_length",
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export_timing(args.dataset, args.output)))


if __name__ == "__main__":
    main()

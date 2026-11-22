"""Generate pass@1 rollouts with native LTX-2.3 or VideoGym-LTX-2.3 weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.backends.ltx.conditioning import encode_prompt_contexts
from inference.backends.ltx.delivery import plan_deliveries
from inference.backends.ltx.sampler import NativeGenerator
from inference.data.inputs import load_inputs
from inference.data.outputs import pending_inputs, valid_video
from inference.data.timing import resolve_timing


def run(args):
    import torch
    from data.common.video_encoding import encode_rgb_frames_to_webm

    _, timings = resolve_timing(
        args.inputs, frames=args.frames, reference_timing=args.reference_timing
    )
    deliveries = plan_deliveries(timings)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise ValueError("native LTX generation requires CUDA")
    device = torch.device(args.device)
    rows = load_inputs(args.inputs, args.shard_index, args.shards, args.limit)
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pending = pending_inputs(output, rows, timings)
    if pending:
        local_files_only = bool(args.local_files_only)
        contexts = encode_prompt_contexts(
            str(args.model_path), pending, device, local_files_only=local_files_only
        )
        engine = NativeGenerator(
            str(args.model_path),
            args.checkpoint,
            contexts,
            device,
            local_files_only=local_files_only,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, row in enumerate(pending):
                destination = output / row["id"] / "prediction.webm"
                destination.parent.mkdir(parents=True, exist_ok=True)
                delivery = deliveries[row["id"]]
                frames = engine.generate(row, args.seed, args.steps, delivery)
                temporary = destination.with_name("prediction.partial.webm")
                encode_rgb_frames_to_webm(
                    list(frames),
                    temporary,
                    fps=delivery.fps,
                    extra_args=["-threads", "1"],
                )
                if not valid_video(temporary, delivery.frames):
                    raise RuntimeError("encoded candidate did not decode completely")
                temporary.replace(destination)
                print(
                    json.dumps(
                        {
                            "phase": "inference",
                            "id": row["id"],
                            "completed": index + 1,
                            "total": len(pending),
                            "prediction": str(destination),
                        }
                    ),
                    flush=True,
                )
    return {
        "status": "complete",
        "shard_index": args.shard_index,
        "shards": args.shards,
        "samples": len(rows),
        "new_samples": len(pending),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    weights = parser.add_mutually_exclusive_group(required=False)
    weights.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional transformer safetensors file or Diffusers model/transformer directory",
    )
    weights.add_argument(
        "--base",
        action="store_true",
        help="Load the transformer bundled inside --model-path",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument(
        "--frames", type=int, help="Fixed Length frame count at 16 FPS (e.g. 81)"
    )
    timing.add_argument(
        "--reference-timing",
        type=Path,
        help="Reference Length evaluation: timing-only JSONL sidecar",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    run(args)


if __name__ == "__main__":
    main()

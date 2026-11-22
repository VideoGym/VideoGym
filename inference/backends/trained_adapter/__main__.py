"""Generate pass@1 rollouts with a checkpoint exported by a Diffusers training adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.configuration.schema import MODELS, load_config


def _config(args):
    overrides = {
        "model": args.model,
        "pretrained_model_name_or_path": args.model_path,
        "warm_start": args.checkpoint,
        "tuning": args.tuning,
        "sharding": "none",
        "device": args.device,
    }
    if args.local_files_only:
        overrides["local_files_only"] = True
    return load_config(args.config, overrides)


def generate(args):
    """Consume public inputs only; labels stay inside the verifier."""
    import numpy as np
    import torch
    from data.common.video_encoding import encode_rgb_frames_to_webm
    from inference.data.inputs import load_inputs
    from inference.data.outputs import pending_inputs, valid_video
    from inference.data.timing import resolve_timing
    from training.data.samples import Sample
    from training.data.video import restore_video
    from training.models import build_adapter

    config = _config(args)
    steps = args.steps if args.steps is not None else (30 if config.model == "ltx-2.3" else 50)
    if steps < 1:
        raise ValueError("generation steps must be positive")
    seed = args.seed if args.seed is not None else (10 if config.model == "ltx-2.3" else config.seed)
    _, timings = resolve_timing(
        args.inputs, frames=args.frames, reference_timing=args.reference_timing
    )
    rows = load_inputs(args.inputs, args.shard_index, args.shards, args.limit)
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    pending = pending_inputs(root, rows, timings)
    for row in pending:
        timing = timings[row["id"]]
        if timing["frames"] > config.video.num_frames:
            raise ValueError(
                f"{row['id']}: increase video.num_frames to preserve the complete delivery"
            )
    if pending:
        adapter = build_adapter(config.to_dict())
        if config.warm_start:
            adapter.load(config.warm_start)
        adapter.eval()
        with (
            torch.no_grad(),
            torch.autocast(
                adapter.device.type,
                dtype=torch.bfloat16,
                enabled=config.dtype == "bfloat16",
            ),
        ):
            for index, row in enumerate(pending):
                timing = timings[row["id"]]
                delivery = {
                    "id": row["id"],
                    "video_num_frames": timing["frames"],
                    "video_fps": timing["fps"],
                }
                sample = Sample(
                    row["id"],
                    row.get("evaluation_task", ""),
                    row["prompt"],
                    Path(row["image"]),
                    Path("unused_reference"),
                    delivery,
                )
                generator = torch.Generator(device=adapter.device).manual_seed(seed)
                prepared = adapter.prepare_condition([sample])
                sigmas = (
                    adapter.sigma_schedule(steps, prepared).to(adapter.device)
                    if hasattr(adapter, "sigma_schedule")
                    else torch.linspace(1, 0, steps + 1, device=adapter.device)
                )
                latents = torch.randn(
                    prepared.latents.shape,
                    device=adapter.device,
                    dtype=prepared.latents.dtype,
                    generator=generator,
                )
                latents = adapter.project(latents, prepared)
                for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
                    latents = adapter.project(
                        latents + (next_sigma - sigma) * adapter(latents, sigma, prepared),
                        prepared,
                    )
                video = adapter.decode(latents, prepared)[0]
                mappings = prepared.mappings
                mapping = mappings[0] if isinstance(mappings, (tuple, list)) else mappings
                restored = restore_video(np.asarray(video, dtype=np.uint8), mapping)
                destination = root / sample.id / "prediction.webm"
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name("prediction.partial.webm")
                encode_rgb_frames_to_webm(
                    list(restored.frames),
                    temporary,
                    fps=restored.fps,
                    extra_args=["-threads", "1"],
                )
                if not valid_video(temporary, timing["frames"]):
                    raise RuntimeError("encoded candidate did not decode completely")
                temporary.replace(destination)
                print(
                    json.dumps(
                        {
                            "phase": "inference",
                            "id": sample.id,
                            "completed": index + 1,
                            "total": len(pending),
                            "prediction": str(destination),
                        }
                    ),
                    flush=True,
                )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": str(root),
                "shard_index": args.shard_index,
                "shards": args.shards,
                "samples": len(rows),
                "new_samples": len(pending),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--model-path")
    parser.add_argument("--checkpoint", help="Weight export produced by training")
    parser.add_argument("--tuning", choices=("full", "lora"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument(
        "--frames", type=int, help="Fixed Length frame count at 16 FPS (e.g. 81)"
    )
    timing.add_argument(
        "--reference-timing", type=Path, help="Reference Length timing-only JSONL sidecar"
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()

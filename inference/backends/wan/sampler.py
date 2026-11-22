from __future__ import annotations
from inference.backends.wan.model import NEGATIVE_PROMPT, resolve_model_inputs
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from inference.data.outputs import pending_inputs, valid_video
from inference.data.inputs import load_inputs
from inference.data.timing import resolve_timing, wan_native_frames


def run(args):
    import torch
    from diffusers import AutoencoderKLWan, DiffusionPipeline, WanImageToVideoPipeline
    from data.common.video_encoding import encode_rgb_frames_to_webm

    all_rows = load_inputs(args.inputs)
    _, timings = resolve_timing(
        args.inputs, frames=args.frames, reference_timing=args.reference_timing
    )
    rows = load_inputs(args.inputs, args.shard_index, args.shards, args.limit)
    local_files_only = bool(getattr(args, "local_files_only", False))
    budget = resolve_model_inputs(
        args.model_path,
        all_rows,
        args.max_sequence_length,
        args.model,
        local_files_only=local_files_only,
    )
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise ValueError("Wan generation requires CUDA")
    rlvr = args.model == "5b-rlvr"
    steps = args.steps if args.steps is not None else (30 if rlvr else 50)
    seed = args.seed if args.seed is not None else (0 if rlvr else 42)
    guidance = 1.0 if rlvr else 5.0
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pending = pending_inputs(output, rows, timings)
    if pending:
        model_path = Path(args.model_path).resolve()
        vae = AutoencoderKLWan.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        )
        pipeline = DiffusionPipeline if rlvr else WanImageToVideoPipeline
        options = (
            {
                "custom_pipeline": str(model_path / "pipeline.py"),
                "trust_remote_code": True,
            }
            if rlvr
            else {}
        )
        pipe = pipeline.from_pretrained(
            model_path,
            vae=vae,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
            **options,
        )
        if args.offload == "model":
            pipe.enable_model_cpu_offload(gpu_id=torch.device(args.device).index or 0)
        else:
            pipe.to(args.device)
        pipe.set_progress_bar_config(disable=True)
        with torch.inference_mode():
            for index, row in enumerate(pending):
                started = time.perf_counter()
                with Image.open(row["image"]) as source:
                    image = source.convert("RGB")
                delivery = timings[row["id"]]
                native_frames = wan_native_frames(delivery["frames"])
                options = (
                    {"sampler": "cps", "cps_eta": 0.7}
                    if rlvr
                    else {"negative_prompt": NEGATIVE_PROMPT}
                )
                generated = pipe(
                    image=image,
                    prompt=row["prompt"],
                    width=512,
                    height=512,
                    num_frames=native_frames,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    max_sequence_length=budget["max_sequence_length"],
                    generator=torch.Generator(
                        device=args.device if rlvr else "cpu"
                    ).manual_seed(seed),
                    output_type="np",
                    **options,
                ).frames[0]
                if (
                    generated.shape != (native_frames, 512, 512, 3)
                    or not np.isfinite(generated).all()
                ):
                    raise RuntimeError(
                        "Wan decoder did not return the complete finite RGB video"
                    )
                frames = (
                    (generated[: delivery["frames"]].clip(0, 1) * 255)
                    .round()
                    .astype(np.uint8)
                )
                destination = output / row["id"] / "prediction.webm"
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name("prediction.partial.webm")
                encode_rgb_frames_to_webm(
                    list(frames), temporary, fps=16, extra_args=["-threads", "1"]
                )
                if not valid_video(temporary, delivery["frames"]):
                    raise RuntimeError("encoded candidate did not decode completely")
                temporary.replace(destination)
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "phase": "inference",
                            "id": row["id"],
                            "completed": index + 1,
                            "total": len(pending),
                            "elapsed_seconds": elapsed,
                        }
                    ),
                    flush=True,
                )
    return {
        "status": "complete",
        "shards": args.shards,
        "shard_index": args.shard_index,
        "samples": len(rows),
        "new_samples": len(pending),
        "limited": args.limit is not None,
    }

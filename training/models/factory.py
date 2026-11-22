"""Load supported native image-to-video pipelines without fallback to T2V."""

from __future__ import annotations

import torch
from training.configuration.schema import MODEL_IDS
from training.models.ltx.adapter import LTX23Adapter
from training.models.wan.adapter import WanAdapter


def build_adapter(config, *, pipeline=None):
    """Build a Diffusers video adapter from config or an injected pipeline."""
    config = dict(config)
    key = config["model"]
    if key not in MODEL_IDS:
        raise ValueError(f"unsupported model {key!r}; choose {list(MODEL_IDS)}")
    if not config.get("pretrained_model_name_or_path"):
        config["pretrained_model_name_or_path"] = MODEL_IDS[key]
    if pipeline is None:
        from diffusers import LTX2ImageToVideoPipeline, WanImageToVideoPipeline

        dtype = getattr(torch, config.get("dtype", "bfloat16"))
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("dtype must be float32 or bfloat16")
        kwargs = {
            "torch_dtype": dtype,
            "local_files_only": config.get("local_files_only", False),
        }
        if key == "ltx-2.3":
            cls = LTX2ImageToVideoPipeline
            kwargs["vocoder"] = None
        else:
            cls = WanImageToVideoPipeline
            kwargs["expand_timesteps"] = key == "wan2.2-ti2v-5b"
            if config.get("boundary_ratio") is not None:
                kwargs["boundary_ratio"] = config["boundary_ratio"]
        pipeline = cls.from_pretrained(
            config["pretrained_model_name_or_path"], **kwargs
        )
        device = config.get("device", "cuda")
        if config.get("sharding") == "fsdp":
            # FSDP auto-wrap materializes/shards denoiser blocks individually.
            # Frozen encoders stay outside the FSDP module tree on the GPU.
            for name, component in pipeline.components.items():
                if name not in {"transformer", "transformer_2"} and isinstance(
                    component, torch.nn.Module
                ):
                    component.to(device)
        else:
            pipeline.to(device)
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
    return (
        LTX23Adapter(pipeline, config)
        if key == "ltx-2.3"
        else WanAdapter(pipeline, config)
    )

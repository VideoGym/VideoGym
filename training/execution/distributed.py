from __future__ import annotations
import os


def _accelerator(config):
    from accelerate import (
        Accelerator,
        DistributedDataParallelKwargs,
        FullyShardedDataParallelPlugin,
    )
    from accelerate.utils import GradientAccumulationPlugin

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if config.sharding == "none" and world > 1:
        raise ValueError("multi-process launch requires sharding=ddp or fsdp")
    if config.sharding == "fsdp" and world < 2:
        raise ValueError(
            "FSDP needs at least two CUDA workers; use sharding=none for one GPU"
        )
    if config.fsdp_cpu_offload and config.batch_samples > world:
        raise ValueError(
            "FSDP CPU offload requires batch_samples <= workers; gradient accumulation is unsupported"
        )
    os.environ["ACCELERATE_USE_FSDP"] = "true" if config.sharding == "fsdp" else "false"
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    plugin = None
    if config.sharding == "fsdp":
        from torch import bfloat16
        from torch.distributed.fsdp import MixedPrecision

        # The adapter explicitly casts denoiser inputs; root casting would
        # also round FP32 flow times, first-frame conditions and VAE targets.
        precision = (
            MixedPrecision(
                param_dtype=bfloat16,
                reduce_dtype=bfloat16,
                buffer_dtype=bfloat16,
                cast_root_forward_inputs=False,
            )
            if config.dtype == "bfloat16"
            else None
        )
        plugin = FullyShardedDataParallelPlugin(
            fsdp_version=1,
            sharding_strategy="FULL_SHARD",
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=[
                (
                    "LTX2VideoTransformerBlock"
                    if config.model == "ltx-2.3"
                    else "WanTransformerBlock"
                )
            ],
            use_orig_params=True,
            state_dict_type="SHARDED_STATE_DICT",
            cpu_offload=config.fsdp_cpu_offload,
            sync_module_states=True,
            mixed_precision_policy=precision,
        )
    return Accelerator(
        cpu=config.device == "cpu",
        mixed_precision="bf16" if config.dtype == "bfloat16" else "no",
        fsdp_plugin=plugin,
        gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=1),
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=config.model == "wan2.2-i2v-a14b"
            )
        ],
    )


def _barrier(accelerator):
    if accelerator.device.type == "cpu" and accelerator.num_processes > 1:
        import torch.distributed as distributed

        distributed.barrier()
    else:
        accelerator.wait_for_everyone()


def _cuda_checkpoint_boundary(device):
    """Synchronize CUDA and release idle cached blocks before checkpoint collectives."""
    if device.type != "cuda":
        return
    import torch

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

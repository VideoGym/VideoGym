from __future__ import annotations
from training.models.common.weights import TrainableWeights
from training.optimization.sft import FlowMatchingObjective
from training.models.common.batch import PreparedBatch
import copy
import math
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn


class VideoAdapter(TrainableWeights, FlowMatchingObjective, nn.Module):
    """Register only denoisers; frozen encoders stay outside the DDP/FSDP tree."""

    def __init__(self, pipeline, config: Mapping[str, Any]):
        super().__init__()
        self.pipeline = pipeline
        self.config = dict(config)
        self.model_key = self.config["model"]
        self.tuning = self.config.get("tuning", "lora")
        self.video_spec = self.config.get("video", {})
        for component in pipeline.components.values():
            if isinstance(component, nn.Module):
                component.requires_grad_(False).eval()
        denoisers = {
            name: getattr(pipeline, name)
            for name in ("transformer", "transformer_2")
            if getattr(pipeline, name, None) is not None
        }
        self.denoisers = nn.ModuleDict(denoisers)
        self._configure_training()

    @property
    def device(self):
        # FSDP may temporarily offload trainable parameters to CPU between
        # forwards; frozen encoder IO still belongs on the execution device.
        if hasattr(self, "config") and self.config.get("device"):
            return torch.device(self.config["device"])
        return next(self.denoisers.parameters()).device

    @property
    def dtype(self):
        return next(self.denoisers.parameters()).dtype

    def _configure_training(self):
        if self.tuning not in {"full", "lora"}:
            raise ValueError("tuning must be full or lora")
        for name, model in self.denoisers.items():
            if self.tuning == "full":
                # AdamW needs FP32 master weights for small SFT/RL updates.
                # Accelerator autocast controls the denoiser compute dtype.
                model.float().requires_grad_(True)
            else:
                from peft import LoraConfig

                targets = self.config.get("lora_targets") or [
                    "to_q",
                    "to_k",
                    "to_v",
                    "to_out.0",
                ]
                module_names = [
                    key
                    for key, module in model.named_modules()
                    if isinstance(module, nn.Linear)
                ]
                missing = [
                    target
                    for target in targets
                    if not any(
                        key == target or key.endswith("." + target)
                        for key in module_names
                    )
                ]
                if missing:
                    raise ValueError(f"{name} has no LoRA target modules: {missing}")
                rank = int(self.config.get("lora_rank", 16))
                model.add_adapter(
                    LoraConfig(
                        r=rank,
                        lora_alpha=int(self.config.get("lora_alpha", rank)),
                        target_modules=targets,
                        init_lora_weights=True,
                    )
                )
                # Train adapters in fp32 even when the frozen base uses bf16.
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.data = parameter.data.float()
                if self.config.get("sharding") == "fsdp":
                    # Transformer-block FSDP requires homogeneous parameter
                    # dtypes even with use_orig_params and frozen base weights.
                    # Keep this CPU conversion before wrapping; mixed precision
                    # still computes attention in bf16 after sharding.
                    model.float()
            if self.config.get("gradient_checkpointing", True):
                model.enable_gradient_checkpointing()

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def gradient_groups(self) -> dict[str, list[nn.Parameter]]:
        """Stable expert/stream groups for distributed update evidence.

        Return parameter references, including empty local FSDP shards. The
        trainer reduces their gradient norms across ranks.
        Each trainable parameter belongs to exactly one group.
        """
        if self.model_key == "ltx-2.3":
            groups = {"video": [], "audio": []}
            for name, parameter in self.denoisers.named_parameters():
                if parameter.requires_grad:
                    groups["audio" if "audio" in name else "video"].append(parameter)
            return groups
        return {
            name: [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            for name, model in self.denoisers.items()
        }

    def inference_scheduler(self, steps: int, prepared: PreparedBatch):
        """Fresh native solver, without mutating the live pipeline scheduler."""
        if type(steps) is not int or steps < 1:
            raise ValueError("inference steps must be a positive integer")
        scheduler = copy.deepcopy(self.pipeline.scheduler)
        scheduler.set_timesteps(steps, device="cpu")
        prepared.conditions["integer_timesteps"] = (
            not scheduler.timesteps.is_floating_point()
        )
        return scheduler

    def sigma_schedule(self, steps: int, prepared: PreparedBatch) -> torch.Tensor:
        """Native flow sigmas for the Euler ODE and Flow-SDE samplers."""
        scheduler = self.inference_scheduler(steps, prepared)
        return self._native_sigmas(scheduler, steps)

    @staticmethod
    def _native_sigmas(scheduler, steps):
        sigmas = torch.as_tensor(scheduler.sigmas).detach().cpu().float().clone()
        if (
            sigmas.shape != (steps + 1,)
            or not bool(torch.isfinite(sigmas).all())
            or not bool(((sigmas >= 0) & (sigmas <= 1)).all())
            or sigmas[-1] != 0
            or not bool((sigmas[:-1] > sigmas[1:]).all())
        ):
            raise ValueError(
                "native scheduler must provide decreasing flow sigmas in [0,1] ending at zero"
            )
        return sigmas

    def _guidance_scale(self, name, default):
        value = self.config.get(name)
        value = default if value is None else float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return value

    def forward(self, latents, sigma=None, prepared=None, **kwargs):
        if isinstance(latents, PreparedBatch):
            return self.flow_loss(latents, **kwargs)
        return self.velocity(latents, sigma, prepared)

    def project(self, latents: torch.Tensor, prepared: PreparedBatch):
        if prepared.known_mask is None:
            return latents
        clean = (
            prepared.known_latents
            if prepared.known_latents is not None
            else prepared.latents
        )
        return latents * (1 - prepared.known_mask) + clean * prepared.known_mask

    def _sigmas(self, sigma, batch):
        result = torch.as_tensor(
            sigma, device=self.device, dtype=torch.float32
        ).reshape(-1)
        if result.numel() == 1:
            result = result.expand(batch)
        if result.numel() != batch or not bool(((result >= 0) & (result <= 1)).all()):
            raise ValueError("sigma must have one value or a [batch] vector in [0, 1]")
        return result

    def _prepare_media(self, samples, spec, condition_only=False):
        from training.data.video import prepare_condition, prepare_sample

        if not isinstance(samples, (list, tuple)):
            samples = [samples]
        spec = self.video_spec if spec is None else spec
        prepare = prepare_condition if condition_only else prepare_sample
        prepared = [prepare(sample, spec) for sample in samples]
        if condition_only:
            # Native adapters only need the public shape to initialize noise.
            # No target video or unused full-resolution zeros cross to CUDA.
            frame_count, height, width, _ = prepared[0].reference_video.frames.shape
            video = torch.empty(
                (len(samples), 3, frame_count, height, width), device="meta"
            )
        else:
            frames = np.stack([item.reference_video.frames for item in prepared])
            video = (
                torch.from_numpy(frames)
                .to(self.device, dtype=torch.float32)
                .permute(0, 4, 1, 2, 3)
            )
            video = video / 127.5 - 1
        images = [item.input_image for item in prepared]
        prompts = [sample.prompt for sample in samples]
        return samples, prepared, video, images, prompts, spec

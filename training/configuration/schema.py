"""Explicit, portable experiment configuration; importing this needs no Torch."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math
import tomllib

MODEL_IDS = {
    "wan2.2-ti2v-5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "wan2.2-i2v-a14b": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "ltx-2.3": "Lightricks/LTX-2.3",
}
MODELS = tuple(MODEL_IDS)


@dataclass(frozen=True)
class VideoSpec:
    width: int = 512
    height: int = 512
    num_frames: int = 81
    fps: float = 16.0

    def validate(self, model: str) -> None:
        spatial = 16 if model == "wan2.2-i2v-a14b" else 32
        temporal = 8 if model == "ltx-2.3" else 4
        if self.width < spatial or self.height < spatial:
            raise ValueError("video dimensions are too small for the native VAE")
        if self.width % spatial or self.height % spatial:
            raise ValueError(f"{model} dimensions must be multiples of {spatial}")
        if self.num_frames < 2 or (self.num_frames - 1) % temporal:
            raise ValueError(f"{model} frame count must be {temporal}k+1, k>=1")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")


@dataclass(frozen=True)
class TrainConfig:
    model: str
    pretrained_model_name_or_path: str
    manifest: str
    output_dir: str
    stage: str = "sft"
    tuning: str = "lora"
    dtype: str = "bfloat16"
    device: str = "cuda"
    local_files_only: bool = False
    gradient_checkpointing: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_targets: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
    boundary_ratio: float | None = None
    guidance_scale: float | None = None
    guidance_scale_2: float | None = None
    audio_guidance_scale: float | None = None
    audio_loss: bool = False
    negative_prompt: str = ""
    max_sequence_length: int | None = None
    batch_samples: int = 8
    epochs: int = 1
    max_updates: int = 1000
    learning_rate: float = 1e-5
    min_lr_ratio: float = 0.1
    lr_scheduler: str = "linear"
    warmup_updates: int = 0
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    seed: int = 42
    save_every: int = 100
    keep_checkpoints: int = 1
    sharding: str = "none"
    fsdp_cpu_offload: bool = False
    tasks: tuple[str, ...] = ()
    samples_per_task: int | None = None
    warm_start: str | None = None
    resume: str | None = None
    video: VideoSpec = field(default_factory=VideoSpec)
    flow_grpo: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model}; choose {MODELS}")
        if self.max_sequence_length is not None and (
            type(self.max_sequence_length) is not int or self.max_sequence_length < 1
        ):
            raise ValueError("max_sequence_length must be a positive integer or None")
        if self.stage not in ("sft", "flow_grpo") or self.tuning not in (
            "full",
            "lora",
        ):
            raise ValueError("stage must be sft/flow_grpo; tuning must be full/lora")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("training supports float32 or bfloat16")
        if self.sharding not in ("none", "ddp", "fsdp"):
            raise ValueError("sharding must be none/ddp/fsdp")
        if type(self.fsdp_cpu_offload) is not bool:
            raise ValueError("fsdp_cpu_offload must be boolean")
        if type(self.audio_loss) is not bool:
            raise ValueError("audio_loss must be boolean")
        if self.fsdp_cpu_offload and (self.sharding != "fsdp" or self.stage != "sft"):
            raise ValueError(
                "FSDP CPU offload is supported only for single-backward SFT"
            )
        if (
            not self.pretrained_model_name_or_path
            or not self.manifest
            or not self.output_dir
        ):
            raise ValueError("model path, manifest and output_dir must be explicit")
        for name in (
            "batch_samples",
            "epochs",
            "max_updates",
            "save_every",
            "lora_rank",
            "lora_alpha",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "adam_epsilon", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.lr_scheduler not in ("linear", "cosine")
            or not 0 <= self.min_lr_ratio <= 1
            or self.warmup_updates < 0
        ):
            raise ValueError("invalid learning rate schedule")
        if (
            self.weight_decay < 0
            or not 0 <= self.adam_beta1 < 1
            or not 0 <= self.adam_beta2 < 1
        ):
            raise ValueError("invalid AdamW parameters")
        if self.samples_per_task is not None and self.samples_per_task < 1:
            raise ValueError("samples_per_task must be positive")
        if self.boundary_ratio is not None and not 0 < self.boundary_ratio < 1:
            raise ValueError("boundary_ratio must be between zero and one")
        for name in ("guidance_scale", "guidance_scale_2", "audio_guidance_scale"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.keep_checkpoints) is not int or self.keep_checkpoints < 0:
            raise ValueError(
                "keep_checkpoints must be a nonnegative integer; zero retains all"
            )
        self.video.validate(self.model)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path, overrides: dict | None = None) -> TrainConfig:
    with Path(path).open("rb") as stream:
        payload = tomllib.load(stream)
    payload.update({k: v for k, v in (overrides or {}).items() if v is not None})
    allowed = {f.name for f in fields(TrainConfig)}
    unknown = payload.keys() - allowed
    if unknown:
        raise ValueError(f"unknown training configuration keys: {sorted(unknown)}")
    video = payload.pop("video", {})
    payload["video"] = VideoSpec(**video)
    for key in ("tasks", "lora_targets"):
        if key in payload:
            if not isinstance(payload[key], (list, tuple)) or any(
                not isinstance(v, str) for v in payload[key]
            ):
                raise ValueError(f"{key} must be an array of strings")
            payload[key] = tuple(payload[key])
    config = TrainConfig(**payload)
    config.validate()
    return config

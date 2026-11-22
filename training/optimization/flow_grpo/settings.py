from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

Reduction = Literal["mean", "sum"]


@dataclass(frozen=True)
class FlowGRPOSettings:
    group_size: int = 4
    rollout_steps: int = 10
    noise_level: float = 0.7
    clip_range: float = 0.0001
    advantage_epsilon: float = 1e-4
    replay_epochs: int = 1
    timestep_fraction: float = 1.0
    rollout_microbatch_size: int = 1
    logprob_reduction: Reduction = "mean"
    trace_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.group_size < 2 or self.rollout_steps < 2:
            raise ValueError("FlowGRPO requires group_size >= 2 and rollout_steps >= 2")
        if not math.isfinite(self.noise_level) or self.noise_level <= 0:
            raise ValueError("FlowGRPO requires positive finite SDE noise_level")
        if (
            not 0 < self.clip_range < 1
            or not math.isfinite(self.advantage_epsilon)
            or self.advantage_epsilon <= 0
        ):
            raise ValueError(
                "clip_range must be in (0,1); advantage_epsilon must be positive and finite"
            )
        if self.replay_epochs < 1 or not 0 < self.timestep_fraction <= 1:
            raise ValueError(
                "replay_epochs must be positive; timestep_fraction must be in (0,1]"
            )
        if self.rollout_microbatch_size < 1:
            raise ValueError("rollout_microbatch_size must be positive")
        if self.logprob_reduction not in ("mean", "sum"):
            raise ValueError("logprob_reduction must be mean or sum")

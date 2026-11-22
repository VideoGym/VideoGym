from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterator

import torch
from torch import Tensor

from training.optimization.flow_grpo.math import (
    clipped_policy_loss,
    flow_sde_statistics,
    gaussian_log_prob,
)
from training.optimization.flow_grpo.rollout import (
    GroupRollout,
    _active_mask,
    _policy_eval,
    _velocity,
)
from training.optimization.flow_grpo.settings import FlowGRPOSettings


@dataclass
class ReplayLoss:
    loss: Tensor
    weight: float  # Sum of weights over an epoch equals one.
    metrics: dict[str, float]
    epoch: int
    step: int
    last_in_epoch: bool


def iter_replay_losses(
    adapter: Any,
    rollout: GroupRollout,
    settings: FlowGRPOSettings | None = None,
    *,
    unwrapped_adapter: Any | None = None,
) -> Iterator[ReplayLoss]:
    """Replay saved state/action pairs with gradients through the current policy."""
    settings = rollout.settings if settings is None else settings
    if (
        settings.noise_level != rollout.settings.noise_level
        or settings.logprob_reduction != rollout.settings.logprob_reduction
    ):
        raise ValueError(
            "replay cannot change the sampled transition distribution or logprob reduction"
        )
    del unwrapped_adapter
    trained_steps = max(
        1, math.ceil((len(rollout.sigmas) - 1) * settings.timestep_fraction)
    )
    sigma_max = float(rollout.sigmas[1])
    with _policy_eval(adapter):
        for epoch in range(settings.replay_epochs):
            for batch in rollout.batches:
                count = batch.old_log_probs.shape[0]
                device = batch.prepared.latents.device
                advantages = rollout.advantages[
                    batch.start_index : batch.start_index + count
                ].to(device)
                for index in range(trained_steps):
                    state = batch.states[index].to(device)
                    action = batch.states[index + 1].to(device)
                    sigma = rollout.sigmas[index].to(device).expand(count)
                    velocity = _velocity(adapter, state, sigma, batch.prepared)
                    mean, std = flow_sde_statistics(
                        state,
                        velocity,
                        sigma,
                        float(rollout.sigmas[index + 1]),
                        settings.noise_level,
                        sigma_max=sigma_max,
                    )
                    active = _active_mask(batch.prepared, state)
                    new_log_prob = gaussian_log_prob(
                        action,
                        mean,
                        std,
                        active_mask=active,
                        reduction=settings.logprob_reduction,
                    )
                    loss, metrics = clipped_policy_loss(
                        new_log_prob,
                        batch.old_log_probs[:, index],
                        advantages,
                        settings.clip_range,
                    )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite FlowGRPO replay loss")
                    metrics["loss"] = loss.detach()
                    yield ReplayLoss(
                        loss,
                        count / (len(rollout.rewards) * trained_steps),
                        {name: float(value) for name, value in metrics.items()},
                        epoch,
                        index,
                        batch is rollout.batches[-1] and index == trained_steps - 1,
                    )

from __future__ import annotations
from training.optimization.flow_grpo.settings import Reduction
import math
import torch
from torch import Tensor


def _scalar_per_sample(value: float | Tensor, sample: Tensor) -> Tensor:
    value = torch.as_tensor(value, device=sample.device, dtype=sample.dtype)
    if value.numel() not in (1, sample.shape[0]):
        raise ValueError("sigma must be scalar or have one value per sample")
    return value.reshape(-1, *([1] * (sample.ndim - 1)))


def flow_sde_statistics(
    latents: Tensor,
    velocity: Tensor,
    sigma: float | Tensor,
    sigma_next: float | Tensor,
    noise_level: float,
    *,
    sigma_max: float,
) -> tuple[Tensor, Tensor]:
    """Return reverse Euler-Maruyama transition mean and standard deviation.

    The schedule's second sigma is ``sigma_max`` and regularizes the otherwise
    singular noise coefficient at sigma=1, as in Flow-GRPO. A zero
    noise level produces the Euler ODE mean but has no Gaussian log density.
    Probability arithmetic is at least float32 even for bf16 model outputs.
    """
    if latents.shape != velocity.shape or latents.ndim < 2:
        raise ValueError("latents and velocity must share a batched shape")
    if not 0 < sigma_max < 1 or not math.isfinite(noise_level) or noise_level < 0:
        raise ValueError(
            "sigma_max must be in (0,1); noise_level must be finite and nonnegative"
        )
    dtype = torch.float64 if latents.dtype == torch.float64 else torch.float32
    state = latents.to(dtype=dtype)
    velocity = velocity.to(dtype=dtype)
    current = _scalar_per_sample(sigma, state)
    following = _scalar_per_sample(sigma_next, state)
    if bool(
        (
            (current <= 0) | (current > 1) | (following < 0) | (following >= current)
        ).any()
    ):
        raise ValueError("sigmas must decrease strictly from (0,1] toward zero")
    dt = following - current
    # Native flow UniPC schedules represent the starting endpoint as 1-1e-6
    # to avoid their log-SNR singularity. Keep that true model sigma, while
    # applying the same first-endpoint diffusion cap as at exactly one.
    near_endpoint = torch.isclose(current, torch.ones_like(current), rtol=0, atol=2e-6)
    denominator_sigma = torch.where(near_endpoint, sigma_max, current)
    diffusion = noise_level * torch.sqrt(current / (1 - denominator_sigma))
    correction = diffusion.square() / (2 * current)
    mean = state + (velocity + correction * (state + (1 - current) * velocity)) * dt
    std = diffusion * torch.sqrt(-dt)
    return mean, std


def _reduce_coordinates(
    values: Tensor, active_mask: Tensor | None, reduction: Reduction
) -> Tensor:
    if active_mask is None:
        return (
            values.flatten(1).mean(1)
            if reduction == "mean"
            else values.flatten(1).sum(1)
        )
    mask = torch.broadcast_to(
        active_mask.to(device=values.device, dtype=torch.bool), values.shape
    )
    count = mask.flatten(1).sum(1)
    if bool((count == 0).any()):
        raise ValueError(
            "each policy sample must have at least one unconstrained coordinate"
        )
    total = torch.where(mask, values, 0).flatten(1).sum(1)
    return total / count if reduction == "mean" else total


def gaussian_log_prob(
    action: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    active_mask: Tensor | None = None,
    reduction: Reduction = "mean",
) -> Tensor:
    """Log density of a saved action; gradients flow only through its mean.

    Fixed coordinates may differ arbitrarily without affecting this density.
    A projected first frame is a Dirac condition, not a Gaussian action.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be mean or sum")
    if bool((std <= 0).any()):
        raise ValueError("deterministic transitions have no Gaussian log probability")
    residual = action.detach().to(mean) - mean
    if active_mask is not None:
        residual = torch.where(
            active_mask.to(device=mean.device, dtype=torch.bool), residual, 0
        )
    elementwise = -0.5 * (residual / std).square()
    elementwise = elementwise - std.log() - 0.5 * math.log(2 * math.pi)
    return _reduce_coordinates(elementwise, active_mask, reduction)


def group_advantages(rewards: Tensor, epsilon: float = 1e-4) -> Tensor:
    """Normalize along the last (group) axis, using population standard deviation.

    Constant-reward groups yield exactly zero, without invented advantages.
    Different prompts must be separate rows, never one shared reward baseline.
    """
    rewards = torch.as_tensor(rewards).detach().float()
    if (
        rewards.ndim == 0
        or rewards.shape[-1] < 2
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError(
            "advantages require groups of at least two and positive finite epsilon"
        )
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("rewards must be finite")
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    return centered / (rewards.std(dim=-1, correction=0, keepdim=True) + epsilon)


def clipped_policy_loss(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    advantages: Tensor,
    clip_range: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """PPO's negative clipped surrogate, preserving both advantage signs."""
    log_ratio = new_log_prob - old_log_prob.detach().to(new_log_prob)
    ratio = log_ratio.exp()
    advantages = advantages.detach().to(new_log_prob)
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1 - clip_range, 1 + clip_range)
    loss = torch.maximum(unclipped, clipped).mean()
    return loss, {
        "policy_loss": loss.detach(),
        "ratio": ratio.detach().mean(),
        "approx_kl": (0.5 * log_ratio.detach().square()).mean(),
        "clip_fraction": ((ratio.detach() - 1).abs() > clip_range).float().mean(),
    }

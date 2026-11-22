from __future__ import annotations
from training.optimization.flow_grpo.math import (
    flow_sde_statistics,
    gaussian_log_prob,
    group_advantages,
)
from training.optimization.flow_grpo.settings import FlowGRPOSettings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence
import torch
from torch import Tensor


@dataclass
class RolloutMicrobatch:
    prepared: Any
    start_index: int
    states: list[Tensor]
    old_log_probs: Tensor  # [batch, denoising steps]


@dataclass
class GroupRollout:
    sample: Any
    batches: list[RolloutMicrobatch]
    sigmas: Tensor  # CPU, [steps + 1]
    rewards: Tensor  # CPU, [group]
    advantages: Tensor  # CPU, [group]
    settings: FlowGRPOSettings


def _core_adapter(adapter: Any, unwrapped_adapter: Any | None) -> Any:
    if unwrapped_adapter is not None:
        return unwrapped_adapter
    for _ in range(8):
        core = next(
            (
                getattr(adapter, name)
                for name in ("module", "_orig_mod", "_fsdp_wrapped_module")
                if getattr(adapter, name, None) is not None
            ),
            adapter,
        )
        if core is adapter:
            return core
        adapter = core
    raise ValueError("cannot unwrap policy; supply unwrapped_adapter explicitly")


@contextmanager
def _policy_eval(adapter: Any) -> Iterator[None]:
    """Match dropout and model modes for rollout and gradient-bearing replay."""
    modules = list(adapter.modules()) if hasattr(adapter, "modules") else []
    modes = [module.training for module in modules]
    if hasattr(adapter, "eval"):
        adapter.eval()
    try:
        yield
    finally:
        # Preserve frozen encoders' eval flags as well as the policy's mode.
        for module, mode in zip(modules, modes):
            module.training = mode


def _velocity(adapter: Any, latents: Tensor, sigma: Tensor, prepared: Any) -> Tensor:
    if isinstance(adapter, torch.nn.Module):
        return adapter(latents, sigma, prepared)
    return adapter.velocity(latents, sigma, prepared)


def _active_mask(prepared: Any, state: Tensor) -> Tensor | None:
    known_mask = getattr(prepared, "known_mask", None)
    return (
        None
        if known_mask is None
        else ~known_mask.to(device=state.device, dtype=torch.bool)
    )


def _schedule(steps: int, sigmas: Sequence[float] | Tensor | None) -> Tensor:
    schedule = (
        torch.linspace(1, 0, steps + 1)
        if sigmas is None
        else torch.as_tensor(sigmas).detach().cpu().float()
    )
    if schedule.ndim != 1 or len(schedule) != steps + 1:
        raise ValueError("sigma schedule must have rollout_steps + 1 values")
    starts_at_noise = bool(
        torch.isclose(schedule[0], torch.tensor(1.0), rtol=0, atol=2e-6)
    )
    if (
        not bool(torch.isfinite(schedule).all())
        or not starts_at_noise
        or schedule[0] > 1
        or schedule[-1] != 0
    ):
        raise ValueError("sigma schedule must be finite and run from 1 to 0")
    if not bool((schedule[:-1] > schedule[1:]).all()):
        raise ValueError("sigma schedule must decrease strictly")
    return schedule


def collect_group(
    adapter: Any,
    sample: Any,
    reward_fn: Callable[[Any, Any], float],
    settings: FlowGRPOSettings,
    *,
    prepared: Any | None = None,
    generator: torch.Generator | None = None,
    sigmas: Sequence[float] | Tensor | None = None,
    unwrapped_adapter: Any | None = None,
) -> GroupRollout:
    """Collect a same-prompt group with the current, frozen policy.

    ``prepared`` contains one sample and supports ``repeat(count)`` for larger
    microbatches. Its ``latents`` determine shape/device, never rollout targets.
    Full decoded videos reach ``reward_fn(video, sample)`` before being released.
    No policy update may occur while collecting a group. Saved states/probabilities
    are detached and copied to ``trace_device``; model conditions remain prepared.
    """
    core = _core_adapter(adapter, unwrapped_adapter)
    schedule = _schedule(settings.rollout_steps, sigmas)
    sigma_max = float(schedule[1])
    batches: list[RolloutMicrobatch] = []
    rewards: list[float] = []
    with _policy_eval(adapter), torch.no_grad():
        if prepared is None:
            prepare = getattr(core, "prepare_condition", core.prepare)
            prepared = prepare([sample])
        if prepared.latents.shape[0] != 1:
            raise ValueError("collect_group expects a single prepared sample")
        for start in range(0, settings.group_size, settings.rollout_microbatch_size):
            count = min(settings.rollout_microbatch_size, settings.group_size - start)
            context = prepared if count == 1 else prepared.repeat(count)
            template = context.latents
            latents = torch.randn(
                template.shape,
                device=template.device,
                dtype=torch.float32,
                generator=generator,
            )
            latents = core.project(latents, context).float()
            active = _active_mask(context, latents)
            states = [latents.detach().to(settings.trace_device, copy=True)]
            old_log_probs: list[Tensor] = []
            for index in range(settings.rollout_steps):
                sigma = schedule[index].to(latents.device).expand(count)
                velocity = _velocity(adapter, latents, sigma, context)
                mean, std = flow_sde_statistics(
                    latents,
                    velocity,
                    sigma,
                    float(schedule[index + 1]),
                    settings.noise_level,
                    sigma_max=sigma_max,
                )
                noise = torch.randn(
                    latents.shape,
                    device=latents.device,
                    dtype=latents.dtype,
                    generator=generator,
                )
                following = core.project(mean + std * noise, context).float()
                log_prob = gaussian_log_prob(
                    following,
                    mean,
                    std,
                    active_mask=active,
                    reduction=settings.logprob_reduction,
                )
                if not bool(
                    torch.isfinite(following).all() & torch.isfinite(log_prob).all()
                ):
                    raise FloatingPointError("nonfinite FlowGRPO rollout transition")
                old_log_probs.append(
                    log_prob.detach().to(settings.trace_device, copy=True)
                )
                states.append(following.detach().to(settings.trace_device, copy=True))
                latents = following
            videos = core.decode(latents, context)
            if len(videos) != count:
                raise ValueError("decode must return one complete video per rollout")
            for video in videos:
                rewards.append(float(reward_fn(video, sample)))
            del videos
            batches.append(
                RolloutMicrobatch(
                    context, start, states, torch.stack(old_log_probs, dim=1)
                )
            )
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    return GroupRollout(
        sample,
        batches,
        schedule,
        reward_tensor,
        group_advantages(reward_tensor, settings.advantage_epsilon),
        settings,
    )

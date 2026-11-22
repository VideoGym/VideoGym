from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any
import torch


@dataclass
class PreparedBatch:
    """Encoded targets and native conditioning; mask 1 means fixed, clean state.

    Wan states are BCTHW. LTX states are B x N concatenated packed video/audio
    features, so rollout and policy replay include both generated modalities.
    """

    latents: torch.Tensor
    conditions: dict[str, Any]
    known_mask: torch.Tensor | None = None
    known_latents: torch.Tensor | None = None
    mappings: list[Any] = field(default_factory=list)
    samples: list[Any] = field(default_factory=list)

    @property
    def loss_mask(self):
        return (
            torch.ones_like(self.latents)
            if self.known_mask is None
            else 1 - self.known_mask
        )

    def _map_tensors(self, function):
        def visit(value):
            if isinstance(value, torch.Tensor):
                return function(value)
            if isinstance(value, dict):
                return {key: visit(item) for key, item in value.items()}
            return value

        return replace(
            self,
            latents=visit(self.latents),
            conditions=visit(self.conditions),
            known_mask=visit(self.known_mask),
            known_latents=visit(self.known_latents),
        )

    def repeat(self, count: int):
        if self.latents.shape[0] != 1 or count < 1:
            raise ValueError("repeat requires one prepared sample and a positive count")
        result = self._map_tensors(
            lambda tensor: tensor.repeat((count,) + (1,) * (tensor.ndim - 1))
        )
        result.mappings = self.mappings * count
        result.samples = self.samples * count
        return result

    def select(self, indices: torch.Tensor):
        batch = self.latents.shape[0]
        result = self._map_tensors(
            lambda tensor: (
                tensor.index_select(0, indices.to(tensor.device))
                if tensor.ndim and tensor.shape[0] == batch
                else tensor
            )
        )
        ids = indices.tolist()
        result.mappings = [self.mappings[i] for i in ids] if self.mappings else []
        result.samples = [self.samples[i] for i in ids] if self.samples else []
        return result


def posterior_mode(output):
    if hasattr(output, "latent_dist"):
        return output.latent_dist.mode()
    if hasattr(output, "latents"):
        return output.latents
    raise TypeError("VAE output must expose latent_dist or latents")


def rgb_video(decoded: torch.Tensor):
    """Native VAE BCTHW [-1, 1] to verifier-ready RGB uint8 BTHWC."""
    return (
        ((decoded.float().clamp(-1, 1) + 1) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 4, 1)
        .cpu()
        .numpy()
    )

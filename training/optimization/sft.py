from __future__ import annotations

import torch


class FlowMatchingObjective:
    """Flow matching on an adapter's projected latents and velocity field."""

    def _flow_loss_at(self, prepared, sigmas, noise):
        scale = sigmas.reshape((-1,) + (1,) * (prepared.latents.ndim - 1))
        noisy = self.project((1 - scale) * prepared.latents + scale * noise, prepared)
        predicted = self.velocity(noisy, sigmas, prepared).float()
        squared_error = (predicted - (noise - prepared.latents).float()).square()
        return self.reduce_flow_error(squared_error, prepared)

    def reduce_flow_error(self, error, prepared):
        mask = prepared.loss_mask.expand_as(error)
        return (error * mask).sum() / mask.sum().clamp_min(1)

    def flow_loss(self, prepared, sigmas=None, noise=None):
        if sigmas is None:
            sigmas = torch.rand(prepared.latents.shape[0], device=self.device)
        sigmas = self._sigmas(sigmas, prepared.latents.shape[0])
        noise = torch.randn_like(prepared.latents) if noise is None else noise
        return self._flow_loss_at(prepared, sigmas, noise)

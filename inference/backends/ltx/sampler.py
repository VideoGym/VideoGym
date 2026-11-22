from __future__ import annotations
from inference.backends.ltx.conditioning import NEGATIVE_PROMPT
from inference.backends.ltx.delivery import deliver_video, pad_input
from inference.backends.ltx.weights import build_meta_transformer, load_transformer_weights
from dataclasses import replace
import numpy as np
from PIL import Image


class NativeGenerator:
    def __init__(
        self,
        model_path,
        checkpoint,
        contexts,
        device,
        *,
        local_files_only=False,
    ):
        import torch
        from diffusers import AutoencoderKLLTX2Video
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        from ltx_core.components.patchifiers import VideoLatentPatchifier
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.guidance.perturbations import (
            BatchedPerturbationConfig,
            Perturbation,
            PerturbationConfig,
            PerturbationType,
        )
        from ltx_core.model.transformer.model import X0Model

        self.device, self.contexts = device, contexts
        transformer = build_meta_transformer(model_path)
        weights = load_transformer_weights(model_path, checkpoint, transformer)
        transformer.load_state_dict(weights, strict=True, assign=True)
        del weights
        self.transformer = (
            transformer.requires_grad_(False)
            .eval()
            .to(device=device, dtype=torch.bfloat16)
        )
        self.model = X0Model(self.transformer)
        self.vae = (
            AutoencoderKLLTX2Video.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                local_files_only=local_files_only,
            )
            .requires_grad_(False)
            .eval()
            .to(device)
        )
        self.vae.enable_tiling()
        self.patchifier = VideoLatentPatchifier(1)
        self.scheduler, self.stepper = LTX2Scheduler(), EulerDiffusionStep()
        self.guider = MultiModalGuider(
            MultiModalGuiderParams(
                cfg_scale=3.0,
                stg_scale=1.0,
                stg_blocks=[28],
                rescale_scale=0.7,
                modality_scale=1.0,
            )
        )
        self.perturbation = BatchedPerturbationConfig(
            [
                PerturbationConfig(
                    [Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, [28])]
                )
            ],
            self.transformer.num_blocks,
            device=device,
            dtype=torch.bfloat16,
        )

    def generate(self, row, seed, steps, delivery):
        import torch
        from diffusers import LTX2ImageToVideoPipeline
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.tools import VideoLatentTools
        from ltx_core.types import VideoLatentShape

        d = delivery
        with Image.open(row["image"]) as image:
            source = np.asarray(image.convert("RGB"))
        canvas = pad_input(source, d)
        pixels = (
            torch.from_numpy(canvas.copy())
            .permute(2, 0, 1)[None, :, None]
            .to(self.device, dtype=torch.float32)
            / 127.5
            - 1
        )
        pixels = pixels.to(torch.bfloat16)
        encoded = self.vae.encode(pixels).latent_dist.mode()
        encoded = LTX2ImageToVideoPipeline._normalize_latents(
            encoded,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )
        shape = VideoLatentShape(
            1,
            128,
            (d.native_frames - 1) // 8 + 1,
            d.native_height // 32,
            d.native_width // 32,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        tools = VideoLatentTools(self.patchifier, shape, d.native_fps)
        clean = tools.create_initial_state(device=self.device, dtype=torch.bfloat16)
        clean = VideoConditionByLatentIndex(
            encoded, strength=1.0, latent_idx=0
        ).apply_to(clean, tools)
        state = GaussianNoiser(generator)(clean, noise_scale=1.0)
        positive = self.contexts[row["prompt"]].to(self.device)
        negative = self.contexts[NEGATIVE_PROMPT].to(self.device)
        sigmas = self.scheduler.execute(steps=steps).to(self.device)
        for index, sigma in enumerate(sigmas[:-1]):
            video = Modality(
                latent=state.latent,
                sigma=sigma[None],
                timesteps=state.denoise_mask * sigma,
                positions=state.positions,
                context=positive,
                context_mask=None,
            )
            pos, _ = self.model(video=video, audio=None, perturbations=None)
            neg, _ = self.model(
                video=replace(video, context=negative), audio=None, perturbations=None
            )
            perturbed, _ = self.model(
                video=video, audio=None, perturbations=self.perturbation
            )
            denoised = self.guider.calculate(pos, neg, perturbed, pos)
            denoised = (
                denoised * state.denoise_mask
                + clean.clean_latent.float() * (1 - state.denoise_mask)
            ).to(denoised.dtype)
            state = replace(
                state, latent=self.stepper.step(video.latent, denoised, sigmas, index)
            )
        video = self.patchifier.unpatchify(state.latent, shape)
        video = LTX2ImageToVideoPipeline._denormalize_latents(
            video,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        ).to(torch.bfloat16)
        # Decode at timestep zero with no extra noise.
        timestep = (
            torch.zeros((1,), device=self.device, dtype=torch.bfloat16)
            if self.vae.config.timestep_conditioning
            else None
        )
        decoded = self.vae.decode(video, temb=timestep, return_dict=False)[0]
        if not bool(torch.isfinite(decoded).all()):
            raise RuntimeError("native decoder returned non-finite model pixels")
        decoded = (
            ((decoded.float().clamp(-1, 1) + 1) * 127.5)
            .round()
            .byte()
            .permute(0, 2, 3, 4, 1)[0]
            .cpu()
            .numpy()
        )
        return deliver_video(decoded, d)

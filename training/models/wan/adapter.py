"""Wan 2.2 native TI2V 5B and dual-expert I2V A14B flow adapters.

Conditioning follows Diffusers' WanImageToVideoPipeline: 5B clamps the
image-latent first frame and uses per-token timesteps; A14B concatenates the
four-channel temporal mask and image-video latents and routes both experts.
"""

from __future__ import annotations

import torch
from PIL import Image

from training.models.common.batch import PreparedBatch, posterior_mode, rgb_video
from training.models.common.adapter import VideoAdapter


class WanAdapter(VideoAdapter):
    def __init__(self, pipeline, config):
        super().__init__(pipeline, config)
        self.expand_timesteps = bool(pipeline.config.expand_timesteps)
        self.boundary_ratio = getattr(pipeline.config, "boundary_ratio", None)
        if self.model_key == "wan2.2-ti2v-5b" and not self.expand_timesteps:
            raise ValueError(
                "Wan TI2V 5B requires native first-frame expanded timesteps"
            )
        if self.model_key == "wan2.2-ti2v-5b" and self.boundary_ratio is not None:
            raise ValueError(
                "Wan TI2V 5B is a single expert and does not use boundary_ratio"
            )
        if self.model_key == "wan2.2-i2v-a14b":
            if "transformer_2" not in self.denoisers or self.boundary_ratio is None:
                raise ValueError(
                    "Wan A14B requires both experts and a routing boundary"
                )
            if not 0 < self.boundary_ratio < 1 or self.expand_timesteps:
                raise ValueError(
                    "A14B uses a boundary in (0,1) and concatenated image conditioning"
                )

    def _normalize(self, latents):
        # Native Wan prepares image/video flow latents in FP32 before applying
        # the VAE statistics, even when the frozen VAE itself computes in bf16.
        latents = latents.float()
        config = self.pipeline.vae.config
        mean = latents.new_tensor(config.latents_mean).view(1, config.z_dim, 1, 1, 1)
        std = latents.new_tensor(config.latents_std).view(1, config.z_dim, 1, 1, 1)
        return (latents - mean) / std

    @torch.no_grad()
    def prepare(self, samples, spec=None):
        return self._prepare(samples, spec, condition_only=False)

    @torch.no_grad()
    def prepare_condition(self, samples, spec=None):
        """Construct image-conditioned state from the prompt/image without opening target video."""
        return self._prepare(samples, spec, condition_only=True)

    def _prepare(self, samples, spec, condition_only):
        samples, media, videos, images, prompts, spec = self._prepare_media(
            samples, spec, condition_only
        )
        pipe = self.pipeline
        guidance = self._guidance_scale("guidance_scale", 5.0)
        guidance_2 = self._guidance_scale("guidance_scale_2", guidance)
        do_cfg = condition_only and (
            guidance != 1 or (self.boundary_ratio is not None and guidance_2 != 1)
        )
        negative_prompt = self.config.get("negative_prompt", "")
        max_sequence_length = self.config.get("max_sequence_length")
        max_sequence_length = (
            512 if max_sequence_length is None else max_sequence_length
        )
        if type(max_sequence_length) is not int or max_sequence_length < 1:
            raise ValueError("Wan max_sequence_length must be a positive integer")
        if pipe.tokenizer is not None:
            from diffusers.pipelines.wan.pipeline_wan_i2v import prompt_clean

            texts = prompts + ([negative_prompt or ""] * len(prompts) if do_cfg else [])
            tokens = pipe.tokenizer(
                [prompt_clean(text) for text in texts],
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]
            required_length = max(len(ids) for ids in tokens)
            if required_length > max_sequence_length:
                raise ValueError(
                    f"Wan conditioning needs {required_length} tokens, exceeding "
                    f"max_sequence_length={max_sequence_length}; set max_sequence_length explicitly "
                    f"to at least {required_length}. Prompt truncation is not allowed."
                )
        height, width = videos.shape[-2:]
        frames = videos.shape[2]
        spatial = pipe.vae_scale_factor_spatial
        temporal = pipe.vae_scale_factor_temporal
        if height % (spatial * 2) or width % (spatial * 2) or (frames - 1) % temporal:
            raise ValueError(
                f"Wan requires spatial multiples of {spatial * 2} and {temporal}N+1 frames"
            )
        if condition_only:
            clean = torch.zeros(
                (
                    len(samples),
                    pipe.vae.config.z_dim,
                    (frames - 1) // temporal + 1,
                    height // spatial,
                    width // spatial,
                ),
                device=self.device,
            )
        else:
            clean = self._normalize(
                posterior_mode(pipe.vae.encode(videos.to(pipe.vae.dtype)))
            )
        native_conditions, masks = [], []
        image_embeds = []
        for image in images:
            pil = Image.fromarray(image)
            native_image = pipe.video_processor.preprocess(
                pil, height=height, width=width
            ).to(self.device, dtype=torch.float32)
            outputs = pipe.prepare_latents(
                native_image,
                batch_size=1,
                num_channels_latents=pipe.vae.config.z_dim,
                height=height,
                width=width,
                num_frames=frames,
                dtype=torch.float32,
                device=self.device,
            )
            native_conditions.append(outputs[1])
            if self.expand_timesteps:
                masks.append(1 - outputs[2])
            if getattr(pipe.transformer.config, "image_dim", None) is not None:
                image_embeds.append(pipe.encode_image(pil, self.device))
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=prompts,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            max_sequence_length=max_sequence_length,
            device=self.device,
        )
        condition = torch.cat(native_conditions)
        known_mask = torch.cat(masks) if masks else None
        conditions = {
            "encoder_hidden_states": prompt_embeds.to(self.dtype),
            "condition": condition,
            "encoder_hidden_states_image": (
                torch.cat(image_embeds).to(self.dtype) if image_embeds else None
            ),
            "do_cfg": do_cfg,
            "inference": condition_only,
            "guidance_scale": guidance,
            "guidance_scale_2": guidance_2,
        }
        if do_cfg:
            conditions["negative_encoder_hidden_states"] = negative_embeds.to(
                self.dtype
            )
        return PreparedBatch(
            clean,
            conditions,
            known_mask=known_mask,
            known_latents=condition.expand_as(clean) if self.expand_timesteps else None,
            mappings=[item.mapping for item in media],
            samples=list(samples),
        )

    def _model_timestep(self, sigma, prepared):
        timestep = sigma * self.pipeline.scheduler.config.num_train_timesteps
        if prepared.conditions.get("inference", False) and prepared.conditions.get(
            "integer_timesteps", False
        ):
            # UniPC's native Wan forward uses integer timesteps; SFT keeps
            # continuous sampled flow times. This also maps its 1-1e-6 first
            # sigma to the native timestep 999 instead of 999.999.
            timestep = timestep.to(torch.int64)
        return timestep

    def _expert_velocity(
        self, model, latents, sigma, prepared, guidance_key="guidance_scale"
    ):
        native = prepared.conditions
        timestep = self._model_timestep(sigma, prepared)
        if self.expand_timesteps:
            model_input = self.project(latents, prepared)
            patch = model.config.patch_size
            mask = (1 - prepared.known_mask)[
                :, 0, :: patch[0], :: patch[1], :: patch[2]
            ]
            timestep = mask.flatten(1) * timestep[:, None]
        else:
            model_input = torch.cat([latents, native["condition"]], dim=1)
        text = native["encoder_hidden_states"]
        image = native.get("encoder_hidden_states_image")
        if native.get("do_cfg", False):
            # Both text branches share the same first-frame image conditioning.
            # Batch them in one expert forward so every distributed worker
            # runs the same FSDP/DDP hook sequence during rollout and replay.
            model_input = torch.cat([model_input, model_input])
            timestep = torch.cat([timestep, timestep])
            text = torch.cat([native["negative_encoder_hidden_states"], text])
            image = None if image is None else torch.cat([image, image])
        prediction = model(
            hidden_states=model_input.to(self.dtype),
            timestep=timestep,
            encoder_hidden_states=text,
            encoder_hidden_states_image=image,
            return_dict=False,
        )[0].float()
        if native.get("do_cfg", False):
            unconditioned, conditioned = prediction.chunk(2)
            return unconditioned + native[guidance_key] * (conditioned - unconditioned)
        return prediction

    def velocity(self, latents, sigma, prepared):
        sigmas = self._sigmas(sigma, latents.shape[0])
        if self.boundary_ratio is None:
            return self._expert_velocity(
                self.denoisers["transformer"], latents, sigmas, prepared
            )
        # Mixed-sigma minibatches must route individual rows, never their mean.
        # Expert predictions and CFG are FP32 even when FSDP casts the
        # incoming state to bf16; keep routed flow velocities in FP32 too.
        results = torch.zeros_like(latents, dtype=torch.float32)
        high = (
            self._model_timestep(sigmas, prepared)
            >= self.boundary_ratio * self.pipeline.scheduler.config.num_train_timesteps
        )
        for name, selected in (("transformer", high), ("transformer_2", ~high)):
            indices = selected.nonzero(as_tuple=True)[0]
            if indices.numel():
                context = prepared.select(indices)
                prediction = self._expert_velocity(
                    self.denoisers[name],
                    latents.index_select(0, indices),
                    sigmas.index_select(0, indices),
                    context,
                    "guidance_scale" if name == "transformer" else "guidance_scale_2",
                )
                results = results.index_copy(0, indices, prediction)
        return results

    def flow_loss(self, prepared, sigmas=None, noise=None):
        if self.boundary_ratio is None or sigmas is not None:
            return super().flow_loss(prepared, sigmas, noise)
        # Stratify two forwards so both experts get gradients even at batch one.
        batch = prepared.latents.shape[0]
        noise = torch.randn_like(prepared.latents) if noise is None else noise
        low = torch.rand(batch, device=self.device) * self.boundary_ratio
        high = self.boundary_ratio + torch.rand(batch, device=self.device) * (
            1 - self.boundary_ratio
        )
        # Interval weights preserve the uniform-sigma flow objective while
        # ensuring each microbatch contributes gradients to both experts.
        return self.boundary_ratio * self._flow_loss_at(prepared, low, noise) + (
            1 - self.boundary_ratio
        ) * self._flow_loss_at(prepared, high, noise)

    @torch.no_grad()
    def decode(self, latents, prepared):
        latents = self.project(latents, prepared)
        config = self.pipeline.vae.config
        mean = latents.new_tensor(config.latents_mean).view(1, config.z_dim, 1, 1, 1)
        std = latents.new_tensor(config.latents_std).view(1, config.z_dim, 1, 1, 1)
        pixels = self.pipeline.vae.decode(
            (latents * std + mean).to(self.pipeline.vae.dtype), return_dict=False
        )[0]
        return rgb_video(pixels)

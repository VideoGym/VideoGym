"""LTX-2.3 first-frame conditioned flow-matching adapter.

VideoGym reference videos are silent task videos. By default ``audio_loss`` is
disabled (matching the VideoGym-LTX-2.3 recipe), while silent audio latents
still participate in the joint LTX-2.3 transformer forward using Diffusers'
native packing, connectors, and positional coordinates.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from PIL import Image

from training.models.common.adapter import VideoAdapter
from training.models.common.batch import PreparedBatch, posterior_mode, rgb_video


class LTX23Adapter(VideoAdapter):
    def __init__(self, pipeline, config):
        super().__init__(pipeline, config)
        for component in ("audio_vae", "connectors"):
            if getattr(pipeline, component, None) is None:
                raise ValueError(f"LTX-2.3 requires its native {component}")

    @torch.no_grad()
    def prepare(self, samples, spec=None):
        return self._prepare(samples, spec, condition_only=False)

    @torch.no_grad()
    def prepare_condition(self, samples, spec=None):
        """Build joint noisy-state shapes and image/text conditions, without targets."""
        return self._prepare(samples, spec, condition_only=True)

    def inference_scheduler(self, steps, prepared):
        """Reuse the native video-token-dependent shift and terminal schedule."""
        from diffusers.pipelines.ltx2.pipeline_ltx2_image2video import (
            calculate_shift,
            retrieve_timesteps,
        )

        if type(steps) is not int or steps < 1:
            raise ValueError("inference steps must be a positive integer")
        scheduler = copy.deepcopy(self.pipeline.scheduler)
        tokens = prepared.conditions["video_shape"][0]
        mu = calculate_shift(
            tokens,
            scheduler.config.get("base_image_seq_len", 1024),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.95),
            scheduler.config.get("max_shift", 2.05),
        )
        retrieve_timesteps(
            scheduler, steps, "cpu", sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu
        )
        return scheduler

    def _prepare(self, samples, spec, condition_only):
        samples, media, videos, images, prompts, spec = self._prepare_media(
            samples, spec, condition_only
        )
        pipe = self.pipeline
        batch, _, frame_count, height, width = videos.shape
        fps = float(spec["fps"] if isinstance(spec, dict) else spec.fps)
        spatial = pipe.vae_spatial_compression_ratio
        temporal = pipe.vae_temporal_compression_ratio
        if (
            height % (spatial * pipe.transformer_spatial_patch_size)
            or width % (spatial * pipe.transformer_spatial_patch_size)
            or (frame_count - 1) % temporal
        ):
            raise ValueError(
                f"LTX requires spatial multiples of {spatial} and {temporal}N+1 frames"
            )
        latent_frames = (frame_count - 1) // temporal + 1
        latent_height, latent_width = height // spatial, width // spatial
        if condition_only:
            clean_video = torch.zeros(
                (
                    batch,
                    pipe.transformer.config.in_channels,
                    latent_frames,
                    latent_height,
                    latent_width,
                ),
                device=self.device,
            )
        else:
            raw_video = posterior_mode(
                pipe.vae.encode(videos.to(pipe.vae.dtype))
            ).float()
            clean_video = pipe._normalize_latents(
                raw_video,
                pipe.vae.latents_mean,
                pipe.vae.latents_std,
                pipe.vae.config.scaling_factor,
            )
        clean_video = pipe._pack_latents(
            clean_video,
            pipe.transformer_spatial_patch_size,
            pipe.transformer_temporal_patch_size,
        ).float()
        conditioned, masks = [], []
        for image in images:
            pixels = pipe.video_processor.preprocess(
                Image.fromarray(image), height=height, width=width
            )
            fixed, mask = pipe.prepare_latents(
                image=pixels.to(self.device, dtype=pipe.vae.dtype),
                batch_size=1,
                num_channels_latents=pipe.transformer.config.in_channels,
                height=height,
                width=width,
                num_frames=frame_count,
                dtype=torch.float32,
                device=self.device,
            )
            conditioned.append(fixed)
            masks.append(mask)
        known_video = torch.cat(conditioned)
        conditioning_mask = torch.cat(masks)
        audio_frames = max(
            1,
            round(
                frame_count
                / fps
                * pipe.audio_sampling_rate
                / pipe.audio_hop_length
                / pipe.audio_vae_temporal_compression_ratio
            ),
        )
        mel_frames = (audio_frames - 1) * pipe.audio_vae_temporal_compression_ratio + 1
        mel = torch.full(
            (
                batch,
                pipe.audio_vae.config.in_channels,
                mel_frames,
                pipe.audio_vae.config.mel_bins,
            ),
            math.log(1e-5),
            device=self.device,
            dtype=pipe.audio_vae.dtype,
        )
        if condition_only:
            clean_audio = torch.zeros(
                (
                    batch,
                    audio_frames,
                    pipe.audio_vae.config.latent_channels
                    * pipe.audio_vae.config.mel_bins
                    // pipe.audio_vae_mel_compression_ratio,
                ),
                device=self.device,
            )
        else:
            raw_audio = posterior_mode(pipe.audio_vae.encode(mel)).float()
            clean_audio = pipe._pack_audio_latents(raw_audio)
            clean_audio = pipe._normalize_audio_latents(
                clean_audio, pipe.audio_vae.latents_mean, pipe.audio_vae.latents_std
            ).float()
        if clean_audio.shape[1] != audio_frames:
            raise ValueError(
                "audio VAE temporal dimensions disagree with native pipeline"
            )
        guidance = self._guidance_scale("guidance_scale", 3.0)
        audio_guidance = self._guidance_scale("audio_guidance_scale", 7.0)
        do_cfg = condition_only and (guidance != 1 or audio_guidance != 1)
        negative_prompt = self.config.get("negative_prompt", "")
        max_sequence_length = self.config.get("max_sequence_length")
        encode_kwargs = {}
        if max_sequence_length is not None:
            encode_kwargs["max_sequence_length"] = int(max_sequence_length)
        if getattr(pipe, "tokenizer", None) is not None:
            limit = int(max_sequence_length or 1024)
            checked = list(prompts)
            if do_cfg and negative_prompt:
                checked.append(negative_prompt)
            for item in checked:
                if len(pipe.tokenizer(item, add_special_tokens=True).input_ids) > limit:
                    raise ValueError(
                        f"prompt exceeds LTX max_sequence_length={limit}; increase max_sequence_length"
                    )
        text, attention_mask, negative_text, negative_mask = pipe.encode_prompt(
            prompt=prompts,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            device=self.device,
            **encode_kwargs,
        )
        if do_cfg:
            text = torch.cat([negative_text, text])
            attention_mask = torch.cat([negative_mask, attention_mask])
        video_text, audio_text, connector_mask = pipe.connectors(
            text,
            attention_mask,
            padding_side=getattr(pipe.tokenizer, "padding_side", "left"),
        )
        video_coords = pipe.transformer.rope.prepare_video_coords(
            batch, latent_frames, latent_height, latent_width, self.device, fps=fps
        )
        audio_coords = pipe.transformer.audio_rope.prepare_audio_coords(
            batch, audio_frames, self.device
        )
        negative_context = {}
        if do_cfg:
            negative_video_text, video_text = video_text.chunk(2)
            negative_audio_text, audio_text = audio_text.chunk(2)
            negative_connector_mask, connector_mask = connector_mask.chunk(2)
            negative_context = {
                "negative_encoder_hidden_states": negative_video_text.to(self.dtype),
                "negative_audio_encoder_hidden_states": negative_audio_text.to(
                    self.dtype
                ),
                "negative_encoder_attention_mask": negative_connector_mask,
            }
        conditions = {
            "video_shape": tuple(clean_video.shape[1:]),
            "audio_shape": tuple(clean_audio.shape[1:]),
            "num_frames": latent_frames,
            "height": latent_height,
            "width": latent_width,
            "fps": fps,
            "audio_num_frames": audio_frames,
            "encoder_hidden_states": video_text.to(self.dtype),
            "audio_encoder_hidden_states": audio_text.to(self.dtype),
            "encoder_attention_mask": connector_mask,
            "conditioning_mask": conditioning_mask,
            "video_coords": video_coords,
            "audio_coords": audio_coords,
            "do_cfg": do_cfg,
            "guidance_scale": guidance,
            "audio_guidance_scale": audio_guidance,
            **negative_context,
        }
        clean = torch.cat([clean_video.flatten(1), clean_audio.flatten(1)], dim=1)
        video_mask = conditioning_mask.unsqueeze(-1).expand_as(clean_video).flatten(1)
        known_mask = torch.cat(
            [video_mask, torch.zeros_like(clean_audio.flatten(1))], dim=1
        )
        known = torch.cat(
            [known_video.flatten(1), torch.zeros_like(clean_audio.flatten(1))], dim=1
        )
        return PreparedBatch(
            clean,
            conditions,
            known_mask,
            known,
            mappings=[item.mapping for item in media],
            samples=list(samples),
        )

    @staticmethod
    def split_state(latents, prepared):
        video_shape, audio_shape = (
            prepared.conditions["video_shape"],
            prepared.conditions["audio_shape"],
        )
        count = math.prod(video_shape)
        video = latents[:, :count].reshape(latents.shape[0], *video_shape)
        audio = latents[:, count:].reshape(latents.shape[0], *audio_shape)
        return video, audio

    def velocity(self, latents, sigma, prepared):
        pipe, context = self.pipeline, prepared.conditions
        sigma = self._sigmas(sigma, latents.shape[0])
        video, audio = self.split_state(self.project(latents, prepared), prepared)
        multiplier = getattr(pipe.transformer.config, "timestep_scale_multiplier", 1000)
        timestep = sigma * multiplier
        video_text = context["encoder_hidden_states"]
        audio_text = context["audio_encoder_hidden_states"]
        mask = context["encoder_attention_mask"]
        conditioning_mask = context["conditioning_mask"]
        video_coords, audio_coords = context["video_coords"], context["audio_coords"]
        if context.get("do_cfg", False):
            video, audio = torch.cat([video, video]), torch.cat([audio, audio])
            timestep = torch.cat([timestep, timestep])
            video_text = torch.cat(
                [context["negative_encoder_hidden_states"], video_text]
            )
            audio_text = torch.cat(
                [context["negative_audio_encoder_hidden_states"], audio_text]
            )
            mask = torch.cat([context["negative_encoder_attention_mask"], mask])
            conditioning_mask = torch.cat([conditioning_mask, conditioning_mask])
            video_coords, audio_coords = torch.cat(
                [video_coords, video_coords]
            ), torch.cat([audio_coords, audio_coords])
        prediction_video, prediction_audio = self.denoisers["transformer"](
            hidden_states=video.to(self.dtype),
            audio_hidden_states=audio.to(self.dtype),
            encoder_hidden_states=video_text,
            audio_encoder_hidden_states=audio_text,
            timestep=timestep[:, None] * (1 - conditioning_mask),
            audio_timestep=timestep,
            sigma=timestep,
            audio_sigma=timestep,
            encoder_attention_mask=mask,
            audio_encoder_attention_mask=mask,
            num_frames=context["num_frames"],
            height=context["height"],
            width=context["width"],
            fps=context["fps"],
            audio_num_frames=context["audio_num_frames"],
            video_coords=video_coords,
            audio_coords=audio_coords,
            isolate_modalities=False,
            use_cross_timestep=True,
            return_dict=False,
        )
        if context.get("do_cfg", False):
            unconditioned_video, conditioned_video = prediction_video.float().chunk(2)
            unconditioned_audio, conditioned_audio = prediction_audio.float().chunk(2)
            prediction_video = unconditioned_video + context["guidance_scale"] * (
                conditioned_video - unconditioned_video
            )
            prediction_audio = unconditioned_audio + context["audio_guidance_scale"] * (
                conditioned_audio - unconditioned_audio
            )
        return torch.cat(
            [prediction_video.float().flatten(1), prediction_audio.float().flatten(1)],
            dim=1,
        )

    def reduce_flow_error(self, error, prepared):
        video, audio = self.split_state(error, prepared)
        mask_video, _ = self.split_state(prepared.loss_mask, prepared)
        video_loss = (video * mask_video).sum() / mask_video.sum().clamp_min(1)
        audio_weight = 1.0 if self.config.get("audio_loss", False) else 0.0
        return video_loss + audio_weight * audio.mean()

    def _unpack_video(self, latents, prepared):
        video, _ = self.split_state(self.project(latents, prepared), prepared)
        c = prepared.conditions
        pipe = self.pipeline
        return pipe._unpack_latents(
            video,
            c["num_frames"],
            c["height"],
            c["width"],
            pipe.transformer_spatial_patch_size,
            pipe.transformer_temporal_patch_size,
        )

    @torch.no_grad()
    def decode(self, latents, prepared):
        pipe = self.pipeline
        video = self._unpack_video(latents, prepared)
        timestep = (
            torch.zeros((video.shape[0],), device=self.device, dtype=pipe.vae.dtype)
            if pipe.vae.config.timestep_conditioning
            else None
        )
        video = pipe._denormalize_latents(
            video,
            pipe.vae.latents_mean,
            pipe.vae.latents_std,
            pipe.vae.config.scaling_factor,
        )
        return rgb_video(
            pipe.vae.decode(video.to(pipe.vae.dtype), timestep, return_dict=False)[0]
        )

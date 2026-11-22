from __future__ import annotations
import json
from pathlib import Path


def native_to_diffusers_key(key):
    """LTX-2.3's lossless upstream transformer parameter renaming."""
    replacements = {
        "patchify_proj": "proj_in",
        "audio_patchify_proj": "audio_proj_in",
        "av_ca_video_scale_shift_adaln_single": "av_cross_attn_video_scale_shift",
        "av_ca_a2v_gate_adaln_single": "av_cross_attn_video_a2v_gate",
        "av_ca_audio_scale_shift_adaln_single": "av_cross_attn_audio_scale_shift",
        "av_ca_v2a_gate_adaln_single": "av_cross_attn_audio_v2a_gate",
        "scale_shift_table_a2v_ca_video": "video_a2v_cross_attn_scale_shift_table",
        "scale_shift_table_a2v_ca_audio": "audio_a2v_cross_attn_scale_shift_table",
        "q_norm": "norm_q", "k_norm": "norm_k",
        "audio_prompt_adaln_single": "audio_prompt_adaln",
        "prompt_adaln_single": "prompt_adaln",
    }
    for original, converted in replacements.items():
        key = key.replace(original, converted)
    if key.startswith("adaln_single."):
        key = key.replace("adaln_single.", "time_embed.", 1)
    if key.startswith("audio_adaln_single."):
        key = key.replace("audio_adaln_single.", "audio_time_embed.", 1)
    return key


def base_weight_map(model_path):
    path = Path(model_path)
    folder = path / "transformer" if (path / "transformer").is_dir() else path
    index = json.loads((folder / "diffusion_pytorch_model.safetensors.index.json").read_text())
    return folder, index["weight_map"]


def _load_sharded_diffusers_weights(source, names):
    from safetensors.torch import load_file

    folder, mapping = base_weight_map(source)
    if set(names) != set(mapping):
        raise ValueError("release does not match the native transformer keys")
    weights = {}
    for filename in sorted(set(mapping.values())):
        for key, tensor in load_file(str(folder / filename), device="cpu").items():
            weights[names[key]] = tensor
    return weights


def load_transformer_weights(model_path, checkpoint, transformer):
    from safetensors.torch import load_file

    native_keys = set(transformer.state_dict())
    names = {native_to_diffusers_key(key): key for key in native_keys}
    if checkpoint is not None:
        source = Path(checkpoint)
        if source.is_dir():
            return _load_sharded_diffusers_weights(source, names)
        raw = load_file(str(source), device="cpu")
        if set(raw) == native_keys:
            return raw
        if set(raw) == set(names):
            return {names[key]: tensor for key, tensor in raw.items()}
        raise ValueError("checkpoint does not match the native transformer keys")
    return _load_sharded_diffusers_weights(model_path, names)


def native_options(config):
    """Translate the LTX-2.3 config without renaming checkpoint tensors."""
    from ltx_core.model.transformer.model import LTXModelType
    from ltx_core.model.transformer.rope import LTXRopeType

    if config.get("_class_name") != "LTX2VideoTransformer3DModel" or config.get(
        "use_prompt_embeddings"
    ):
        raise ValueError(
            "expected the per-modality projected LTX-2.3 transformer config"
        )
    return dict(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=config["num_attention_heads"],
        attention_head_dim=config["attention_head_dim"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        num_layers=config["num_layers"],
        cross_attention_dim=config["cross_attention_dim"],
        norm_eps=config["norm_eps"],
        positional_embedding_theta=config["rope_theta"],
        positional_embedding_max_pos=[
            config["pos_embed_max_pos"],
            config["base_height"],
            config["base_width"],
        ],
        timestep_scale_multiplier=config["timestep_scale_multiplier"],
        use_middle_indices_grid=True,
        audio_num_attention_heads=config["audio_num_attention_heads"],
        audio_attention_head_dim=config["audio_attention_head_dim"],
        audio_in_channels=config["audio_in_channels"],
        audio_out_channels=config["audio_out_channels"],
        audio_cross_attention_dim=config["audio_cross_attention_dim"],
        audio_positional_embedding_max_pos=[config["audio_pos_embed_max_pos"]],
        av_ca_timestep_scale_multiplier=config["cross_attn_timestep_scale_multiplier"],
        rope_type=LTXRopeType(config["rope_type"]),
        double_precision_rope=config["rope_double_precision"],
        apply_gated_attention=config["gated_attn"],
        cross_attention_adaln=config["cross_attn_mod"],
        use_prompt_adaln_single=config.get("use_prompt_adaln_single", True),
        ff_bias=config.get("ff_bias", True),
        audio_ff_bias=config.get("audio_ff_bias", True),
    )


def build_meta_transformer(model_path):
    import torch
    from ltx_core.model.transformer.model import LTXModel

    config = json.loads((Path(model_path) / "transformer/config.json").read_text())
    with torch.device("meta"):
        return LTXModel(**native_options(config))

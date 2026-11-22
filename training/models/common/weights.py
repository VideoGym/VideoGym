from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import shutil
import torch

_WRAPPER_SEGMENTS = frozenset(
    {"_fsdp_wrapped_module", "_checkpoint_wrapped_module", "_orig_mod"}
)


def public_parameter_name(name: str) -> str:
    return ".".join(
        segment for segment in name.split(".") if segment not in _WRAPPER_SEGMENTS
    )


class TrainableWeights:
    """Read and write the trainable weights of a video adapter."""

    def save(self, path, state_dict=None, *, model_checkpoint=None):
        """Save trainable weights, or hardlink an existing full model DCP state.

        A model-only DCP export consumes the checkpoint's saved model shards
        directly; it never gathers or duplicates a full FSDP weight tensor.
        """
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 1,
            "model": self.model_key,
            "tuning": self.tuning,
            "denoisers": list(self.denoisers),
            "config": self.config,
        }
        if model_checkpoint is not None:
            if self.tuning != "full" or state_dict is not None:
                raise ValueError(
                    "model DCP exports require full tuning and no gathered state dictionary"
                )
            from torch.distributed.checkpoint import FileSystemReader

            original = Path(model_checkpoint).expanduser().resolve()
            checkpoint = FileSystemReader(original).read_metadata()
            if not checkpoint.state_dict_metadata or any(
                not key.startswith("model.") for key in checkpoint.state_dict_metadata
            ):
                raise ValueError(
                    "weight-only DCP exports must contain only the model payload"
                )
            weights = destination / "model_state"
            weights.mkdir()
            files = {
                Path(entry.relative_path) for entry in checkpoint.storage_data.values()
            }
            files.add(Path(".metadata"))
            for relative in sorted(files):
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        "DCP storage filenames must stay inside the model checkpoint"
                    )
                target = weights / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(original / relative, target)
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
                    shutil.copy2(original / relative, target)
            metadata.update(
                format="fsdp-dcp", payload_key="model", weight_path="model_state"
            )
        else:
            from safetensors.torch import save_file

            state = self.state_dict() if state_dict is None else state_dict
            state = {public_parameter_name(name): value for name, value in state.items()}
            selected = (
                set(state)
                if self.tuning == "full"
                else {
                    public_parameter_name(name)
                    for name, parameter in self.named_parameters()
                    if parameter.requires_grad
                }
            )
            weights = {
                name: value.detach().cpu().contiguous()
                for name, value in state.items()
                if name in selected
            }
            if selected - weights.keys():
                raise ValueError(
                    "gathered state dictionary is missing trainable parameters"
                )
            save_file(weights, str(destination / "trainable_weights.safetensors"))
            metadata["format"] = "safetensors"
        (destination / "adapter_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

    def save_checkpoint_weights(self, path, model_checkpoint):
        """Explicit runtime capability for a deduplicated full FSDP export."""
        return self.save(path, model_checkpoint=model_checkpoint)

    def load(self, path):
        from safetensors.torch import load_file

        source = Path(path)
        metadata = json.loads((source / "adapter_config.json").read_text())
        for key, expected in (
            ("model", self.model_key),
            ("tuning", self.tuning),
            ("denoisers", list(self.denoisers)),
        ):
            if metadata[key] != expected:
                raise ValueError(f"checkpoint {key} does not match adapter")
        source_config = metadata["config"]
        checks = {}
        if self.tuning == "lora":
            checks.update(
                {
                    "lora_rank": 16,
                    "lora_alpha": self.config.get("lora_rank", 16),
                    "lora_targets": ["to_q", "to_k", "to_v", "to_out.0"],
                }
            )
        for key, default in checks.items():
            expected = self.config.get(key, default)
            actual = source_config.get(key, default)
            if key == "lora_targets":
                expected, actual = list(expected or default), list(actual or default)
            if expected != actual:
                raise ValueError(
                    f"checkpoint {key} does not match the configured LoRA"
                )
        if metadata.get("format", "safetensors") == "fsdp-dcp":
            if self.tuning != "full" or metadata.get("payload_key") != "model":
                raise ValueError(
                    "model DCP checkpoints require full tuning and the model payload"
                )
            import torch.distributed.checkpoint as dcp

            reader = dcp.FileSystemReader(source / metadata["weight_path"])
            checkpoint = reader.read_metadata()
            state = self.state_dict()
            if set(checkpoint.state_dict_metadata) != {
                "model." + name for name in state
            }:
                raise ValueError(
                    "DCP checkpoint does not contain exactly the expected model weights"
                )
            if any(
                checkpoint.state_dict_metadata["model." + name].size != value.shape
                for name, value in state.items()
            ):
                raise ValueError(
                    "DCP checkpoint model weight shapes do not match the adapter"
                )
            dcp.load(
                {"model": state},
                storage_reader=reader,
                no_dist=not torch.distributed.is_initialized(),
            )
            self.load_state_dict(state, strict=True)
            return self
        if metadata.get("format", "safetensors") != "safetensors":
            raise ValueError("unsupported adapter weight format")
        state = load_file(str(source / "trainable_weights.safetensors"))
        expected = (
            set(self.state_dict())
            if self.tuning == "full"
            else {
                name
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
            }
        )
        if set(state) != expected:
            raise ValueError(
                "checkpoint does not contain exactly the expected trainable weights"
            )
        self.load_state_dict(state, strict=self.tuning == "full")
        return self

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path


def _initialize_fsdp_adam_state(optimizer, model, checkpoint):
    """Create the saved owned-shard Adam layout before Accelerate DCP loading.

    DCP only loads keys present in its destination layout. A fresh AdamW has
    no moments, so Accelerate's FSDP1 loader otherwise reads an empty state.
    Metadata selects saved FQNs: unused parameters and empty local shards
    must stay stateless. This allocates zeros without taking an optimizer step.
    """
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from training.models.common.weights import public_parameter_name

    raw = getattr(optimizer, "optimizer", optimizer)
    if not isinstance(raw, torch.optim.AdamW):
        raise TypeError("FSDP recovery layout requires the launcher's AdamW optimizer")
    metadata = FileSystemReader(Path(checkpoint) / "optimizer_0").read_metadata()
    saved = set()
    for path in metadata.planner_data.values():
        if len(path) >= 4 and path[:2] == ("optimizer", "state"):
            if not isinstance(path[2], str):
                raise RuntimeError("FSDP optimizer checkpoint must use parameter FQNs")
            saved.add(public_parameter_name(path[2]))
    names = {
        id(parameter): public_parameter_name(name)
        for name, parameter in model.named_parameters()
    }
    if saved - set(names.values()):
        raise RuntimeError(
            "FSDP optimizer checkpoint contains parameters absent from the current model"
        )
    for group in raw.param_groups:
        for parameter in group["params"]:
            if (
                not parameter.requires_grad
                or not parameter.numel()
                or names.get(id(parameter)) not in saved
            ):
                continue
            if parameter.dtype != torch.float32:
                raise RuntimeError(
                    "FSDP Adam recovery requires FP32 local master parameters"
                )
            state = raw.state[parameter]
            state.setdefault(
                "step",
                torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=(
                        parameter.device
                        if group.get("capturable") or group.get("fused")
                        else "cpu"
                    ),
                ),
            )
            for key in ("exp_avg", "exp_avg_sq"):
                if key not in state:
                    state[key] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
            if group.get("amsgrad") and "max_exp_avg_sq" not in state:
                state["max_exp_avg_sq"] = torch.zeros_like(
                    parameter, memory_format=torch.preserve_format
                )
    return saved


@contextmanager
def _strict_fsdp1_checkpoint_loading():
    """Keep modern DCP layout when this rank has zero local tensor shards.

    DefaultLoadPlanner's legacy-format heuristic sees deliberately omitted
    no-local-shard tensors as missing keys and can wrongly switch optimizer
    groups to a v2.3 whole-list leaf. Our checkpoints use modern flattened
    groups: load all requested keys strictly, without that legacy fallback.
    Release obsolete owned-shard Adam placeholders before native conversion
    creates the restored owned shards from the loaded row-shard layout.
    """
    from unittest.mock import patch
    from torch.distributed.checkpoint import default_planner
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    import torch

    class StrictModernLoadPlanner(default_planner.DefaultLoadPlanner):
        def create_local_plan(self):
            if self.metadata is None:
                raise RuntimeError("DCP load planner has no checkpoint metadata")
            return default_planner.create_default_local_load_plan(
                self.state_dict, self.metadata, strict=True
            )

    original_to_load = FSDP.optim_state_dict_to_load

    def to_load(
        model,
        optim,
        optim_state_dict,
        is_named_optimizer=False,
        load_directly=False,
        group=None,
    ):
        # Accelerate calls this only after its strict DCP load has finished.
        # The loaded row-shard layout now owns the real moments; the original
        # owned-shard placeholders are no longer inputs to native conversion.
        # Drop them before native reshaping allocates the replacement shards.
        raw = getattr(optim, "optimizer", optim)
        if isinstance(raw, torch.optim.AdamW) and not is_named_optimizer:
            raw.state.clear()
        return original_to_load(
            model=model,
            optim=optim,
            optim_state_dict=optim_state_dict,
            is_named_optimizer=is_named_optimizer,
            load_directly=load_directly,
            group=group,
        )

    # Accelerate imports this class inside its FSDP model/optimizer loaders.
    # Native model/optimizer/RNG recovery remains intact. Replace the planner
    # heuristic and free obsolete moments at the native conversion boundary.
    with (
        patch.object(default_planner, "DefaultLoadPlanner", StrictModernLoadPlanner),
        patch.object(FSDP, "optim_state_dict_to_load", staticmethod(to_load)),
    ):
        yield

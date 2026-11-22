from __future__ import annotations
from training.execution.checkpoint.optimizer import (
    _initialize_fsdp_adam_state,
    _strict_fsdp1_checkpoint_loading,
)
from training.execution.distributed import (
    _accelerator,
    _barrier,
    _cuda_checkpoint_boundary,
)
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import shutil
from training.configuration.schema import TrainConfig
from training.execution.sampling import epoch_order, learning_rate, rank_slots


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


@dataclass
class Progress:
    updates: int = 0
    optimizer_steps: int = 0
    consumed_samples: int = 0
    epoch: int = 0
    offset: int = 0

    def state_dict(self):
        return vars(self).copy()

    def load_state_dict(self, state):
        for key in vars(self):
            setattr(self, key, state[key])


def build_plan(config: TrainConfig, samples) -> dict:
    config.validate()
    if not samples:
        raise ValueError("training selection is empty")
    from collections import Counter

    maximum_frames = max(s.canonical_sample["video_num_frames"] for s in samples)
    if maximum_frames > config.video.num_frames:
        temporal = 8 if config.model == "ltx-2.3" else 4
        needed = math.ceil((maximum_frames - 1) / temporal) * temporal + 1
        raise ValueError(
            f"complete reference videos require video.num_frames >= {needed}"
        )

    return {
        "version": "videogym.video.training.v1",
        "config": json.loads(json.dumps(config.to_dict())),
        "sample_count": len(samples),
        "maximum_reference_frames": maximum_frames,
        "task_counts": dict(sorted(Counter(s.task_name for s in samples).items())),
        "planned_updates": min(
            config.max_updates,
            config.epochs * math.ceil(len(samples) / config.batch_samples),
        ),
        "conditioning": "native_first_frame",
        "supervision": "complete_reference_video",
        "reward": (
            "complete_video_task_verifier" if config.stage == "flow_grpo" else None
        ),
        "ltx_audio_loss": config.audio_loss if config.model == "ltx-2.3" else None,
    }


def _compatible_plan(previous: dict, current: dict) -> None:
    old, new = previous["config"].copy(), current["config"].copy()
    for cfg in (old, new):
        cfg.pop("resume", None)
    if (
        old != new
        or previous.get("sample_count") != current.get("sample_count")
        or previous.get("task_counts") != current.get("task_counts")
    ):
        raise ValueError(
            "resume requires matching training configuration and dataset selection"
        )


def train(
    config: TrainConfig,
    *,
    samples=None,
    reward=None,
) -> dict:
    import torch
    from accelerate.utils import gather_object, set_seed
    from training.data.manifest import load_training_manifest
    from training.models import build_adapter

    config.validate()
    accelerator = _accelerator(config)
    if config.device == "cuda" and accelerator.device.type != "cuda":
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    set_seed(config.seed, device_specific=True)
    samples = (
        samples
        if samples is not None
        else load_training_manifest(
            config.manifest,
            tasks=config.tasks,
            samples_per_task=config.samples_per_task,
        )
    )
    plan = build_plan(config, samples)
    settings = None
    if config.stage == "flow_grpo":
        from training.optimization.flow_grpo.settings import FlowGRPOSettings
        from training.optimization.flow_grpo.rollout import collect_group
        from training.optimization.flow_grpo.replay import iter_replay_losses
        from training.optimization.rewards import VerifierReward

        settings = FlowGRPOSettings(**config.flow_grpo)
        scorer = reward or VerifierReward()
    root = Path(config.output_dir).expanduser().resolve()
    if accelerator.is_main_process:
        if config.resume:
            if not (root / "plan.json").exists():
                raise ValueError("resume output_dir must contain the original plan")
            _compatible_plan(json.loads((root / "plan.json").read_text()), plan)
        elif root.exists() and any(root.iterdir()):
            raise ValueError("new training output_dir must be empty")
        root.mkdir(parents=True, exist_ok=True)
        if not config.resume:
            write_json(root / "plan.json", plan)
    _barrier(accelerator)
    adapter_config = config.to_dict()
    adapter_config["device"] = str(accelerator.device)
    core = build_adapter(adapter_config)
    # Resume retains the original warm-start identity in the sealed plan,
    # but its model/optimizer/RNG initialization comes from the checkpoint.
    if config.warm_start and not config.resume:
        core.load(Path(config.warm_start))
    optimizer = torch.optim.AdamW(
        core.trainable_parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    model, optimizer = accelerator.prepare(core, optimizer)
    parameter_groups = (
        core.gradient_groups()
        if hasattr(core, "gradient_groups")
        else {"policy": list(core.trainable_parameters())}
    )
    progress = Progress()
    accelerator.register_for_checkpointing(progress)
    if config.resume:
        checkpoint = Path(config.resume).expanduser().resolve()
        if not (checkpoint / "committed.json").is_file():
            raise ValueError("resume checkpoint is incomplete")
        committed = json.loads((checkpoint / "committed.json").read_text())
        if "seed" in committed and committed["seed"] != config.seed:
            raise ValueError("resume requires matching training seed")
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(config.seed + accelerator.process_index)

    # Rollout RNG is part of the same registered recovery state as Torch RNG.
    class RolloutRNG:
        def state_dict(self):
            return {
                "states": gather_object([generator.get_state()]),
                "world_size": accelerator.num_processes,
            }

        def load_state_dict(self, state):
            if state["world_size"] != accelerator.num_processes:
                raise ValueError(
                    "exact rollout RNG recovery requires the original world size"
                )
            generator.set_state(state["states"][accelerator.process_index])

    rollout_rng = RolloutRNG()
    accelerator.register_for_checkpointing(rollout_rng)
    if config.resume:
        # Register every custom object before restoring its state.
        if config.sharding == "fsdp":
            _initialize_fsdp_adam_state(optimizer, model, checkpoint)
            with _strict_fsdp1_checkpoint_loading():
                accelerator.load_state(str(checkpoint))
        else:
            accelerator.load_state(str(checkpoint))

    def checkpoint_now():
        destination = root / "checkpoints" / f"step-{progress.updates:08d}"
        _cuda_checkpoint_boundary(accelerator.device)
        _barrier(accelerator)
        accelerator.save_state(str(destination))
        _barrier(accelerator)
        if accelerator.is_main_process:
            write_json(
                destination / "committed.json",
                {"seed": config.seed, "progress": progress.state_dict()},
            )
            if config.keep_checkpoints:
                committed = sorted(
                    path
                    for path in destination.parent.glob("step-*")
                    if (path / "committed.json").is_file()
                )
                for obsolete in committed[: -config.keep_checkpoints]:
                    shutil.rmtree(obsolete)
        _barrier(accelerator)
        return destination

    def export_weights(destination, committed_checkpoint):
        _cuda_checkpoint_boundary(accelerator.device)
        reuse = (
            config.sharding == "fsdp"
            and config.tuning == "full"
            and callable(getattr(core, "save_checkpoint_weights", None))
        )
        state = None if reuse else accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            if reuse:
                core.save_checkpoint_weights(
                    destination, committed_checkpoint / "pytorch_model_fsdp_0"
                )
            else:
                core.save(destination, state_dict=state)
        _barrier(accelerator)

    last_checkpoint = None
    while progress.epoch < config.epochs and progress.updates < plan["planned_updates"]:
        order = epoch_order(len(samples), config.seed, progress.epoch)
        indices = order[progress.offset : progress.offset + config.batch_samples]
        if not indices:
            progress.epoch += 1
            progress.offset = 0
            continue
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(config, progress.updates, plan["planned_updates"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_value = torch.zeros((), device=accelerator.device)
        metrics = []
        rollouts = []
        for index, real in rank_slots(
            indices, accelerator.process_index, accelerator.num_processes
        ):
            sample = samples[index]
            weight = accelerator.num_processes / len(indices) if real else 0.0
            with torch.no_grad():
                prepared = (
                    core.prepare([sample])
                    if config.stage == "sft"
                    else core.prepare_condition([sample])
                )
            if config.stage == "sft":
                with accelerator.autocast():
                    loss = model(prepared)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite SFT loss for {sample.id}")
                accelerator.backward(loss * weight)
                loss_value += loss.detach() * weight
            else:

                def reward_fn(video, requested_sample):
                    if not real:
                        return 0.0
                    mapping = prepared.mappings
                    if isinstance(mapping, (list, tuple)):
                        mapping = mapping[0] if mapping else None
                    result = scorer.score(requested_sample, video, mapping=mapping)
                    evidence = {
                        key: value
                        for key, value in result.metrics.items()
                        if key
                        in (
                            "training_video_mapping",
                            "whole_video_loaded",
                            "whole_video_process_correct",
                            "video_frame_count",
                            "video_candidate_frame_count",
                            "video_contract",
                            "raw_reward_vector",
                            "failure_reason",
                            "hard_gate_failures",
                            "video_process_reason",
                            "verifier_profile",
                        )
                    }
                    metrics.append(
                        {
                            "sample_id": sample.id,
                            "reward": result.reward,
                            "components": result.components,
                            "generated_frame_count": len(video),
                            "metrics": evidence,
                        }
                    )
                    return result.reward

                model.eval()
                with accelerator.autocast():
                    sigmas = (
                        core.sigma_schedule(settings.rollout_steps, prepared)
                        if hasattr(core, "sigma_schedule")
                        else None
                    )
                    rollout = collect_group(
                        model,
                        sample,
                        reward_fn,
                        settings,
                        prepared=prepared,
                        generator=generator,
                        unwrapped_adapter=core,
                        sigmas=sigmas,
                    )
                rollouts.append((rollout, weight))
        replay_epochs = settings.replay_epochs if settings is not None else 1
        norms = []
        group_norms = {name: 0.0 for name in parameter_groups}
        for replay_epoch in range(replay_epochs):
            if config.stage == "flow_grpo":
                optimizer.zero_grad(set_to_none=True)
                for rollout, weight in rollouts:
                    with accelerator.autocast():
                        for replay in iter_replay_losses(
                            model,
                            rollout,
                            replace(settings, replay_epochs=1),
                            unwrapped_adapter=core,
                        ):
                            if not torch.isfinite(replay.loss):
                                raise FloatingPointError(
                                    f"nonfinite FlowGRPO replay loss for {rollout.sample.id}"
                                )
                            accelerator.backward(replay.loss * replay.weight * weight)
                            loss_value += (
                                replay.loss.detach()
                                * replay.weight
                                * weight
                                / replay_epochs
                            )
            norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if norm is None or not torch.isfinite(norm):
                raise FloatingPointError("missing or nonfinite training gradients")
            norms.append(float(norm))
            for name, parameters in parameter_groups.items():
                pieces = [
                    p.grad.detach().float().square().sum()
                    for p in parameters
                    if p.grad is not None
                ]
                squared = (
                    torch.stack(pieces).sum().to(accelerator.device)
                    if pieces
                    else torch.zeros((), device=accelerator.device)
                )
                squared = accelerator.reduce(
                    squared, reduction="sum" if config.sharding == "fsdp" else "mean"
                )
                group_norms[name] = max(group_norms[name], float(squared.sqrt()))
            if config.stage == "flow_grpo" and float(norm) == 0:
                continue
            optimizer.step()
            if accelerator.optimizer_step_was_skipped:
                raise FloatingPointError("optimizer skipped the update")
            progress.optimizer_steps += 1
        group_updates = {
            name: {"grad_norm": norm} for name, norm in group_norms.items()
        }
        progress.updates += 1
        progress.consumed_samples += len(indices)
        progress.offset += len(indices)
        step_metrics = gather_object(metrics) if config.stage == "flow_grpo" else []
        log = {
            **progress.state_dict(),
            "sample_ids": [samples[i].id for i in indices],
            "loss": float(accelerator.reduce(loss_value, reduction="mean")),
            "learning_rate": lr,
            "grad_norm": max(norms),
            "update_groups": group_updates,
            "rollout_rewards": step_metrics,
        }
        if accelerator.device.type == "cuda":
            log["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                accelerator.device
            )
            log["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
                accelerator.device
            )
        if accelerator.is_main_process:
            with (root / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps(log, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in log.items()
                        if k not in ("sample_ids", "rollout_rewards")
                    }
                ),
                flush=True,
            )
        if progress.updates % config.save_every == 0:
            last_checkpoint = checkpoint_now()
    if (
        last_checkpoint is None
        or last_checkpoint.name != f"step-{progress.updates:08d}"
    ):
        last_checkpoint = checkpoint_now()
    # This export is the weight-only SFT→RL/inference consumer of the final state.
    export_weights(root / "adapter", last_checkpoint)
    return {
        "output_dir": str(root),
        "progress": progress.state_dict(),
        "checkpoint": str(last_checkpoint),
    }

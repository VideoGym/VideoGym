"""Training data export, configuration and SFT/FlowGRPO execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from training.configuration.schema import MODELS, load_config


def _config(args):
    names = (
        "model",
        "stage",
        "tuning",
        "manifest",
        "output_dir",
        "max_updates",
        "sharding",
        "warm_start",
        "resume",
    )
    overrides = {name: getattr(args, name, None) for name in names}
    overrides["pretrained_model_name_or_path"] = getattr(args, "model_path", None)
    return load_config(args.config, overrides)


def _print(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))


def export(args):
    from training.data.manifest import export_training_manifest

    _print(
        export_training_manifest(
            args.manifest,
            args.output,
            tasks=args.tasks,
            samples_per_task=args.samples_per_task,
        )
    )


def plan(args):
    from training.data.manifest import load_training_manifest
    from training.execution.trainer import build_plan, write_json

    config = _config(args)
    samples = load_training_manifest(
        config.manifest, tasks=config.tasks, samples_per_task=config.samples_per_task
    )
    result = build_plan(config, samples)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
    _print(result)


def run(args):
    from training.execution.trainer import train

    _print(train(_config(args)))


def launch(args):
    config = _config(args)
    if args.processes < 1:
        raise ValueError("processes must be positive")
    if args.processes > 1 and config.sharding == "none":
        raise ValueError("choose --sharding ddp or fsdp for multiple workers")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(args.processes),
        "-m",
        "training.cli",
        "run",
        "--config",
        str(Path(args.config).resolve()),
    ]
    for name in (
        "model",
        "stage",
        "tuning",
        "manifest",
        "output_dir",
        "max_updates",
        "sharding",
        "warm_start",
        "resume",
        "model_path",
    ):
        value = getattr(args, name, None)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    if args.dry_run:
        _print({"command": command, "config": config.to_dict(), "executed": False})
        return
    subprocess.run(command, check=True, env=os.environ.copy())


def _experiment_options(parser):
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--model-path")
    parser.add_argument("--stage", choices=("sft", "flow_grpo"))
    parser.add_argument("--tuning", choices=("full", "lora"))
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--sharding", choices=("none", "ddp", "fsdp"))
    parser.add_argument("--warm-start")
    parser.add_argument("--resume")


def build_parser(subparsers):
    parser = subparsers.add_parser(
        "training", help="Image-conditioned SFT and FlowGRPO"
    )
    commands = parser.add_subparsers(dest="training_command", required=True)
    command = commands.add_parser(
        "export-data", help="Validate and export In-Domain training videos"
    )
    command.add_argument("--manifest", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--tasks", nargs="+")
    command.add_argument("--samples-per-task", type=int)
    command.set_defaults(func=export)
    command = commands.add_parser(
        "plan", help="Resolve experiment configuration and selected samples"
    )
    _experiment_options(command)
    command.add_argument("--output", type=Path)
    command.set_defaults(func=plan)
    command = commands.add_parser("run", help="Run in one worker or under torchrun")
    _experiment_options(command)
    command.set_defaults(func=run)
    command = commands.add_parser("launch", help="Launch local distributed training")
    _experiment_options(command)
    command.add_argument("--processes", type=int, default=1)
    command.add_argument("--dry-run", action="store_true")
    command.set_defaults(func=launch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser(subparsers)
    args = parser.parse_args(["training", *sys.argv[1:]])
    args.func(args)


if __name__ == "__main__":
    main()

"""Build and evaluate the configurable VideoGym generalization suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.catalog.index import (
    PROTOCOL,
    TRACKS,
    catalog_summary,
    evaluation_tasks,
)
from evaluation.execution.datasets.generation import generate_dataset
from evaluation.execution.datasets.selection import subset_dataset
from evaluation.execution.datasets.validation import audit_dataset
from evaluation.execution.scoring.candidates import score


def _list_command(args):
    training_tasks = (
        json.loads(args.training_profile.read_text())["task_counts"]
        if args.training_profile
        else None
    )
    cases = evaluation_tasks(args.config, training_tasks=training_tasks)
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "catalog": catalog_summary(cases),
                "tasks": {name: case.to_dict() for name, case in cases.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser(subparsers):
    parser = subparsers.add_parser("benchmark", help="VideoGym generalization evaluation")
    commands = parser.add_subparsers(dest="benchmark_command", required=True)
    from evaluation.cross_benchmark.vbvr_pro.cli import build_parser as build_vbvr_parser

    build_vbvr_parser(commands.add_parser("vbvr", help="VBVR-Pro cross-benchmark evaluation"))
    listing = commands.add_parser("list", help="Print evaluation task definitions")
    listing.add_argument("--config", type=Path)
    listing.add_argument("--training-profile", type=Path)
    listing.set_defaults(func=_list_command)

    catalog = commands.add_parser(
        "check-catalog",
        help="Check configured construction targets and executable routes",
    )
    catalog.add_argument("--config", type=Path)
    catalog.set_defaults(func=check_catalog)

    profile = commands.add_parser(
        "profile", help="Measure actual training membership and source parameters"
    )
    profile.add_argument("--manifest", required=True, type=Path)
    membership = profile.add_mutually_exclusive_group(required=True)
    membership.add_argument(
        "--members-file", type=Path, help="One trained canonical sample ID per line"
    )
    membership.add_argument(
        "--all-training-rows",
        action="store_true",
        help="Every row in this manifest was used for training",
    )
    profile.add_argument("--dataset-name", required=True)
    profile.add_argument("--output", required=True, type=Path)
    profile.set_defaults(func=profile_command)

    generate = commands.add_parser(
        "generate", help="Generate evaluation questions against actual training coverage"
    )
    generate.add_argument("--output-root", required=True)
    generate.add_argument("--training-profile", required=True)
    generate.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Task selection configuration",
    )
    generate.add_argument("--tasks", nargs="+")
    generate.add_argument("--tracks", nargs="+", choices=TRACKS)
    generate.add_argument("--families", nargs="+")
    generate.add_argument("--count-per-task", type=int)
    generate.add_argument("--workers", type=int)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--max-attempts", type=int)
    generate.add_argument(
        "--extend-from",
        type=Path,
        help="Retain every question from an existing dataset and fill larger task budgets",
    )
    generate.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed task/tier shards with the same seed and training membership",
    )
    generate.set_defaults(
        func=lambda args: print(json.dumps(generate_dataset(args), indent=2))
    )

    subset = commands.add_parser(
        "subset", help="Select a deterministic task-total subset"
    )
    subset.add_argument("--dataset", required=True, type=Path)
    subset.add_argument("--output-root", required=True, type=Path)
    subset.add_argument("--count-per-task", type=int, default=5)
    subset.add_argument("--tasks", nargs="+", help="Select tasks from the dataset")
    subset.set_defaults(
        func=lambda args: print(
            json.dumps(
                subset_dataset(
                    args.dataset, args.output_root, args.count_per_task, tasks=args.tasks
                ),
                indent=2,
            )
        )
    )

    audit = commands.add_parser(
        "audit", help="Replay parameters and check training overlap"
    )
    audit.add_argument("--dataset", required=True)
    audit.add_argument("--training-profile", type=Path)
    audit.set_defaults(
        func=lambda args: print(
            json.dumps(
                audit_dataset(
                    Path(args.dataset).expanduser().resolve(),
                    training_profile=args.training_profile,
                ),
                indent=2,
            )
        )
    )

    evaluate = commands.add_parser(
        "score", help="Score model videos or calibrate reference videos"
    )
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--training-profile", type=Path)
    evaluate.add_argument("--tasks", nargs="+")
    evaluate.add_argument("--families", nargs="+")
    evaluate.add_argument("--tracks", nargs="+", choices=TRACKS, default=list(TRACKS))
    evaluate.add_argument(
        "--sample-ids-file",
        type=Path,
        help="Score only these sample IDs (one per line), after task/track/family filters",
    )
    candidate = evaluate.add_mutually_exclusive_group(required=True)
    candidate.add_argument(
        "--predictions-root",
        help="Directory containing <sample-id>/prediction.webm or mp4",
    )
    candidate.add_argument(
        "--reference",
        action="store_true",
        help="Score reference videos directly",
    )
    evaluate.add_argument(
        "--model", help="Checkpoint identity of the model being evaluated"
    )
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--bootstrap-samples", type=int, default=1000)
    evaluate.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU scoring processes; each keeps one OpenCV thread",
    )
    evaluate.set_defaults(func=lambda args: print(json.dumps(score(args), indent=2)))


def check_catalog(args):
    from data.generate import load_generator_class
    from data.registry import get_task_spec
    from evaluation.catalog.contracts import build_base_catalog

    build_base_catalog(args.config)
    cases = evaluation_tasks(args.config)
    result = catalog_summary(cases)
    for case in cases.values():
        load_generator_class(get_task_spec(case.task_name))
        if case.track in ("composition", "cross_task"):
            if not case.novelty:
                raise ValueError(f"{case.name}: missing novelty specification")
            if case.track == "composition" and not case.primitives:
                raise ValueError(f"{case.name}: missing composition source tasks")
    print(json.dumps({"passed": True, **result}, ensure_ascii=False, indent=2))


def profile_command(args):
    from evaluation.execution.datasets.training_exposure import profile_training

    members = (
        set(args.members_file.read_text().splitlines()) if args.members_file else None
    )
    print(
        json.dumps(
            profile_training(
                args.manifest,
                args.output,
                members=members,
                all_training_rows=args.all_training_rows,
                dataset_name=args.dataset_name,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    build_parser(parser.add_subparsers(required=True))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

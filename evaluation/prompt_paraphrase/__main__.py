"""Prepare, qualify, and score Text Prompt Paraphrase and Vision Prompt Paraphrase."""

import argparse
import json
from pathlib import Path

from evaluation.prompt_paraphrase.text_paraphrase.variants import compare, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare-text")
    preparation.add_argument("--dataset", type=Path, required=True)
    preparation.add_argument("--output-root", type=Path, required=True)
    preparation.add_argument(
        "--rewrites",
        type=Path,
        required=True,
        help="Source-bound rewrite corpus JSONL",
    )
    for name in ("qualify-text", "score-text", "rescore-text-baseline", "merge-text"):
        text = commands.add_parser(name)
        text.add_argument("--dataset", type=Path, required=True)
        text.add_argument("--output-root", type=Path, required=True)
        if name in {"qualify-text", "score-text", "rescore-text-baseline"}:
            text.add_argument("--workers", type=int, default=1)
        if name == "rescore-text-baseline":
            source = text.add_mutually_exclusive_group(required=True)
            source.add_argument(
                "--predictions-root",
                type=Path,
                help="Directory containing In-Domain <sample-id>/prediction.webm or mp4",
            )
            source.add_argument(
                "--baseline-scores",
                type=Path,
                help="Existing In-Domain scores.jsonl referencing candidate videos",
            )
            text.add_argument("--model", required=True)
        if name == "score-text":
            text.add_argument("--predictions-root", type=Path, required=True)
            text.add_argument("--model", required=True)
            text.add_argument("--shard-index", type=int, default=0)
            text.add_argument("--shards", type=int, default=1)
        if name == "merge-text":
            text.add_argument("--shard-roots", nargs="+", type=Path, required=True)
            text.add_argument("--id-scores", type=Path, required=True)
    visual = commands.add_parser("prepare-visual")
    visual.add_argument("--dataset", type=Path, required=True)
    visual.add_argument("--output-root", type=Path, required=True)
    visual.add_argument("--workers", type=int, default=1)
    visual.add_argument("--seed", type=int, default=20261027)
    for command in ("qualify-visual", "score-visual", "rescore-visual-baseline", "merge-visual"):
        scoring = commands.add_parser(command)
        scoring.add_argument("--dataset", type=Path, required=True)
        scoring.add_argument("--output-root", type=Path, required=True)
        if command != "merge-visual":
            scoring.add_argument("--workers", type=int, default=1)
        if command == "rescore-visual-baseline":
            source = scoring.add_mutually_exclusive_group(required=True)
            source.add_argument(
                "--predictions-root",
                type=Path,
                help="Directory containing In-Domain <sample-id>/prediction.webm or mp4",
            )
            source.add_argument(
                "--baseline-scores",
                type=Path,
                help="Existing In-Domain scores.jsonl referencing candidate videos",
            )
            scoring.add_argument("--model", required=True)
        if command == "merge-visual":
            scoring.add_argument("--shard-roots", nargs="+", type=Path, required=True)
            scoring.add_argument("--id-scores", type=Path, required=True)
        if command == "score-visual":
            scoring.add_argument("--predictions-root", type=Path, required=True)
            scoring.add_argument("--model", required=True)
            scoring.add_argument("--shard-index", type=int, default=0)
            scoring.add_argument("--shards", type=int, default=1)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--id-scores", type=Path, required=True)
    comparison.add_argument("--variant-scores", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-text":
        result = prepare(args.dataset, args.output_root, args.rewrites)
    elif args.command in {"qualify-text", "score-text", "rescore-text-baseline", "merge-text"}:
        from evaluation.prompt_paraphrase.text_paraphrase import scoring

        if args.command == "qualify-text":
            result = scoring.calibrate(args.dataset, args.output_root, args.workers)
        elif args.command == "rescore-text-baseline":
            result = scoring.rescore_baseline(
                args.dataset,
                args.output_root,
                args.model,
                args.workers,
                predictions_root=args.predictions_root,
                baseline_scores=args.baseline_scores,
            )
        elif args.command == "score-text":
            result = scoring.score(
                args.dataset,
                args.output_root,
                args.predictions_root,
                args.model,
                args.workers,
                args.shard_index,
                args.shards,
            )
        else:
            result = scoring.merge(args.dataset, args.shard_roots, args.output_root, args.id_scores)
    elif args.command == "prepare-visual":
        from evaluation.prompt_paraphrase.vision_prompt_paraphrase.dataset import prepare_visual

        result = prepare_visual(args.dataset, args.output_root, seed=args.seed, workers=args.workers)
    elif args.command in {"qualify-visual", "score-visual", "rescore-visual-baseline", "merge-visual"}:
        from evaluation.prompt_paraphrase.vision_prompt_paraphrase import scoring

        if args.command == "qualify-visual":
            result = scoring.calibrate(args.dataset, args.output_root, args.workers)
        elif args.command == "rescore-visual-baseline":
            result = scoring.rescore_baseline(
                args.dataset,
                args.output_root,
                args.model,
                args.workers,
                predictions_root=args.predictions_root,
                baseline_scores=args.baseline_scores,
            )
        elif args.command == "merge-visual":
            result = scoring.merge(args.dataset, args.shard_roots, args.output_root, args.id_scores)
        else:
            result = scoring.score(
                args.dataset,
                args.output_root,
                args.predictions_root,
                args.model,
                args.workers,
                args.shard_index,
                args.shards,
            )
    else:
        result = compare(args.id_scores, args.variant_scores)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

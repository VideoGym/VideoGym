import argparse
import json
from pathlib import Path

from evaluation.cross_benchmark.vbvr_pro.inputs import prepare_vbvr_pro
from evaluation.cross_benchmark.vbvr_pro.scoring import score_vbvr_pro


def build_parser(parser):
    sub = parser.add_subparsers(dest="vbvr_command", required=True)
    download = sub.add_parser("download", help="Download the pinned VBVR-Pro Bench dataset")
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--endpoint", default="https://huggingface.co")
    prepare = sub.add_parser("prepare", help="Export public inputs and Reference Length timing")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    score = sub.add_parser("score", help="Score rollouts under deterministic full-video verification")
    for name in ("dataset", "predictions", "output"):
        score.add_argument(f"--{name}", type=Path, required=True)
    score.add_argument("--workers", type=int, required=True)
    controls = sub.add_parser(
        "controls", help="Verify reference videos and static-input negatives on VBVR-Pro Bench"
    )
    controls.add_argument("--gt-root", type=Path, required=True)
    controls.add_argument("--output", type=Path, required=True)
    controls.add_argument("--workers", type=int, default=1)
    continuous = sub.add_parser(
        "score-continuous",
        help="Run the upstream VBVR-Pro continuous partial-credit evaluator on scored rollouts",
    )
    continuous.add_argument("--strict", type=Path, required=True)
    continuous.add_argument("--upstream", type=Path, required=True)
    continuous.add_argument("--gt-root", type=Path, required=True)
    continuous.add_argument("--output", type=Path, required=True)
    continuous.add_argument("--workers", type=int, default=1)
    parser.set_defaults(func=run)
    return parser


def run(args):
    cmd = args.vbvr_command
    if cmd == "download":
        from evaluation.cross_benchmark.vbvr_pro.benchmark.download import prepare_dataset

        result = {"manifest": str(prepare_dataset(args.output, args.endpoint))}
        print(json.dumps(result))
    elif cmd == "prepare":
        print(json.dumps(prepare_vbvr_pro(args.manifest, args.output_root)))
    elif cmd == "controls":
        from evaluation.cross_benchmark.vbvr_pro.controls import run_controls

        code = run_controls(args.gt_root, args.output, args.workers)
        if code:
            raise SystemExit(code)
    elif cmd == "score-continuous":
        from evaluation.cross_benchmark.vbvr_pro.continuous import run_continuous

        run_continuous(args.strict, args.upstream, args.gt_root, args.output, args.workers)
    else:
        if args.workers < 1:
            raise ValueError("workers must be positive")
        print(
            json.dumps(
                score_vbvr_pro(
                    args.dataset,
                    args.predictions,
                    args.output,
                    args.workers,
                )
            )
        )


def main():
    parser = build_parser(
        argparse.ArgumentParser(description="VBVR-Pro cross-benchmark evaluation")
    )
    run(parser.parse_args())

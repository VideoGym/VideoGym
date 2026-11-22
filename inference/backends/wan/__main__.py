"""Generate pass@1 rollouts with Wan2.2 base or VBVR-Pro post-trained models."""

from __future__ import annotations

import argparse
from pathlib import Path

from inference.backends.wan.model import MODEL_RELEASES
from inference.backends.wan.sampler import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_RELEASES), default="a14b")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument(
        "--frames", type=int, help="Fixed Length frame count at 16 FPS (e.g. 81)"
    )
    timing.add_argument(
        "--reference-timing", type=Path, help="Reference Length timing-only JSONL sidecar"
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offload", choices=("none", "model"), default="none")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    if args.max_sequence_length is not None and args.max_sequence_length < 1:
        parser.error("max-sequence-length must be positive")
    run(args)


if __name__ == "__main__":
    main()

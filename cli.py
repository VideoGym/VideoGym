from __future__ import annotations

import argparse

from data import generate as data_generate
from evaluation import cli as benchmark
from training import cli as training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VideoGym: procedural task synthesis, post-training, full-video verification, and seven-level generalization evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="Procedural task synthesis")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_generate.build_parser(data_subparsers)
    benchmark.build_parser(subparsers)
    training.build_parser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

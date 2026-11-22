"""Publish paired continuous scores and full-video verification verdicts across models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib
from typing import Sequence

from evaluation.cross_benchmark.vbvr_pro.pair_continuous import compare
from evaluation.cross_benchmark.vbvr_pro.reports import read_jsonl


def parse_pairs(raw_pairs: Sequence[Sequence[str]]) -> list[tuple[str, Path, Path]]:
    entries: list[tuple[str, Path, Path]] = []
    seen: set[str] = set()
    for item in raw_pairs:
        if len(item) != 3:
            raise ValueError("Each --pair requires MODEL CONTINUOUS_JSON STRICT_JSONL")
        model, continuous_path, strict_path = item
        if not model or model in seen:
            raise ValueError(f"Empty or duplicate model identifier: {model!r}")
        seen.add(model)
        entries.append((model, Path(continuous_path), Path(strict_path)))
    if not entries:
        raise ValueError("At least one --pair entry is required")
    return entries


def summarize(rows):
    incorrect = [r for r in rows if r["strict_binary_score"] == 0]
    return {
        "samples": len(rows),
        "continuous_mean": sum(r["continuous_score"] for r in rows) / len(rows),
        "strict_correct": sum(r["strict_binary_score"] for r in rows),
        "strict_pass_rate": sum(r["strict_binary_score"] for r in rows) / len(rows),
        "strict_incorrect": len(incorrect),
        "continuous_mean_on_strict_incorrect": (
            sum(r["continuous_score"] for r in incorrect) / len(incorrect)
            if incorrect
            else None
        ),
        "continuous_ge_0_8_and_strict_incorrect": sum(
            r["continuous_score"] >= 0.8 for r in incorrect
        ),
    }


def paired_rows(entries: Sequence[tuple[str, Path, Path]]) -> dict[str, list[dict]]:
    if not entries:
        raise ValueError("At least one (model, continuous_path, strict_path) entry is required")
    catalog_path = Path(__file__).with_name("relations.toml")
    catalog = tomllib.loads(catalog_path.read_text())
    categories = {r["task"]: r["category"] for r in catalog["tasks"]}
    by_model: dict[str, list[dict]] = {}
    for model, continuous_path, strict_path in entries:
        if model in by_model:
            raise ValueError(f"Duplicate model identifier: {model}")
        continuous = json.loads(Path(continuous_path).read_text())
        strict = read_jsonl(Path(strict_path))
        paired = compare(continuous, strict)
        if not paired["summary"]["complete"]:
            raise ValueError(f"Unresolved comparison: {model}")
        cont_by_id = {
            f"{r['folder']}/{r['task_name']}/{Path(r['video_file']).stem}": r
            for r in continuous["samples"]
        }
        samples = paired["samples"]
        for row in samples:
            sid = row["sample_id"]
            row["track"] = categories.get(sid.split("/")[1], "id")
            row["continuous_dimensions"] = cont_by_id[sid]["dimensions"]
        by_model[model] = samples
    return by_model


def publish(entries: Sequence[tuple[str, Path, Path]], output: Path) -> dict:
    by_model = paired_rows(entries)
    cells = []
    for model, samples in by_model.items():
        cells.append(
            {
                "model": model,
                "protocol": "reference_length",
                "overall": summarize(samples),
                "native_id": summarize([r for r in samples if r["track"] == "id"]),
                "native_ood": summarize([r for r in samples if r["track"] != "id"]),
                "by_generalization": {
                    key: summarize([r for r in samples if r["track"] == key])
                    for key in (
                        "prompt_paraphrase",
                        "extrapolation",
                        "composition",
                        "cross_task",
                    )
                },
            }
        )
    result = {
        "schema": "videogym.paired-continuous-strict",
        "complete": True,
        "samples": sum(c["overall"]["samples"] for c in cells),
        "cells": cells,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        required=True,
        metavar=("MODEL", "CONTINUOUS_JSON", "STRICT_JSONL"),
        help="Model name, upstream continuous score JSON, and full-video scores.jsonl",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    publish(parse_pairs(args.pair), args.output)


if __name__ == "__main__":
    main()

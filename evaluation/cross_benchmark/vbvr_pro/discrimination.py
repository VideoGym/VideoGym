"""Measure whether continuous partial-credit scores distinguish valid solutions from failures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Sequence

from evaluation.cross_benchmark.vbvr_pro.comparison import paired_rows, parse_pairs

THRESHOLDS = (0.8, 0.9, 0.95, 0.99, 1.0)


def discrimination(rows, label="strict_binary_score"):
    resolved = [r for r in rows if r[label] is not None]
    positive = [r["continuous_score"] for r in resolved if r[label] == 1]
    negative = [r["continuous_score"] for r in resolved if r[label] == 0]
    pairs = len(positive) * len(negative)
    thresholds = []
    for threshold in THRESHOLDS:
        correct = sum(s >= threshold for s in positive)
        incorrect = sum(s >= threshold for s in negative)
        total = correct + incorrect
        thresholds.append(
            {
                "threshold": threshold,
                "correct": correct,
                "incorrect": incorrect,
                "selected": total,
                "incorrect_share": incorrect / total if total else None,
                "correct_recall": correct / len(positive) if positive else None,
            }
        )
    return {
        "samples": len(rows),
        "resolved": len(resolved),
        "correct": len(positive),
        "incorrect": len(negative),
        "mean_correct": mean(positive) if positive else None,
        "mean_incorrect": mean(negative) if negative else None,
        "auroc": (
            sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / pairs
            if pairs
            else None
        ),
        "thresholds": thresholds,
        "histogram": [
            {
                "lower": i / 20,
                "upper": (i + 1) / 20,
                "correct": sum(
                    i / 20 <= s < (i + 1) / 20 or i == 19 and s == 1 for s in positive
                ),
                "incorrect": sum(
                    i / 20 <= s < (i + 1) / 20 or i == 19 and s == 1 for s in negative
                ),
            }
            for i in range(20)
        ],
    }


def question_pairs(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    pairs = []
    for sid, group in sorted(grouped.items()):
        for correct in (r for r in group if r["strict_binary_score"] == 1):
            for incorrect in (r for r in group if r["strict_binary_score"] == 0):
                p, n = (
                    correct["continuous_score"],
                    incorrect["continuous_score"],
                )
                pairs.append(
                    {
                        "sample_id": sid,
                        "correct_model": correct["model"],
                        "incorrect_model": incorrect["model"],
                        "correct_reward": p,
                        "incorrect_reward": n,
                        "gap": p - n,
                    }
                )
    return {
        "pairs": len(pairs),
        "questions": len({r["sample_id"] for r in pairs}),
        "inversions": sum(r["gap"] < 0 for r in pairs),
        "ties": sum(r["gap"] == 0 for r in pairs),
        "within_five_points": sum(abs(r["gap"]) <= 0.05 for r in pairs),
        "both_perfect": sum(
            r["correct_reward"] == r["incorrect_reward"] == 1 for r in pairs
        ),
        "pairwise_auc": (
            sum((r["gap"] > 0) + 0.5 * (r["gap"] == 0) for r in pairs) / len(pairs)
            if pairs
            else None
        ),
        "nonpositive_gap_pairs": [r for r in pairs if r["gap"] <= 0],
    }


def build(
    entries: Sequence[tuple[str, Path, Path]],
    output: Path,
) -> dict:
    by_model = paired_rows(entries)
    rows = [dict(row, model=model) for model, items in by_model.items() for row in items]
    result = {
        "schema": "videogym.scorer-discrimination",
        "overall": discrimination(rows),
        "by_model": {
            model: discrimination([r for r in rows if r["model"] == model])
            for model in by_model
        },
        "by_split": {
            k: discrimination([r for r in rows if (r["track"] == "id") == (k == "id")])
            for k in ("id", "ood")
        },
        "same_question": question_pairs(rows),
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(parse_pairs(args.pair), args.output)


if __name__ == "__main__":
    main()

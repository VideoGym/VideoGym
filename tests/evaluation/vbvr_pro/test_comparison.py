"""Official pairing, diagnostic scope and coverage preserve metric meaning."""

import pytest

from evaluation.cross_benchmark.vbvr_pro.benchmark.catalog import catalog
from evaluation.cross_benchmark.vbvr_pro import pair_continuous as comparison


def inputs():
    identities = [(task, folder, instance) for task, folder in catalog().items() for instance in range(5)]
    continuous = {"samples": []}
    strict = []
    for task, folder, instance in identities:
        continuous["samples"].append(
            {"folder": folder, "task_name": task, "video_file": f"{instance:05d}.mp4", "score": 0.9}
        )
        strict.append(
            {
                "sample_id": f"{folder}/{task}/{instance:05d}",
                "status": "incorrect",
                "score": 0,
                "checks": {"correct_order": False},
            }
        )
    return continuous, strict


def test_official_comparison_requires_all_frozen_identities_and_pairs_by_identity():
    continuous, strict = inputs()
    continuous["samples"][0]["score"] = 1.0
    strict[0].update(status="correct", score=1, checks={"correct_order": True})
    correct_id = strict[0]["sample_id"]
    strict.reverse()
    result = comparison.compare(continuous, strict)
    summary = result["summary"]
    assert summary["scope"] == "official_500"
    assert summary["num_samples"] == 500
    assert summary["continuous_mean"] == pytest.approx((499 * 0.9 + 1) / 500)
    assert summary["strict_binary_pass_rate"] == 1 / 500
    assert summary["complete"]
    pair = next(row for row in result["samples"] if row["sample_id"] == correct_id)
    assert pair["continuous_score"] == 1
    assert pair["strict_status"] == "correct"
    assert pair["strict_binary_score"] == 1
    split = correct_id.split("/")[0]
    assert result["by_split"][split]["strict_correct"] == 1
    assert result["by_split"][split]["continuous_ge_0_8_and_strict_incorrect"] == 249


def test_comparison_preserves_missing_denominator_and_unresolved_bounds():
    continuous, strict = inputs()
    statuses = [("correct", 1), ("incorrect", 0), ("missing", 0), ("error", None), ("unverified", None)]
    continuous["samples"] = continuous["samples"][: len(statuses)]
    strict = strict[: len(statuses)]
    for row, (status, score) in zip(strict, statuses):
        row.update(status=status, score=score)
    summary = comparison.compare(continuous, strict, diagnostic_subset=True)["summary"]
    assert summary["num_samples"] == 5
    assert summary["counts"] == dict.fromkeys((status for status, _ in statuses), 1)
    assert summary["num_missing"] == 1
    assert summary["num_unresolved"] == 2
    assert summary["coverage"] == 0.6
    assert summary["strict_binary_pass_rate"] is None
    assert summary["strict_binary_pass_rate_bounds"] == [0.2, 0.6]
    assert summary["strict_resolved_binary_pass_rate"] == pytest.approx(1 / 3)
    assert not summary["complete"]
    summary = comparison.compare({"samples": continuous["samples"][:3]}, strict[:3], diagnostic_subset=True)["summary"]
    assert summary["strict_binary_pass_rate"] == pytest.approx(1 / 3)
    assert summary["num_missing"] == 1
    assert not summary["complete"]

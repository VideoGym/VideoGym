"""Pair official continuous scores and full-video verification verdicts by sample identity."""

from pathlib import Path

from evaluation.cross_benchmark.vbvr_pro.benchmark.catalog import catalog
from evaluation.cross_benchmark.vbvr_pro.reports import summarize


def compare(continuous, strict, *, diagnostic_subset=False):
    continuous_by_id = {}
    for row in continuous["samples"]:
        identity = f"{row['folder']}/{row['task_name']}/{Path(row['video_file']).stem}"
        if identity in continuous_by_id:
            raise ValueError(f"Duplicate continuous sample: {identity}")
        continuous_by_id[identity] = row
    strict_by_id = {}
    for row in strict:
        identity = row["sample_id"]
        if identity in strict_by_id:
            raise ValueError(f"Duplicate strict sample: {identity}")
        strict_by_id[identity] = row
    if not continuous_by_id or continuous_by_id.keys() != strict_by_id.keys():
        raise ValueError("Comparison requires exactly the same nonempty sample set")
    official_ids = {f"{folder}/{task}/{instance:05d}" for task, folder in catalog().items() for instance in range(5)}
    if strict_by_id.keys() - official_ids:
        raise ValueError("Comparison contains identities outside the official 500")
    if not diagnostic_subset and strict_by_id.keys() != official_ids:
        raise ValueError("Comparison requires the official 500; use diagnostic_subset=True for a subset")
    pairs = []
    for identity in sorted(continuous_by_id):
        cont, full = continuous_by_id[identity], strict_by_id[identity]
        if cont.get("error"):
            raise ValueError(f"Continuous execution failed for {identity}: {cont['error']}")
        score = cont["score"]
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError(f"Continuous score must be finite and bounded: {identity}")
        pairs.append(
            {
                "sample_id": identity,
                "continuous_score": score,
                "strict_status": full["status"],
                "strict_binary_score": full["score"],
                "failed_obligations": [name for name, passed in full["checks"].items() if not passed],
            }
        )
    strict_summary = summarize(list(strict_by_id.values()))
    summary = {
        "scope": "diagnostic_subset" if diagnostic_subset else "official_500",
        **strict_summary,
        "continuous_mean": sum(p["continuous_score"] for p in pairs) / len(pairs),
    }
    for name in ("binary_pass_rate", "binary_pass_rate_bounds", "resolved_binary_pass_rate"):
        summary[f"strict_{name}"] = summary.pop(name)
    review = sorted(
        (p for p in pairs if p["strict_status"] == "incorrect"),
        key=lambda p: p["continuous_score"],
        reverse=True,
    )
    by_split = {}
    for split in sorted({p["sample_id"].split("/")[0] for p in pairs}):
        selected = [p for p in pairs if p["sample_id"].split("/")[0] == split]
        incorrect = [p for p in selected if p["strict_status"] == "incorrect"]
        by_split[split] = {
            "samples": len(selected),
            "continuous_mean": sum(p["continuous_score"] for p in selected) / len(selected),
            "strict_correct": sum(p["strict_status"] == "correct" for p in selected),
            "strict_incorrect": len(incorrect),
            "continuous_mean_on_strict_incorrect": (
                sum(p["continuous_score"] for p in incorrect) / len(incorrect) if incorrect else None
            ),
            "continuous_ge_0_8_and_strict_incorrect": sum(p["continuous_score"] >= 0.8 for p in incorrect),
        }
    return {"summary": summary, "by_split": by_split, "samples": pairs, "review_priority": review}

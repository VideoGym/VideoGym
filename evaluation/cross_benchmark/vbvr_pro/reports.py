from collections import Counter, defaultdict
import json
from pathlib import Path
import tomllib
import numpy as np


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(
            value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_value
        )
        + "\n"
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, allow_nan=False, default=_json_value
                )
                + "\n"
            )
            stream.flush()


def group_vbvr_pro_generalization(results):
    catalog = tomllib.loads((Path(__file__).parent / "relations.toml").read_text())
    relations = {row["task"]: row["category"] for row in catalog["tasks"]}
    ood = [row for row in results if row["split"] == "Out_of_Domain"]
    if {row["task"] for row in ood} != relations.keys():
        raise ValueError("OOD results do not cover the 50 classified tasks")
    groups = {name: [] for name in ("prompt_paraphrase", "extrapolation", "composition", "cross_task")}
    for row in ood:
        groups[relations[row["task"]]].append(row)
    return groups


def summarize_vbvr_pro(results):
    summary = report(results)
    groups = group_vbvr_pro_generalization(results)
    summary["by_generalization"] = {
        name: summarize(rows) for name, rows in groups.items()
    }
    return summary


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    counts = Counter(row["status"] for row in rows)
    n = len(rows)
    correct = counts["correct"]
    unresolved = counts["error"] + counts["unverified"]
    resolved = n - unresolved
    return {
        "num_samples": n,
        "counts": dict(counts),
        "num_correct": correct,
        "num_unresolved": unresolved,
        "num_missing": counts["missing"],
        "coverage": resolved / n,
        "binary_pass_rate": correct / n if unresolved == 0 else None,
        "binary_pass_rate_bounds": [correct / n, (correct + unresolved) / n],
        "resolved_binary_pass_rate": correct / resolved if resolved else None,
        "complete": unresolved == 0 and counts["missing"] == 0,
    }


def report(rows: list[dict]) -> dict:
    groups = {"by_split": defaultdict(list), "by_task": defaultdict(list)}
    for row in rows:
        groups["by_split"][row["split"]].append(row)
        groups["by_task"][row["task"]].append(row)
    return {
        "overall": summarize(rows),
        **{
            category: {key: summarize(value) for key, value in sorted(group.items())}
            for category, group in groups.items()
        },
    }

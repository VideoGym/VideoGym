"""Execution-group membership and score aggregation helpers."""

from __future__ import annotations

import json
from pathlib import Path

_GROUPS_PATH = Path(__file__).resolve().parents[1] / "data/common/task_groups.json"
MEMBERS = json.loads(_GROUPS_PATH.read_text())
GROUP_BY_TASK = {task: group for group, tasks in MEMBERS.items() for task in tasks}


def task_group(key: str) -> str:
    """Resolve an evaluation or source task key to its execution group."""
    if key in GROUP_BY_TASK:
        return GROUP_BY_TASK[key]
    for prefix in ("id_", "ext_", "mechanism_"):
        if key.startswith(prefix):
            return GROUP_BY_TASK[key[len(prefix) :]]
    if key.endswith("_composition"):
        return GROUP_BY_TASK["eval_" + key.removesuffix("_composition")]
    return GROUP_BY_TASK["eval_cross_" + key]


def _summarize(values):
    samples = sum(v["samples"] for v in values)
    correct = sum(v["correct"] for v in values)
    return {
        "tasks": len(values),
        "samples": samples,
        "correct": correct,
        "rate": 100 * correct / samples if samples else None,
        "histogram": [sum(v["correct"] == i for v in values)
                      for i in range(max((v["samples"] for v in values), default=5) + 1)],
    }


def grouped_scores(task_scores):
    """Aggregate per-task scores into Single-Step Group and Multi-Step Group summaries."""
    return {
        group: _summarize([v for k, v in task_scores.items() if task_group(k) == group])
        for group in MEMBERS
    }


def grouped_pairs(rows):
    """Aggregate paired source-to-target rows by execution group."""
    output = {}
    for group in MEMBERS:
        selected = [r for r in rows if r["group"] == group]
        n = sum(r["samples"] for r in selected)
        source = sum(r["source_correct"] for r in selected)
        correct = sum(r["correct"] for r in selected)
        output[group] = {
            "tasks": len(selected),
            "samples": n,
            "source_correct": source,
            "correct": correct,
            "source_rate": 100 * source / n if n else None,
            "rate": 100 * correct / n if n else None,
        }
    return output

from __future__ import annotations
import random
from collections import defaultdict

DIMENSIONS = ("whole_video_process", "video_final_state", "media_contract")


def _summary(rows, bootstrap_samples=1000):
    if not rows:
        raise ValueError("cannot report an empty evaluation")
    groups = defaultdict(list)
    for row in rows:
        groups[row["evaluation_task"]].append(row)
    means = {
        task: sum(r["success"] for r in values) / len(values)
        for task, values in groups.items()
    }
    clusters = {}
    for task, values in groups.items():
        scenes = defaultdict(list)
        for row in values:
            scenes[row["scene_group"]].append(row["success"])
        clusters[task] = list(scenes.values())
    rng = random.Random(0)
    draws = []
    for _ in range(bootstrap_samples):
        task_draws = []
        for scenes in clusters.values():
            sample = [v for scene in rng.choices(scenes, k=len(scenes)) for v in scene]
            task_draws.append(sum(sample) / len(sample))
        draws.append(sum(task_draws) / len(task_draws))
    draws.sort()
    available = {
        key: [row for row in rows if row.get("reward_available", {}).get(key, True)]
        for key in DIMENSIONS
    }
    final_rows = available["video_final_state"]
    return {
        "samples": len(rows),
        "scene_groups": sum(map(len, clusters.values())),
        "success_rate": sum(r["success"] for r in rows) / len(rows),
        "task_macro_success": sum(means.values()) / len(means),
        "task_macro_cluster_bootstrap_95ci": (
            [
                draws[int(0.025 * len(draws))],
                draws[min(len(draws) - 1, int(0.975 * len(draws)))],
            ]
            if draws
            else None
        ),
        "per_task_success": means,
        "dimension_pass_rates": {
            key: (
                sum(r["rewards"][key] == 1 for r in values) / len(values)
                if values
                else None
            )
            for key, values in available.items()
        },
        "dimension_evaluated_samples": {
            key: len(values) for key, values in available.items()
        },
        "final_correct_process_wrong": (
            sum(
                r["rewards"]["video_final_state"] == 1
                and r["rewards"]["whole_video_process"] != 1
                for r in final_rows
            )
            / len(final_rows)
            if final_rows
            else None
        ),
    }


def summarize(rows, bootstrap_samples=1000):
    tracks, tasks, tiers = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        tracks[row["track"]].append(row)
        tasks[row["evaluation_task"]].append(row)
        tiers[(row["evaluation_task"], row.get("difficulty_tier", "default"))].append(
            row
        )
    return {
        "overall": _summary(rows, bootstrap_samples) if rows else None,
        "tracks": {k: _summary(v, bootstrap_samples) for k, v in tracks.items()},
        "tasks": {k: _summary(v, bootstrap_samples) for k, v in tasks.items()},
        "difficulty_tiers": {
            task: {
                tier: _summary(values, bootstrap_samples)
                for (name, tier), values in tiers.items()
                if name == task
            }
            for task in tasks
        },
    }

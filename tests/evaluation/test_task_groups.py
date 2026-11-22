import json
from collections import Counter
from pathlib import Path

from evaluation.task_groups import grouped_scores, task_group
from data.common.task_semantics import TASK_GROUP_BY_TASK, get_task_group

ROOT = Path(__file__).resolve().parents[2]


def test_group_histogram_supports_twenty_questions_per_task():
    scores = {
        "id_midpoint": {"samples": 20, "correct": 18},
        "polygon_color_composition": {"samples": 20, "correct": 7},
    }
    groups = grouped_scores(scores)
    assert sum(sum(group["histogram"]) for group in groups.values()) == 2
    assert sum(sum(i * count for i, count in enumerate(group["histogram"]))
               for group in groups.values()) == 25


def test_grouping_conserves_scores_across_catalog_and_external_splits():
    task_scores = {
        task: {"samples": 20, "correct": i % 21}
        for i, task in enumerate(TASK_GROUP_BY_TASK)
    }
    groups = grouped_scores(task_scores)
    assert sum(g["samples"] for g in groups.values()) == 20 * len(task_scores)
    assert sum(g["correct"] for g in groups.values()) == sum(
        score["correct"] for score in task_scores.values()
    )
    tasks = json.loads(
        (ROOT / "evaluation/cross_benchmark/vbvr_pro/benchmark/tasks.json").read_text()
    )
    assert len(tasks) == 100
    assert Counter((split, get_task_group(task)) for task, split in tasks.items()) == {
        ("In-Domain_50", "single_step"): 26,
        ("In-Domain_50", "multi_step"): 24,
        ("Out-of-Domain_50", "single_step"): 41,
        ("Out-of-Domain_50", "multi_step"): 9,
    }
    members = json.loads((ROOT / "data/common/task_groups.json").read_text())
    assert not set(members["single_step"]) & set(members["multi_step"])
    comp_single = [t for t in members["single_step"] if t.startswith("eval_") and not t.startswith("eval_cross_")]
    comp_multi = [t for t in members["multi_step"] if t.startswith("eval_") and not t.startswith("eval_cross_")]
    assert (len(comp_single), len(comp_multi)) == (14, 20)


def test_extrapolation_uses_target_contract_instead_of_source_group():
    assert task_group("ray_reflect") == "single_step"
    assert task_group("ext_ray_reflect") == "multi_step"
    assert get_task_group("ray_reflect", evaluation_task="ext_ray_reflect") == "multi_step"
    scores = grouped_scores({
        "id_ray_reflect": {"samples": 5, "correct": 4},
        "ext_ray_reflect": {"samples": 20, "correct": 7},
    })
    assert (scores["single_step"]["correct"], scores["single_step"]["samples"]) == (4, 5)
    assert (scores["multi_step"]["correct"], scores["multi_step"]["samples"]) == (7, 20)

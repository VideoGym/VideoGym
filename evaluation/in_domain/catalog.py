"""In-Domain tasks reuse the registered training generators with disjoint evaluation seeds."""

from evaluation.catalog.index import EvaluationTask


def in_domain_tasks():
    from data.registry import TASK_SPECS

    return [
        EvaluationTask(
            "id_" + task,
            "In-Domain / " + task.replace("_", " "),
            "id",
            task,
            description="Unseen problem instances within trained task patterns and parameter ranges.",
        )
        for task in sorted(TASK_SPECS)
    ]

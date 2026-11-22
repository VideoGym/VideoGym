"""Evaluation generator registry, separate from the training inventory."""


EVAL_TASK_SPECS = {}


def register_evaluation_specs(specs):
    from data.registry import TASK_SPECS

    for spec in specs.values() if isinstance(specs, dict) else specs:
        if spec.name in TASK_SPECS:
            raise ValueError(f"evaluation generator overlaps training: {spec.name}")
        previous = EVAL_TASK_SPECS.get(spec.name)
        if previous is not None and previous != spec:
            raise ValueError(f"conflicting evaluation generator: {spec.name}")
        EVAL_TASK_SPECS[spec.name] = spec


def load_builtin_specs():
    from evaluation.composition.catalog import TASK_SPECS as compositions
    from evaluation.cross_task.catalog import (
        TASK_SPECS as cross_tasks,
        FEATURE_EXTRACTORS,
    )
    from evaluation.shared.features import register_feature_extractors

    register_evaluation_specs(compositions)
    register_evaluation_specs(cross_tasks)
    register_feature_extractors(FEATURE_EXTRACTORS)


load_builtin_specs()

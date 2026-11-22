"""AST-level task contract extraction and catalog metadata assembly."""

from __future__ import annotations

import ast
from pathlib import Path

from data.common.task_semantics import TASK_FAMILY_LABELS, TASK_GROUPS, get_task_group

REQUIRED_FIELDS = ("logic", "design_intent")


class ContractError(ValueError):
    """Task or package metadata disagrees with authoritative contracts."""


def literal_contract(path: Path) -> dict:
    """Evaluate literal TASK_CONTRACT dicts without importing heavy generators."""
    values = {}

    def evaluate(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {evaluate(k): evaluate(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return evaluate(node.left) + evaluate(node.right)
        raise ContractError(f"non-literal contract expression in {path}")

    for statement in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.Assign):
            try:
                value = evaluate(statement.value)
            except (ContractError, KeyError):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    contract = values.get("TASK_CONTRACT")
    if not isinstance(contract, dict):
        raise ContractError(f"missing literal TASK_CONTRACT in {path}")
    for field in REQUIRED_FIELDS:
        if not isinstance(contract.get(field), str) or not contract[field].strip():
            raise ContractError(f"{path}: missing {field} contract")
    return contract


def build_base_catalog(config=None, training_tasks=None) -> dict:
    """Assemble all 11 task families, 246 In-Domain contracts, and evaluation tasks."""
    from data.registry import TASK_FAMILY_SPECS, TASK_SPECS, get_task_spec
    import data
    from evaluation.catalog.index import TRACK_NAMES, evaluation_tasks

    data_root = Path(data.__file__).resolve().parent
    definitions = evaluation_tasks(config, training_tasks)
    families = []
    for name, family in TASK_FAMILY_SPECS.items():
        tasks = []
        for task in sorted(family.tasks, key=lambda t: t.name):
            contract_path = data_root / name / task.name / "__init__.py"
            contract = literal_contract(contract_path)
            if (
                contract.get("task_name") != task.name
                or contract.get("task_family") != name
            ):
                raise ContractError(f"contract identity mismatch: {contract_path}")
            tasks.append(
                {
                    "id": task.name,
                    "task": task.name,
                    "task_family": name,
                    "task_group": get_task_group(task.name),
                    "design_intent": contract["design_intent"],
                    "logic": contract["logic"],
                    "samples": [],
                }
            )
        families.append(
            {
                "name": name,
                "display_name": TASK_FAMILY_LABELS.get(
                    name, name.replace("_", " ").title()
                ),
                "description": family.description,
                "tasks": tasks,
            }
        )
    return {
        "schema": "videogym.explorer.v1",
        "task_groups": [{"id": key, **group} for key, group in TASK_GROUPS.items()],
        "evaluation": {
            "tracks": TRACK_NAMES,
            "tasks": [
                {
                    **case.to_dict(),
                    "task_family": get_task_spec(case.task_name).task_family,
                    "task_group": get_task_group(
                        case.task_name, evaluation_task=case.name
                    ),
                    "source_task_ids": []
                    if case.track == "cross_task"
                    else list(
                        (case.task_name,)
                        if case.track in ("id", "extrapolation")
                        else case.primitives
                    ),
                    "samples": [],
                }
                for case in definitions.values()
            ],
        },
        "families": families,
        "totals": {
            "tasks": len(TASK_SPECS),
            "families": len(families),
            "samples": 0,
        },
    }

"""Both rule removals, computed from the same public scene before model inference."""

from copy import deepcopy
import math


def answers(source):
    from evaluation.shared.scenes.catalog import TASKS, module
    task = TASKS[source["key"]]
    m = module(task.key)
    if task.family in ("next_state", "board", "rearrangement", "physics", "spatial", "dynamics"):
        return m.ablations(source)
    if task.family == "sequence":
        joint = m.solve(source)
        first = dict(joint, color=source["colors"][-1])
        field = "height" if "height" in joint else "vertex_count"
        second = dict(joint, **{field: source["values"][-1]})
        return joint, first, second
    if task.family == "navigation":
        plans = [m.solve(dict(source, variant=v)) for v in ("main", "source_goal", "second_goal")]
        return tuple([list(s[:2]) for s in p["states"]] if p else None for p in plans)
    if task.family == "geometry":
        return tuple(next(c["point"] for c in source["candidates"]
                          if c["label"] == m.solve(dict(source, variant=v)))
                     for v in ("main", "source_goal", "second_goal"))
    raise ValueError(task.family)


def evidence(source):
    from evaluation.shared.scenes.catalog import TASKS, module
    task = TASKS[source["key"]]
    m = module(task.key)
    if hasattr(m, "solution_set_evidence"):
        return m.solution_set_evidence(source)
    if hasattr(m, "rule_controls"):
        joint = m.solve(source)
        controls = m.rule_controls(source)
        removed = {name: m.solve(value) for name, value in controls.items()}
        passed = len(removed) == len(set(task.primitives)) and all(value != joint for value in removed.values())
        if task.family == "geometry":
            positions = {c["label"]: c["point"] for c in source["candidates"]}
            passed = passed and all(math.dist(positions[joint], positions[v]) >= 24 for v in removed.values())
        return {"joint":deepcopy(joint),"rule_removals":removed,"passed":bool(passed)}
    joint, first, second = answers(source)
    family = TASKS[source["key"]].family
    if family == "geometry":
        gaps = [math.dist(joint, value) for value in (first,second)]
        passed = min(gaps) >= 24
    elif family == "rearrangement":
        stats = module(source["key"]).plans(source)
        passed = all(r[0] is not None for r in stats) and stats[0][0] > max(stats[1][0], stats[2][0])
        gaps = None
    elif family == "navigation":
        m = module(source["key"])
        controls = [m.solve(dict(source,variant=v)) for v in ("source_goal","second_goal")]
        passed = all(p is not None and not m.path_legal(source,p["states"]) for p in controls)
        gaps = None
    else:
        passed = joint != first and joint != second
        gaps = None
    return {"joint":deepcopy(joint),"first_only":first,"second_only":second,"coordinate_gaps":gaps,"passed":bool(passed)}

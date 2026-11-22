"""Native animation authors and CPU-only counterfactual controls."""

from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from evaluation.shared.scenes.catalog import TASKS, module


def wrong_answer(source):
    family = TASKS[source["key"]].family
    m = module(source["key"])
    answer = deepcopy(m.solve(source))
    if family == "constraints":
        return answer[:-1]
    if source["key"] in ("crane_reverse", "dock_reverse"):
        return deepcopy(source["target"])
    if family == "graph":
        selected = list(answer.get("nodes", []))
        others = [
            n["label"]
            for n in source["graph"]["nodes"]
            if n["label"] not in selected
        ]
        answer["nodes"] = selected[:-1] + others[:1] if others else selected[:-1]
    elif family == "geometry":
        labels = "ABCDE"
        choices = [labels[i : i + len(answer)] for i in range(6 - len(answer))]
        answer = next(c for c in choices if set(c) != set(answer))
    elif family == "physics":
        answer[0] += 1
    elif family == "spatial":
        answer = "wrong"
    elif family == "dynamics":
        answer[0][0] = "R" if answer[0][0] != "R" else "B"
    elif family == "rearrangement":
        answer = answer[:-1]
    elif family == "board":
        r, c = next(
            (r, c)
            for r, row in enumerate(answer["board"])
            for c, value in enumerate(row)
            if value is None
        )
        answer["board"][r][c] = "B"
    elif family == "next_state":
        answer[0][0] = 1 - answer[0][0]
    elif family == "sequence":
        if "height" in answer:
            answer["height"] = (
                answer["height"] + 2 if answer["height"] <= 6 else answer["height"] - 2
            )
        elif "vertex_count" in answer:
            answer["vertex_count"] = 3 if answer["vertex_count"] != 3 else 7
    return answer


def frames(source, answer=None):
    task = TASKS[source["key"]]
    m = module(source["key"])
    if hasattr(m, "frames"):
        return m.frames(source, answer)
    if task.family == "navigation":
        plan = m.solve(source)
        states = plan["states"]
        g = m.board(source["size"])
        if answer == "wrong":
            states = states[:-1] + [m.initial(source)]
        positions = [tuple(s[:2]) for s in states]
        teleports = [
            i for i, (a, b) in enumerate(zip(states, states[1:])) if a[4] >= 20
        ]
        scenario = SimpleNamespace(agent_path=positions)
        return g._animate_path(
            scenario,
            lambda cell, center, index: m.render(source, states[index]),
            teleport_steps=teleports,
        )
    if task.family == "sequence":
        from data.common.pipeline.media import build_complete_footprint_fade_frames

        first, last = m.endpoints(source, answer)
        result = build_complete_footprint_fade_frames(
            first,
            last,
            hold_frames=5,
            fade_frames=5,
            transition_repeats=2,
            final_hold_frames=10,
        )
    elif task.family == "graph":
        first = np.asarray(m.render(source, 0, answer))
        last = np.asarray(m.render(source, 1, answer))
        result = (
            [first.copy()] * 5
            + [np.asarray(m.render(source, i / 31, answer)) for i in range(1, 32)]
            + [last.copy()] * 5
        )
    else:
        stages = m.constructions(source)
        n = len(stages) + 1
        result = [np.asarray(m.render(source, 0, answer))] * 5
        for i in range(len(stages)):
            result.extend(
                np.asarray(m.render(source, (i + step / 8) / n, answer))
                for step in range(1, 9)
            )
        result.extend(
            np.asarray(m.render(source, (len(stages) + step / 4) / n, answer))
            for step in range(1, 5)
        )
        result.extend([np.asarray(m.render(source, 1, answer))] * 8)
    while len(result) % 4 != 1:
        result.append(result[-1].copy())
    return result


def negatives(source, reference):
    damaged = [f.copy() for f in reference]
    damaged[len(damaged) // 2][8:48, 8:48] = (255, 0, 255)
    yield "transient_foreign_object", damaged
    yield "unchanged_input", [reference[0]] * len(reference)
    yield "reversed_process", reference[::-1]
    cut = len(reference) // 2
    yield "terminal_hard_cut", [reference[0]] * cut + [reference[-1]] * (
        len(reference) - cut
    )
    family = TASKS[source["key"]].family
    yield "wrong_answer", frames(
        source, "wrong" if family == "navigation" else wrong_answer(source)
    )
    if source["variant"] == "main":
        if hasattr(module(source["key"]), "rule_controls"):
            for name, control in module(source["key"]).rule_controls(source).items():
                yield name, frames(control)
            return
        from evaluation.shared.scenes.ablation import answers

        _, first, second = answers(source)
        if family == "sequence":
            yield "first_rule_only", frames(source, first)
            yield "second_rule_only", frames(source, second)
        else:
            yield "first_rule_only", frames(dict(source, variant="source_goal"))
            yield "second_rule_only", frames(dict(source, variant="second_goal"))


def positives(source):
    """Different valid answers/processes must not be rejected as negatives."""
    m = module(source["key"])
    if TASKS[source["key"]].family == "constraints":
        from data.constraint_grid.core.forced_deduction import (
            build_sequential_commit_frames,
        )

        joint = m.solutions(source)[0]
        for i, answer in enumerate(joint[1:], 1):
            yield f"valid_alternate_solution_{i}", frames(source, answer)
        answer = m.solve(source)
        yield "valid_reverse_commit_order", frames(source, answer[::-1])
        yield "valid_simultaneous_commit", build_sequential_commit_frames(
            [m.picture(source, []), m.picture(source, answer)], minimum_final_hold=8
        )
    elif source["key"] in ("crane_reverse", "dock_reverse"):
        from collections import deque

        last = deque(m.programs(source), maxlen=1)[0]
        yield "valid_alternate_optimal_program", m.replay(source, last)[1]

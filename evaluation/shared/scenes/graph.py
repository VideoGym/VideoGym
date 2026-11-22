"""New graph goals, rendered and observed by the unchanged native graph family."""

import random
from data.graph_theory.core.semantics import (
    generate_graph_source,
    solve_graph_source,
    _tree_parent_depth,
)
from data.graph_theory.core.rendering import render_unified_graph
from data.graph_theory.core.replay import UnifiedGraphReplay
from data.graph_theory.core.verifier import _verify_video
from evaluation.shared.scenes.catalog import TASKS


def adjacency(s, excluded=()):
    out = {str(n["label"]): [] for n in s["nodes"] if str(n["label"]) not in excluded}
    for e in s["edges"]:
        u, v = (e["u"], e["v"])
        if u in out and v in out:
            out[u].append(v)
            out[v].append(u)
    return out


def paths(s, start, goal, excluded=()):
    a = adjacency(s, excluded)
    out = []

    def visit(path):
        if path[-1] == goal:
            out.append(path)
            return
        for n in a.get(path[-1], ()):
            if n not in path:
                visit(path + [n])

    if start in a:
        visit([start])
    return out


def descendants(s):
    root, parent, depth = _tree_parent_depth(s)

    def below(n, q):
        while n is not None:
            if n == q:
                return True
            n = parent[n]
        return False

    return (root, parent, depth, below)


def solve(source):
    s = source["graph"]
    key = source["key"]
    roles = s["roles"]
    a = adjacency(s)
    labels = sorted(a)
    if source.get("variant") == "source_goal":
        return solve_graph_source(source["host"], s)
    nodes = None
    if key == "ancestor_subtree_diameter":
        root, parent, depth, below = descendants(s)
        queries = roles["queries"]
        ancestor = max((v for v in labels if all(below(q, v) for q in queries)), key=depth.get)
        mode = source.get("omitted_rule")
        selected = [v for v in labels if below(v, ancestor)]
        if mode == "closest_common_ancestor":
            selected = [v for v in labels if any(below(v, q) for q in queries)]
        elif mode == "subtree_size":
            selected = labels
        if mode == "diameter_endpoints":
            return dict(valid=True, unique=True, answer=selected, nodes=selected, paths=[], support={})
        candidates = [p for i, u in enumerate(selected) for v in selected[i+1:]
                      for p in paths(s, u, v, set(labels)-set(selected))]
        if not candidates:
            return None
        longest = max(map(len, candidates))
        winners = [p for p in candidates if len(p) == longest]
        if len(winners) != 1:
            return None
        path = winners[0]
        endpoints = sorted((path[0], path[-1]), key=int)
        return dict(valid=True, unique=True, answer=endpoints, nodes=endpoints,
                    paths=[path], support={"ancestor": ancestor, "subtree": selected})
    elif key == "subtree_leaves":
        root, parent, depth, below = descendants(s)
        leaves = [
            v
            for v in labels
            if v != root and (not any((parent.get(w) == v for w in labels)))
        ]
        nodes = [v for v in leaves if below(v, roles["query"])]
        if source.get("omitted_rule") == "leaf_count":
            nodes = [v for v in labels if below(v, roles["query"])]
        elif source.get("omitted_rule") == "subtree_size":
            nodes = leaves
    if not nodes:
        return None
    return dict(
        valid=True,
        unique=True,
        answer=sorted(nodes),
        nodes=sorted(nodes),
        paths=[],
        support={},
    )


def drawing_task(source):
    return "diameter_endpoints" if source["key"] == "ancestor_subtree_diameter" else "leaf_count"


def render(source, progress=0.0, answer=None):
    graph = dict(source["graph"], task_name=drawing_task(source))
    return render_unified_graph(
        graph, solve(source) if answer is None else answer, progress=progress
    )


def sample(key, seed, variant="main"):
    task = TASKS[key]
    rng = random.Random(seed)
    for attempt in range(1500):
        g = generate_graph_source(task.host, rng)
        s = {"key": key, "host": task.host, "graph": g, "variant": variant}
        answer = solve(s)
        if not answer:
            continue
        if key == "ancestor_subtree_diameter":
            joint = solve(dict(s, variant="main"))
            if (not joint or len(joint["support"]["subtree"]) < 6
                    or len(g["nodes"]) - len(joint["support"]["subtree"]) < 2
                    or len(joint["paths"][0]) < 5
                    or set(joint["nodes"]) == set(g["roles"]["queries"])):
                continue
            adjacent = adjacency(g)
            if sum(len(adjacent[v]) == 1 for v in joint["support"]["subtree"]) < 3:
                continue
            controls = [solve(control) for control in rule_controls(dict(s, variant="main")).values()]
            if any(not control or (control["nodes"], control["paths"]) == (joint["nodes"], joint["paths"]) for control in controls):
                continue
        if variant != "source_goal" and key == "subtree_leaves":
            root, parent, depth, below = descendants(g)
            unrestricted = {
                v
                for v in depth
                if v != root and (not any((parent.get(w) == v for w in depth)))
            }
            if set(answer["nodes"]) == unrestricted:
                continue
        if variant != "source_goal":
            if key == "subtree_leaves":
                if set(answer["nodes"]) == set(
                    solve(dict(s, variant="source_goal"))["nodes"]
                ):
                    continue
        from evaluation.shared.scenes.ablation import evidence

        if not evidence(dict(s, variant="main"))["passed"]:
            continue
        return s
    raise RuntimeError(f"{key}: native graph sampling exhausted")


def prompt(source):
    task = TASKS[source["key"]]
    description = (
        "A native rooted tree: the green-ringed top node is the root and each edge descends from parent to child."
        if "root" in source["graph"]["roles"]
        else "A native numbered undirected graph or unrooted tree."
    )
    action = (
        "Fade in the complete diameter path in teal; then fade in both complete orange endpoint rings together, preserving the teal path."
        if source["key"] == "ancestor_subtree_diameter"
        else "Fade in one complete orange ring around each selected node, all rings appearing together."
    )
    return (
        "Input Image: "
        + description
        + " Every segment is one unweighted edge, and the numbers are node labels. Additional role marks have only the meanings stated in the Rule.\n\nRule: "
        + task.rule
        + "\n\nRequired Output: "
        + action
        + " Begin with the unchanged input and end with a stable solved hold. Keep every node, label, source edge and role marker fixed. No extra marks, hard cuts, erasures, camera changes or reordered updates."
    )


def rule_controls(source):
    return {
        name: dict(source, omitted_rule=name)
        for name in TASKS[source["key"]].primitives
    }


def grade(source, path):
    answer = solve(source)
    if answer is None:
        raise ValueError("invalid source graph goal")
    task = drawing_task(source)
    graph = dict(source["graph"], task_name=task)
    metric = _verify_video(path, UnifiedGraphReplay(task, graph, answer))
    required = bool(
        metric.get("video_initial_hold")
        and metric.get("video_final_hold")
        and metric.get("video_continuous_process")
    )
    metric["visible_holds_and_continuity"] = required
    metric["whole_video_process_correct"] = float(
        bool(metric.get("whole_video_process_correct")) and required
    )
    return metric

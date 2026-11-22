"""Ordered binary cellular-rule compositions on the shared native bit-grid canvas."""

from copy import deepcopy
import random
import numpy as np

from data.next_state_prediction.core import semantics as native
from data.next_state_prediction.core.rendering import render_state, render_video_frames
from evaluation.shared.task_io import read_video

PROGRAMS = {
    "life_rowtable": ("life", "rowtable"),
    "life_xor": ("life", "xor"),
    "xor_life": ("xor", "life"),
    "life_xor_life": ("life", "xor", "life"),
}


def trace(s, condition="joint"):
    states = [deepcopy(s["initial"])]
    for rule in PROGRAMS[s["key"]]:
        if condition in ("joint", rule):
            if rule == "rowtable":
                old = states[-1]
                grid = [native._step_rule_table(dict(row=row,generation=old["generation"]),s["params"])["row"] for row in old["grid"]]
                states.append(dict(grid=grid,generation=old["generation"]+1))
            else:
                step = native._step_lifelike if rule == "life" else native._step_xor
                states.append(step(states[-1], s["params"]))
    return states


def ablations(s):
    return tuple(trace(s, c)[-1]["grid"] for c in ("joint", "life", "rowtable" if s["key"] == "life_rowtable" else "xor"))


def sample(key, seed, variant="main"):
    rng = random.Random(seed)
    for _ in range(300):
        params, initial = native._sample_lifelike(rng)
        params["include_self"] = bool(rng.randrange(2))
        if key == "life_rowtable":
            params["rule"] = [rng.randrange(2) for _ in range(8)]
        s = {"key":key,"host":"lifelike_rule_rollout","variant":variant,"initial":initial,"params":params}
        joint, first, second = ablations(s)
        stages = trace(s)
        if joint != first and joint != second and len({str(x["grid"]) for x in stages}) == len(stages):
            return s
    raise RuntimeError("could not sample a nondegenerate ordered binary rule composition")


def solve(s):
    condition = {"main":"joint","source_goal":"life","second_goal":("rowtable" if s["key"] == "life_rowtable" else "xor")}[s["variant"]]
    return trace(s,condition)[-1]["grid"]


def render(s, progress=0, answer=None):
    states=trace(s,{"main":"joint","source_goal":"life","second_goal":("rowtable" if s["key"] == "life_rowtable" else "xor")}[s["variant"]])
    state=states[0] if progress == 0 else states[-1]
    if answer is not None and progress != 0:
        state=dict(state,grid=answer)
    return render_state("lifelike_rule_rollout",state,s["params"])


def frames(s, answer=None):
    states=trace(s,{"main":"joint","source_goal":"life","second_goal":("rowtable" if s["key"] == "life_rowtable" else "xor")}[s["variant"]])
    if answer is not None:
        states[-1]=dict(states[-1],grid=answer)
    return [np.asarray(f) for f in render_video_frames("lifelike_rule_rollout",states,s["params"])]


def prompt(s):
    p=s["params"]
    rules=PROGRAMS[s["key"]]
    if s["variant"] != "main":
        selected="life" if s["variant"] == "source_goal" else ("rowtable" if s["key"] == "life_rowtable" else "xor")
        rules=tuple(rule for rule in rules if rule==selected)
    if s["key"] == "life_rowtable":
        definitions = {
            "life": (f"Count the eight old neighbouring bits; 0 becomes 1 for counts in {p['birth']}, "
                     f"and 1 survives for counts in {p['survive']}; otherwise write 0."),
            "rowtable": (f"Independently in each row, replace each cell using its old left/centre/right triple: "
                         f"entries 000 through 111 map to {p['rule']} respectively."),
        }
        boundary = "Both axes wrap for Life; each row wraps for the row table." if p["boundary"] == "wrap" else "Outside neighbours are 0 for both rules."
        return ("Input Image: The native numbered square grid uses pale cells for 0 and dark-blue cells for 1.\n\nRule: "
                + " Then ".join(definitions[rule] for rule in rules) + " " + boundary
                + " Each update consumes the complete preceding grid, synchronously.\n\nRequired Output: "
                "Hold the input, show and hold each complete successor in order, and end in a stable hold. "
                "Updates are atomic. Preserve all cell positions, colours, labels, card and camera; add no rule panel or counter.")
    return (
        "Input Image: The native numbered square bit grid uses pale cells for 0 and dark-blue cells for 1. "
        "All cells share the same coordinate system.\n\nRule: Apply the following rule sequence exactly once from left to right: "
        + ", then ".join(rules) + ". Each rule consumes the complete grid produced by the preceding rule. "
        f"life counts the eight old surrounding bits: an old 0 becomes 1 exactly for counts in {p['birth']}; "
        f"an old 1 stays 1 exactly for counts in {p['survive']}; otherwise write 0. "
        + ("For life both axes wrap. " if p["boundary"]=="wrap" else "For life missing outside neighbours are 0. ")
        + "xor writes the parity of the four old north/east/south/west bits"
        + (" plus the old centre bit" if p["include_self"] else "")
        + "; for xor both axes wrap. Compute each entire successor synchronously from its complete predecessor. "
        "\n\nRequired Output: Start with the unchanged input. Show and hold every complete successor grid in the stated order, "
        "using the original cell colors and positions. State changes are atomic grid updates. End with the last grid in a stable hold. "
        "Preserve the card, grid, coordinate labels and camera. Add no rule card, counter, caption, or new symbol."
    )


def grade(s, path):
    from data.next_state_prediction.core.evaluator import (
        _discrete_sequence_metrics, _dynamic_mask, _terminal_image_metrics,
    )
    from data.common.verifiers.region_gates import static_region_errors, STATIC_REGION_FLOOR

    observed = read_video(path)
    condition = {"main":"joint", "source_goal":"life", "second_goal":("rowtable" if s["key"] == "life_rowtable" else "xor")}[s["variant"]]
    pictures = [np.asarray(render_state("lifelike_rule_rollout", state, s["params"])) for state in trace(s, condition)]
    dynamic = _dynamic_mask(pictures)
    sequence = _discrete_sequence_metrics(observed, pictures, dynamic)
    terminal = _terminal_image_metrics(observed[-1], pictures[0], pictures[-1])
    final = terminal["terminal_image_correct"] == 1
    preserved = all(static_region_errors(f, pictures[0], ~dynamic)[0] <= STATIC_REGION_FLOOR for f in observed)
    process = final and preserved and sequence["video_discrete_sequence_correct"] == 1
    return {"whole_video_loaded":True, "whole_video_process_correct":float(process),
            "video_final_state_correct":float(final), "source_state_replayed":True,
            "static_preserved":preserved, **terminal, **sequence}

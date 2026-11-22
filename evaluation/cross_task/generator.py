"""Executable family-derived tasks; no source-ID generator dependencies."""

import random

from data.common.base import AbstractPuzzleGenerator
from data.common.typed_answers import format_canonical_json_answer
from evaluation.cross_task.catalog import DESIGNS
from evaluation.cross_task.engine import family, grade, reference_states, render, solve
from evaluation.shared.task_io import eval_record, write_record


def sample_source(rng, key):
    return family({"goal": key}).sample(rng, key)


def prompt(source):
    rule = DESIGNS[source["goal"]].rule
    if source["goal"] == "petri_target_marking":
        rule += " The left A-D circles are the current counts of four shared places; the right A-D circles are target counts of those SAME places, not four additional places. Blue links consume from the left identities and green links produce into the right identities."
    elif source["goal"] == "recurrent_rotor_router":
        rule += f" Start at node {source['start'] + 1} and execute exactly {source['steps']} moves. The bottom-right answer box is the final node; each node's lower-right box is its departure count."
    elif source["goal"] == "layered_refraction":
        rule += f" The incident sine relative to the vertical normal is {source['sine']:.2f}."
    elif source["goal"] == "causal_counterfactual":
        rule += " The node symbols ∧, ∨ and ⊕ denote AND, OR and XOR; the blue digit inside each node is its factual observed value."
    elif source["goal"] == "hidden_markov_path":
        rule += " The unshown self-transition probability is one minus the shown outgoing transition probability. Below each mode, the two colored symbol disks and fractions give its emission probabilities."
    elif source["goal"] == "delayed_wave_superposition":
        rule += " The A and B rows below the network list amplitudes at t=0,1,2 from left to right."
    return (
        "Input Image: "
        + DESIGNS[source["goal"]].family.replace("_", " ")
        + " objects and their task-relevant labels are shown. Spatial grouping distinguishes the fixed givens and the writable answer region.\n\nRule: "
        + rule
        + "\n\nRequired Output: Begin with the unchanged input. Construct the requested answer one mark or writable cell at a time, in any order. A mark may appear directly or fade in steadily; completed marks must remain. Do not change any source content, clues, labels or the camera. Hold the complete final answer. The marking order is an output convention, not a claimed reasoning trace."
    )


class CrossTaskGenerator(AbstractPuzzleGenerator):
    goal = None

    def __init__(self, output_dir, *, seed=None, video=True, **kwargs):
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.video = video

    def create_puzzle(self):
        source = sample_source(self.rng, self.goal)
        answers = solve(source)
        if not answers:
            raise ValueError("unsatisfiable source")
        return write_record(
            self,
            source,
            reference_states(source),
            render,
            format_canonical_json_answer({"construction": answers[0]}),
            prompt(source),
        )


def extract_features(semantic_input):
    source = semantic_input.get("semantic_state", semantic_input)
    answers = solve(source)
    return {
        "answer_regions": len(answers[0]),
        "constructed_marks": sum(v != -1 for v in answers[0]),
        "canonical_optima": len(answers),
        "mechanism": DESIGNS[source["goal"]].mechanism,
    }


def verify(samples, pred_root, *, modality="ti2v"):
    if modality != "ti2v":
        raise ValueError("TI2V only")
    return [eval_record(sample, pred_root, grade) for sample in samples]


for key in DESIGNS:
    name = "".join(x.title() for x in key.split("_")) + "Generator"
    globals()[name] = type(
        name,
        (CrossTaskGenerator,),
        {"goal": key, "task_name": "eval_cross_" + key, "__module__": __name__},
    )

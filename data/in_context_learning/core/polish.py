"""Family-level authoring polish for in-context-learning records.

The task renderers intentionally keep their executable state machines local.
This module owns only the repeated model-facing conventions: concise prompts,
natural reference reasoning, and removal of duplicate/cosmetic frames.
It never changes the semantic answer or supplies evidence to a verifier.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


from data.common.grid_coordinates import GRID_XY_CONVENTION_SENTENCE
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.common.typed_answers import FOUR_CHOICE_CODEC, format_answer_line
from data.in_context_learning.core.direct_delivery import (
    DIRECT_CHOICE_TASKS as _DIRECT_CHOICE_TASKS,
    VISUAL_CHOICE_TASKS as _VISUAL_CHOICE_TASKS,
    apply_direct_choice_delivery,
)

_DIRECT_INPUT_COPY: dict[str, str] = {
    "analogy_completion": "The problem image shows A changing to B and a query shape C with its transformed result missing.",
    "object_subtraction": "The problem image shows the before-after demonstrations and the target before-scene, with fixed object slots.",
    "odd_one_out_shape": "The problem image shows the same four shapes as one unlabelled comparison set.",
    "set_inclusion_diagram": "The problem image shows the complete visible memberships of sets X and Y.",
    "venn_intersection_count": "The problem image shows two overlapping labelled sets and all visible dot markers.",
    "venn_symmetric_difference": "The problem image shows one unshaded two-set diagram with its universe and overlap visible.",
}

_DIRECT_RULE_COPY: dict[str, str] = {
    "analogy_completion": "Infer the single rigid rotation or reflection mapping A to B, then apply it to C without changing cell count or colour.",
    "object_subtraction": "Infer which single object is removed in every demonstration, then remove that object from the target while all survivors keep their slots.",
    "set_inclusion_diagram": "Match tokens by number, then determine whether one set contains the other, they partially overlap, or they are disjoint.",
    "venn_intersection_count": "Count each dot whose centre lies inside both circles exactly once; dots in only one circle do not count.",
    "venn_symmetric_difference": "The symmetric difference contains the A-only and B-only regions, excluding the overlap and the exterior.",
}

_DIRECT_TERMINAL_COPY: dict[str, str] = {
    "analogy_completion": "the transformed query shape filled directly in the missing result panel",
    "object_subtraction": "the target after-scene with exactly the inferred object removed",
    "odd_one_out_shape": "the four-object collection with one direct ring around the unique outlier",
    "set_inclusion_diagram": "the set relation constructed directly from the visible memberships",
    "venn_intersection_count": "the shared lens and all counted intersection dots marked directly",
    "venn_symmetric_difference": "the complete A-only and B-only regions shaded directly",
}

# Dynamic text is deliberately scarce in ICL renderers.  This inventory is
# persisted with every sample so visual QA can distinguish an intentional,
# bounded glyph from an accidental title, answer banner, or metadata dump.
# Static labels are whitelisted per task; dynamic entries name only source
# fields whose values may be rendered.  They never include the selected option
# or another saved answer field.
_GLYPH_CONTRACTS: dict[str, dict[str, Any]] = {
    "analogy_completion": {"static": ["A", "B", "C", "D", "?"], "dynamic": ["rigid_transform_badge:enum"]},
    "hue": {"static": ["?"], "dynamic": ["palette_names:enum", "row_column_labels:int[1,5]"]},
    "legend_operator_program": {"static": [], "dynamic": ["operator_tokens:enum[A,B,C]", "binary_cells:enum[0,1]"]},
    "object_subtraction": {"static": ["1", "2", "3", "↓", "A", "B", "C", "D"], "dynamic": []},
    "odd_one_out_shape": {"static": ["A", "B", "C", "D"], "dynamic": []},
    # The turn count is drawn as "xN" with a multiplication sign; no rotation
    # arrow is rendered, so no arrow character belongs in this whitelist.
    "rotate_matrix": {"static": ["×"], "dynamic": ["input_cells:int[10,99]", "quarter_turn_count:int[1,3]"]},
    "set_inclusion_diagram": {"static": ["A", "B", "C", "D", "X", "Y", "∩"], "dynamic": ["member_ids:int[1,8]", "shared_count:int[0,6]"]},
    "shape_outline_fill": {"static": ["A", "B", "C", "D", "?"], "dynamic": []},
    "spiral_matrix": {"static": [], "dynamic": ["input_cells:int[10,98]"]},
    "triple_intersection_red": {"static": ["A", "B", "C", "x →", "y ↓"], "dynamic": ["coordinate_ticks:int[0,32]-step-4"]},
    # A/B are the two set labels.  No numeric option value or option-card
    # label is rendered after this task moves its choices into Prompt text.
    "venn_intersection_count": {"static": ["A", "B"], "dynamic": []},
    "venn_symmetric_difference": {"static": ["A", "B", "C", "D"], "dynamic": []},
}


_TASK_COPY: dict[str, tuple[str, str, str]] = {
    "analogy_completion": (
        "The top panels show A changing to B and a query shape C. Four small-grid candidates appear below.",
        "Infer the single rigid rotation or reflection that maps A to B, ignoring translation within a small grid, and apply the same transform to C. Cell count and color stay unchanged.",
        "the transformed query shape and the selected candidate",
    ),
    "hue": (
        "A color legend and an indexed grid are shown. One cell is missing while every other cell is a visible clue. "
        + GRID_XY_CONVENTION_SENTENCE,
        "Treat the legend order as a repeating color cycle. Infer the consistent row and column shifts from the visible cells and use them to fill the missing cell.",
        "the missing color and completed grid",
    ),
    "legend_operator_program": (
        "A legend demonstrates what each operator token does to a binary grid. A target grid and an ordered token program appear beside it.",
        "Infer every used token from its before-after example, then apply the program from left to right. Each operation updates the whole grid at once.",
        "the final binary grid",
    ),
    "object_subtraction": (
        "Several before-after demonstrations show four objects in fixed slots. A target scene and four possible after-scenes appear below.",
        "Infer which single visible object is removed in every demonstration, then apply the same rule to the target. Survivors keep their slots; the removed slot becomes empty.",
        "the target after-scene and the selected candidate",
    ),
    "odd_one_out_shape": (
        "Four candidate shapes are shown with one short property definition.",
        "Compare only that named property. Three candidates share one value and exactly one has a different value; select the different candidate.",
        "the selected candidate",
    ),
    "rotate_matrix": (
        "A square matrix of two-digit values is shown, with the requested number of quarter-turns printed in amber above it.",
        "Turn the whole matrix clockwise, as seen in the image, by that many quarter-turns, taking one quarter-turn at a time. Every value moves with its cell; the matrix keeps its size and no value is changed, added, removed, or duplicated.",
        "the rotated matrix",
    ),
    "set_inclusion_diagram": (
        "Rows X and Y list numbered tokens, and four candidate set diagrams appear below.",
        "Match tokens by number to determine which are shared and which occur in only one row. Choose the diagram that represents the resulting relation between X and Y.",
        "the recovered set relation and the selected candidate",
    ),
    "shape_outline_fill": (
        "An analogy shows A changing to B and a query shape C with its result missing.",
        "Infer the fill-style change from A to B and apply only that change to C. Shape, color, size, and position stay the same.",
        "the transformed query shape",
    ),
    "spiral_matrix": (
        "A square matrix of distinct two-digit values is shown.",
        "Starting at the top-left cell, read the outer boundary clockwise, then read each remaining inner boundary the same way, until every cell has been read exactly once. The answer is the cell values in that reading order.",
        "every cell read and the complete traced reading path",
    ),
    "triple_intersection_red": (
        "Three labeled circle outlines A, B, and C overlap on a fixed coordinate grid; each circle has a visible center mark, with x increasing rightward and y increasing downward.",
        "Boundary points count as inside. The target region is exactly the set of points lying inside all three circles; any point outside one or more circles is excluded. Obtain the central grid point by averaging the three visible center x-coordinates and the three visible center y-coordinates separately, then rounding each average to the nearest integer.",
        "the typed triple-overlap location and the region inside all three circles filled with one flat translucent red",
    ),
    "venn_intersection_count": (
        "Two overlapping labeled circles contain visible dot markers, with four numeric candidates below.",
        "Count each dot whose center lies inside both circles exactly once. Dots in only one circle do not count; choose the unique candidate equal to the intersection count.",
        "the counted intersection and the selected candidate",
    ),
    "venn_symmetric_difference": (
        "An unshaded two-set diagram and four region-shading candidates are shown.",
        "The symmetric difference contains the A-only and B-only regions, but not the overlap or the exterior. Choose the unique candidate with exactly those regions shaded.",
        "the shaded symmetric difference and the selected candidate",
    ),
}


# Tasks whose reference video is an exact-state timeline (equal-duration
# complete states, direct switches); all other ICL media use continuous
# motion or element-local fades.
_DISCRETE_TIMING_TASKS = frozenset({"rotate_matrix", "spiral_matrix"})

# TI2V process copy for the non-direct tasks only; direct-choice tasks build
# their process sentence in the direct branch of ``_prompt``.
_PROCESS: dict[str, str] = {
    "hue": "Hold the input, fade one complete normal-sized region of the inferred color uniformly at the missing cell's fixed footprint until it replaces the question mark, then hold the completed grid.",
    "legend_operator_program": "Show one cumulative grid state after each program token, in token order.",
    "rotate_matrix": "Show the matrix after each clockwise quarter-turn in order.",
    "shape_outline_fill": "Hold the input, fade C's unchanged shape into the missing slot as a neutral outline, then change only that local preview smoothly to the demonstrated fill style and hold the result.",
    "spiral_matrix": "Step the reading position from cell to cell in clockwise outer-to-inner order, extending the traced path by one whole segment on each step.",
    "triple_intersection_red": "Fade the complete region common to A, B, and C into one flat translucent red together as one global update.",
}


def _formal_answer_line(record: dict[str, Any]) -> str:
    from data.common.typed_answers import parse_answer_line

    if parse_answer_line(record.get("answer")) is not None:
        return str(record["answer"])
    answer = str(record.get("answer") or "").strip().rstrip(".")
    return format_answer_line(answer)


def _structured(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("structured_answer")
    return dict(value) if isinstance(value, dict) else {}


def _flatten_integer_glyphs(value: Any) -> list[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_integer_glyphs(item))
        return result
    return []


def _rendered_glyph_inventory(
    task: str,
    structured: dict[str, Any],
    *,
    option_labels: list[str],
) -> dict[str, Any]:
    contract = copy.deepcopy(_GLYPH_CONTRACTS[task])
    observed: list[str] = []
    if task == "analogy_completion":
        transform_glyphs = {
            "rot90cw": "90° ↻",
            "rot90ccw": "90° ↺",
            "rot180": "180°",
            "reflect_v": "↔",
            "reflect_h": "↕",
        }
        transform = str(structured.get("transform") or "")
        if transform in transform_glyphs:
            observed.append(transform_glyphs[transform])
    elif task == "hue":
        observed.extend(str(value) for value in structured.get("palette_names") or [])
        board_size = int(structured.get("board_size") or 0)
        observed.extend(str(index) for index in range(1, board_size + 1))
    elif task == "legend_operator_program":
        observed.extend(str(value) for value in structured.get("program_tokens") or [])
        observed.extend(str(value) for value in (structured.get("operator_by_token") or {}).keys())
    elif task == "rotate_matrix":
        # The turn count is tagged so it stays distinguishable from a cell
        # value; both are integers and would otherwise merge into one
        # unattributable sorted list.
        observed.extend(_flatten_integer_glyphs(structured.get("input_matrix")))
        observed.append(f"×{int(structured.get('rotations') or 0)}")
    elif task == "set_inclusion_diagram":
        meta = structured.get("task_meta") or {}
        observed.extend(_flatten_integer_glyphs(meta.get("x_elements")))
        observed.extend(_flatten_integer_glyphs(meta.get("y_elements")))
        observed.append(str(len(meta.get("shared_elements") or [])))
    elif task == "spiral_matrix":
        observed.extend(_flatten_integer_glyphs(structured.get("input_matrix")))
    elif task == "triple_intersection_red":
        observed.extend(str(index) for index in range(0, 33, 4))
    observed = sorted({value.strip() for value in observed if value and value.strip()})
    if len(observed) > 128 or any("\n" in value or len(value) > 32 for value in observed):
        raise ValueError(f"{task} dynamic rendered-glyph inventory exceeds its bounded schema")
    if observed and not contract["dynamic"]:
        raise ValueError(f"{task} records dynamic glyphs without a bounded dynamic schema")
    inventory = {
        "schema_version": "icl_rendered_glyph_inventory_v1",
        "static_whitelist": sorted(set(contract["static"])),
        "bounded_dynamic_schema": list(contract["dynamic"]),
        "observed_dynamic_values": observed,
        "answer_fields_excluded": True,
    }
    if task in _DIRECT_CHOICE_TASKS:
        input_plate_labels = option_labels if task in _VISUAL_CHOICE_TASKS else []
        inventory.update(
            {
                "input_plate_option_labels": input_plate_labels,
                "input_plate_option_label_scope": (
                    "input_only" if input_plate_labels else "none"
                ),
                "option_labels_in_terminal_or_ti2v": False,
            }
        )
    return inventory


_TASK_ANSWER_EXAMPLES = {
    "analogy_completion": "Answer: X.",
    "hue": "Answer: ULTRAVIOLET.",
    "object_subtraction": "Answer: X.",
    "odd_one_out_shape": "Answer: X.",
    "set_inclusion_diagram": "Answer: X.",
    "shape_outline_fill": "Answer: octagon:striped.",
    "venn_intersection_count": "Answer: X.",
    "venn_symmetric_difference": "Answer: X.",
}


def _answer_format_example(task: str, structured: dict[str, Any]) -> str:
    answer_schema = structured.get("answer_schema")
    schema_example = (
        str(answer_schema.get("format_example") or "").strip()
        if isinstance(answer_schema, dict)
        else ""
    )
    task_example = (
        FOUR_CHOICE_CODEC.example
        if task in _DIRECT_CHOICE_TASKS
        else _TASK_ANSWER_EXAMPLES.get(task, "")
    )
    if schema_example and task_example and schema_example != task_example:
        raise ValueError(
            f"{task} AnswerContract format_example disagrees with the family-level fixed example"
        )
    example = schema_example or task_example
    if not example.startswith("Answer: ") or not example.endswith(".") or "\n" in example:
        raise ValueError(f"{task} does not define one fixed Answer syntax example")
    return example


def _prompt(
    task: str,
    mode: str,
    answer_line: str,
    structured: dict[str, Any],
    *,
    option_labels: list[str] | None = None,
) -> str:
    input_copy, rule_copy, visual_delivery = _TASK_COPY[task]
    if task in _DIRECT_CHOICE_TASKS:
        labels = [str(value) for value in option_labels or []]
        if labels != list(FOUR_CHOICE_CODEC.labels):
            raise ValueError(f"{task} must expose candidate labels A, B, C, and D in order")
        input_copy = _DIRECT_INPUT_COPY[task]
        rule_copy = _DIRECT_RULE_COPY.get(task, rule_copy)
        visual_delivery = _DIRECT_TERMINAL_COPY[task]
    if task == "odd_one_out_shape":
        rule_label = str(structured.get("rule_label") or "the named property").strip().lower()
        rule_definition = str(structured.get("rule_definition") or "").strip()
        rule_copy = (
            f"Compare only {rule_label}: {rule_definition} "
            "Three shapes share one value and exactly one has a different value. "
            "The shapes also differ in size, orientation and how irregular their outlines "
            "look; none of that decides the answer."
        )
    visual_delivery = visual_delivery.rstrip().rstrip(".")
    if task in _DIRECT_CHOICE_TASKS:
        if task == "odd_one_out_shape":
            process = (
                "Hold the unlabelled four-shape source, fade in one complete fixed-colour ring as a "
                "whole around the unique outlier, and hold the marked collection."
            )
        elif task == "venn_symmetric_difference":
            process = (
                "Hold the unshaded source, fade in the complete A-only and B-only crescents "
                "synchronously at their final footprints, and hold the completed diagram."
            )
        else:
            process = (
                f"Hold the source scene, fade in {visual_delivery} as a whole at its complete final "
                "footprint, and hold the solved source scene."
            )
    else:
        process = _PROCESS.get(
            task,
            f"Hold the input, reveal only rule-bearing cumulative changes, commit {visual_delivery}, and hold the final state.",
        )
    output = f"Return one video of the solving process. {process}".rstrip(".")
    output = output.rstrip().rstrip(".")
    terminal = "."
    body = f"Input Image: {input_copy}\n\nRule: {rule_copy}\n\nRequired Output: {output}{terminal}"
    if mode == "ti2v":
        body += "\n\n" + build_ti2v_continuity_paragraph(
            timing="discrete" if task in _DISCRETE_TIMING_TASKS else "continuous",
            moving_nouns="the elements the rules change",
            static_nouns="everything the rules leave unchanged, including the source evidence, labels, background, and camera",
            hold_noun="the final answer state",
        )
    return body


def _synchronize_task_contract(record: dict[str, Any]) -> None:
    """Keep persisted authoring metadata aligned with the emitted artifacts.

    Canonical manifests must describe the rows that actually exist, not the
    pre-polish implementation.  Metadata-integrity measurements are retained
    as contamination diagnostics, but are explicitly outside candidate
    correctness and the reward vector.
    """

    contract = record.get("task_contract")
    structured = record.get("structured_answer")
    if not isinstance(contract, dict) or not isinstance(structured, dict):
        return
    task_name = str(record.get("task_name") or "")

    if task_name in _DIRECT_CHOICE_TASKS:
        modality_outputs = contract.get("modality_outputs")
        if not isinstance(modality_outputs, dict):
            modality_outputs = {}
        modality_outputs.update(
            {
                "ti2v": (
                    "One fixed-camera video from the single source-only input: reveal the rule-bearing "
                    "process, construct the solved source state directly, and hold it."
                ),
                "solution_image": (
                    "A direct solved source scene with no option cards, option labels, green selection "
                    "border, or retained reasoning annotation."
                ),
            }
        )
        contract["modality_outputs"] = modality_outputs
        contract["preservation_constraints"] = [
            "The terminal image and video preserve every source fact that the task rule does not change.",
            "Candidate plates are supplied input evidence only and never become terminal content or TI2V frames.",
        ]
        contract["transition_constraints"] = [
            "Begin with a readable hold of the single source-only input.",
            "Show every new answer element at its complete final footprint and raise its local opacity uniformly; the two symmetric-difference crescents fade synchronously, while odd-one-out fades one complete fixed-colour ring about the selected shape.",
            "Whole-frame endpoint blends, radial growth, directional wipes or scans, and hard commits are invalid; only the source-derived answer element may fade locally.",
            "End with a readable hold of the complete solved source scene without cards, selection borders, hard cuts, or camera drift.",
        ]
        contract["modality_input_descriptions"] = {
            "ti2v": "One source-only problem image with no hypothesis plates or prompt-visible choice list.",
            "solution_image": "The same source-only problem image used to author a direct solved terminal.",
        }
        answer_contract = contract.get("answer")
        if isinstance(answer_contract, dict):
            answer_contract["verifier_profile"] = "icl_direct_choice_source_replay_final_delivery_v1"
            contract["answer"] = answer_contract
        contract["input_media_policy"] = record.get("input_media_policy")
        contract["option_selection_visual"] = "direct_solved_source_state_v1"

    verifier_contract = contract.get("verifier")
    if isinstance(verifier_contract, dict):
        components = verifier_contract.get("component_scores")
        has_metadata_diagnostic = isinstance(components, dict) and any(
            "metadata_integrity_score" in list(names or [])
            for names in components.values()
            if isinstance(names, (list, tuple))
        )
        if has_metadata_diagnostic:
            verifier_contract["diagnostic_components"] = ["metadata_integrity_score"]
            verifier_contract["binary_predicate"] = (
                "all formal-delivery semantic hard gates pass; "
                "metadata_integrity_score is diagnostic only and cannot change "
                "binary_correct or any reward component"
            )
        contract["verifier"] = verifier_contract

    record["task_contract"] = contract


def polish_icl_record(record: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    """Return a model-facing-polished copy without changing answer semantics."""

    polished = copy.deepcopy(record)
    task = str(polished.get("task_name") or "")
    if task not in _TASK_COPY:
        return polished
    answer_line = _formal_answer_line(polished)
    if task in _DIRECT_CHOICE_TASKS:
        option_label = FOUR_CHOICE_CODEC.parse(answer_line)
        if option_label is None:
            raise ValueError(f"{task} reference answer is not a strict A-D choice delivery")
        answer_line = FOUR_CHOICE_CODEC.format(option_label)
    polished["answer"] = answer_line
    structured = _structured(polished)
    answer_schema = structured.get("answer_schema")
    if isinstance(answer_schema, dict):
        if task in _DIRECT_CHOICE_TASKS:
            answer_schema["format_template"] = FOUR_CHOICE_CODEC.schema
            answer_schema["parser_profile"] = "strict_option_letter_a_d"
        answer_schema["format_example"] = _answer_format_example(task, structured)
        structured["answer_schema"] = answer_schema
    glyph_inventory = _rendered_glyph_inventory(
        task,
        structured,
        option_labels=[str(value) for value in polished.get("option_labels") or []],
    )
    structured["rendered_glyph_inventory"] = glyph_inventory
    polished["structured_answer"] = structured
    # Canonicalization deliberately removes authoring QA metadata from the
    # semantic verifier source, but preserves this top-level field in
    # ``extra`` for render review and provenance.
    polished["rendered_glyph_inventory"] = glyph_inventory
    glyph_scope = (
        "exact font-rendered strings across every formal input image "
        "(including independent visual option plates), authored reasoning "
        "states, terminal image, and reference video"
        if task in _DIRECT_CHOICE_TASKS
        else "exact font-rendered strings across the primary input, authored "
        "reasoning states, terminal image, and reference video"
    )
    polished["rendered_glyph_schema"] = {
        "schema_version": "icl_rendered_glyph_schema_v1",
        "scope": glyph_scope,
        "static_whitelist": (
            "only strings listed in rendered_glyph_inventory.static_whitelist"
        ),
        "bounded_dynamic_schema": (
            "only values admitted by rendered_glyph_inventory.bounded_dynamic_schema; "
            "observed_dynamic_values records the current sample"
        ),
        "exclusions": (
            "no task title, rule prose, answer banner, step narration, changing "
            "subtitle, renderer metadata, or hidden answer field"
        ),
    }
    for field, mode in (
        ("video_prompt", "ti2v"),
        ("ti2v_prompt", "ti2v"),
        ("solution_video_prompt", "ti2v"),
    ):
        if field in polished and polished.get(field) is not None:
            polished[field] = _prompt(
                task,
                mode,
                answer_line,
                _structured(polished),
                option_labels=[str(value) for value in polished.get("option_labels") or []],
            )


    contract = polished.get("task_contract")
    if isinstance(contract, dict):
        contract["prompt_authoring"] = "concise_input_rule_required_output_v2"
        contract["option_selection_visual"] = (
            "direct_solved_source_state_v1"
            if task in _DIRECT_CHOICE_TASKS
            else "not_applicable"
        )
        answer_contract = contract.get("answer")
        if isinstance(answer_contract, dict):
            if task in _DIRECT_CHOICE_TASKS:
                answer_contract["format_template"] = FOUR_CHOICE_CODEC.schema
                answer_contract["parser_profile"] = "strict_option_letter_a_d"
            answer_contract["format_example"] = _answer_format_example(task, structured)
            contract["answer"] = answer_contract
        polished["task_contract"] = contract
    _synchronize_task_contract(polished)
    return polished


def polished_icl_generator(cls: type) -> type:
    """Decorate a registered generator without changing its constructor API."""

    original_create = cls.create_puzzle
    declared_task = cls.__module__.split(".")[-2]

    def create_puzzle(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        # Keep each task's native solved-state renderer and temporal trace.
        # The family-local delivery adapter removes the candidate-card region
        # from those artifacts instead of replacing them with a selection
        # animation.
        record = original_create(self, *args, **kwargs)
        if not isinstance(record, dict):
            raise TypeError(f"{cls.__name__}.create_puzzle() must return a record dict")
        if declared_task in _DIRECT_CHOICE_TASKS:
            record = apply_direct_choice_delivery(record, output_dir=Path(self.output_dir))
        return polish_icl_record(record, output_dir=Path(self.output_dir))

    cls.create_puzzle = create_puzzle
    return cls


__all__ = ["polish_icl_record", "polished_icl_generator"]

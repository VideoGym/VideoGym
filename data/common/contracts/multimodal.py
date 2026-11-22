from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from data.common.prompts import (
    build_sectioned_prompt,
    remove_model_facing_serialization_recipes,
    soften_precise_video_timing,
)
from data.task_spec import CANONICAL_ANSWER_KINDS
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph

# Default noun slots for the shared TI2V continuity paragraph.  Contracts in
# this machinery all animate continuous motion, progressive drawing, or an
# element-local appearance commit, so ``continuous`` timing is the accurate
# family default; a contract overrides individual slots via ``ti2v_continuity``.
_TI2V_CONTINUITY_DEFAULTS = {
    "timing": "continuous",
    "moving_nouns": "the objects the rules change",
    "static_nouns": "everything the rules leave unchanged, including the labels, background, and camera",
    "hold_noun": "the final answer state",
}


ACTIVE_MODALITIES = ("ti2v",)
_FORBIDDEN_MODEL_MEDIA_RECIPE_RE = re.compile(
    r"(?:"
    r"\b(?:frame\s*count|frames?\s+\d+|\d+\s*[- ]?frames?|\d+(?:\.\d+)?\s*fps|"
    r"4\s*N\s*\+\s*1)\b|"
    # Seconds are often part of the task semantics (countdowns, travel time,
    # or a physics rule).  Reject them only when the surrounding language
    # makes them an output-media duration recipe.
    r"\b(?:video|animation|clip|runtime|media\s+duration)\b[^.\n]{0,48}"
    r"\b\d+(?:\.\d+)?\s*seconds?\b|"
    r"\b\d+(?:\.\d+)?\s*seconds?\b[^.\n]{0,48}"
    r"\b(?:video|animation|clip|runtime|long)\b|"
    r"\b(?:at\s+most|no\s+longer\s+than|lasting|lasts?|runs?\s+for)\s+"
    r"\d+(?:\.\d+)?\s*seconds?\b|"
    r"\b(?:hold|pause|display|freeze)\b[^.\n]{0,24}\bfor\s+"
    r"\d+(?:\.\d+)?\s*seconds?\b|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|sixteen)\s*-\s*frame\b|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|sixteen)\s+"
    r"(?:(?:identical|unchanged|readable|opening|closing|final)\s+){0,3}frames\b|"
    r"\b(?:single|one)\s*[- ]\s*frame\b|\b(?:one|single)\s+(?:abrupt|instantaneous|isolated)\s+frame\b|"
    r"\b(?:on|in)\s+(?:the\s+)?next\s+frame\b|\bframe\s+by\s+frame\b|\bper\s*[- ]\s*frame\b|"
    r"\b(?:allocate\s+motion\s+|duplicate(?:d)?\s+(?:the\s+)?(?:contact\s+)?|"
    r"several\s+(?:local\s+)?activation\s+|very\s+next\s+|first\s+moving\s+)frames?\b|"
    r"\b(?:px|pixels?)\b|\b(?:final|every|between)\s+frames?\b|"
    r"\bmedium(?:-|\s)+(?:width|thickness)\b|"
    r"\bsampled\b|\bgenerated\s+(?:instance|sample)\b|"
    r"\b(?:segments?|rays?|arcs?|lines?|circles?|outlines?|connectors?|extensions?|"
    r"altitudes?|tangents?|bisectors?|radii)\s+of\s+"
    r"(?:from|extending|centered|connecting|through)\b|"
    r"\b\d{3,4}\s*(?:by|x|×)\s*\d{3,4}\b"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerContract:
    kind: str
    format_template: str
    parser_profile: str
    verifier_profile: str
    format_example: str
    accepts_multiple_valid_answers: bool = False

    def validate(self) -> None:
        for field_name in ("format_template", "parser_profile", "verifier_profile"):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"Answer contract field '{field_name}' must be non-empty.")
        if self.kind not in CANONICAL_ANSWER_KINDS:
            raise ValueError(
                "Answer contract kind must use one canonical answer kind from "
                f"{CANONICAL_ANSWER_KINDS}; got {self.kind!r}."
            )
        if "Answer:" not in self.format_template:
            raise ValueError("Answer contract format_template must include the normalized 'Answer:' prefix.")
        if not str(self.format_example or "").strip():
            raise ValueError("Answer contract format_example must be non-empty.")
        if (
            not self.format_example.startswith("Answer: ")
            or not self.format_example.endswith(".")
            or "\n" in self.format_example
        ):
            raise ValueError(
                "Answer contract format_example must be one complete `Answer: <payload>.` line."
            )
        if "<" in self.format_example or ">" in self.format_example:
            raise ValueError(
                "Answer contract format_example must instantiate every symbolic field."
            )
        if self.format_example == self.format_template:
            raise ValueError(
                "Answer contract format_example must be concrete rather than repeat the format template."
            )


@dataclass(frozen=True)
class VisualReasoningContract:
    input_description: str
    symbol_legend: Sequence[str]
    rules: Sequence[str]
    modality_outputs: Mapping[str, str]
    preservation_constraints: Sequence[str]
    transition_constraints: Sequence[str]
    answer: AnswerContract
    render_profile: str
    active_modalities: Sequence[str] = ACTIVE_MODALITIES
    modality_input_descriptions: Mapping[str, str] = field(default_factory=dict)
    # Optional overrides for the shared TI2V continuity paragraph noun slots
    # (``timing``, ``moving_nouns``, ``static_nouns``, ``hold_noun``).
    ti2v_continuity: Mapping[str, str] = field(default_factory=dict)

    def prompt_modes(self) -> tuple[str, ...]:
        return (*tuple(str(mode) for mode in self.active_modalities), "solution_image")

    def validate(self) -> None:
        if not self.input_description.strip():
            raise ValueError("Visual reasoning contract requires an input description.")
        active_modalities = tuple(str(mode) for mode in self.active_modalities)
        if not active_modalities:
            raise ValueError("Visual reasoning contract requires at least one active modality.")
        if len(set(active_modalities)) != len(active_modalities):
            raise ValueError("Visual reasoning contract active modalities must be unique.")
        if any(mode not in ACTIVE_MODALITIES for mode in active_modalities):
            raise ValueError("Visual reasoning contract active modalities must be TI2V.")
        invalid_input_modes = [
            mode for mode in self.modality_input_descriptions if mode not in self.prompt_modes()
        ]
        if invalid_input_modes:
            raise ValueError(
                "Visual reasoning contract has input descriptions for unsupported modes: "
                + ", ".join(sorted(invalid_input_modes))
            )
        if any(not str(value).strip() for value in self.modality_input_descriptions.values()):
            raise ValueError("Modality-specific input descriptions must be non-empty.")
        invalid_continuity_keys = [
            key for key in self.ti2v_continuity if key not in _TI2V_CONTINUITY_DEFAULTS
        ]
        if invalid_continuity_keys:
            raise ValueError(
                "Visual reasoning contract has unknown ti2v_continuity slots: "
                + ", ".join(sorted(invalid_continuity_keys))
            )
        if any(not str(value).strip() for value in self.ti2v_continuity.values()):
            raise ValueError("ti2v_continuity slot overrides must be non-empty.")
        for field_name, values in (
            ("symbol_legend", self.symbol_legend),
            ("rules", self.rules),
            ("preservation_constraints", self.preservation_constraints),
        ):
            if not values or any(not str(value).strip() for value in values):
                raise ValueError(f"Visual reasoning contract field '{field_name}' must contain non-empty entries.")
        if "ti2v" in active_modalities and (
            not self.transition_constraints
            or any(not str(value).strip() for value in self.transition_constraints)
        ):
            raise ValueError("Visual reasoning contract field 'transition_constraints' must contain non-empty entries for TI2V.")
        missing_modes = [
            mode for mode in self.prompt_modes() if not str(self.modality_outputs.get(mode) or "").strip()
        ]
        if missing_modes:
            raise ValueError(f"Visual reasoning contract is missing output contracts for: {', '.join(missing_modes)}")
        if not self.render_profile.strip():
            raise ValueError("Visual reasoning contract requires a render profile.")
        self.answer.validate()

    def build_prompt(self, mode: str) -> str:
        self.validate()
        if mode not in self.prompt_modes():
            raise ValueError(f"Unsupported visual reasoning prompt mode: {mode}")

        authored_input_description = str(
            self.modality_input_descriptions.get(mode, self.input_description)
        ).strip()
        input_description = (
            f"{authored_input_description} "
            f"Symbol legend: {' '.join(str(item).strip() for item in self.symbol_legend)}"
        )
        primary_output = remove_model_facing_serialization_recipes(
            str(self.modality_outputs[mode]).strip()
        )
        required_parts = [primary_output]
        if mode in {"ti2v", "solution_image"}:
            preservation = " ".join(str(item).strip() for item in self.preservation_constraints)
            if preservation.lower().startswith(("keep ", "preserve ", "leave ")):
                required_parts.append(preservation)
            else:
                required_parts.append(f"Keep {preservation}")
        if mode == "ti2v":
            transitions = " ".join(str(item).strip() for item in self.transition_constraints)
            if transitions and transitions in primary_output:
                # The contract's TI2V output body already narrates the same
                # visual sequence; repeating it verbatim would state one
                # delivery twice.
                pass
            elif transitions.lower().startswith(
                ("begin ", "first ", "move ", "show ", "draw ", "grow ", "reveal ", "hold ")
            ):
                required_parts.append(transitions)
            else:
                required_parts.append(f"Follow this visual sequence: {transitions}")
        continuity_paragraph = None
        if mode == "ti2v":
            continuity_slots = dict(_TI2V_CONTINUITY_DEFAULTS)
            continuity_slots.update({key: str(value) for key, value in self.ti2v_continuity.items()})
            continuity_paragraph = build_ti2v_continuity_paragraph(**continuity_slots)
        return build_sectioned_prompt(
            input_description,
            " ".join(required_parts),
            mode=mode,
            rule_description=" ".join(str(item).strip() for item in self.rules),
            continuity_paragraph=continuity_paragraph,
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["active_modalities"] = list(self.active_modalities)
        if not self.modality_input_descriptions:
            # Preserve the pushed contract serialization for every task that
            # does not opt into modality-specific input prose.
            payload.pop("modality_input_descriptions", None)
        if not self.ti2v_continuity:
            payload.pop("ti2v_continuity", None)
        return payload


def audit_contract_prompts(
    contract: VisualReasoningContract,
    prompts: Mapping[str, str],
    *,
    answer_payload: str | None = None,
) -> list[str]:
    """Return deterministic violations for prompts built from this contract."""

    issues: list[str] = []
    try:
        contract.validate()
    except ValueError as exc:
        return [str(exc)]

    for mode in contract.prompt_modes():
        key = f"{mode}_prompt"
        if mode not in ACTIVE_MODALITIES and key not in prompts:
            continue
        prompt = str(prompts.get(key) or "").strip()
        if not prompt:
            issues.append(f"missing prompt: {key}")
            continue
        has_input_section = "Input Image:" in prompt
        has_output_section = "Required Output:" in prompt
        if not has_input_section or not has_output_section:
            issues.append(f"unsectioned prompt: {key}")
        has_rule_section = "Rule:" in prompt
        has_delivery = "Required Output:" in prompt
        if "Symbol legend:" not in prompt or not has_rule_section or not has_delivery:
            issues.append(f"incomplete executable specification: {key}")
        if mode in {"ti2v", "solution_image"}:
            preservation = " ".join(str(item).strip() for item in contract.preservation_constraints)
            # ``build_sectioned_prompt`` deliberately removes model-facing
            # media constants (canvas size, fps, frame count, duration).  The
            # contract audit must check the same normalized prose, otherwise a
            # correctly scrubbed prompt is rejected because its internal
            # authoring contract still mentions those runtime constants.
            preservation = soften_precise_video_timing(preservation)
            if preservation and preservation not in prompt:
                issues.append(f"missing preservation contract: {key}")
        if mode == "ti2v":
            transitions = " ".join(str(item).strip() for item in contract.transition_constraints)
            transitions = soften_precise_video_timing(transitions)
            if transitions and transitions not in prompt:
                issues.append(f"missing transition contract: {key}")
        if answer_payload and answer_payload.strip() and answer_payload.strip() in prompt:
            issues.append(f"answer leakage: {key}")
        if _FORBIDDEN_MODEL_MEDIA_RECIPE_RE.search(prompt):
            issues.append(f"model-facing media recipe: {key}")
        if len(re.findall(r"\b(?:fixed|static)\s+camera\b", prompt, flags=re.IGNORECASE)) > 1:
            issues.append(f"repeated fixed-camera constraint: {key}")
    return issues


__all__ = [
    "ACTIVE_MODALITIES",
    "AnswerContract",
    "VisualReasoningContract",
    "audit_contract_prompts",
]

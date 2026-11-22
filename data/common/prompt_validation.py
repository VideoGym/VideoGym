from __future__ import annotations

import re

from data.common.ti2v_continuity import looks_like_ti2v_continuity_paragraph


_SECTION_LABELS = ("Input Image:", "Rule:", "Required Output:")
# Hold/preservation/prohibition boilerplate is TI2V-only and lives in one
# unlabeled final paragraph.  These phrases identify that
# boilerplate wherever it appears: a prompt must contain none of
# them, and a TI2V prompt may contain them only inside its final paragraph.
_TI2V_CONTINUITY_MARKERS = (
    re.compile(r"\bdo\s+not\s+teleport\b", flags=re.IGNORECASE),
    re.compile(r"\bteleport-like\b", flags=re.IGNORECASE),
    re.compile(r"\bcross-?fade\b", flags=re.IGNORECASE),
    re.compile(r"\bflicker\w*\b", flags=re.IGNORECASE),
    re.compile(r"\bhard[- ]cut\b", flags=re.IGNORECASE),
    re.compile(r"\bmotion\s+blur\b", flags=re.IGNORECASE),
    re.compile(r"\bopacity\s+(?:lerp|blend\w*|fad\w*)\b", flags=re.IGNORECASE),
    re.compile(r"\bexact\s+discrete\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:pan,?\s+or\s+zoom|zoom,?\s+or\s+pan|no\s+zoom,?\s+no\s+pan)\b", flags=re.IGNORECASE),
    re.compile(r"\bbriefly\s+(?:hide|disappear|hidden)\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?:fixed|static)[- ]camera\b|\bcamera\s+(?:drift|motion)\b|\bdrift\s+the\s+camera\b",
        flags=re.IGNORECASE,
    ),
)
_INTERNAL_ROUTING = re.compile(
    r"\b(?:text_cot|interleaved_cot|teacher_cot|final_only_sft|SFT|RLVR|GDPO|TI2V|teacher trace|reasoning replay|parser-facing)\b",
    flags=re.IGNORECASE,
)
_MEDIA_CONFIGURATION = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*fps\b|\bframes?\s+per\s+second\b|"
    r"\b4\s*N\s*\+|\bframe\s+count\b|\bexactly\s+\d+\s+frames?\b|"
    r"\bframes?\s+\d+\b|\bat\s+most\s+10\s+seconds?\b|"
    r"\bno\s+longer\s+than\s+10\s+seconds?\b|"
    r"\b\d{3,4}\s*(?:x|×|by)\s*\d{3,4}\b|"
    r"\b(?:px|pixels?)\b|\bline\s+width\b|\bmedium(?:-|\s+)(?:thickness|width)\b|"
    r"\b(?:single|one)\s*[- ]\s*frame\b|\b(?:one|single)\s+(?:abrupt|instantaneous|isolated)\s+frame\b|"
    r"\b(?:on|in)\s+(?:the\s+)?next\s+frame\b|\bframe\s+by\s+frame\b|\bper\s*[- ]\s*frame\b|"
    r"\b(?:allocate\s+motion\s+|duplicate(?:d)?\s+(?:the\s+)?(?:contact\s+)?|"
    r"several\s+(?:local\s+)?activation\s+|very\s+next\s+|first\s+moving\s+)frames?\b|"
    r"\bpixel\s*[- ]\s*(?:identical|perfect|exact|level)\b|"
    r"\b(?:final|every|between)\s+frames?\b|"
    r"\brender(?:er|ing)\s+(?:tolerance|detail)\b)",
    flags=re.IGNORECASE,
)
_PARALLEL_INPUT = re.compile(
    r"\b(?:multiple|parallel|separate|provided|attached|additional)\s+"
    r"(?:input|reasoning|option|candidate)\s+images?\b",
    flags=re.IGNORECASE,
)
_REASONING_IMAGE_INPUT = re.compile(
    r"\b(?:multiple|parallel|separate|provided|attached|additional)?\s*reasoning\s+images?\b",
    flags=re.IGNORECASE,
)
_GENERIC_RULE_FALLBACK = re.compile(
    r"\bselect\s+the\s+unique\s+labeled\s+candidate\s+whose\s+center\s+satisfies\s+"
    r"the\s+stated\s+geometric\s+relation\b",
    flags=re.IGNORECASE,
)
_PIPELINE_PROVENANCE = re.compile(
    r"\bpublished\b|\bsampled\b|\bformal delivery\b|\bauthored reasoning\b|"
    r"\bgenerated\s+(?:sample|samples|instance|instances|input|inputs|puzzle|puzzles|board|boards|"
    r"state|states|source|sources|arrangement|arrangements|placement|placements|move|moves|output|outputs|"
    r"reasoning\s+media)\b|"
    r"\b(?:training|evaluation|test|data|sample)\s+distribution\b|"
    r"\brenderer-path\b",
    flags=re.IGNORECASE,
)
_GRAMMAR_DEBRIS = re.compile(
    r"\b(?:segments?|rays?|arcs?|lines?|circles?|outlines?|connectors?|extensions?|"
    r"altitudes?|tangents?|bisectors?|radii)\s+of\s+"
    r"(?:from|extending|centered|connecting|through)\b",
    flags=re.IGNORECASE,
)
_ANSWER_MARKER_ANTIPATTERNS = (
    re.compile(
        r"\bgreen\s+(?:answer|selection|correct-option)\s+(?:check|tick|mark|border)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bred\s+(?:answer|selection|wrong-option)\s+(?:x|cross|mark|border)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\btrace\s+(?:the\s+)?(?:correct|matching|selected)(?:\s+card)?\s+border\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bclockwise\s+around\s+the\s+(?:correct|matching|selected)\b", flags=re.IGNORECASE),
)
_COT_RECIPE = (
    re.compile(r"\bbefore each image,\s*(?:name|list|state|describe)\b", flags=re.IGNORECASE),
    re.compile(r"\bterminal text block\b[^.]*\b(?:state|list|name|explain)\b", flags=re.IGNORECASE),
    re.compile(r"\bfirst check\b[^.]*\bthen (?:calculate|inspect|check|derive)\b", flags=re.IGNORECASE),
    re.compile(r"\bprecede (?:the|each|every|that) image\b[^.]*\b(?:reasoning|text)\b", flags=re.IGNORECASE),
    re.compile(r"\bgive sample-specific reasoning\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:first|then|next),?\s+explain\b", flags=re.IGNORECASE),
    re.compile(r"\bleave (?:one|a) blank line\b", flags=re.IGNORECASE),
    re.compile(r"\bend (?:that|the|this|same) (?:reasoning|text)\b", flags=re.IGNORECASE),
)
_LOWERCASE_AFTER_SENTENCE = re.compile(r"(?<!\.)\.(?!\.)\s+([a-z][a-z-]*)")
_DOUBLE_DOT_RANGE = re.compile(r"(?<!\.)\.\.(?!\.)")
_ABBREVIATION_BEFORE_PERIOD = re.compile(r"(?:\bi\.e|\be\.g|\bvs)\.$", flags=re.IGNORECASE)
_REDUNDANT_SCHEMA_PERIOD = re.compile(
    r"(?:`Answer:[^`\n]*\.`\.|\"Answer:[^\"\n]*\.\"\.|'Answer:[^'\n]*\.'\.)"
)
_LEGACY_COORDINATE_PATH_SEPARATOR = re.compile(
    r"\([^()\n]{1,80}\)\s*\|\s*\([^()\n]{1,80}\)"
)
_LEGACY_ROW_COLUMN_PAIR_SCHEMA = re.compile(
    r"\(\s*<\s*(?:row|r)\s*>\s*,\s*<\s*(?:column|col|c)\s*>\s*\)",
    flags=re.IGNORECASE,
)
_CONCRETE_NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
_COMPACT_CONCRETE_COORDINATE_PAIR = re.compile(
    rf"\({_CONCRETE_NUMBER},{_CONCRETE_NUMBER}\)"
)
_COMPACT_COORDINATE_LIST_SEPARATOR = re.compile(
    rf"\({_CONCRETE_NUMBER}, {_CONCRETE_NUMBER}\),"
    rf"\({_CONCRETE_NUMBER}, {_CONCRETE_NUMBER}\)"
)
_PIPE_ALTERNATIVE_PLACEHOLDER = re.compile(r"<[^>\n]*\|[^>\n]*>")
def _plain_prose(text: str) -> str:
    """Remove inline schemas before checking ordinary sentence typography."""

    return re.sub(r"`[^`\n]*`", "<schema>", text)


def _has_lowercase_sentence_start(text: str) -> bool:
    for match in _LOWERCASE_AFTER_SENTENCE.finditer(_plain_prose(text)):
        prefix = _plain_prose(text)[: match.start() + 1].rstrip()
        if _ABBREVIATION_BEFORE_PERIOD.search(prefix):
            continue
        return True
    return False


def _input_image_description(text: str) -> str:
    """Return only the model-facing ``Input Image`` section body.

    Parallel-media wording is an input-contract concern. Scanning the whole
    Prompt would also match legitimate output duties in ``Required Output``.
    """

    input_marker = "Input Image:"
    rule_marker = "Rule:"
    start = text.find(input_marker)
    if start < 0:
        return ""
    start += len(input_marker)
    end = text.find(rule_marker, start)
    if end < 0:
        end = len(text)
    return text[start:end].strip()


def prompt_contract_violations(
    prompt: str,
    *,
    modality: str,
) -> list[str]:
    """Return model-facing Prompt contract violations.

    This checks serialization-independent rules only.  It intentionally does
    not judge task semantics; Family and Task audits remain responsible for
    whether the stated rule is complete and correct.
    """

    text = str(prompt or "").strip()
    mode = str(modality or "").strip().lower()
    issues: list[str] = []
    if mode not in {"ti2v"}:
        return [f"unsupported modality {modality!r}"]
    if not text:
        return ["empty prompt"]

    section_positions = []
    for label in _SECTION_LABELS:
        count = text.count(label)
        if count != 1:
            issues.append(f"{label[:-1]} section count is {count}, expected 1")
        section_positions.append(text.find(label))
        if count == 1:
            section_body = text.split(label, 1)[1].lstrip()
            if section_body and section_body[0].islower():
                issues.append(f"{label[:-1]} section starts with a lowercase word")
    if all(position >= 0 for position in section_positions) and section_positions != sorted(section_positions):
        issues.append("sections are not ordered Input Image -> Rule -> Required Output")

    if _INTERNAL_ROUTING.search(text):
        issues.append("contains internal training or routing language")
    if _MEDIA_CONFIGURATION.search(text):
        issues.append("contains internal media configuration")
    input_description = _input_image_description(text)
    if _REASONING_IMAGE_INPUT.search(input_description):
        issues.append("describes reasoning outputs as prompt inputs")
    elif _PARALLEL_INPUT.search(input_description):
        issues.append("describes parallel or reasoning-image inputs instead of the primary input")
    if _GENERIC_RULE_FALLBACK.search(text):
        issues.append("uses a generic geometric-relation fallback instead of an executable Rule")
    if _PIPELINE_PROVENANCE.search(text):
        issues.append("contains data-pipeline provenance or distribution language")
    if _GRAMMAR_DEBRIS.search(text):
        issues.append("contains malformed construction phrase composition")
    if len(re.findall(r"\b(?:fixed|static)\s+camera\b", text, flags=re.IGNORECASE)) > 1:
        issues.append("repeats the fixed-camera constraint")
    if any(pattern.search(text) for pattern in _ANSWER_MARKER_ANTIPATTERNS):
        issues.append("uses a nonstandard check/X/traced-border answer marker")
    if any(pattern.search(text) for pattern in _COT_RECIPE):
        issues.append("prescribes a task-specific chain-of-thought recipe")
    if _has_lowercase_sentence_start(text):
        issues.append("starts a sentence with a lowercase word")
    if _DOUBLE_DOT_RANGE.search(_plain_prose(text)):
        issues.append("uses double-dot range notation in prose")
    if _REDUNDANT_SCHEMA_PERIOD.search(text):
        issues.append("adds a redundant prose period after the Answer schema's terminal period")
    if _LEGACY_COORDINATE_PATH_SEPARATOR.search(text):
        issues.append("uses a pipe-separated coordinate path instead of one bracketed (x, y) list")
    if _LEGACY_ROW_COLUMN_PAIR_SCHEMA.search(text):
        issues.append("uses a row/column pair schema instead of the shared (x, y) grid-position notation")
    if _COMPACT_CONCRETE_COORDINATE_PAIR.search(text):
        issues.append("omits the required space after the comma in a concrete coordinate pair")
    if _COMPACT_COORDINATE_LIST_SEPARATOR.search(text):
        issues.append("omits the required space between consecutive coordinate pairs")
    if _PIPE_ALTERNATIVE_PLACEHOLDER.search(text):
        issues.append("uses an opaque pipe-alternative placeholder instead of naming the finite vocabulary in prose")

    if mode == "ti2v" and re.search(r'\bexact\s+format\s+"?Answer:', text, flags=re.IGNORECASE):
        issues.append("TI2V asks for a video-external typed answer")
    if mode == "ti2v":
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        final_block = blocks[-1] if blocks else ""
        body = "\n\n".join(blocks[:-1])
        if final_block.startswith(_SECTION_LABELS) or not looks_like_ti2v_continuity_paragraph(final_block):
            issues.append("TI2V prompt does not end with the shared unlabeled continuity paragraph")
        elif "Required Output:" not in body:
            issues.append("TI2V continuity paragraph is not preceded by a Required Output section")
        if any(pattern.search(body) for pattern in _TI2V_CONTINUITY_MARKERS):
            issues.append("TI2V continuity boilerplate appears outside the final continuity paragraph")
    return issues


def validate_model_prompt(
    prompt: str,
    *,
    modality: str,
) -> None:
    issues = prompt_contract_violations(prompt, modality=modality)
    if issues:
        raise ValueError(f"{modality} prompt contract failed: {'; '.join(issues)}")


__all__ = ["prompt_contract_violations", "validate_model_prompt"]

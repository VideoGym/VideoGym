from __future__ import annotations

import re
from typing import Optional

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
)
from data.common.typed_answers import ANSWER_SYNTAX_EXAMPLE_CUE


PATH_PUZZLE_INPUT_DESCRIPTION = (
    "The single input image shows the maze layout, the two red endpoints, and blue 1-based row and column numbers "
    f"along its edges. {GRID_XY_CONVENTION_SENTENCE}"
)

PATH_REASONING_INPUT_DESCRIPTION = (
    PATH_PUZZLE_INPUT_DESCRIPTION
)

def _normalize_prompt_text(prompt: Optional[str]) -> str:
    text = " ".join(str(prompt or "").split())
    return text.strip()


def _strip_trailing_video_sentences(text: str) -> str:
    result = _normalize_prompt_text(text)
    patterns = (
        r"\s*In portrait, static camera, no zoom, no pan\.\s*$",
        r"\s*In portrait\.\s*$",
        r"\s*Static camera perspective, no zoom or pan\.\s*$",
        r"\s*Static camera\.\s*$",
    )
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            updated, count = re.subn(pattern, "", result, flags=re.IGNORECASE)
            if count:
                result = updated.strip()
                changed = True
    return result


def soften_precise_video_timing(text: Optional[str]) -> str:
    result = _normalize_prompt_text(text)
    if not result:
        return ""
    # Remove compound media-contract phrases before generic ``N-frame``
    # rewrites can accidentally turn ``4N+1 frame count`` into ``4N+count``.
    result = re.sub(
        r"\b(?:using|with|whose\s+(?:total\s+)?frame\s+count\s+is|and)\s+(?:a\s+)?4\s*N\s*\+\s*1"
        r"(?:\s+frame\s+count|\s+frames?|\s*[- ]frame)?\b",
        "",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\b(?:of|for|lasting|no\s+longer\s+than|with\s+(?:a\s+)?duration\s+of|whose\s+duration\s+is)\s+"
        r"(?:at\s+most\s+)?\d+(?:\.\d+)?\s+seconds?\b",
        "",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\b(?:duration\s+)?at\s+most\s+\d+(?:\.\d+)?\s+seconds?\b",
        "",
        result,
        flags=re.IGNORECASE,
    )
    replacements = (
        (
            r"\b(?:for|over)\s+about\s+one\s+and\s+a\s+half\s+seconds\b",
            "smoothly",
        ),
        (
            r"\b(?:for|over)\s+about\s+[\d.]+\s+seconds\b",
            "briefly",
        ),
        (
            r"\b(?:for|over)\s+about\s+one\s+second\b",
            "briefly",
        ),
        (
            r"\bfor\s+one\s+second\b",
            "briefly",
        ),
        (
            r"\bfor\s+1\s+second\b",
            "briefly",
        ),
    )
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = re.sub(
        r",?\s*hence\s+\d+(?:\.\d+)?\s+seconds?\s+and\s+(?:a\s+)?valid\s+4\s*N\s*\+\s*1\s+length\s*[;,]?",
        "",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bover\s+frames?\s+\d+\s*[-–]\s*\d+\s*,?\s*",
        "during the next phase, ",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"^Use the supplied\s+\d+\s*(?:by|x|×)\s*\d+\s+input image as frame\s*1\s+and render exactly\s+\d+\s+frames at\s+\d+(?:\.\d+)?\s*fps with a fixed camera\.\s*",
        "",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bFrames?\s+\d+\s*[-–]\s*\d+\s+are\s+an?\s+exact unchanged input hold",
        "Begin with an exact unchanged input hold",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bFrames?\s+\d+\s*[-–]\s*\d+\s+are\s+(?:an?\s+)?pixel-exact\s+(?:unchanged\s+)?input hold",
        "Begin with a pixel-exact input hold",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bDuring frames?\s+\d+\s*[-–]\s*\d+\s*,?\s*",
        "during the next phase, ",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bFrames?\s+\d+\s*[-–]\s*\d+\s+hold\b",
        "Finally, hold",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bFrames?\s+\d+\s+(?:through|to)\s+\d+\b",
        "the next phase",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bFrames?\s+\d+\s*[-–]\s*\d+\s+are\s+",
        "The next phase is ",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\bFrames?\s+\d+\s*[-–]\s*\d+\s+",
        "Next, ",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\b(?:before|after|by)\s+frame\s+\d+\b", "outside its stated phase", result, flags=re.IGNORECASE)
    result = re.sub(r"\b(?:at\s+)?the\s+end\s+of\s+frame\s+\d+\b", "When that phase ends", result, flags=re.IGNORECASE)
    result = re.sub(r"\bframe\s+1\b", "the starting frame", result, flags=re.IGNORECASE)
    result = re.sub(r"\bframe\s+\d+\b", "the phase endpoint", result, flags=re.IGNORECASE)
    result = re.sub(r"\bexactly\s+\d+\s+frames\b", "a short sequence", result, flags=re.IGNORECASE)
    result = re.sub(r"\b(?:an?\s+)?ordered\s+\d+\s*[- ]frame\s+program\b", "an ordered visual program", result, flags=re.IGNORECASE)
    result = re.sub(r"\b\d+\s*[- ]frame\s+", "", result, flags=re.IGNORECASE)
    result = re.sub(
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|sixteen)\s*[- ]frame\s+"
        r"(?=(?:construction|placement|motion|turn|slot|sequence|hold|circle|arc|line|drawing|evidence|comparison|selection|border|phase|update|reveal)\b)",
        "",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|sixteen)\s+"
        r"(?:(?:identical|unchanged|readable|opening|closing|final)\s+){0,3}frames\b",
        "a readable hold",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\b\d+\s+frames\b", "the sequence", result, flags=re.IGNORECASE)
    result = re.sub(r"\b(?:all|throughout all)\s+the sequence\b", "throughout the full sequence", result, flags=re.IGNORECASE)
    result = re.sub(
        r"\b(?:the\s+)?(?:total\s+)?frame count must be\s+4\s*N\s*\+\s*1\b",
        "the sequence must remain continuous",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\b4\s*N\s*\+\s*1\b",
        "the internal sequence contract",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\b4\s*N\s*\+\s*count\b", "", result, flags=re.IGNORECASE)
    result = re.sub(
        r"\b(?:the\s+)?internal sequence contract(?:\s+frame count|\s+frames?|\s+sequence length)?\b",
        "",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\s+whose sequence length is the internal sequence contract and whose duration is at most\s+\d+(?:\.\d+)?\s+seconds\b",
        "",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\s+at\s+\d+(?:\.\d+)?\s*fps\b", "", result, flags=re.IGNORECASE)
    result = re.sub(r"\s+at\s+\d+(?:\.\d+)?\s*frames?\s+per\s+second\b", "", result, flags=re.IGNORECASE)
    result = re.sub(r"\s+at\s+the sequence\s+per\s+second\b", "", result, flags=re.IGNORECASE)
    result = re.sub(r"\b\d+(?:\.\d+)?\s*fps\b", "", result, flags=re.IGNORECASE)
    # Remove pixel-scale media resolutions without erasing load-bearing task
    # geometry such as a 3 by 3 block or an 8x8 board.
    result = re.sub(r"\b\d{3,4}\s*(?:by|x|×)\s*\d{3,4}\b", "", result, flags=re.IGNORECASE)
    result = re.sub(r"\bholds\s+briefly\b", "briefly holds", result, flags=re.IGNORECASE)
    result = re.sub(r"\bholds\s+smoothly\b", "holds", result, flags=re.IGNORECASE)
    result = re.sub(r"\bbriefly\s+briefly\b", "briefly", result, flags=re.IGNORECASE)
    result = re.sub(r"\bframe count\b", "sequence length", result, flags=re.IGNORECASE)
    result = re.sub(r"\bover\s+Next,\s*", "during the next phase, ", result, flags=re.IGNORECASE)
    result = result.replace(". during the next phase", ". During the next phase")
    # Normal prose never needs whitespace before punctuation, but an answer
    # placeholder such as ``[ID, ID, ...]`` intentionally has a space before
    # the ellipsis.  Do not collapse that typed-grammar example.
    result = re.sub(r"\s+([,;:]|\.(?!\.))", r"\1", result)
    result = re.sub(r"\b(?:video|sequence)\s+of\s+using\b", "video", result, flags=re.IGNORECASE)
    result = re.sub(r"\busing\s+(?=[,.;])", "", result, flags=re.IGNORECASE)
    result = re.sub(r"\bvideo\s+count\b", "video", result, flags=re.IGNORECASE)
    # These are useful implementation/audit terms but read like pipeline
    # instructions when surfaced to a model.  Keep the underlying semantic
    # requirement while using ordinary task language in every Prompt.
    result = re.sub(r"\bpixel[- ]exactly\b", "exactly", result, flags=re.IGNORECASE)
    result = re.sub(r"\bpixel[- ]exact\b", "unchanged", result, flags=re.IGNORECASE)
    result = re.sub(r"\bsolver ticks?\b", "reasoning states", result, flags=re.IGNORECASE)
    result = re.sub(r"\bkeyframes?\b", "states", result, flags=re.IGNORECASE)
    result = re.sub(r"\bterminal text\b", "final answer text", result, flags=re.IGNORECASE)
    result = re.sub(r"\bformal final delivery\b", "final delivery", result, flags=re.IGNORECASE)
    result = re.sub(r"\bsingle final visual deliverable\b", "final image", result, flags=re.IGNORECASE)
    result = re.sub(r"\bunchanged\s+unchanged\b", "unchanged", result, flags=re.IGNORECASE)
    result = re.sub(r"\s+(?:and|with|using)\s*([.;])", r"\1", result, flags=re.IGNORECASE)
    result = re.sub(r"\s{2,}", " ", result)
    return " ".join(result.split()).strip()


def remove_model_facing_serialization_recipes(text: Optional[str]) -> str:
    """Remove task-authored envelope/checklist prose from a model Prompt.

    Exact ``<think>``/``Answer:`` placement belongs to the shared phase wrapper
    and serializer. Task prompts may name meaningful visual states, but must
    not dictate a verifier-style narration before each image or restate the
    item-ordering contract.
    """

    normalized = _normalize_prompt_text(text)
    if not normalized:
        return ""
    sentences = _split_sentences(normalized)
    kept: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        # Preserve the visual state responsibility while removing an authored
        # narration recipe such as "First explain ... and show <state>".
        # Phase wrappers alone decide whether reasoning is emitted.
        sentence = re.sub(
            r"^(?:First|Then|Next),?\s+explain\b.*?\band\s+show\b",
            "Show",
            sentence,
            flags=re.IGNORECASE,
        )
        lowered = sentence.lower()
        if re.search(r"\bbefore (?:each|every|image\s*\d+) image\b", lowered):
            continue
        if re.search(r"^precede\b.*\b(?:reasoning|text)\b", lowered):
            continue
        if re.search(r"\bgive sample-specific reasoning\b", lowered):
            continue
        if re.search(r"\b(?:state|list)\b.*\b(?:compute|sum|compare|derive)\b", lowered) and (
            "answer" in lowered or "final line" in lowered or "shared clock" in lowered
        ):
            continue
        # Match whole words for ``terminal reasoning`` so phrases like
        # ``nonterminal reasoning image`` are preserved.
        if "terminal text block" in lowered or re.search(r"\bterminal reasoning\b", lowered):
            continue
        if "same text block" in lowered and ("answer" in lowered or "end" in lowered):
            continue
        if "blank line before" in lowered and "answer" in lowered:
            continue
        if "leave one blank line" in lowered and "answer" in lowered:
            continue
        if re.search(r"\b(?:append|emit|output|place)\b[^.]*\b(?:duplicate|another) answer block\b", lowered):
            continue
        if "exact parser-facing line" in lowered and ("then end" in lowered or "end the same" in lowered):
            continue
        kept.append(sentence)
    result = " ".join(kept).strip()
    return re.sub(
        r"\bFor example:\s*(?=[`\"']?Answer:)",
        ANSWER_SYNTAX_EXAMPLE_CUE + " ",
        result,
        flags=re.IGNORECASE,
    )


def _split_sentences(text: Optional[str]) -> list[str]:
    normalized = _normalize_prompt_text(text)
    if not normalized:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]


def strip_answer_instruction(prompt: Optional[str]) -> str:
    text = _normalize_prompt_text(prompt)
    if not text:
        return ""
    patterns = (
        r"\s*Answer with a single option(?: from [A-Z](?:-[A-Z])?)?\.\s*$",
        r"\s*Answer with exactly one option(?: from [A-Z](?:-[A-Z])?)?\.\s*$",
        r"\s*Answer with one option(?: from [A-Z](?:-[A-Z])?)?\.\s*$",
        r"\s*Answer with the final answer only\.\s*$",
        r"\s*Answer directly\.\s*$",
        r"\s*Must answer now without asking for clarifications\.\s*$",
    )
    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    result = " ".join(result.split()).strip()
    if result and not result.endswith("."):
        result += "."
    return result


def build_still_image_prompt_from_video_description(prompt: Optional[str]) -> str:
    text = strip_answer_instruction(prompt)
    if not text:
        return ""
    text = _strip_trailing_video_sentences(text)
    text = re.sub(
        r"\bThe video begins with a short hold on [^.]*?, and there is no construction line or extra shape added during the middle phase\.",
        "Do not add any construction line or extra shape.",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:The video|The animation) (?:holds|first holds|begins with|starts with) [^.]*?\bthen\b\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:The video|The animation)\s+starts\s+from\s+the unchanged [^.]*?\s+image and progressively draws\s+",
        "Draw ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:The video|The animation)\s+starts\s+from\s+the unchanged [^.]*?\s+image and progressively traces\s+",
        "Draw ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:The video|The animation)\s+starts\s+from\s+the unchanged [^.]*?\s+image and progressively renders\s+",
        "Render ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*After the route reaches the goal, the completed red line holds for one second(?: with the red endpoint cells still visible)?\.",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*When the route is complete, the frame holds for one second on the finished path\.",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*After the line reaches the center, the completed route holds for one second\.",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*The (?:solved|finished|completed) route briefly holds\.",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*The (?:solved|finished|completed) line briefly holds\.",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r",?\s*then briefly holds\.",
        ".",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*briefly holds\.",
        ".",
        text,
        flags=re.IGNORECASE,
    )

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    rewritten: list[str] = []
    for sentence in sentences:
        updated = sentence
        lowered = updated.lower()
        if lowered.startswith("draws "):
            updated = f"Draw {updated[6:]}"
        elif lowered.startswith("extends "):
            updated = f"Extend {updated[8:]}"
        elif lowered.startswith("reveals "):
            updated = f"Reveal {updated[8:]}"
        elif lowered.startswith("there is no construction line or extra shape"):
            updated = "Do not add any construction line or extra shape."
        elif lowered.startswith("the central mask smoothly narrows and disappears so "):
            updated = f"Remove the central mask so {updated[len('the central mask smoothly narrows and disappears so '):]}"
        elif lowered.startswith("the central mask narrows and disappears so "):
            updated = f"Remove the central mask so {updated[len('the central mask narrows and disappears so '):]}"
        else:
            progressive = re.match(r"^(a|an|one)\s+(.+?)\s+is progressively (drawn|traced)\s+(.*)$", updated, flags=re.IGNORECASE)
            if progressive:
                article = progressive.group(1)
                body = progressive.group(2)
                tail = progressive.group(4)
                updated = f"Draw {article} {body} {tail}".strip()

        final_prefixes = (
            "In the final revealed frame, ",
            "In the final frame, ",
            "In the final state, ",
        )
        for prefix in final_prefixes:
            if updated.startswith(prefix):
                updated = f"In the solved image, {updated[len(prefix):]}"
                break
        if updated.startswith("The final state simply changes "):
            updated = f"In the solved image, change {updated[len('The final state simply changes '):]}"

        if updated and updated[0].islower():
            updated = updated[0].upper() + updated[1:]
        rewritten.append(updated)

    return " ".join(rewritten).strip()


def split_visual_description_and_output(prompt: Optional[str]) -> tuple[str, str]:
    sentences = _split_sentences(prompt)
    if not sentences:
        return "", ""
    output_prefixes = (
        "the video ",
        "the animation ",
        "in the final ",
        "the final state ",
        "in the solved image",
        "draw ",
        "reveal ",
        "remove ",
        "extend ",
        "do not add ",
        "change ",
    )
    input_sentences: list[str] = []
    output_sentences: list[str] = []
    output_started = False
    for sentence in sentences:
        lowered = sentence.lower()
        is_output = lowered.startswith(output_prefixes)
        if "is progressively drawn" in lowered or "is progressively traced" in lowered:
            is_output = True
        normalized_sentence = sentence
        if lowered.startswith("in portrait"):
            normalized_sentence = re.sub(r"^In portrait,?\s*", "", sentence, flags=re.IGNORECASE).strip()
            normalized_sentence = re.sub(r"^[\s.,;:]+", "", normalized_sentence).strip()
            if normalized_sentence and normalized_sentence[0].islower():
                normalized_sentence = normalized_sentence[0].upper() + normalized_sentence[1:]
        is_camera = bool(normalized_sentence) and normalized_sentence.lower().startswith("static camera")
        if is_output:
            output_started = True
        if output_started or is_camera:
            if normalized_sentence:
                output_sentences.append(normalized_sentence)
        else:
            input_sentences.append(normalized_sentence)
    if not output_sentences:
        return " ".join(sentences), ""
    return " ".join(input_sentences).strip(), " ".join(output_sentences).strip()


def split_reasoning_description_and_output(prompt: Optional[str]) -> tuple[str, str]:
    sentences = _split_sentences(prompt)
    if not sentences:
        return "", ""
    output_prefixes = (
        "determine ",
        "identify ",
        "find ",
        "return ",
        "present ",
        "answer ",
        "mentally ",
        "what ",
        "which ",
        "first determine ",
    )
    input_sentences: list[str] = []
    output_sentences: list[str] = []
    output_started = False
    for sentence in sentences:
        lowered = sentence.lower()
        if lowered.startswith(output_prefixes):
            output_started = True
        if output_started:
            output_sentences.append(sentence)
        else:
            input_sentences.append(sentence)
    if not input_sentences:
        return "", " ".join(sentences).strip()
    return " ".join(input_sentences).strip(), " ".join(output_sentences).strip()


_REDUNDANT_ANSWER_SCHEMA_PERIOD = re.compile(
    r"((?:`|\"|')Answer:[^\n]{0,400}?\.(?:`|\"|'))\.(?=\s|$)"
)


def normalize_model_prompt_typography(prompt: Optional[str]) -> str:
    """Apply presentation-only cleanup after every Family prompt rewrite.

    The transformation is deliberately narrow: it capitalizes only the first
    alphabetic character of a declared section and removes only a redundant
    prose period that immediately follows an Answer schema whose exact period
    is already inside its quote/code delimiter.  Task semantics are untouched.
    """

    text = str(prompt or "").strip()
    for label in ("Input Image:", "Rule:", "Required Output:"):
        text = re.sub(
            rf"({re.escape(label)}\s*)([a-z])",
            lambda match: match.group(1) + match.group(2).upper(),
            text,
            count=1,
        )
    return _REDUNDANT_ANSWER_SCHEMA_PERIOD.sub(r"\1", text)


def build_sectioned_prompt(
    input_description: Optional[str],
    output_description: Optional[str],
    *,
    mode: str,
    input_section_label: str = "Input Image:",
    rule_description: Optional[str] = None,
    continuity_paragraph: Optional[str] = None,
) -> str:
    if continuity_paragraph is not None and mode != "ti2v":
        raise ValueError("continuity_paragraph applies only to ti2v prompts")
    input_text = soften_precise_video_timing(input_description)
    output_text = soften_precise_video_timing(output_description)
    rule_text = soften_precise_video_timing(rule_description)

    def polish_section_start(value: str) -> str:
        stripped = str(value or "").strip()
        if stripped and stripped[0].islower():
            stripped = stripped[0].upper() + stripped[1:]
        # The exact terminal period belongs inside the quoted/backticked
        # Answer schema.  A second prose period after the closing delimiter is
        # visually redundant and was widespread in model-facing prompts.
        stripped = _REDUNDANT_ANSWER_SCHEMA_PERIOD.sub(r"\1", stripped)
        return stripped

    input_text = polish_section_start(input_text)
    output_text = polish_section_start(output_text)
    rule_text = polish_section_start(rule_text)
    embedded_rule = re.match(
        r"^(?:Rules?|Rule):\s*(.+?)(?:\s+(?:Required output|Deliverable):\s*(.+))?$",
        output_text,
        flags=re.IGNORECASE,
    )
    if not rule_text and embedded_rule:
        rule_text = embedded_rule.group(1).strip()
        if embedded_rule.group(2):
            output_text = embedded_rule.group(2).strip()
    if mode not in {"ti2v", "solution_image"}:
        return " ".join(part for part in (input_text, output_text) if part).strip()

    normalized_label = input_section_label.replace("**", "").strip()
    if normalized_label.lower() == "input images:":
        normalized_label = "Input Image:"
    # The task-specific sections already describe the input, rules, and
    # delivery.  A generic preamble before ``Input Image:`` merely repeated
    # camera/CoT boilerplate and made otherwise clean prompts harder to read.
    # Phase wrappers remain responsible for the single think instruction.
    parts: list[str] = []
    if input_text:
        parts.append(f"{normalized_label} {input_text}")
    if rule_text:
        parts.append(f"Rule: {rule_text}")
    if output_text:
        parts.append(f"Required Output: {output_text}")
    if continuity_paragraph is not None:
        # The shared TI2V continuity paragraph is a separate, unlabeled final
        # block; it is never part of the Required Output section body.
        paragraph = " ".join(str(continuity_paragraph).split()).strip()
        if paragraph:
            parts.append(paragraph)
    return normalize_model_prompt_typography("\n\n".join(parts))


def build_point_target_prompt(reasoning_prompt: Optional[str], *, mode: str) -> str:
    base = strip_answer_instruction(reasoning_prompt)
    if not base:
        base = "Determine which candidate solves the puzzle."
    if mode == "ti2v":
        return (
            f"{base} Generate a short solution video that starts from the unchanged puzzle image, "
            "reveals only the minimal additional solution geometry needed by the solution, if any, and ends by "
            "highlighting only the correct candidate with one green outer border while the other candidates remain unchanged."
        )
    if mode == "solution_image":
        return (
            f"{base} Render the solved image by revealing only the minimal additional solution geometry needed by the "
            "solution, if any, and by highlighting only the correct candidate with one green outer border "
            "while the other candidates remain unchanged."
        )
    return base


def build_path_prompt(*, mode: str) -> str:
    if mode == "ti2v":
        return (
            "Rule: Follow only legal visible adjacency from the red start to the red goal without crossing a black wall or "
            "violating a task-specific mechanic. Required Output: Grow one continuous red route from the start through each "
            "visited cell to the goal in traversal order."
        )
    if mode == "solution_image":
        return (
            "Rule: Follow only legal visible adjacency from the red start to the red goal without crossing a wall or violating "
            "a task-specific mechanic. Required Output: Draw the complete red route on the labelled input and preserve every "
            "blue row and column number, wall, endpoint, mechanic glyph, and other unchanged element."
        )
    return ""

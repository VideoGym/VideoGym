"""Natural, sectioned model-facing prompts for constraint-grid records. This
normalizer is deliberately family-local. It preserves each task's authored
semantics while emitting one compact Input Image -> Rule -> Required Output
contract for every modality."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
)
from data.common.ti2v_continuity import (
    build_ti2v_continuity_paragraph,
    looks_like_ti2v_continuity_paragraph,
)


_LABEL_RE = re.compile(r"\*{0,2}(Input Image|Rules?|Required Output)\*{0,2}:\s*", re.IGNORECASE)
_WRAPPER_RE = re.compile(
    r"^(?:Use the provided input image[^.]*\.|Edit the puzzle image[^.]*\.)\s*",
    re.IGNORECASE,
)
# Per-task noun slots for the shared TI2V continuity paragraph.  Rebuilt tasks
# name their exact moving marks and scenery; the family default covers tasks
# that have not yet been through their forced-deduction rebuild.
_DEFAULT_CONTINUITY_NOUNS = {
    "moving_nouns": "the answer marks the rules add",
    "static_nouns": (
        "everything the rules leave unchanged, including the clues, boundaries, labels, "
        "background, and camera"
    ),
    "hold_noun": "the solved final state",
}
_TASK_CONTINUITY_NOUNS = {
    "aquarium_completion": {
        "moving_nouns": "the blue water fills and small gray dry rings being committed",
        "static_nouns": (
            "the tank borders, pastel tank shades, counts, edge numbers, already-given water and "
            "dry cells, background, and camera"
        ),
        "hold_noun": "the fully decided board",
    },
    "fillomino_completion": {
        "moving_nouns": "the dark digits being committed into blank cells",
        "static_nouns": (
            "the grid lines, navy given digits, edge numbers, background, and camera"
        ),
        "hold_noun": "the fully numbered grid",
    },
    "campsite_completion": {
        "moving_nouns": "the red tents being pitched",
        "static_nouns": (
            "the trees, boxed counts, edge numbers, already-pitched tents, ground cells, "
            "background, and camera"
        ),
        "hold_noun": "the completed campsite",
    },
    "light_up_akari_completion": {
        "moving_nouns": "the golden bulbs being committed and the warm tint of the cells they light",
        "static_nouns": (
            "the walls and their digits, edge numbers, already-lit bulbs and their light, "
            "background, and camera"
        ),
        "hold_noun": "the fully lit board",
    },
    "masyu_completion": {
        "moving_nouns": "the red loop segments being drawn",
        "static_nouns": (
            "the black and white circles, grid, edge numbers, already-drawn segments, "
            "background, and camera"
        ),
        "hold_noun": "the completed loop",
    },
    "maze_flow_pairs": {
        "moving_nouns": "the colored route segments being committed",
        "static_nouns": (
            "the endpoint dots, grid, edge numbers, already-drawn route segments, "
            "background, and camera"
        ),
        "hold_noun": "the fully routed board",
    },
    "nurikabe_completion": {
        "moving_nouns": "the dark sea shades and small gray island dots being committed",
        "static_nouns": (
            "the island numbers, edge numbers, already-shaded sea, already-dotted island cells, "
            "background, and camera"
        ),
        "hold_noun": "the fully decided board",
    },
    "nqueens_completion": {
        "moving_nouns": "the dark Q markers being committed",
        "static_nouns": (
            "the navy fixed queens, the checkered squares, edge numbers, background, and camera"
        ),
        "hold_noun": "the completed placement",
    },
    "shikaku_completion": {
        "moving_nouns": "the bold rectangle outlines being committed",
        "static_nouns": (
            "the grid lines, clue circles and their numbers, already-drawn outlines, edge "
            "numbers, background, and camera"
        ),
        "hold_noun": "the completed rectangle partition",
    },
    "slitherlink_completion": {
        "moving_nouns": "the red loop segments being drawn",
        "static_nouns": (
            "the lattice dots, digit clues, edge numbers, already-drawn segments, background, "
            "and camera"
        ),
        "hold_noun": "the completed loop",
    },
    "star_battle_completion": {
        "moving_nouns": "the gold stars being committed",
        "static_nouns": (
            "the colored regions and their boundaries, edge numbers, already-placed stars, "
            "background, and camera"
        ),
        "hold_noun": "the completed star placement",
    },
    "yajilin_completion": {
        "moving_nouns": "the dark cell shades and red loop segments being committed",
        "static_nouns": (
            "the boxed arrow clues, grid, edge numbers, already-shaded cells, already-drawn "
            "segments, background, and camera"
        ),
        "hold_noun": "the finished board",
    },
    "yin_yang_completion": {
        "moving_nouns": "the black and white stones being committed",
        "static_nouns": (
            "the parchment cells, edge numbers, ringed given stones, background, and camera"
        ),
        "hold_noun": "the fully stoned board",
    },
}


def _continuity_paragraph(task_name: str) -> str:
    nouns = _TASK_CONTINUITY_NOUNS.get(task_name, _DEFAULT_CONTINUITY_NOUNS)
    return build_ti2v_continuity_paragraph(timing="continuous", **nouns)


_ALL_CONTINUITY_PARAGRAPHS = tuple(
    {
        _continuity_paragraph(name)
        for name in list(_TASK_CONTINUITY_NOUNS) + ["__default__"]
    }
)

# Public typed grammars for every completion task. Keeping this exhaustive
# table at the Family boundary makes each schema independently auditable.
CONSTRAINT_GRID_TYPED_SCHEMAS = {
    "aquarium_completion": "Answer: [[<row values>], ...].",
    "fillomino_completion": "Answer: [[<row values>], ...].",
    "campsite_completion": "Answer: [(x1, y1), (x2, y2), ...].",
    "light_up_akari_completion": "Answer: [(x1, y1), (x2, y2), ...].",
    "masyu_completion": "Answer: [(x1, y1), (x2, y2), ...].",
    "maze_flow_pairs": (
        "Answer: <color_name>=[(x1, y1), (x2, y2), ...]; "
        "<color_name>=[(x1, y1), (x2, y2), ...]."
    ),
    "nqueens_completion": "Answer: [(x1, y1), (x2, y2), ...].",
    "nurikabe_completion": "Answer: [[<row values>], ...].",
    "shikaku_completion": "Answer: [((x1, y1), (x2, y2)), ...].",
    "slitherlink_completion": "Answer: [((x1, y1), (x2, y2)), ...].",
    "star_battle_completion": "Answer: [(x1, y1), (x2, y2), ...].",
    "yajilin_completion": "Answer: [(x1, y1), (x2, y2), ...].",
    "yin_yang_completion": "Answer: [[<row values>], ...].",
}

CONSTRAINT_GRID_TYPED_EXAMPLES = {
    "aquarium_completion": "Answer: [[0, 1, 1], [0, 0, 1], ...].",
    "fillomino_completion": "Answer: [[1, 2, 2], [3, 3, 3], ...].",
    "campsite_completion": "Answer: [(1, 2), (3, 4), ...].",
    "light_up_akari_completion": "Answer: [(1, 2), (3, 4), ...].",
    "masyu_completion": "Answer: [(1, 2), (1, 3), (2, 3), ...].",
    "nqueens_completion": "Answer: [(1, 2), (2, 4), (3, 1), ...].",
    "nurikabe_completion": "Answer: [[0, 1, 1], [0, 0, 1], ...].",
    "shikaku_completion": (
        "Answer: [((1, 1), (2, 3)), ((3, 1), (4, 2)), ...]."
    ),
    "slitherlink_completion": (
        "Answer: [((1, 1), (1, 2)), ((1, 2), (2, 2)), ...]."
    ),
    "star_battle_completion": "Answer: [(1, 2), (2, 4), (3, 1), ...].",
    "yajilin_completion": "Answer: [(1, 2), (1, 3), (2, 3), ...].",
    "yin_yang_completion": "Answer: [[0, 1, 1], [0, 0, 1], ...].",
}

def _clean(text: object) -> str:
    value = str(text or "").strip()
    value = value.replace("**Rules:**", "Rule:").replace("**Rule:**", "Rule:")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _dedupe_sentences(text: str) -> str:
    """Remove exact repeated prose without weakening task-specific semantics.

    A few legacy multimodal builders compose the solution-image instruction
    with the typed-answer instruction.  When both halves already contain the
    same delivery sentence, that composition produces verbatim repetition.
    Exact sentence deduplication is safe at the Family boundary: it never
    paraphrases a rule or merges merely similar constraints, and it is
    idempotent when generation annotates a record more than once.
    """

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(text or "").strip())
        if sentence.strip()
    ]
    seen: set[str] = set()
    kept: list[str] = []
    for sentence in sentences:
        key = re.sub(r"\s+", " ", sentence).casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append(sentence)
    return " ".join(kept).strip()


def _sections(text: object) -> Dict[str, str]:
    value = _clean(text)
    matches = list(_LABEL_RE.finditer(value))
    result: Dict[str, str] = {}
    for index, match in enumerate(matches):
        label = match.group(1).lower()
        key = "input" if label.startswith("input") else "rule" if label.startswith("rule") else "output"
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        body = value[match.end() : end].strip()
        if body:
            result[key] = body
    if not matches and value:
        result["body"] = value
    return result


def _split_embedded_rule(input_text: str) -> Tuple[str, str]:
    marker = re.search(r"\bRules?:\s*", input_text, flags=re.IGNORECASE)
    if marker:
        return input_text[: marker.start()].strip(), input_text[marker.end() :].strip()

    # Many old descriptions use the first sentence for visible content and
    # the remainder for executable constraints.  Keep that natural boundary.
    sentences = re.split(r"(?<=[.!?])\s+", input_text, maxsplit=1)
    if len(sentences) == 2:
        first, remainder = sentences[0].strip(), sentences[1].strip()
        if re.search(
            r"\b(?:must|may not|cannot|place|fill|draw|shade|complete|each row|each column|the number|the goal)\b",
            remainder,
            flags=re.IGNORECASE,
        ):
            return first, remainder
    return input_text.strip(), ""


def _required_body(raw_prompt: object) -> str:
    parsed = _sections(raw_prompt)
    body = parsed.get("output") or parsed.get("body") or ""
    body = _WRAPPER_RE.sub("", body).strip()
    # The shared phase serializer owns reasoning tags and output order.
    body = re.sub(
        r"\b(?:First|Then)\s+(?:render|generate|show)[^.]*\.\s*",
        "",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(r"^Then\s+", "", body, flags=re.IGNORECASE)
    return body.strip()


def normalize_constraint_grid_prompts(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a record whose three public prompts share one semantic rule."""

    normalized = dict(record)
    raw_prompts = {
        mode: _clean(record.get(f"{mode}_prompt"))
        for mode in ("ti2v",)
    }
    parsed = {mode: _sections(value) for mode, value in raw_prompts.items()}

    input_text = next(
        (parsed[mode].get("input", "") for mode in ("ti2v",) if parsed[mode].get("input")),
        "",
    )
    rule_text = next(
        (parsed[mode].get("rule", "") for mode in ("ti2v",) if parsed[mode].get("rule")),
        "",
    )
    if input_text and not rule_text:
        input_text, rule_text = _split_embedded_rule(input_text)
    if not input_text:
        input_text = (
            "The image shows one partially completed constraint grid. Visible numbers, symbols, marked cells, "
            "region boundaries, and row or column clues are fixed."
        )
    if not rule_text:
        raise ValueError("Constraint-grid prompt has no executable rule text")

    task_name = str(record.get("task_name") or "")
    input_text = input_text.replace(GRID_XY_CONVENTION_SENTENCE, "").strip()
    input_text = f"{input_text} {GRID_XY_CONVENTION_SENTENCE}".strip()
    continuity = _continuity_paragraph(task_name)
    for mode in ("ti2v",):
        output = _required_body(raw_prompts[mode])
        # Generation annotates worker records and merged records.  Remove the
        # family-owned suffix before reapplying it so normalization is exactly
        # idempotent rather than duplicating model-facing instructions.
        for paragraph in _ALL_CONTINUITY_PARAGRAPHS:
            output = output.replace(paragraph, "").strip()
        # A generator-level builder (e.g. the shared path pipeline) may already
        # have attached a continuity paragraph with its own noun slots; strip
        # any recognized paragraph so exactly one family-owned copy remains.
        paragraph_start = output.find("Begin with a readable hold of the unchanged input")
        if paragraph_start >= 0 and looks_like_ti2v_continuity_paragraph(output[paragraph_start:]):
            output = output[:paragraph_start].rstrip()
        output = _dedupe_sentences(output)

        prompt = (
            f"Input Image: {input_text}\n\n"
            f"Rule: {rule_text}\n\n"
            f"Required Output: {output}"
        )
        if mode == "ti2v":
            # The shared continuity paragraph is the one unlabeled final block
            # of every TI2V prompt; re-normalization stays exactly idempotent
            # because the paragraph was stripped from the parsed body above.
            prompt = f"{prompt}\n\n{continuity}"
        normalized[f"{mode}_prompt"] = prompt

    return normalized


__all__ = [
    "CONSTRAINT_GRID_TYPED_EXAMPLES",
    "CONSTRAINT_GRID_TYPED_SCHEMAS",
    "normalize_constraint_grid_prompts",
]

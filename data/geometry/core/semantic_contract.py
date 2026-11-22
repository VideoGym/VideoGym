"""Family-local semantic contracts for Geometry reference artifacts.

This module intentionally contains no training or evaluator routing.  It keeps
the Geometry generators from deriving prompts and visual reasoning states from
renderer configuration or an arbitrary number of drawing commands.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="continuous",
    moving_nouns="the construction elements the rules add or move",
    static_nouns="everything the rules leave unchanged, including the source geometry, labels, background, and camera",
    hold_noun="the completed final state",
)


_SPACE_RE = re.compile(r"\s+")
_ANSWER_RE = re.compile(r"(?:\n\s*)?Answer:\s*[A-Z](?:\s*,\s*[A-Z])*\.\s*$", re.IGNORECASE)


# Each tuple is the number of renderer commands that belong to one semantic
# construction phase.  A mismatch is a
# semantic-drift error: silently slicing or uniformly sampling a changed
# renderer trace would manufacture supervision.  This grouping never implies
# simultaneous TI2V drawing: video always executes the commands one by one.
SEMANTIC_PHASE_COMMAND_COUNTS: Dict[str, tuple[int, ...]] = {
    "incenter": (1, 1, 1, 2),
    "double_reflection": (1, 1),
    "double_rotation": (1, 1),
    "excircle_center": (2, 1),
    "external_angle_bisector": (1, 1),
    "glide_reflection": (1, 1),
    "incircle_tangent_point": (1, 1, 1, 2),
    "inscribed_angle_arc": (1, 2),
    "nine_point_center": (3, 6, 1),
}


# Semantic construction steps in rule order for multi-stage geometry tasks.
SEMANTIC_PHASE_ROLES: Dict[str, tuple[str, ...]] = {
    "circle_tangent_line": (
        "the radius from the center to the marked contact point",
        "that retained radius plus the perpendicular tangent and right-angle relation before candidate selection",
    ),
    "circle_tangent_point": (
        "the valid tangent from the external point to its circle contact",
        "that retained tangent plus the contact radius and right-angle relation before candidate selection",
    ),
    "circumcenter": (
        "the first side's perpendicular bisector",
        "the first two retained perpendicular bisectors",
        "all three retained perpendicular bisectors at their common intersection",
        "those retained bisectors plus the circumcircle before candidate selection",
    ),
    "double_reflection": (
        "the source reflected across the first mirror to its intermediate position",
        "that retained first reflection plus the second reflection at the composed destination",
    ),
    "double_rotation": (
        "the source rotated about the first marked pivot to its intermediate position",
        "that retained first rotation plus the second rotation at the composed destination",
    ),
    "excircle_center": (
        "both adjacent side extensions retained",
        "those retained extensions plus the excircle tangent to them and the opposite side before candidate selection",
    ),
    "external_angle_bisector": (
        "the internal bisector of the smaller visible angle",
        "that retained internal bisector plus the perpendicular external-bisector line before candidate selection",
    ),
    "glide_reflection": (
        "the source reflected across the marked axis to its intermediate position",
        "that retained reflection translated along the axis by the visible glide distance",
    ),
    "incenter": (
        "the first complete internal angle-bisector segment from a vertex through the incenter to the opposite side",
        "the first two retained complete internal angle-bisector segments",
        "all three retained complete internal angle-bisector segments through their common incenter",
        "those retained bisectors plus the center marker and incircle before candidate selection",
    ),
    "incircle_tangent_point": (
        "the first complete internal angle-bisector segment from a vertex through the incenter to the opposite side",
        "the first two retained complete internal angle-bisector segments",
        "all three retained complete internal angle-bisector segments through their common incenter",
        "those retained bisectors plus the center marker and incircle before tangent-point selection",
    ),
    "inscribed_angle_arc": (
        "the circle center marked without selecting a candidate",
        "both retained radii exposing the corresponding central angle before candidate selection",
    ),
    "isosceles_trapezoid": (
        "the shorter base drawn from V to X parallel to the longer base JU",
        "that retained base plus the closing leg from X to U, exposing UX equal to JV in the strictly convex quadrilateral",
    ),
    "nine_point_center": (
        "all three triangle altitudes",
        "those retained altitudes plus every side midpoint and altitude foot",
        "all retained construction points plus the nine-point circle before candidate selection",
    ),
    "orthocenter": (
        "the first altitude perpendicular to its completed opposite-side line",
        "the first two retained altitudes",
        "all three retained altitudes at their common intersection before candidate selection",
    ),
}


def _clean_spaces(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text or "")).strip()


def strip_media_configuration(text: str) -> str:
    """Remove renderer-only dimensions/timing while retaining geometric data."""

    value = str(text or "")
    value = re.sub(r"\b(?:one\s+)?\d{2,4}\s*(?:by|[x×])\s*\d{2,4}\b(?:\s*(?:pixel|pixels))?", "", value, flags=re.I)
    value = re.sub(r"\b(?:render|produce|create)\s+exactly\s+\d+\s+frames?\b", "render a continuous video", value, flags=re.I)
    value = re.sub(r"\b(?:at\s+)?\d+(?:\.\d+)?\s*fps\b", "", value, flags=re.I)
    value = re.sub(r"\bframes?\s+\d+(?:\s*[-–]\s*\d+)?\b", "the next phase", value, flags=re.I)
    value = re.sub(r"\bduring\s+the\s+next\s+\d+(?:\.\d+)?\s+seconds?\b", "during the next phase", value, flags=re.I)
    value = re.sub(r"\bfor\s+\d+(?:\.\d+)?\s+seconds?\b", "briefly", value, flags=re.I)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    return _clean_spaces(value)


def strip_reference_authoring_instructions(text: str) -> str:
    """Drop prompt text that prescribes a renderer/CoT audit transcript."""

    value = strip_media_configuration(text)
    # Task contracts may name these details for authoring and visual QA, but
    # they are renderer/provenance language rather than executable geometry.
    # Normalize them once at the Family boundary so every point-target task
    # receives the same concise model-facing wording.
    # Remove complete noun-phrase modifiers before the bare adjectives.  If
    # ``medium thickness`` is stripped first, phrases such as ``segment of
    # medium thickness from ...`` degrade into the ungrammatical ``segment of
    # from ...``.  Keep this Family-level instead of repairing every task.
    value = re.sub(
        r"\s+(?:of|with)\s+(?:a\s+)?medium\s+(?:thickness|width)\b",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"\bmedium(?:[- ](?:width|thickness|thin))\s*", "", value, flags=re.I)
    # Also heal prompts authored before the ordering fix.  This deliberately
    # targets only construction nouns followed by impossible ``of`` + verb or
    # preposition compositions, not legitimate geometry such as "arc of a
    # circle" or "line of symmetry".
    value = re.sub(
        r"\b(segments?|rays?|arcs?|lines?|circles?|outlines?|connectors?|extensions?|"
        r"altitudes?|tangents?|bisectors?|radii)\s+of\s+"
        r"(?=(?:from|extending|centered|connecting|through)\b)",
        r"\1 ",
        value,
        flags=re.I,
    )
    value = re.sub(r"\bthe\s+generator\s+guarantees\b", "the visible geometry ensures", value, flags=re.I)
    value = re.sub(r"\bgenerated\s+(?:instance|sample)\b", "supplied instance", value, flags=re.I)
    value = re.sub(r"\bsampled\b", "shown", value, flags=re.I)
    value = re.sub(r"\bin\s+the\s+final\s+frame\b", "in the terminal state", value, flags=re.I)
    value = re.sub(r"\bevery\s+frame\b", "the entire sequence", value, flags=re.I)
    value = re.sub(r"\bbetween\s+frames\b", "during the sequence", value, flags=re.I)
    value = re.sub(
        r"(?:^|\s)(?:In portrait,\s*)?(?:use\s+a\s+fixed|static)\s+camera"
        r"(?:\s+perspective)?(?:,?\s*no\s+zoom(?:\s+or\s+pan)?)?\.?(?=\s|$)",
        " ",
        value,
        flags=re.I,
    )
    sentences = re.split(r"(?<=[.!?])\s+", value)
    blocked_prefixes = (
        "respond with sample-specific reasoning",
        "then insert one blank line",
        "put no text before",
        "put no text after",
        "before each image",
        "generate an interleaved multimodal solution",
        "in the terminal step",
        "end the text immediately",
        "make sure every generated image",
        "provide the final text answer",
        "answer with one option",
    )
    kept = [
        sentence
        for sentence in sentences
        if sentence and not sentence.strip().lower().startswith(blocked_prefixes)
    ]
    return _clean_spaces(" ".join(kept))


def _extract_section(prompt: str, label: str) -> str:
    pattern = re.compile(
        rf"(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*:\s*(.*?)(?=\n\s*\n(?:\*\*)?[A-Z][^\n:]*(?:\*\*)?\s*:|\Z)",
        re.I | re.S,
    )
    match = pattern.search(str(prompt or ""))
    return _clean_spaces(match.group(1)) if match else ""


def _remove_delivery_clause(text: str) -> str:
    value = strip_reference_authoring_instructions(text)
    value = re.sub(
        r"\s*(?:Provide|Return|Finish|End|Answer)\s+(?:the\s+)?(?:final\s+)?(?:text\s+)?answer\b.*$",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"\s*Answer\s+with\s+one\s+option\b.*$", "", value, flags=re.I)
    return _clean_spaces(value)


def _normalize_construction_palette_language(text: str) -> str:
    """Name newly drawn solution strokes with the shared cyan palette."""

    value = str(text or "")
    pattern = re.compile(
        r"\b(draws?|drawing|drawn|extends?|extending)([^.!?]{0,52}?)\bblack\s+"
        r"(connector(?:\s+segments?)?|line(?:\s+segments?)?|segments?|lines?)\b",
        re.I,
    )
    value = pattern.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}cyan construction {match.group(3).lower()}"
        ),
        value,
    )
    broader = re.compile(
        r"\b(draws?|drawing|drawn|traces?|tracing|extends?|extending|constructs?|constructing)"
        r"([^.!?]{0,88}?)\b(?:black|dark[- ]gr[ae]y)\b"
        r"(?=(?:\s+[\w-]+){0,5}\s+(?:mark|ray|segments?|lines?|arcs?|medians?|altitudes?|"
        r"diagonals?|connectors?|extensions?|tangents?|bisectors?|sides?|outlines?|circles?|radii|radius)\b)",
        re.I,
    )
    value = broader.sub(
        lambda match: f"{match.group(1)}{match.group(2)}cyan construction",
        value,
    )
    return _clean_spaces(value)


def build_point_prompts(
    *,
    task_name: str,
    input_description: str,
    rule_description: str,
    solution_description: str,
    video_description: str,
) -> Dict[str, str]:
    """Build compact prompts with one shared semantic rule across modalities."""

    input_text = strip_reference_authoring_instructions(input_description)
    if not input_text:
        input_text = "The supplied image contains the complete visible geometry and labeled candidate markers."
    rule_text = _remove_delivery_clause(rule_description)
    if not rule_text:
        raise ValueError(
            "Geometry point prompts require a task-specific executable Rule; "
            "a generic geometric-relation fallback is not permitted"
        )

    prefix = f"Input Image:\n{input_text}\n\nRule:\n{rule_text}"

    solution_text = _extract_section(solution_description, "Required output") or solution_description
    solution_text = strip_reference_authoring_instructions(solution_text)
    solution_text = _normalize_construction_palette_language(solution_text)
    if not solution_text:
        solution_text = "Preserve the input and mark only the selected candidate in the terminal image."

    video_text = _extract_section(video_description, "Required output") or video_description
    video_text = strip_reference_authoring_instructions(video_text)
    video_text = _normalize_construction_palette_language(video_text)
    if not video_text:
        video_text = "Preserve the input scene, then commit the solved construction and selected candidate."

    return {
        "solution_image": f"{prefix}\n\nRequired Output:\n{solution_text}",
        "ti2v": (
            f"{prefix}\n\nRequired Output:\n{video_text} "
            "Draw every new line, polyline, arc, and circle one stroke at a time. Reveal each line or arc "
            "continuously from one fixed endpoint to the other, and draw a complete circle continuously from one "
            "fixed starting point in one direction until it closes; never grow a stroke from both ends, fade a "
            "complete stroke into view, or draw multiple strokes at once. Show each semantic construction or "
            "commit in rule order."
            f"\n\n{_TI2V_CONTINUITY}"
        ),
    }


def semantic_phase_roles(task_name: str) -> tuple[str, ...]:
    """Return the explicit non-terminal image duties for one Geometry task."""

    roles = SEMANTIC_PHASE_ROLES.get(str(task_name))
    if roles is None or not roles:
        raise RuntimeError(f"{task_name}: visual construction phases require explicit semantic prompt roles")
    if len(roles) > 6:
        raise RuntimeError(f"{task_name}: {len(roles)} non-terminal phases exceed the six-phase limit")
    normalized = tuple(_clean_spaces(role) for role in roles)
    if any(not role for role in normalized) or len(set(normalized)) != len(normalized):
        raise RuntimeError(f"{task_name}: semantic prompt roles must be non-empty and distinct")
    blocked = re.compile(
        r"(?:Answer:|text_cot|interleaved_cot|teacher_cot|final_only_sft|\bSFT\b|\bRLVR\b|\bpixels?\b|\bfps\b|\bframes?\b|\b\d+(?:\.\d+)?\s+seconds?\b)",
        re.I,
    )
    if any(blocked.search(role) for role in normalized):
        raise RuntimeError(f"{task_name}: semantic prompt role leaks an answer or pipeline/media term")
    return normalized


def semantic_command_groups(task_name: str, commands: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group renderer commands into semantic phases without implying TI2V synchrony."""

    counts = SEMANTIC_PHASE_COMMAND_COUNTS.get(task_name)
    if counts is None:
        if commands:
            raise RuntimeError(
                f"{task_name}: visual reasoning commands exist but no semantic phase contract is declared"
            )
        return []
    if sum(counts) != len(commands):
        raise RuntimeError(
            f"{task_name}: semantic phase contract expects {sum(counts)} commands, got {len(commands)}"
        )
    groups: List[List[Dict[str, Any]]] = []
    offset = 0
    for count in counts:
        groups.append(list(commands[offset : offset + count]))
        offset += count
    return groups


def canvas_region_phrase(x: Any, y: Any, *, width: float, height: float) -> str:
    """Name a location on a top-left-origin canvas in reader terms.

    Shared by authored reasoning across scene-counting tasks so location
    language stays uniform; thirds partition the canvas into a 3x3 grid.
    """

    column = min(2, max(0, int(float(x) * 3.0 / float(width))))
    row = min(2, max(0, int(float(y) * 3.0 / float(height))))
    horizontal = ("left", "middle", "right")[column]
    vertical = ("upper", "middle", "lower")[row]
    if horizontal == "middle" and vertical == "middle":
        return "near the center"
    if horizontal == "middle":
        return f"in the {vertical} middle"
    if vertical == "middle":
        return f"at the middle {horizontal}"
    return f"in the {vertical} {horizontal}"


def _human_join(values: Sequence[str]) -> str:
    items = [str(value) for value in values]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"

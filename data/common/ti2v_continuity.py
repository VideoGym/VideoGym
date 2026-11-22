"""Shared TI2V continuity paragraph builder.

Every TI2V prompt ends with one shared hold/preservation/prohibition paragraph
appended by ``build_sectioned_prompt`` after ``Required Output:``:

* ``"discrete"`` — the state space is discrete (cell-to-cell steps or turn-based
  commits), so the video switches between exact complete states of equal duration
  with no in-between rendering;
* ``"continuous"`` — the state evolves continuously (trajectories, sliding,
  drawing, folding) or commits as a single element-local fade.
"""

from __future__ import annotations

TIMING_DISCRETE = "discrete"
TIMING_CONTINUOUS = "continuous"
_TIMINGS = (TIMING_DISCRETE, TIMING_CONTINUOUS)

_DISCRETE_SENTENCE = (
    "Show every change as exact complete states: hold each state for the same duration, then switch directly "
    "to the next, never rendering in-between positions, motion blur, or opacity blending, and commit every "
    "moving object, attached marker, counter, and dependent world change together at each switch."
)


def build_ti2v_continuity_paragraph(
    *,
    timing: str,
    moving_nouns: str,
    static_nouns: str,
    hold_noun: str,
    allow_rule_teleports: bool = False,
) -> str:
    """Build the one shared final paragraph of a TI2V prompt.

    ``moving_nouns`` names what the sample moves or updates, ``static_nouns``
    names what must stay fixed (scenery, labels, background, camera), and
    ``hold_noun`` names the terminal state the video must end on.  All three
    are plain lowercase noun phrases from the calling task or family; this
    builder adds no task-specific vocabulary of its own.
    """

    mode = str(timing or "").strip().lower()
    if mode not in _TIMINGS:
        raise ValueError(f"timing must be one of {_TIMINGS}, got {timing!r}")
    moving = " ".join(str(moving_nouns or "").split())
    static = " ".join(str(static_nouns or "").split())
    hold = " ".join(str(hold_noun or "").split())
    if not moving or not static or not hold:
        raise ValueError("moving_nouns, static_nouns, and hold_noun must all be non-empty")

    sentences = [
        f"Begin with a readable hold of the unchanged input and end with a steady hold of {hold}.",
    ]
    if mode == TIMING_DISCRETE:
        sentences.append(_DISCRETE_SENTENCE)
    sentences.append(f"Keep {static} fixed throughout.")
    motion_prohibition = "Do not teleport, duplicate, resize,"
    if allow_rule_teleports:
        sentences.append("Teleport only through the portals explicitly defined by the Rule.")
        motion_prohibition = "Do not duplicate, resize,"
    sentences.append(
        f"{motion_prohibition} swap the identities of, or briefly hide {moving}; do not skip, "
        "reorder, or repeat an update; and do not hard-cut, crossfade the full frame, flicker, pan, or zoom."
    )
    return " ".join(sentences)


def looks_like_ti2v_continuity_paragraph(text: str) -> bool:
    """Recognize a paragraph produced by :func:`build_ti2v_continuity_paragraph`.

    Used by the prompt validator to require exactly one continuity paragraph
    as the final block of every TI2V prompt.  The check is structural, not a
    byte comparison, so noun slots stay free.
    """

    value = " ".join(str(text or "").split())
    return (
        value.startswith("Begin with a readable hold of the unchanged input")
        and ("Do not teleport, duplicate," in value or
             ("Teleport only through the portals explicitly defined by the Rule." in value and "Do not duplicate, resize," in value))
        and value.endswith("crossfade the full frame, flicker, pan, or zoom.")
    )


__all__ = [
    "TIMING_CONTINUOUS",
    "TIMING_DISCRETE",
    "build_ti2v_continuity_paragraph",
    "looks_like_ti2v_continuity_paragraph",
]

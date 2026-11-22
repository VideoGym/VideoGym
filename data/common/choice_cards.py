"""Explicit input-media policies for tasks with independent answer cards. The default data contract remains a single primary input image. A generator
must opt in with one of the policy names below before canonical serialization
will consume modality-specific input image lists."""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence


VISUAL_CHOICE_CARDS_POLICY = "visual_choice_cards_v1"
TEXT_CHOICE_CARDS_POLICY = "text_choice_cards_v1"
CHOICE_CARD_INPUT_POLICIES = frozenset(
    {VISUAL_CHOICE_CARDS_POLICY, TEXT_CHOICE_CARDS_POLICY}
)


def consecutive_option_labels(count: int) -> list[str]:
    """Return the stable A-based label vocabulary for a bounded card set."""

    if not 2 <= int(count) <= 26:
        raise ValueError("independent option-card sets must contain 2 to 26 cards")
    return [chr(ord("A") + index) for index in range(int(count))]


def normalize_option_labels(values: Sequence[object]) -> list[str]:
    """Normalize and validate a complete ordered option-label vocabulary."""

    labels = [str(value).strip().upper() for value in values]
    if labels != consecutive_option_labels(len(labels)):
        raise ValueError(
            "option-card labels must be the consecutive ordered vocabulary "
            f"A-{chr(ord('A') + len(labels) - 1) if labels else '?'}"
        )
    return labels


def attach_visual_choice_card_inputs(
    record: MutableMapping[str, Any],
    *,
    problem_image: str,
    option_images: Sequence[str],
    option_labels: Sequence[object],
    ti2v_problem_image: str | None = None,
) -> None:
    """Attach the visual-card input bundle to a record."""

    labels = normalize_option_labels(option_labels)
    option_paths = [str(path) for path in option_images]
    if len(option_paths) != len(labels) or any(not path for path in option_paths):
        raise ValueError("visual card records require one non-empty image path per label")
    primary = str(problem_image)
    video_primary = str(ti2v_problem_image or problem_image)
    if not primary or not video_primary:
        raise ValueError("visual card records require problem-only input images")
    record["input_media_policy"] = VISUAL_CHOICE_CARDS_POLICY
    record["option_labels"] = labels
    record["option_image_paths"] = option_paths
    record["ti2v_input_images"] = [video_primary]


def attach_text_choice_card_inputs(
    record: MutableMapping[str, Any],
    *,
    problem_image: str,
    option_labels: Sequence[object],
    ti2v_problem_image: str | None = None,
) -> None:
    """Attach single-image inputs when the finite options live in Prompt text."""

    labels = normalize_option_labels(option_labels)
    primary = str(problem_image)
    video_primary = str(ti2v_problem_image or problem_image)
    if not primary or not video_primary:
        raise ValueError("text card records require problem-only input images")
    record["input_media_policy"] = TEXT_CHOICE_CARDS_POLICY
    record["option_labels"] = labels
    record.pop("option_image_paths", None)
    record["ti2v_input_images"] = [video_primary]


__all__ = [
    "CHOICE_CARD_INPUT_POLICIES",
    "TEXT_CHOICE_CARDS_POLICY",
    "VISUAL_CHOICE_CARDS_POLICY",
    "attach_text_choice_card_inputs",
    "attach_visual_choice_card_inputs",
    "consecutive_option_labels",
    "normalize_option_labels",
]

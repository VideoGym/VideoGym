"""Shared final-delivery cleanup and option styling for rendered Geometry tasks."""

from __future__ import annotations

import re
from typing import Any, MutableMapping, Sequence

from PIL import ImageDraw

from data.geometry.core.semantic_contract import (
    strip_media_configuration,
)


ANSWER_GREEN = (45, 154, 86)
NEUTRAL_BORDER = (154, 160, 170)

def draw_answer_border(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    scale: int = 1,
    radius: int = 9,
    width: int = 5,
    progress: float = 1.0,
) -> None:
    """Draw one whole-card green border, with optional smooth color commit."""

    value = max(0.0, min(1.0, float(progress)))
    color = tuple(
        int(round(start + (end - start) * value))
        for start, end in zip(NEUTRAL_BORDER, ANSWER_GREEN)
    )
    scaled = tuple(int(round(float(item) * scale)) for item in bbox)
    draw.rounded_rectangle(
        scaled,
        radius=max(1, int(round(radius * scale))),
        fill=None,
        outline=color,
        width=max(1, int(round(width * scale))),
    )


def finalize_geometry_sample(sample: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Enforce Geometry's CoT/final boundary on a generated record.

    This is reference serialization QA, not candidate verification.  It never
    changes the semantic answer or evaluator inputs.
    """

    task_name = str(sample.get("task_name") or "")

    for key in ("ti2v_prompt", "solution_image_prompt"):
        value = sample.get(key)
        if isinstance(value, str):
            # Strip block by block: the shared TI2V continuity paragraph is a
            # separate final ``\n\n`` block and must survive whitespace cleanup.
            cleaned_prompt = "\n\n".join(
                strip_media_configuration(block)
                for block in value.split("\n\n")
                if strip_media_configuration(block)
            )
            section_counts = {
                label: len(re.findall(rf"(?i){re.escape(label)}\s*:", cleaned_prompt))
                for label in ("Input Image", "Rule", "Required Output")
            }
            if any(count != 1 for count in section_counts.values()):
                raise ValueError(
                    f"{task_name}: {key} must contain exactly one Input Image, Rule, and Required Output "
                    f"section; got {section_counts}"
                )
            sample[key] = cleaned_prompt
    return sample


__all__ = ["ANSWER_GREEN", "draw_answer_border", "finalize_geometry_sample"]

"""Modality-specific choice presentation for discrete dynamics tasks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.common.prompts import build_sectioned_prompt


def build_composite_mcq_prompts(
    *,
    input_desc: str,
    rule_summary: str,
    ti2v_output: str,
    solution_image_output: str = "",
    continuity_timing: str,
    continuity_moving: str,
    continuity_static: str,
    continuity_hold: str,
) -> Dict[str, str]:
    """Build the TI2V prompt for one A-D visual-choice dynamics task.

    The TI2V continuity paragraph nouns and timing are required arguments so
    every task supplies its own visible vocabulary.
    """
    _ = solution_image_output

    scene_description = str(input_desc or "").strip()
    scene_description = re.sub(
        r"^The\s+(?:(?:single|provided)\s+)?(?:input|problem)\s+image\s+"
        r"(?:shows|contains)\s+",
        "",
        scene_description,
        flags=re.IGNORECASE,
    )
    scene_description = re.sub(
        r"^The\s+image\s+(?:shows|contains)\s+",
        "",
        scene_description,
        flags=re.IGNORECASE,
    )
    if scene_description:
        scene_description = scene_description[:1].upper() + scene_description[1:]
        if scene_description[-1] not in ".?!":
            scene_description += "."
    video_input = (
        f"The single input image is the problem-only starting scene: {scene_description}"
    )
    return {
        "ti2v_prompt": build_sectioned_prompt(
            video_input,
            str(ti2v_output).strip(),
            mode="ti2v",
            rule_description=rule_summary,
            continuity_paragraph=build_ti2v_continuity_paragraph(
                timing=continuity_timing,
                moving_nouns=continuity_moving,
                static_nouns=continuity_static,
                hold_noun=continuity_hold,
            ),
        ),
    }


def finalize_choice_artifacts(
    *,
    output_dir: Path,
    artifacts: Dict[str, Any],
    option_tiles: Sequence[Any],
    correct_index: int,
    payload: Dict[str, Any],
    video_fps: int,
) -> Dict[str, Any]:
    """Keep native problem, terminal, keyframes, and video unchanged."""

    _ = output_dir, video_fps
    if len(option_tiles) != 4 or not 0 <= int(correct_index) < 4:
        raise ValueError("Dynamics visual choice input requires four terminal hypotheses")
    option_paths = artifacts.get("options_paths")
    if not isinstance(option_paths, list) or len(option_paths) != 4:
        raise ValueError("Dynamics artifacts must expose four saved A-D option images")
    payload["choice_input_layout"] = "problem_then_four_separate_option_images_v1"
    payload["ti2v_input_policy"] = "single_problem_only_image"
    payload["formal_visual_delivery"] = "direct_solved_scene_or_process"
    artifacts["option_image_paths"] = list(option_paths)
    artifacts["ti2v_input_images"] = [artifacts["puzzle_path"]]
    return artifacts


def attach_choice_input_fields(
    record: Any,
    *,
    artifacts: Mapping[str, Any],
    relativize: Any,
) -> Dict[str, Any]:
    """Serialize a task record and attach explicit choice-card input lists."""

    if isinstance(record, dict):
        result = dict(record)
    elif hasattr(record, "to_dict"):
        result = dict(record.to_dict())
    else:
        raise TypeError("Dynamics choice records must be dictionaries or expose to_dict()")

    def rel(path: Any) -> str:
        return str(relativize(Path(path)))

    ti2v_paths = list(artifacts.get("ti2v_input_images") or [artifacts["puzzle_path"]])
    if len(ti2v_paths) != 1:
        raise ValueError("Dynamics TI2V requires exactly one source image")
    result["input_media_policy"] = "visual_choice_cards_v1"
    result["ti2v_input_images"] = [rel(path) for path in ti2v_paths]
    return result


__all__ = [
    "attach_choice_input_fields",
    "build_composite_mcq_prompts",
    "finalize_choice_artifacts",
]

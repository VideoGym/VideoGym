"""Family-local Prompt contracts for Spatial Navigation tasks."""

from __future__ import annotations

from typing import Dict

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
)
from data.common.prompts import build_sectioned_prompt
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph


def build_spatial_navigation_prompt_bundle(
    *,
    task_name: str,
    input_description: str,
    rule_description: str,
    ti2v_output: str,
    moving_nouns: str = "the agent and every object the rules move",
) -> Dict[str, str]:
    """Build the model-facing TI2V prompt from one family-owned contract.

    Task code remains authoritative for objects, rules, state-event roles, and
    motion language.  This helper owns only the family-wide coordinate sentence,
    section layout, and the TI2V continuity wrapper.
    """

    task_name = str(task_name).strip()
    input_description = str(input_description).strip()
    rule_description = str(rule_description).strip()
    ti2v_output = str(ti2v_output).strip()
    if not task_name:
        raise ValueError("Spatial Navigation prompt construction requires a task name")
    fields = {
        "input description": input_description,
        "rule description": rule_description,
        "TI2V output": ti2v_output,
    }
    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise ValueError(f"{task_name}: empty Spatial Navigation Prompt field(s): {', '.join(missing)}")
    if GRID_XY_CONVENTION_SENTENCE not in input_description:
        input_description = f"{input_description} {GRID_XY_CONVENTION_SENTENCE}"

    ti2v_continuity = build_ti2v_continuity_paragraph(
        timing="discrete",
        moving_nouns=str(moving_nouns).strip(),
        static_nouns="everything the rules leave unchanged, including the grid, labels, background, and camera",
        hold_noun="the solved final state",
    )

    return {
        "ti2v_prompt": build_sectioned_prompt(
            input_description,
            ti2v_output,
            mode="ti2v",
            rule_description=rule_description,
            continuity_paragraph=ti2v_continuity,
        ),
    }


__all__ = ["build_spatial_navigation_prompt_bundle"]

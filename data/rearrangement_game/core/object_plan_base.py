"""Rearrangement-owned adapter for the general object-plan pipeline."""

from __future__ import annotations

from typing import Dict

from data.common.pipeline.object_plans import ObjectPlanGenerator
from data.rearrangement_game.core.prompt_contract import prompts_from_generator


class RearrangementObjectPlanGenerator(ObjectPlanGenerator):
    """Bind reusable object-plan mechanics to Rearrangement Prompt semantics."""

    def _build_task_prompts(self) -> Dict[str, str]:
        if self.TASK_FAMILY != "rearrangement_game":
            raise ValueError(
                f"{type(self).__name__} uses the Rearrangement Prompt adapter "
                f"but declares Task Family {self.TASK_FAMILY!r}"
            )
        return prompts_from_generator(self)


__all__ = ["RearrangementObjectPlanGenerator"]

"""Source-replay and formal final-delivery routing for maze_layered."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from .generator import MazeLayeredGenerator
from data.common.pipeline.path_tasks import PathEvaluationResult, PathTaskEvaluator
from data.common.verifiers.routes import VerifierRoute


class MazeLayeredEvaluator(PathTaskEvaluator):
    """Rehydrate both layers and staircase mappings for source replay."""

    RED_DOMINANCE = 80

    def _build_generator(self, record: Dict[str, Any]) -> MazeLayeredGenerator:
        rows, cols = record["grid_size"]
        width, _ = record["canvas_dimensions"]
        generator = MazeLayeredGenerator(
            output_dir=self.base_dir,
            canvas_width=width,
            rows=rows,
            cols=cols,
            staircase_count=max(1, len(record.get("staircase_positions", []))),
            seed=None,
            ti2v_prompt=str(record.get("ti2v_prompt") or ""),
            video=False,
        )
        generator.rehydrate_from_record(record)
        return generator


TASK_NAME = "maze_layered"
VERIFIER = VerifierRoute("data.spatial_navigation.core.path_verifier", "verify_path_trace")


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = ["MazeLayeredEvaluator", "PathEvaluationResult", "TASK_NAME", "VERIFIER", "evaluate"]

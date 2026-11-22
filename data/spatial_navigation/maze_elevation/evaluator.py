"""Source-replay and formal final-delivery routing for maze_elevation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from .generator import MazeElevationGenerator
from data.common.pipeline.path_tasks import PathEvaluationResult, PathTaskEvaluator
from data.common.verifiers.routes import VerifierRoute


class MazeElevationEvaluator(PathTaskEvaluator):
    """Rehydrate the visible elevation geometry for source replay."""

    RED_DOMINANCE = 80

    def _build_generator(self, record: Dict[str, Any]) -> MazeElevationGenerator:
        rows, cols = record["grid_size"]
        cell_size = record["cell_size"]
        width, height = record["canvas_dimensions"]
        generator = MazeElevationGenerator(
            output_dir=self.base_dir,
            canvas_width=width,
            aspect=width / height,
            size=cell_size,
            rows=rows,
            cols=cols,
            cell_size=cell_size,
            ti2v_prompt=str(record.get("ti2v_prompt") or ""),
            video=False,
        )
        generator.rehydrate_from_record(record)
        return generator


TASK_NAME = "maze_elevation"
VERIFIER = VerifierRoute("data.spatial_navigation.core.path_verifier", "verify_path_trace")


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = ["MazeElevationEvaluator", "PathEvaluationResult", "TASK_NAME", "VERIFIER", "evaluate"]

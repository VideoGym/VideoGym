"""Evaluator for Roman-aqueduct pipe-flow maze puzzles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from .generator import MazeAqueductPipesGenerator
from data.common.pipeline.path_tasks import PathEvaluationResult, PathTaskEvaluator
from data.common.verifiers.routes import VerifierRoute


class MazeAqueductPipesEvaluator(PathTaskEvaluator):
    """Provide record/asset resolution for the dedicated offline verifier.

    ``data.spatial_navigation.core.path_verifier`` owns the strict aqueduct
    contract: it independently recovers visible pipe kinds and the unique
    rotatable path, parses the exact ordered ``(x, y)`` answer, checks the
    semantic final image, and decodes the full rotation-then-flow video once.
    The inherited evaluator is retained for shared metadata loading, source-path
    resolution, and for rehydrating the renderer whose own tiles calibrate the
    verifier's sandstone-body floor.
    """

    def _build_generator(self, record: Dict[str, Any]) -> MazeAqueductPipesGenerator:
        width, height = record["canvas_dimensions"]
        return MazeAqueductPipesGenerator(
            output_dir=self.base_dir,
            canvas_width=int(width),
            aspect=(int(width) / int(height)) if int(height) else None,
            grid_size=int(record["grid_size"]),
            cell_size=int(record["cell_size"]),
            video=False,
        )


TASK_NAME = "maze_aqueduct_pipes"
VERIFIER = VerifierRoute("data.spatial_navigation.core.path_verifier", "verify_path_trace")


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    """Evaluate only the declared formal delivery through source replay."""

    return VERIFIER(samples, pred_root, modality=modality)


__all__ = ["MazeAqueductPipesEvaluator", "PathEvaluationResult", "TASK_NAME", "VERIFIER", "evaluate"]

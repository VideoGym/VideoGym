"""Source-replay and formal final-delivery routing for maze_toroidal."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from data.common.pipeline.path_tasks import PathTaskEvaluator
from data.common.verifiers.routes import VerifierRoute
from data.spatial_navigation.maze_toroidal.generator import MazeToroidalGenerator


class ToroidalMazeEvaluator(PathTaskEvaluator):
    """Rehydrate wrap-around geometry for source replay."""

    def _build_generator(self, record: Dict[str, Any]) -> MazeToroidalGenerator:
        generator = MazeToroidalGenerator(
            canvas_width=record["canvas_dimensions"][0],
            size=record.get("cell_size", 40),
            rows=record["grid_size"][0],
            cols=record["grid_size"][1],
            seed=record.get("seed"),
        )
        generator.rehydrate_from_record(record)
        return generator


TASK_NAME = "maze_toroidal"
VERIFIER = VerifierRoute("data.spatial_navigation.core.path_verifier", "verify_path_trace")


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = ["ToroidalMazeEvaluator", "TASK_NAME", "VERIFIER", "evaluate"]

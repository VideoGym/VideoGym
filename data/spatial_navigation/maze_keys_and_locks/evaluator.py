"""Source-replay and formal final-delivery routing for maze_keys_and_locks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from .generator import MazeKeysAndLocksGenerator
from data.common.pipeline.path_tasks import PathEvaluationResult, PathTaskEvaluator
from data.common.verifiers.routes import VerifierRoute


class MazeKeysAndLocksEvaluator(PathTaskEvaluator):
    """Rehydrate keys and gates while text replay enforces key-before-gate order."""

    RED_DOMINANCE = 80

    def _build_generator(self, record: Dict[str, Any]) -> MazeKeysAndLocksGenerator:
        rows, cols = record["grid_size"]
        width, height = record["canvas_dimensions"]
        generator = MazeKeysAndLocksGenerator(
            output_dir=self.base_dir,
            canvas_width=width,
            aspect=width / height,
            rows=rows,
            cols=cols,
            cell_size=int(record["cell_size"]),
            key_pairs=max(2, len(record.get("keys") or [])),
            ti2v_prompt=str(record.get("ti2v_prompt") or ""),
            video=False,
        )
        generator.rehydrate_from_record(record)
        return generator


TASK_NAME = "maze_keys_and_locks"
VERIFIER = VerifierRoute("data.spatial_navigation.core.path_verifier", "verify_path_trace")


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = ["MazeKeysAndLocksEvaluator", "PathEvaluationResult", "TASK_NAME", "VERIFIER", "evaluate"]

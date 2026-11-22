"""Source-replay and formal final-delivery routing for maze_hexagon."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import numpy as np

from .generator import MazeHexagonGenerator
from data.common.pipeline.path_tasks import PathEvaluationResult, PathTaskEvaluator
from data.common.verifiers.routes import VerifierRoute


class MazeHexagonEvaluator(PathTaskEvaluator):
    """Rehydrate the hex maze and close only raster gaps at slanted joins."""

    RED_DOMINANCE = 75
    CONNECTIVITY_OFFSETS = (
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )
    RED_MASK_CLOSE_KERNEL_SIZE = 3

    def _build_generator(self, record: Dict[str, Any]) -> MazeHexagonGenerator:
        width, height = record["canvas_dimensions"]
        return MazeHexagonGenerator(
            output_dir=self.base_dir,
            radius=record["radius"],
            cell_radius=record["cell_radius"],
            wall_thickness=record["wall_thickness"],
            canvas_width=width,
            aspect=width / height,
            ti2v_prompt=str(record.get("ti2v_prompt") or ""),
            video=False,
        )

    def _prepare_red_mask(self, red_mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((self.RED_MASK_CLOSE_KERNEL_SIZE,) * 2, dtype=np.uint8)
        refined = cv2.morphologyEx(red_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
        return refined.astype(bool)


TASK_NAME = "maze_hexagon"
VERIFIER = VerifierRoute("data.spatial_navigation.core.path_verifier", "verify_path_trace")


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = ["MazeHexagonEvaluator", "PathEvaluationResult", "TASK_NAME", "VERIFIER", "evaluate"]

"""Source-replay and formal final-delivery routing for maze_labyrinth."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .generator import MazeLabyrinthGenerator
from .answer_codec import parse_polar_path_answer
from data.common.pipeline.path_tasks import PathEvaluationResult, PathTaskEvaluator
from data.common.verifiers.routes import VerifierRoute


class MazeLabyrinthEvaluator(PathTaskEvaluator):
    """Rehydrate circular rings and segments for source replay."""

    RED_DOMINANCE = 75

    @staticmethod
    def parse_candidate_route(
        record: Dict[str, Any],
        prediction: str,
    ) -> tuple[Optional[list[int]], Optional[list[dict[str, object]]]]:
        """Parse the task-owned polar JSON and map it to source cell ids."""

        positions = parse_polar_path_answer(prediction)
        if positions is None:
            return None, None
        try:
            total_rings = int(record["total_rings"])
            segments = int(record["segments"])
        except (KeyError, TypeError, ValueError):
            return None, None
        if total_rings < 2 or segments < 1:
            return None, None
        position_to_cell_id = {(1, None): 1}
        for ring in range(2, total_rings + 1):
            for wedge in range(1, segments + 1):
                position_to_cell_id[(ring, wedge)] = (
                    2 + (ring - 2) * segments + (wedge - 1)
                )
        public_positions = [
            {"ring": ring, "wedge": wedge}
            for ring, wedge in positions
        ]
        cell_ids: list[int] = []
        for position in positions:
            cell_id = position_to_cell_id.get(position)
            if cell_id is None:
                return None, public_positions
            cell_ids.append(cell_id)
        return cell_ids, public_positions

    def _build_generator(self, record: Dict[str, Any]) -> MazeLabyrinthGenerator:
        width, height = record["canvas_dimensions"]
        return MazeLabyrinthGenerator(
            output_dir=self.base_dir,
            rings=record["rings"],
            segments=record["segments"],
            ring_width=record["ring_width"],
            wall_thickness=record["wall_thickness"],
            canvas_width=width,
            aspect=width / height,
            ti2v_prompt=str(record.get("ti2v_prompt") or ""),
            video=False,
        )


TASK_NAME = "maze_labyrinth"
VERIFIER = VerifierRoute("data.spatial_navigation.core.path_verifier", "verify_path_trace")


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> list[Any]:
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = ["MazeLabyrinthEvaluator", "PathEvaluationResult", "TASK_NAME", "VERIFIER", "evaluate"]

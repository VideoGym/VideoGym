"""Evaluator for color_group_with_locks."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional

from data.common.base import AbstractPuzzleEvaluator, PathLike
from data.rearrangement_game.core.answer_indexing import from_visible_plan


def _parse_moves(text: str) -> List[int]:
    if not text:
        return []
    payload = text
    if "answer:" in payload.lower():
        payload = payload[payload.lower().index("answer:") + len("answer:"):]
    bracket = re.search(r"\[.*\]", payload, flags=re.DOTALL)
    if bracket:
        try:
            parsed = ast.literal_eval(bracket.group(0))
            if isinstance(parsed, (list, tuple)):
                return [int(x) for x in parsed if isinstance(x, (int, float))]
        except Exception:
            pass
    return [int(x) for x in re.findall(r"\b\d+\b", payload)]


def _replay(initial: List[str], moves: List[int]) -> Optional[List[str]]:
    cur = list(initial)
    for i in moves:
        if not (0 <= i < len(cur) - 1):
            return None
        if cur[i] == "L" or cur[i + 1] == "L":
            return None
        cur[i], cur[i + 1] = cur[i + 1], cur[i]
    return cur


class ColorGroupWithLocksEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        text = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() and candidate_path.is_file() else str(candidate)
        submitted = _parse_moves(text)
        # The submitted payload counts slots from 1; replay is 0-based and an
        # out-of-domain ordinal (an old zero-based 0, for instance) fails closed.
        moves = from_visible_plan(TASK_NAME, submitted)
        initial = list(record.get("initial_state", []))
        target = list(record.get("target_state", []))
        result = _replay(initial, moves) if moves is not None else None
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": submitted,
            "valid": result is not None and result == target,
            "result_state": result,
        }


__all__ = ["ColorGroupWithLocksEvaluator"]

TASK_NAME = "color_group_with_locks"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]

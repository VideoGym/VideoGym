"""Evaluator for parking_lot_swap_minimal_moves."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional

from data.common.base import AbstractPuzzleEvaluator, PathLike

_INT_PATTERN = re.compile(r"\b(\d+)\b")


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
    return [int(x) for x in _INT_PATTERN.findall(payload)]


def _replay(initial: List, moves: List[int]) -> Optional[List]:
    cur = list(initial)
    for car_id in moves:
        try:
            ci = cur.index(car_id)
            ei = cur.index(None)
        except ValueError:
            return None
        if abs(ci - ei) != 1:
            return None
        cur[ei], cur[ci] = cur[ci], None
    return cur


class ParkingLotSwapMinimalMovesEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        text = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() and candidate_path.is_file() else str(candidate)
        moves = _parse_moves(text)
        initial = list(record.get("initial_state", []))
        target = list(record.get("target_state", []))
        result = _replay(initial, moves)
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": moves,
            "valid": result is not None and result == target,
            "result_state": result,
            "target_state": target,
        }


__all__ = ["ParkingLotSwapMinimalMovesEvaluator"]

TASK_NAME = "parking_lot_swap_minimal_moves"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]

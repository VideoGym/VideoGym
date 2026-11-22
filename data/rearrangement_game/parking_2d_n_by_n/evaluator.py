"""Evaluator for parking_2d_n_by_n's strict coordinate move list."""

from __future__ import annotations

import re
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike
from data.rearrangement_game.core.answer_indexing import from_visible_plan

ROWS = 3
COLS = 3


_MOVE_PATTERN = r"\([1-9][0-9]*, \([1-3], [1-3]\)\)"
_ANSWER_RE = re.compile(rf"Answer: \[({_MOVE_PATTERN}(?:, {_MOVE_PATTERN})*)\]\.")
_MOVE_RE = re.compile(r"\(([1-9][0-9]*), \(([1-3]), ([1-3])\)\)")


def _parse_moves(text: str) -> List[Tuple[int, Tuple[int, int]]]:
    if not isinstance(text, str):
        return []
    match = _ANSWER_RE.fullmatch(text)
    if match is None:
        return []
    return [
        (int(car_id), (int(x), int(y)))
        for car_id, x, y in _MOVE_RE.findall(match.group(1))
    ]


def _replay(initial: List, moves: List[Tuple[int, int]]) -> Optional[List]:
    cur = list(initial)
    for car_id, target_idx in moves:
        try:
            ci = cur.index(car_id)
        except ValueError:
            return None
        if not (0 <= target_idx < len(cur)):
            return None
        if cur[target_idx] is not None:
            return None
        cr, cc = ci // COLS, ci % COLS
        tr, tc = target_idx // COLS, target_idx % COLS
        if abs(cr - tr) + abs(cc - tc) != 1:
            return None
        cur[target_idx], cur[ci] = car_id, None
    return cur


class Parking2DNByNEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        text = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() and candidate_path.is_file() else str(candidate)
        submitted = _parse_moves(text)
        moves = from_visible_plan(TASK_NAME, submitted)
        initial = list(record.get("initial_state", []))
        target = list(record.get("target_state", []))
        result = _replay(initial, moves) if moves is not None else None
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": moves,
            "coordinate_domain_valid": moves is not None,
            "valid": result is not None and result == target,
            "result_state": result,
            "target_state": target,
        }


__all__ = ["Parking2DNByNEvaluator"]

TASK_NAME = "parking_2d_n_by_n"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]

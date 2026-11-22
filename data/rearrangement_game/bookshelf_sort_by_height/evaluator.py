"""Evaluator for bookshelf_sort_by_height.

A submitted swap names both slots from 1; replay below is 0-based. The single
conversion happens in :func:`_parse_moves`, which fails closed on any ordinal
outside the published 1..N domain rather than raising.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike
from data.rearrangement_game.core.answer_indexing import from_visible_plan

TASK_NAME = "bookshelf_sort_by_height"

_PAIR_PATTERN = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def _submitted_pairs(text: str) -> List[Tuple[int, int]]:
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
                return [(int(e[0]), int(e[1])) for e in parsed
                        if isinstance(e, (list, tuple)) and len(e) == 2]
        except Exception:
            pass
    return [(int(a), int(b)) for a, b in _PAIR_PATTERN.findall(payload)]


def _parse_moves(text: str) -> Optional[List[Tuple[int, int]]]:
    """Return 0-based swaps, or ``None`` when a submitted slot is out of domain."""

    recovered = from_visible_plan(TASK_NAME, _submitted_pairs(text))
    if recovered is None:
        return None
    return [(int(left), int(right)) for left, right in recovered]


def _replay(initial: List[int], moves: List[Tuple[int, int]]) -> Optional[List[int]]:
    cur = list(initial)
    n = len(cur)
    for i, j in moves:
        if not (0 <= i < n and 0 <= j < n) or i == j:
            return None
        cur[i], cur[j] = cur[j], cur[i]
    return cur


class BookshelfSortByHeightEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        text = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() and candidate_path.is_file() else str(candidate)
        moves = _parse_moves(text)
        initial = list(record.get("initial_state", []))
        target = list(record.get("target_state", []))
        result = _replay(initial, moves) if moves is not None else None
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": [[a, b] for a, b in moves] if moves is not None else None,
            "slot_domain_valid": moves is not None,
            "valid": result is not None and result == target,
            "result_state": result,
            "target_state": target,
        }


__all__ = ["BookshelfSortByHeightEvaluator"]

VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]

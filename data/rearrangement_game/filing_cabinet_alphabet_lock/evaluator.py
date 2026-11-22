"""Evaluator for filing_cabinet_alphabet_lock."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike

DRAWERS = {"A", "B", "C", "D", "E"}


def _parse_moves(text: str) -> List[Tuple[str, str]]:
    if not text:
        return []
    payload = text
    if "answer:" in payload.lower():
        payload = payload[payload.lower().index("answer:") + len("answer:"):]
    bracket = re.search(r"\[.*\]", payload, flags=re.DOTALL)
    if not bracket:
        return []
    try:
        parsed = ast.literal_eval(bracket.group(0))
    except Exception:
        return []
    out: List[Tuple[str, str]] = []
    if isinstance(parsed, (list, tuple)):
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                a, b = item
                if isinstance(a, str) and isinstance(b, str) and a in DRAWERS and b in DRAWERS:
                    out.append((a, b))
    return out


def _replay(initial: Dict[str, Any], moves: List[Tuple[str, str]]) -> Optional[Dict[str, List[int]]]:
    drawers = {d: list(initial["drawers"].get(d, [])) for d in DRAWERS}
    locked = set(initial.get("locked", []))
    for src, dst in moves:
        if src == dst:
            return None
        if src in locked or dst in locked:
            return None
        if not drawers[src]:
            return None
        top = drawers[src].pop()
        drawers[dst].append(top)
    return drawers


class FilingCabinetAlphabetLockEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        text = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() and candidate_path.is_file() else str(candidate)
        moves = _parse_moves(text)
        initial = record.get("initial_state", {})
        target = record.get("target_state", {})
        result = _replay(initial, moves)
        target_drawers = target.get("drawers", {}) if isinstance(target, dict) else {}
        valid = result is not None and all(list(result[d]) == list(target_drawers.get(d, [])) for d in DRAWERS)
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": moves,
            "valid": valid,
            "result_state": result,
        }


__all__ = ["FilingCabinetAlphabetLockEvaluator"]

TASK_NAME = "filing_cabinet_alphabet_lock"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]

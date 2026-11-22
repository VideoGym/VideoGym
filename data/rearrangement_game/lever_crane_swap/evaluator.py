"""Evaluator for lever_crane_swap."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional

from data.common.base import AbstractPuzzleEvaluator, PathLike

VALID = {"PL", "PR", "DL", "DR"}


def _parse_moves(text: str) -> List[str]:
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
                return [str(x).upper() for x in parsed if isinstance(x, str) and str(x).upper() in VALID]
        except Exception:
            pass
    return [m.upper() for m in re.findall(r"\b(PL|PR|DL|DR)\b", payload)]


def _replay(initial_state: List, moves: List[str]) -> Optional[Dict[str, Any]]:
    l = list(initial_state[0])
    r = list(initial_state[1])
    buf: Optional[int] = None
    last_pick: Optional[str] = None
    for mv in moves:
        if mv == "PL":
            if buf is not None or not l:
                return None
            if last_pick == "L":
                return None
            buf = l.pop()
            last_pick = "L"
        elif mv == "PR":
            if buf is not None or not r:
                return None
            if last_pick == "R":
                return None
            buf = r.pop()
            last_pick = "R"
        elif mv == "DL":
            if buf is None:
                return None
            l.append(buf)
            buf = None
        elif mv == "DR":
            if buf is None:
                return None
            r.append(buf)
            buf = None
        else:
            return None
    if buf is not None:
        return None
    return {"left": l, "right": r}


class LeverCraneSwapEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        text = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() and candidate_path.is_file() else str(candidate)
        moves = _parse_moves(text)
        initial = list(record.get("initial_state", [[], []]))
        target = list(record.get("target_state", [[], []]))
        result = _replay(initial, moves)
        valid = (
            result is not None
            and list(result["left"]) == list(target[0])
            and list(result["right"]) == list(target[1])
        )
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": moves,
            "valid": valid,
            "result_state": result,
        }


__all__ = ["LeverCraneSwapEvaluator"]

TASK_NAME = "lever_crane_swap"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]

"""Evaluator for the pancake_flip rearrangement task."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional

from data.common.base import AbstractPuzzleEvaluator, PathLike

_INT_PATTERN = re.compile(r"\b(\d+)\b")


def _parse_flips(text: str) -> List[int]:
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


def _flip_top(stack: List[int], k: int) -> Optional[List[int]]:
    if k < 2 or k > len(stack):
        return None
    out = list(stack)
    out[-k:] = list(reversed(out[-k:]))
    return out


class PancakeFlipEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        if candidate_path.exists() and candidate_path.is_file():
            text = candidate_path.read_text(encoding="utf-8")
        else:
            text = str(candidate)
        moves = _parse_flips(text)
        cur = list(record.get("initial_state", []))
        for k in moves:
            nxt = _flip_top(cur, k)
            if nxt is None:
                return {
                    "puzzle_id": puzzle_id,
                    "moves_parsed": moves,
                    "valid": False,
                    "result_state": cur,
                    "target_state": record.get("target_state"),
                    "error": f"invalid k={k} for stack length {len(cur)}",
                }
            cur = nxt
        target = list(record.get("target_state", []))
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": moves,
            "valid": cur == target,
            "result_state": cur,
            "target_state": target,
        }


__all__ = ["PancakeFlipEvaluator"]

TASK_NAME = "pancake_flip"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
